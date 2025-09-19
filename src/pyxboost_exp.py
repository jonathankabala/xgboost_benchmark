import gc
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from utils import Metrics, ensure_data_and_split, flatten_xgb_config


def get_dmatrix_cpu(py_dir_name, config):
    """
    returns dtrain, dtest for CPU training.
    creates & caches DMatrix binaries if not present; otherwise loads them quickly.
    """

    x_path = py_dir_name / "X.npy"
    y_path = py_dir_name / "y.npy"
    train_idx_path = py_dir_name / "train_idx.npy"
    test_idx_path = py_dir_name / "test_idx.npy"

    ensure_data_and_split(x_path, y_path, train_idx_path, test_idx_path, config)

    dtrain_bin_dir = py_dir_name / "dtrain.bin"
    dtest_bin_dir = py_dir_name / "dtest.bin"

    if dtrain_bin_dir.exists() and dtest_bin_dir.exists():
        dtrain = xgb.DMatrix(str(dtrain_bin_dir))
        dtest = xgb.DMatrix(str(dtest_bin_dir))
        return dtrain, dtest

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    train_idx = np.load(train_idx_path, mmap_mode="r")
    test_idx = np.load(test_idx_path, mmap_mode="r")

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx])

    del X, y, train_idx, test_idx
    gc.collect()

    dtrain.save_binary(str(dtrain_bin_dir))
    dtest.save_binary(str(dtest_bin_dir))
    return dtrain, dtest


def get_quantile_dmatrix_gpu(py_dir_name, config):
    """
    returns dtrain, dtest for GPU training using QuantileDMatrix.
    QuantileDMatrix cannot be saved/loaded; we reload data and build fresh each run.

    GPU training goes fast, so this is acceptable still.
    """

    x_path = py_dir_name / "X.npy"
    y_path = py_dir_name / "y.npy"
    train_idx_path = py_dir_name / "train_idx.npy"
    test_idx_path = py_dir_name / "test_idx.npy"

    # create data if not exist
    ensure_data_and_split(x_path, y_path, train_idx_path, test_idx_path, config)

    # i could make this faster by returning X, Y and the indices from ensure_data_and_split to avoid reloading from disk
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    train_idx = np.load(train_idx_path, mmap_mode="r")
    test_idx = np.load(test_idx_path, mmap_mode="r")

    dtrain = xgb.QuantileDMatrix(X[train_idx], label=y[train_idx])
    dtest = xgb.QuantileDMatrix(X[test_idx], label=y[test_idx])
    return dtrain, dtest


def get_py_data(config, args):

    py_dir_name = (
        Path(args.data_dir) / f"py_data" / f"samples_{config.sample.n_samples}"
    )
    py_dir_name.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.device == "gpu":
        dtrain, dtest = get_quantile_dmatrix_gpu(py_dir_name, config)
    else:
        dtrain, dtest = get_dmatrix_cpu(py_dir_name, config)
    t1 = time.perf_counter()
    data_time = t1 - t0

    return (dtrain, dtest), data_time


class IterTimer(xgb.callback.TrainingCallback):
    def __init__(self):
        self.iter_times = []
        self._t0 = None

    def before_training(self, model):
        self.iter_times.clear()
        self._t0 = time.perf_counter()
        return model

    def after_iteration(self, model, epoch: int, evals_log: dict):
        t1 = time.perf_counter()
        self.iter_times.append(t1 - self._t0)
        self._t0 = time.perf_counter()
        return False


def run_training_once(params, dtrain, num_boost_round, do_one_step=False):
    """

    run training for num_boost_round or one boosting step if do_one_step is True.

    params: dict of xgb.train params
    dtrain: training DMatrix or QuantileDMatrix (if on gpu)
    num_boost_round: int, number of boosting rounds to run if do_one_step is True.
    Returns:
        booster, total_time, per_iter_times

    """
    iter_timer = IterTimer()
    callbacks = [iter_timer]
    t0 = time.perf_counter()
    rounds = 1 if do_one_step else num_boost_round
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[],
        callbacks=callbacks,
    )
    t1 = time.perf_counter()
    total = t1 - t0
    per_iter = iter_timer.iter_times.copy()
    return booster, total, per_iter


def py_one_run(
    args,
    config,
):

    (dtrain, dtest), data_time = get_py_data(config, args)

    params = {**config.common, **(config.cpu if args.device == "cpu" else config.gpu)}

    params.setdefault("tree_method", "hist")  # in xgboost 2.0, gpu_hist is deprecated
    params["seed"] = config.sample.random_state
    params["nthread"] = (
        args.threads
    )  # still limit the number of gpu even if we are using gpu since some other operations are on cpu

    if args.device == "gpu":
        # we isolate each worker with CUDA_VISIBLE_DEVICES to one GPU.
        # inside the child process, the single visible device is ordinal 0.
        params["device"] = "cuda:0"  # or simply "cuda" in this isolated context

    # one step timing for warmup
    _, one_step_time, one_step_per_iter = run_training_once(
        params, dtrain, 5, do_one_step=False
    )

    # ----- measured run with GC disabled -----
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        booster, full_total, per_iter_times = run_training_once(
            params, dtrain, config.common.num_boost_round, do_one_step=False
        )
    finally:
        if gc_was_enabled:
            gc.enable()

    all_params = flatten_xgb_config(booster)

    y_prob_train = booster.predict(dtrain)
    train_logloss = log_loss(dtrain.get_label(), y_prob_train, labels=[0.0, 1.0])

    y_prob = booster.predict(dtest)  # probabilities for binary:logistic

    y_true = dtest.get_label()
    y_pred = (y_prob >= 0.5).astype(np.int32)
    test_logloss = log_loss(y_true, y_prob, labels=[0.0, 1.0])

    acc = accuracy_score(y_true, y_pred)

    avg_per_iter = np.mean(per_iter_times)
    std_per_iter = np.std(per_iter_times) if len(per_iter_times) > 1 else 0.0
    median_per_iter = np.median(per_iter_times) if len(per_iter_times) > 0 else 0.0

    boosts_per_sec_mean = 1.0 / avg_per_iter if avg_per_iter > 0 else 0.0

    times = Metrics(
        data_time_s=data_time,
        full_train_total_s=full_total,
        boost_step_estm_s=full_total / config.common.num_boost_round,
        n_boost_per_sec=config.common.num_boost_round / full_total,
        boost_step_avg_s=avg_per_iter,
        boost_step_s_std=std_per_iter,
        boost_step_s_median=median_per_iter,
        n_boost_per_sec_avg=boosts_per_sec_mean,
        test_accuracy=acc,
        train_logloss=train_logloss,
        test_logloss=test_logloss,
    ).to_dict()

    # proactively release GPU memory at end of run
    del booster, dtrain, dtest
    gc.collect()

    return times, all_params
