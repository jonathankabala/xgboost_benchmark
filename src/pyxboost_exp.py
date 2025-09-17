import time, gc

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

from utils import (
    make_synthetic_binary,
    build_train_test_dmatrices,
)

class IterTimer(xgb.callback.TrainingCallback):
    def __init__(self): 
        self.iter_times = []; 
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
    
    # data generation
    t0 = time.perf_counter()
    X, y = make_synthetic_binary(
        n_samples=config.sample.n_samples,
        n_features=config.sample.n_features,
        seed=config.sample.random_state,
    )
    t1 = time.perf_counter()
    gen_time = t1 - t0

    # DMatrix construction
    dtrain, dtest, times = build_train_test_dmatrices(
        X, y, test_pct=config.sample.test_size, seed=config.sample.random_state, gpu=(args.device == "gpu")
    )

    del X, y; gc.collect()  # encourage memory release if possible

    params = {**config.common, **(config.cpu if args.device == "cpu" else config.gpu)}
    params.setdefault("tree_method", "hist")
    if args.device == "cpu":
        params["nthread"] = args.threads
        
    else:
        # we isolate each worker with CUDA_VISIBLE_DEVICES to one GPU.
        # inside the child process, the single visible device is ordinal 0.
        params["device"] = "cuda:0"           # or simply "cuda" in this isolated context

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
    # print(f"Full training wall time for {config.common.num_boost_round} rounds {full_total:.3f} s")

    # if per_iter_times:
    #     print(f"First five per iteration times {per_iter_times[:5]}")

    if args.device == "gpu":
        booster.set_param({"predictor": "gpu_predictor"})

    # Train loss
    trp0 = time.perf_counter()
    y_prob_train = booster.predict(dtrain)
    trp1 = time.perf_counter()
    train_logloss = log_loss(dtrain.get_label(), y_prob_train, labels=[0.0, 1.0])

    tp0 = time.perf_counter()
    y_prob = booster.predict(dtest)  # probabilities for binary:logistic
    tp1 = time.perf_counter()

    y_true = dtest.get_label()
    y_pred = (y_prob >= 0.5).astype(np.int32)
    test_logloss = log_loss(y_true, y_prob, labels=[0.0, 1.0])

    acc = accuracy_score(y_true, y_pred)
    # print(f"test prediction time {tp1 - tp0:.3f} s | test accuracy {acc:.4f}")

    avg_per_iter = np.mean(per_iter_times)
    std_per_iter = np.std(per_iter_times) if len(per_iter_times) > 1 else 0.0
    median_per_iter = np.median(per_iter_times) if len(per_iter_times) > 0 else 0.0


    num_rounds = config.common.num_boost_round
    boosts_per_sec_total = num_rounds / full_total
    boosts_per_sec_mean = 1.0 / avg_per_iter if avg_per_iter > 0 else 0.0
    boosts_per_sec_median = 1.0 / median_per_iter if median_per_iter > 0 else 0.0

    times = {
        "data_gen_s": gen_time,
        "one_step_train_s": one_step_time,
        "full_train_total_s": full_total,
        "per_iter_s": avg_per_iter,
        "per_iter_s_std": std_per_iter,
        "per_iter_s_median": median_per_iter,
        "boosts_per_sec_total": boosts_per_sec_total,
        "boosts_per_sec_mean": boosts_per_sec_mean,
        "boosts_per_sec_median": boosts_per_sec_median,
        "test_accuracy": float(acc),
        "predict_test": tp1 - tp0,
        "predict_train": trp1 - trp0,
        "train_logloss": float(train_logloss),
        "test_logloss": float(test_logloss),
    }


    # proactively release GPU memory at end of run
    del booster, dtrain, dtest
    gc.collect()

    return times