#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
import platform
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

from configs import get_config

from utils import (
    make_synthetic_binary,
    build_train_test_dmatrices
)

class IterTimer(xgb.callback.TrainingCallback):
    def __init__(self):
        self.iter_times = []
        self._t0 = None

    def before_training(self, model):
        self.iter_times.clear()
        self._t0 = None
        return model

    def after_iteration(self, model, epoch: int, evals_log: dict):
        t1 = time.perf_counter()
        if self._t0 is not None:
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

def summarize_env():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "omp_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

def main():
    parser = argparse.ArgumentParser(description="XGBoost CPU and single GPU timing benchmark")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    parser.add_argument("--summary-json", type=str, default="benchmark_summary.json")
    args = parser.parse_args()
    
    config = get_config()
 
    print("\nBenchmark configuration")
    # log these configurations
    print(json.dumps(vars(args), indent=2))

    # data generation
    t0 = time.perf_counter()
    X, y = make_synthetic_binary(
        n_samples=config.sample.n_samples,
        n_features=config.sample.n_features,
        seed=config.sample.random_state,
    )
    t1 = time.perf_counter()
    gen_time = t1 - t0
    print(f"\nsynthetic data generated in {gen_time:.3f} s")

    # DMatrix construction
    dtrain, dtest, times = build_train_test_dmatrices(
        X, y, test_pct=config.sample.test_size, seed=config.sample.random_state, gpu=(args.device == "gpu")
    )
    print(f"DMatrix construction time {times['dmatrix_train_s'] + times['dmatrix_test_s']:.3f} s")
    del X, y; gc.collect()  # encourage memory release if possible
   
    # params
    params = {**config.common, **(config.cpu if args.device == "cpu" else config.gpu)}
   
    # one step timing for warmup
    _, one_step_time, one_step_per_iter = run_training_once(
        params, dtrain, config.common.num_boost_round, do_one_step=True
    )

    print(f"\none boosting step wall time {one_step_time:.3f} s")

    # full training timing
    booster, full_total, per_iter_times = run_training_once(
        params, dtrain, config.common.num_boost_round, do_one_step=False
    )
    print(f"Full training wall time for {config.common.num_boost_round} rounds {full_total:.3f} s")

    if per_iter_times:
        print(f"First five per iteration times {per_iter_times[:5]}")

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
    print(f"test prediction time {tp1 - tp0:.3f} s | test accuracy {acc:.4f}")


    # report and persist summary
    summary = {
        "env": summarize_env(),
        "config": vars(args),
        "times": {
            "data_gen_s": gen_time,
            "one_step_train_s": one_step_time,
            "full_train_total_s": full_total,
            "per_iter_s": per_iter_times,
            "validation_logloss": float(acc),
            "predict_test": tp1 - tp0,
            "predict_train": trp1 - trp0,
            "train_logloss": float(train_logloss),
            "test_logloss": float(test_logloss),
        }
    }
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote summary to {args.summary_json}")

if __name__ == "__main__":
    main()
