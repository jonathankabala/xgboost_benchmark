#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import psutil
import time
import platform
from pathlib import Path
from tqdm import tqdm
from multiprocessing import get_context


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb


from configs import get_config

from utils import (
    make_synthetic_binary,
    build_train_test_dmatrices,
    get_system_info
)


# def worker(q, args, config):
#     times = one_run(args=args, config=config)
#     q.put(times)

# def worker(q, args, config):
#     # ----- stability preamble -----
#     try:
#         import psutil, os
#         if getattr(args, "core_start", None) is not None:
#             cores = list(range(args.core_start, args.core_start + args.threads))
#             psutil.Process(os.getpid()).cpu_affinity(cores)
#             os.environ["GOMP_CPU_AFFINITY"] = " ".join(map(str, cores))
#     except Exception:
#         pass

#     os.environ.setdefault("OMP_PROC_BIND", "true")
#     os.environ.setdefault("OMP_PLACES", "cores")
#     os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
#     os.environ.setdefault("MALLOC_ARENA_MAX", "1")

#     # Optional: short, consistent pause between runs
#     time.sleep(0.3)
#     # --------------------------------
#     times = one_run(args=args, config=config)
#     q.put(times)


def worker(q, args, config):
    """
    CPU worker (respects core pinning and BLAS/OMP isolation).
    """
    if args.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("OMP_PROC_BIND", "true")
        os.environ.setdefault("OMP_PLACES", "cores")
        os.environ.setdefault("OMP_DYNAMIC", "false")
        os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
        os.environ.setdefault("MALLOC_ARENA_MAX", "1")
        if args.isolate_blas:
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        if getattr(args, "core_start", None) is not None and args.device == "cpu":
            cores = list(range(args.core_start, args.core_start + args.threads))
            psutil.Process(os.getpid()).cpu_affinity(cores)
            os.environ["GOMP_CPU_AFFINITY"] = " ".join(map(str, cores))
        if args.nice is not None:
            psutil.Process(os.getpid()).nice(args.nice)
    except Exception:
        pass

    time.sleep(0.2)  # tiny settle
    times = one_run(args=args, config=config)
    q.put(times)


def gpu_worker(q, args, config, gpu_id: int):
    """
    GPU worker. Hard-isolate to a single GPU via CUDA_VISIBLE_DEVICES.
    inside this process, that GPU is ordinal 0; we set params['device']='cuda:0'.
    """
    # stable device ordering + hard isolation
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # avoid cross-GPU fabric probing (we're single-GPU per process)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    # set niceness if requested
    try:
        if args.nice is not None:
            psutil.Process(os.getpid()).nice(args.nice)
    except Exception:
        pass

    # clone args minimally without mutating parent
    # we don't need argparse.Namespace here—just change the device view.
    class A: pass
    args_local = A()
    for k, v in vars(args).items():
        setattr(args_local, k, v)
    args_local.device = "gpu"

    time.sleep(0.2)
    times = one_run(args=args_local, config=config)
    q.put(times)


# class IterTimer(xgb.callback.TrainingCallback):
#     def __init__(self):
#         self.iter_times = []
#         self._t0 = None

#     def before_training(self, model):
#         self.iter_times.clear()
#         self._t0 = None
#         return model

#     def after_iteration(self, model, epoch: int, evals_log: dict):
#         t1 = time.perf_counter()
#         if self._t0 is not None:
#             self.iter_times.append(t1 - self._t0)
#         self._t0 = time.perf_counter()
#         return False
    
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

def summarize_env():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "omp_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

def one_run(
    args,
    config,
):
    
    # print("\nBenchmark configuration")
    # log these configurations
    # print(json.dumps(vars(args), indent=2))


    # data generation
    t0 = time.perf_counter()
    X, y = make_synthetic_binary(
        n_samples=config.sample.n_samples,
        n_features=config.sample.n_features,
        seed=config.sample.random_state,
    )
    t1 = time.perf_counter()
    gen_time = t1 - t0
    # print(f"\nsynthetic data generated in {gen_time:.3f} s")

    # DMatrix construction
    dtrain, dtest, times = build_train_test_dmatrices(
        X, y, test_pct=config.sample.test_size, seed=config.sample.random_state, gpu=(args.device == "gpu")
    )
    # print(f"DMatrix construction time {times['dmatrix_train_s'] + times['dmatrix_test_s']:.3f} s")
    del X, y; gc.collect()  # encourage memory release if possible

    # params
    # params = {**config.common, **(config.cpu if args.device == "cpu" else config.gpu)}
    
    # if args.device == "cpu":
    #     params["nthread"] = args.threads

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

def run_cpu_benchmark(args, config):
    if args.device == "cpu":
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    ctx = get_context("spawn")
    time_stats = []
    with tqdm(total=args.n_runs, desc="CPU runs", leave=False) as pbar:
        for _ in range(args.n_runs):
            q = ctx.Queue()
            p = ctx.Process(target=worker, args=(q, args, config))
            p.start()
            result = q.get()
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"Run failed with exit code {p.exitcode}")
            time_stats.append(result)
            pbar.update(1)

    return time_stats




def run_gpu_benchmark(args, config, num_gpus: int = 4):
    """
    run args.n_runs total, with up to num_gpus runs in parallel.
    rach run gets a unique GPU by masking visibility to that device.
    """
    ctx = get_context("spawn")
    time_stats = []
    in_flight = []  # list[(Process, Queue, gpu_id)]
    launched = 0
    next_gpu = 0

    def launch(gid):
        q = ctx.Queue()
        p = ctx.Process(target=gpu_worker, args=(q, args, config, gid))
        p.start()
        in_flight.append((p, q, gid))

    with tqdm(total=args.n_runs, desc="GPU runs", leave=False) as pbar:
        # prime up to num_gpus workers
        while launched < args.n_runs and len(in_flight) < num_gpus:
            launch(next_gpu % num_gpus)
            launched += 1
            next_gpu += 1

        # collect/refill loop
        while in_flight:
            p, q, gid = in_flight.pop(0)
            result = q.get()  # wait for that worker
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"GPU run on device {gid} failed with exit code {p.exitcode}")
            time_stats.append(result)
            pbar.update(1)

            if launched < args.n_runs:
                launch(next_gpu % num_gpus)
                launched += 1
                next_gpu += 1

    return time_stats

def main():

    parser = argparse.ArgumentParser(description="XGBoost CPU and single-GPU timing benchmark")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--summary-json", type=str, default="benchmark_summary.json")
    parser.add_argument("--n-runs", type=int, default=5, help="number of runs")
    parser.add_argument("--threads", type=int, default=5, help="threads per CPU run")

    parser.add_argument("--core-start", type=int, default=0,
                        help="first core index to pin the worker+threads (CPU only)")
    parser.add_argument("--isolate-blas", action="store_true",
                        help="force single-thread BLAS during DMatrix prep (CPU only)")
    parser.add_argument("--nice", type=int, default=None,
                        help="set process niceness (lower is higher priority)")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="number of GPUs to use concurrently for GPU mode")

    args = parser.parse_args()
    config = get_config()


    if args.device == "cpu":
        time_stats = run_cpu_benchmark(args, config)
    else:
        # run with up to args.n_gpus (i.e., 4) GPUs in parallel (configurable via --n-gpus)
        time_stats = run_gpu_benchmark(args, config, num_gpus=args.n_gpus)

    # report and persist summary
    summary = {
        "env": summarize_env(),
        "system": get_system_info(),
        "config": vars(args),
    }

    df = pd.DataFrame(time_stats)
    benchmark_detailed_file = f"{args.device}_benchmark_detailed_times.csv"
    df.to_csv(benchmark_detailed_file, index=False)

    with open(f"{args.device}_{args.summary_json}", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote summary to {args.summary_json}")

    print(f"\nwrote summary to {args.summary_json}")
    print(f"wrote per-run details to {benchmark_detailed_file}")

if __name__ == "__main__":
    main()