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
from datetime import datetime
from zoneinfo import ZoneInfo


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import h2o


from configs import get_config

from pyxboost_exp import py_one_run
from h2o_xgboost_exp import h2o_one_run
from utils import (
    get_system_info
)


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

    if args.experiment_type == "pyxgboost":
        times = py_one_run(args=args, config=config)
    elif args.experiment_type == "h2oxgboost":
        raise NotImplementedError("h2oxgboost benchmark not implemented yet")
    else:
        raise ValueError(f"unknown experiment type {args.experiment_type}")
    
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
    if args.experiment_type == "pyxgboost":
        times = py_one_run(args=args_local, config=config)
    elif args.experiment_type == "h2oxgboost":
        raise NotImplementedError("h2oxgboost benchmark not implemented yet")
    else:
        raise ValueError(f"unknown experiment type {args.experiment_type}")
    q.put(times)

def summarize_env():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "omp_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


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
    parser.add_argument("--experiment-type", choices=["pyxgboost", "h2oxgboost"], default="pyxgboost")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    parser.add_argument("--out-dir", type=str, default="logs_dev", help="output directory")
    parser.add_argument(
        "--load-data",
        action="store_true",
        help="load data instead of creating it"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="directory to load data from"
    )

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

    if args.experiment_type == "h2oxgboost":
        h2o.init()

    h2o_one_run(args, config)

    import ipdb
    ipdb.set_trace()
    
    if args.device == "cpu":
        time_stats = run_cpu_benchmark(args, config)
    else:
        # run with up to args.n_gpus (i.e., 4) GPUs in parallel (configurable via --n-gpus)
        time_stats = run_gpu_benchmark(args, config, num_gpus=args.n_gpus)
   
    austin_tz = ZoneInfo("America/Chicago")
    current_time = datetime.now(austin_tz)
    experiment_folder = f"experiment_{current_time.strftime('%Y%m%d_%H%M%S')}"

    args.out_dir = Path(args.out_dir) / args.experiment_type / experiment_folder
    args.out_dir.mkdir(parents=True, exist_ok=True) 

    df = pd.DataFrame(time_stats)
    benchmark_detailed_file = f"{args.out_dir}/{args.device}_benchmark_detailed_times.csv"
    df.to_csv(benchmark_detailed_file, index=False)

     # report and persist summary
    args.out_dir = str(args.out_dir) # since I am saving this to json, args.out_dir needs to be str not Path
    summary = {
        "env": summarize_env(),
        "system": get_system_info(),
        "config": vars(args),
    }

    summary_json = f"{args.out_dir}/{args.device}_benchmark_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote summary to {summary_json}")
    print(f"wrote per-run details to {benchmark_detailed_file}")

if __name__ == "__main__":
    main()