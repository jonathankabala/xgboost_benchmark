#!/usr/bin/env python3
import argparse
import gc
import json
import os
os.environ["H2O_JAR_PATH"] = "/home/h2o/h2o-3.46.0.7/h2o.jar"
import platform
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from zoneinfo import ZoneInfo

import h2o
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

from configs import get_config
from h2o_xgboost_exp import h2o_one_run
from pyxboost_exp import py_one_run
from utils import format_number, get_system_info, summarize_env


# def _ensure_h2o():
#     try:
#         # connects to a running cluster at default http://localhost:54321
#         h2o.connect()
#     except Exception:
#         # if nothing is running yet, start one
#         h2o.init(ip="localhost", port=54321)

def _init_h2o_isolated(gpu_id: int | None = None):
    """
    Start a fresh, private H2O cluster in this process with a port band that
    won't overlap with other workers. Space bands by 100 ports.
    """
    import tempfile, uuid, time as _time
    import h2o, os

    tmp_root = tempfile.mkdtemp(prefix=f"h2o_{os.getpid()}_")

    # Reserve wide, non-overlapping bands to avoid H2O's internal port scanning collisions.
    # GPU workers: fixed by gid (e.g., 55100, 55200, 55300, 55400)
    # CPU workers: put them in a separate band (56000+) if you ever run H2O on CPU.
    if gpu_id is not None:
        band_base = 55100 + gpu_id * 100
    else:
        band_base = 56000 + (os.getpid() % 10) * 100  # unlikely to collide

    name = f"bench_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    # H2O can take a couple seconds to come up; give it ample room.
    last_err = None
    for delta in range(0, 50):  # 50 ports in this worker's private band
        port = band_base + delta
        try:
            h2o.init(
                ip="127.0.0.1",
                port=port,
                name=name,
                ice_root=tmp_root,
                bind_to_localhost=True,
                strict_version_check=False,
                # min_mem_size="1g",
                # max_mem_size="4g",
            )
            # h2o.no_progress()
            return
        except Exception as e:
            last_err = e
            _time.sleep(0.25)  # give the JVM time to bind before trying next port
    raise RuntimeError(f"Failed to start isolated H2O cluster after retries: {last_err}")



def cpu_worker(q, args, config):
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

    # if args.experiment_type == "h2oxgboost":
    #     _ensure_h2o()  # i just want to connect to the parent cluster that I started with h2o.init()

    # if args.experiment_type == "pyxgboost":
    #     times, model_params = py_one_run(args=args, config=config)
    # elif args.experiment_type == "h2oxgboost":
    #     times, model_params = h2o_one_run(args=args, config=config)
    # else:
    #     raise ValueError(f"unknown experiment type {args.experiment_type}")

    # q.put((times, model_params))

    time.sleep(0.2)  # tiny settle

    try:
        if args.experiment_type == "h2oxgboost":
            # start a private H2O cluster just for THIS process
            _init_h2o_isolated(gpu_id=None)

        if args.experiment_type == "pyxgboost":
            times, model_params = py_one_run(args=args, config=config)
        elif args.experiment_type == "h2oxgboost":
            times, model_params = h2o_one_run(args=args, config=config)
        else:
            raise ValueError(f"unknown experiment type {args.experiment_type}")

        q.put((times, model_params))
    finally:
        if args.experiment_type == "h2oxgboost":
            try:
                import h2o
                h2o.remove_all()
                h2o.cluster().shutdown()
            except Exception:
                pass


# def gpu_worker(q, args, config, gpu_id: int):
#     """
#     GPU worker. hard-isolate to a single GPU via CUDA_VISIBLE_DEVICES.
#     inside this process, that GPU is ordinal 0; we set params['device']='cuda:0'.
#     """
#     # stable device ordering + hard isolation
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     # avoid cross-GPU fabric probing (we're single-GPU per process)
#     os.environ.setdefault("NCCL_P2P_DISABLE", "1")
#     os.environ.setdefault("NCCL_IB_DISABLE", "1")

#     # set niceness if requested
#     try:
#         if args.nice is not None:
#             psutil.Process(os.getpid()).nice(args.nice)
#     except Exception:
#         pass

#     # clone args minimally without mutating parent
#     # we don't need argparse.Namespace here—just change the device view.
#     class A:
#         pass

#     args_local = A()
#     for k, v in vars(args).items():
#         setattr(args_local, k, v)
#     args_local.device = "gpu"

#     time.sleep(0.2)

#     if args.experiment_type == "h2oxgboost":
#         _ensure_h2o()  # i just want to connect to the parent cluster that I started with h2o.init()

#     if args.experiment_type == "pyxgboost":
#         times, model_params = py_one_run(args=args_local, config=config)
#     elif args.experiment_type == "h2oxgboost":
#         times, model_params = h2o_one_run(args=args_local, config=config)
#     else:
#         raise ValueError(f"unknown experiment type {args.experiment_type}")
#     q.put((times, model_params))

def gpu_worker(q, args, config, gpu_id: int):
    # isolate GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    try:
        if args.nice is not None:
            psutil.Process(os.getpid()).nice(args.nice)
    except Exception:
        pass

    # clone args without mutation and mark as GPU
    class A: ...
    args_local = A()
    for k, v in vars(args).items():
        setattr(args_local, k, v)
    args_local.device = "gpu"

    time.sleep(0.2)

    try:
        if args.experiment_type == "h2oxgboost":
            # Private H2O cluster bound to localhost & unique port
            _init_h2o_isolated(gpu_id=gpu_id)

        if args.experiment_type == "pyxgboost":
            times, model_params = py_one_run(args=args_local, config=config)
        elif args.experiment_type == "h2oxgboost":
            times, model_params = h2o_one_run(args=args_local, config=config)
        else:
            raise ValueError(f"unknown experiment type {args.experiment_type}")

        q.put((times, model_params))
    finally:
        if args.experiment_type == "h2oxgboost":
            try:
                import h2o
                h2o.remove_all()
                h2o.cluster().shutdown()
            except Exception:
                pass



def run_cpu_benchmark(args, config):
    if args.device == "cpu":
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    ctx = get_context("spawn")
    time_stats = []
    with tqdm(total=args.n_runs, desc="CPU runs", leave=False) as pbar:
        for _ in range(args.n_runs):
            q = ctx.Queue()
            p = ctx.Process(target=cpu_worker, args=(q, args, config))
            p.start()
            result, model_params = q.get()
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"CPU run failed with exit code {p.exitcode}")
            time_stats.append(result)
            pbar.update(1)

    return time_stats, model_params


def run_gpu_benchmark(args, config, num_gpus: int = 4):
    """
    run args.n_runs total, with up to num_gpus runs in parallel.
    each run gets a unique GPU by masking visibility to that device.
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
            result, model_params = q.get()  # wait for that worker
            p.join()
            if p.exitcode != 0:
                raise SystemExit(
                    f"GPU run on device {gid} failed with exit code {p.exitcode}"
                )
            time_stats.append(result)
            pbar.update(1)

            if launched < args.n_runs:
                launch(next_gpu % num_gpus)
                launched += 1
                next_gpu += 1

    return time_stats, model_params


def main():

    t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="XGBoost CPU and single-GPU timing benchmark"
    )
    parser.add_argument(
        "--experiment-type", choices=["pyxgboost", "h2oxgboost"], default="pyxgboost"
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    parser.add_argument(
        "--out-dir", type=str, default="logs_dev", help="output directory"
    )
    parser.add_argument(
        "--load-data", action="store_true", help="load data instead of creating it"
    )

    parser.add_argument(
        "--data-dir", type=str, default="data", help="directory to load data from"
    )

    parser.add_argument("--n-runs", type=int, default=5, help="number of runs")
    parser.add_argument("--threads", type=int, default=5, help="threads per CPU run")

    parser.add_argument(
        "--core-start",
        type=int,
        default=0,
        help="first core index to pin the worker+threads (CPU only)",
    )
    parser.add_argument(
        "--isolate-blas",
        action="store_true",
        help="force single-thread BLAS during DMatrix prep (CPU only)",
    )
    parser.add_argument(
        "--nice",
        type=int,
        default=None,
        help="set process niceness (lower is higher priority)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help="number of GPUs to use concurrently for GPU mode",
    )

    args = parser.parse_args()
    config = get_config()

    # if args.experiment_type == "h2oxgboost":
    #     h2o.init()

    # py_one_run(args=args, config=config)

    # h2o_one_run(args, config)

    # import ipdb
    # ipdb.set_trace()

    # class A: pass
    # args_local = A()
    # for k, v in vars(args).items():
    #     setattr(args_local, k, v)
    # args_local.device = "gpu"
    # times = py_one_run(args=args_local, config=config)

    # import ipdb
    # ipdb.set_trace()

    if args.device == "cpu":
        time_stats, model_params = run_cpu_benchmark(args, config)
    else:
        time_stats, model_params = run_gpu_benchmark(args, config, num_gpus=args.n_gpus)

    t1 = time.perf_counter()
    total_run_time = t1 - t0

    austin_tz = ZoneInfo("America/Chicago")
    current_time = datetime.now(austin_tz)
    experiment_folder = f"{args.device}_{format_number(config.sample.n_samples)}_{current_time.strftime('%Y%m%d_%H%M%S')}"

    args.out_dir = Path(args.out_dir) / args.experiment_type / experiment_folder
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(time_stats)
    benchmark_detailed_file = (
        f"{args.out_dir}/{args.device}_benchmark_detailed_times.csv"
    )
    df.to_csv(benchmark_detailed_file, index=False)

    # report and persist summary
    args.out_dir = str(
        args.out_dir
    )  # since I am saving this to json, args.out_dir needs to be str not Path
    summary = {
        "total_time": total_run_time,
        "env": summarize_env(),
        "system": get_system_info(),
        "config": vars(args),
        "model_params": model_params,
    }

    summary_json = f"{args.out_dir}/{args.device}_benchmark_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote summary to {summary_json}")
    print(f"wrote per-run details to {benchmark_detailed_file}")


if __name__ == "__main__":
    main()
