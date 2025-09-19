import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cpuinfo
import GPUtil
import numpy as np
import psutil
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class Metrics:
    data_time_s: Optional[float] = None
    full_train_total_s: Optional[float] = None
    boost_step_estm_s: Optional[float] = None  # estimated time for one boosting step
    n_boost_per_sec: Optional[float] = None
    boost_step_avg_s: Optional[float] = None
    boost_step_s_std: Optional[float] = None
    boost_step_s_median: Optional[float] = None
    n_boost_per_sec_avg: Optional[float] = None
    n_boost_per_sec_std: Optional[float] = None
    n_boost_per_sec_median: Optional[float] = None
    test_accuracy: Optional[float] = None
    train_logloss: Optional[float] = None
    test_logloss: Optional[float] = None

    def to_dict(self, drop_none: bool = True) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None} if drop_none else d


def summarize_env():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "omp_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def make_synthetic_binary(
    n_samples: int,
    n_features: int = 1,
    mean_a: float = 1.0,
    std_a: float = 1.0,
    mean_b: float = 2.0,
    std_b: float = 2.0,
    seed: int = 42,
    shuffle: bool = True,
):
    """
    generate a binary dataset with two overlapping normal classes.

    class A ~ normal(mean_a, std_a), label 0.0
    class B ~ normal(mean_b, std_b), label 1.0

    X: (n_samples, n_features) float32
    y: (n_samples,)           uint8 in {0, 1}
    """
    assert n_samples % 2 == 0, "use an even n_samples to get an exact 50/50 split."

    n_a = n_samples // 2
    n_b = n_samples - n_a
    rng = np.random.default_rng(seed)

    Xa = rng.normal(mean_a, std_a, size=(n_a, n_features)).astype(
        np.float32, copy=False
    )
    Xb = rng.normal(mean_b, std_b, size=(n_b, n_features)).astype(
        np.float32, copy=False
    )
    X = np.vstack([Xa, Xb])
    y = np.empty(n_samples, dtype=np.uint8)
    y[:n_a] = 0
    y[n_a:] = 1

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X, y


def build_train_test_dmatrices(
    X,
    y,
    test_pct: float = 0.2,
    seed: int = 42,
    gpu: bool = False,  # set True if you will use tree_method="gpu_hist"
):
    """
    Returns: dtrain, dtest, times
    times includes: split_s, dmatrix_train_s, dmatrix_test_s, n_train, n_test
    """
    assert 0.0 < test_pct < 1.0
    times = {}

    # Generate train/test row indices only, then construct DMatrices sequentially.
    t0 = time.perf_counter()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_pct, random_state=seed)
    # dummy features for splitter; it only needs y for stratification
    train_idx, test_idx = next(sss.split(np.zeros_like(y), y))
    times["split_s"] = time.perf_counter() - t0

    # Build train DMatrix (optionally QuantileDMatrix for GPU)
    t0 = time.perf_counter()
    if gpu:
        dtrain = xgb.QuantileDMatrix(X[train_idx], label=y[train_idx])
    else:
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    times["dmatrix_train_s"] = time.perf_counter() - t0
    n_train = int(train_idx.size)
    del train_idx
    gc.collect()

    # Build test DMatrix
    t0 = time.perf_counter()
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx])
    times["dmatrix_test_s"] = time.perf_counter() - t0
    n_test = int(test_idx.size)
    del test_idx
    gc.collect()

    times["n_train"] = n_train
    times["n_test"] = n_test
    times["test_pct"] = float(test_pct)

    return dtrain, dtest, times


def ensure_data_and_split(x_path, y_path, train_idx_path, test_idx_path, config):

    if not (x_path.exists() and y_path.exists()):
        X, y = make_synthetic_binary(
            n_samples=config.sample.n_samples,
            n_features=config.sample.n_features,
            seed=config.sample.random_state,
            shuffle=True,
        )
        np.save(x_path, X, allow_pickle=False)
        np.save(y_path, y, allow_pickle=False)
        del X, y
        gc.collect()

    # single fixed stratified split (same seed), saved once
    if not (train_idx_path.exists() and test_idx_path.exists()):
        y = np.load(y_path, mmap_mode="r")
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.sample.test_size,
            random_state=config.sample.random_state,
        )

        train_idx, test_idx = next(sss.split(np.zeros_like(y), y))
        np.save(
            train_idx_path, train_idx.astype(np.int32, copy=False), allow_pickle=False
        )
        np.save(
            test_idx_path, test_idx.astype(np.int32, copy=False), allow_pickle=False
        )
        del y, train_idx, test_idx
        gc.collect()


def get_system_info():
    info = {}

    # Basic system/platform info
    info["System"] = platform.system()
    info["Node Name"] = platform.node()
    info["Release"] = platform.release()
    info["Version"] = platform.version()
    info["Machine"] = platform.machine()
    info["Processor"] = platform.processor()

    # CPU details
    cpu_info = cpuinfo.get_cpu_info()
    info["CPU Brand"] = cpu_info.get("brand_raw", "N/A")
    info["Architecture"] = cpu_info.get("arch", "N/A")
    info["Bits"] = cpu_info.get("bits", "N/A")
    info["Count (logical)"] = psutil.cpu_count(logical=True)
    info["Count (physical)"] = psutil.cpu_count(logical=False)
    freq = psutil.cpu_freq()
    if freq:
        info["Max Frequency (MHz)"] = freq.max
        info["Min Frequency (MHz)"] = freq.min
        info["Current Frequency (MHz)"] = freq.current
    info["Total CPU Usage (%)"] = psutil.cpu_percent()

    # Memory info
    svmem = psutil.virtual_memory()
    info["Total Memory (GB)"] = round(svmem.total / (1024**3), 2)
    info["Available Memory (GB)"] = round(svmem.available / (1024**3), 2)

    # GPU info (if NVIDIA GPU is available)
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            info[f"GPU {i} Name"] = gpu.name
            info[f"GPU {i} Driver"] = gpu.driver
            info[f"GPU {i} Memory Total (MB)"] = gpu.memoryTotal
            # info[f"GPU {i} Memory Used (MB)"] = gpu.memoryUsed
            info[f"GPU {i} Memory Free (MB)"] = f"{gpu.memoryFree}"
            # info[f"GPU {i} Utilization (%)"] = gpu.load * 100
            # info[f"GPU {i} Temperature (°C)"] = gpu.temperature
    except Exception as e:
        info["GPU Info"] = f"Could not retrieve GPU info ({e})"

    return info


def _smart_cast(v: str):
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        if s in ("null", "none"):
            return None
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except Exception:
                pass
        try:
            if any(c in s for c in ".e"):
                return float(s)
        except Exception:
            pass
    return v


def _flatten(obj, out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _flatten(item, out)
    else:
        # use only the last part of the key path → no prefix
        out_key = str(obj)  # fallback if needed
    return out


def flatten_xgb_config(booster: xgb.Booster) -> dict:
    cfg = json.loads(booster.save_config())
    out = {}

    def recurse(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    recurse(v)
                else:
                    out[k] = _smart_cast(v)
        elif isinstance(node, list):
            for item in node:
                recurse(item)

    recurse(cfg)
    return out


def format_number(n: int) -> str:
    suffixes = ["", "K", "M", "B", "T", "P"]  # extend as needed
    i = 0
    while abs(n) >= 1000 and i < len(suffixes) - 1:
        n /= 1000.0
        i += 1
    return f"{n:.0f}{suffixes[i]}"
