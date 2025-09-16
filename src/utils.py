
import time, gc

import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import platform
import psutil
import cpuinfo
import GPUtil

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

    Xa = rng.normal(mean_a, std_a, size=(n_a, n_features)).astype(np.float32, copy=False)
    Xb = rng.normal(mean_b, std_b, size=(n_b, n_features)).astype(np.float32, copy=False)
    X = np.vstack([Xa, Xb])
    y = np.empty(n_samples, dtype=np.uint8)
    y[:n_a] = 0
    y[n_a:] = 1

    
    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X, y

def build_dmatrix(X, y):
    t0 = time.perf_counter()
    dtrain = xgb.DMatrix(X, label=y)
    t1 = time.perf_counter()
    return dtrain, t1 - t0



def build_train_test_dmatrices(
    X,
    y,
    test_pct: float = 0.2,
    seed: int = 42,
    gpu: bool = False # set True if you will use tree_method="gpu_hist"
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
    del train_idx; gc.collect()

    # Build test DMatrix
    t0 = time.perf_counter()
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx])
    times["dmatrix_test_s"] = time.perf_counter() - t0
    n_test = int(test_idx.size)
    del test_idx; gc.collect()

    times["n_train"] = n_train
    times["n_test"] = n_test
    times["test_pct"] = float(test_pct)
    
    return dtrain, dtest, times


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
