import time, gc
from pathlib import Path

import h2o
from h2o.estimators import H2OXGBoostEstimator
from xgboost import train

from utils import (
    make_synthetic_binary,
    Metrics
)

def get_h2o_data(config, args):

    py_dir_name = Path(args.data_dir) / f"h2o_data" / f"samples_{config.sample.n_samples}"
    h2o_file_dir = f"file://{py_dir_name.resolve()}" # otherwise h2o will save it to the direcotry you put the javer process code (ie., ./h2o-3.46.0.7) 
    
    def create_hf():
        X, y = make_synthetic_binary(
            n_samples=config.sample.n_samples,
            n_features=config.sample.n_features,
            seed=config.sample.random_state,
        )

        columns = [f"x{i}" for i in range(X.shape[1])]
        hf = h2o.H2OFrame(X, column_names=columns)   
        hy = h2o.H2OFrame(y.reshape(-1, 1), column_names=["target"])

        del X, y; gc.collect()  # encourage memory release if possible
        hf = hf.cbind(hy)

        if not py_dir_name.exists():
            py_dir_name.mkdir(parents=True, exist_ok=True)
            h2o.export_file(hf, path=str(h2o_file_dir), format="parquet", force=True)
        return hf
    
    t0 = time.perf_counter()
    if args.load_data:

        # this is if we want to create data each time we make a run
        # this will be very slow than reading the data from disk
        hf = create_hf()
    else:
        if not py_dir_name.exists():
            hf = create_hf()
        else:
            hf = h2o.import_file(str(h2o_file_dir))
    hf["target"] = hf["target"].asfactor()

    train, valid = hf.split_frame(
        ratios=[1-config.sample.test_size], 
        seed=config.sample.random_state)

    t1 = time.perf_counter()
    data_time = t1 - t0

    # print(f"loading file: {args.load_data}, took {gen_time:.3f} sec\n\n")
    
    del hf; gc.collect()  # encourage memory release if possible
    return (train, valid), data_time

def h2o_map_params(args, config):

    distribution = "bernoulli" if config.common.objective == "binary:logistic" else NotImplementedError(f"h2o does not set objetives like the xgboost library. for now we only support the binary:logistic objective")

    params = {
        "booster": config.common.booster,
        "distribution": distribution,
        "learn_rate": config.common.eta,
        "reg_alpha": config.common.alpha,
        "reg_lambda": config.common.lmbda,
        "gamma": config.common.gamma,
        "col_sample_rate": config.common.colsample_bylevel,
        "colsample_bynode": config.common.colsample_bynode,
        "col_sample_rate_per_tree": config.common.colsample_bytree,
        "grow_policy": config.common.grow_policy,
        "max_bins": config.common.max_bin,
        "max_delta_step": config.common.max_delta_step,
        "max_depth": config.common.max_depth,
        "max_leaves": config.common.max_leaves,
        "min_rows": config.common.min_child_weight,
        "ntrees": config.common.num_boost_round,
        "sample_type": config.common.sampling_method,
        "scale_pos_weight": config.common.scale_pos_weight,
        "sample_rate": config.common.subsample,
        "quiet_mode": True if config.common.verbosity == 0 else False,
        "tree_method": config.gpu.tree_method if args.device == "gpu" else config.cpu.tree_method,
        # "min_split_loss": config.common.min_split_loss,
        "nthread": args.threads,
        "backend": "gpu" if args.device == "gpu" else "cpu",
        "gpu_id": 0 # this will be 0 for these experiments since we explore one gpu at a time in this experiment

    }

    return params


def run_training_once(params, train, num_boost_round=None):
    """
    
    run training for num_boost_round or one boosting step if do_one_step is True.

    params: dict of xgb.train params
    dtrain: training DMatrix or QuantileDMatrix (if on gpu)
    num_boost_round: int, number of boosting rounds to run if do_one_step is True.
    Returns:    
        booster, total_time, per_iter_times
    
    """

    if num_boost_round is not None:
        params = params.copy()
        params["ntrees"] = num_boost_round # we will do one tree at a time

    t0 = time.perf_counter()
    booster = H2OXGBoostEstimator(**params)
    booster.train(x=train.columns[:-1], y="target", training_frame=train)
    t1 = time.perf_counter()
    total = t1 - t0
    return booster, total


def h2o_one_run(
    args,
    config,
):
    # data loading / generation
    (train, valid), data_time = get_h2o_data(config, args)

    # paramter mapping
    # some h2o params are different from xgboost params
    params = h2o_map_params(args, config)

    # warmup run
    _ = run_training_once(params, train, num_boost_round=5)

    # main training run
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
       booster, full_total = run_training_once(params, train)
    finally:
        if gc_was_enabled:
            gc.enable()

   
    train_results = booster.model_performance(train)
    valid_results = booster.model_performance(valid)

    train_loss = train_results.logloss()
    test_loss = valid_results.logloss()
    acc = valid_results.accuracy()[0][1]

    times = Metrics(
        data_time_s=data_time,
        full_train_total_s=full_total,
        boost_step_estm_s=full_total / config.common.num_boost_round,
        n_boost_per_sec=config.common.num_boost_round / full_total,
        test_accuracy=float(acc),
        train_logloss=float(train_loss),
        test_logloss=float(test_loss),
    ).to_dict()

    return times, booster.get_params()