import time, gc
from pathlib import Path

import h2o

from utils import (
    make_synthetic_binary
)

def get_h2o_data(config, args):

    py_dir_name = Path(args.data_dir) / f"h2o_data_{config.sample.n_samples}_parquet"
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


def h2o_one_run(
    args,
    config,
):
    # data loading / generation
    (train, valid), data_time = get_h2o_data(config, args)

    # paramter mapping

    # train warmup

    # train for real

    # predict
    import ipdb
    ipdb.set_trace()
