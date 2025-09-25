import numpy as np
import pandas as pd
import os


if __name__ == "__main__":

    data_dir = "data/py_data/samples_10000000"
    # Load arrays
    X = np.load(f"{data_dir}/X.npy")
    y = np.load(f"{data_dir}/y.npy")
    train_idx = np.load(f"{data_dir}/train_idx.npy")
    test_idx = np.load(f"{data_dir}/test_idx.npy")

    # Build full DataFrame
    df = pd.DataFrame(X)
    df["target"] = y

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    out_dir = "data/driverless"
    os.makedirs(out_dir, exist_ok=True)
    
    train_df.to_parquet(f"{out_dir}/train_samples_10000000.parquet", index=False)
    test_df.to_parquet(f"{out_dir}/test_samples_10000000.parquet", index=False)

