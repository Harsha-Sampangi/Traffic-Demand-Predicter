"""
Step 2 — Load the data and confirm shapes.
"""
import pandas as pd


def load_data(data_dir="dataset"):
    """Load train, test, and sample submission CSVs."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    sub = pd.read_csv(f"{data_dir}/sample_submission.csv")

    assert train.shape == (77299, 11), f"Unexpected train shape: {train.shape}"
    assert test.shape == (41778, 10), f"Unexpected test shape: {test.shape}"

    print("Train shape :", train.shape)
    print("Test shape  :", test.shape)
    print("Sub columns :", sub.columns.tolist())
    print("Train cols  :", train.columns.tolist())
    print("\nMissing in train:\n", train.isnull().sum())
    print("\nMissing in test:\n", test.isnull().sum())
    print("\nDemand stats:\n", train["demand"].describe())

    return train, test, sub
