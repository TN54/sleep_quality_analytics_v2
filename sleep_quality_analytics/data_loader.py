# data_loader.py
# This module is responsible for loading and checking the dataset

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # allows execution from UI layer
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"


def load_data(file_path=None):
    """
    Load CSV dataset and return dataframe.
    Params:
        file_path: str | Path | None - path to the CSV file. Uses packaged dataset when omitted.
    Returns:
        df: pandas DataFrame
    """
    resolved_path = Path(file_path) if file_path else _DEFAULT_DATA_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {resolved_path}. "
            "Ensure the CSV exists or pass an explicit path to load_data()."
        )

    df = pd.read_csv(resolved_path)
    print("Dataset Loaded Successfully!")
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    return df

def check_missing_values(df):
    """
    Check for missing values in the dataset.
    """
    missing = df.isnull().sum()
    print("\nMissing Values in Each Column:")
    print(missing)
    return missing

