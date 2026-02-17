import pandas as pd
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)

def load_data(
    file_path: str,
    encoding: str = "utf-8",
    delimiter: str = ","
) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame with validation and error handling.

    Args:
        file_path (str): Path to the CSV file.
        encoding (str): File encoding (default: utf-8).
        delimiter (str): CSV delimiter (default: comma).

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If the dataset is empty.
        RuntimeError: If reading fails.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)

        if df.empty:
            raise ValueError("Loaded dataset is empty.")

        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file: {e}")
