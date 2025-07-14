# load_data.py
import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def load_parquet_data(train_path: str, eval_path: Optional[str] = None, verbose: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load training and optional evaluation data from Parquet files.
    Also converts any ndarray columns to list for Streamlit compatibility.

    Args:
        train_path (str): Path to training parquet file.
        eval_path (str, optional): Path to evaluation parquet file.
        verbose (bool): Whether to print status messages.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]: (train_df, eval_df or None)
    """
    # --- Load Training Data ---
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Training file not found: {train_path}")
    if verbose:
        print(f"📂 Loading training data from {train_path}...")
    train_df = pd.read_parquet(train_path)

    # --- Load Evaluation Data (if provided) ---
    eval_df = None
    if eval_path:
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"❌ Evaluation file not found: {eval_path}")
        if verbose:
            print(f"📂 Loading evaluation data from {eval_path}...")
        eval_df = pd.read_parquet(eval_path)

    # --- Fix: Convert ndarray columns to list ---
    def convert_array_columns(df: pd.DataFrame) -> pd.DataFrame:
        for col in ['hours_sale', 'hours_stock_status']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        return df

    train_df = convert_array_columns(train_df)
    if eval_df is not None:
        eval_df = convert_array_columns(eval_df)

    # --- Verbose Output ---
    if verbose:
        print(f"✅ train_df loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
        if eval_df is not None:
            print(f"✅ eval_df loaded: {eval_df.shape[0]} rows, {eval_df.shape[1]} columns")

    return train_df, eval_df


# --- Optional CLI Run for Debugging ---
if __name__ == "__main__":
    TRAIN_PATH = "../data/train.parquet"
    EVAL_PATH = "../data/eval.parquet"

    train_df, eval_df = load_parquet_data(TRAIN_PATH, EVAL_PATH)

    print("\n🔎 Sample training data:")
    print(train_df.head())

    if eval_df is not None:
        print("\n🔎 Sample evaluation data:")
        print(eval_df.head())
