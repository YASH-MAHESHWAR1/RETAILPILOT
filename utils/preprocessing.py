import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
from .load_data import load_parquet_data
from typing import Tuple, Optional

# ========== TRAIN FEATURE ENGINEERING ==========
def prepare_train_features(df):
    df = df.copy()
    df['dt'] = pd.to_datetime(df['dt'])

    df['hours_sale'] = df['hours_sale'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df['hours_stock_status'] = df['hours_stock_status'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    df = df.sort_values(['store_id', 'product_id', 'dt'])

    # Time-based features
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['month'] = df['dt'].dt.month
    df['day_of_month'] = df['dt'].dt.day
    df['quarter'] = df['dt'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Lag/Moving Average
    for window in [1, 3, 5, 7]:
        df[f'sales_lag_{window}'] = df.groupby(['store_id', 'product_id'])['sale_amount'].shift(window)
        df[f'stock_lag_{window}'] = df.groupby(['store_id', 'product_id'])['stock_hour6_22_cnt'].shift(window)

    for window in [3, 5, 7]:
        df[f'sales_ma_{window}'] = (
            df.groupby(['store_id', 'product_id'])['sale_amount']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f'stock_ma_{window}'] = (
            df.groupby(['store_id', 'product_id'])['stock_hour6_22_cnt']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    # Label encoding
    for col in ['city_id', 'store_id', 'management_group_id',
                'first_category_id', 'second_category_id', 'third_category_id', 'product_id']:
        df[col] = LabelEncoder().fit_transform(df[col])

    df['product_category'] = (
        df['first_category_id'].astype(str) + '_' +
        df['second_category_id'].astype(str) + '_' +
        df['third_category_id'].astype(str)
    )

    # Drop missing values in essential columns
    required_cols = ['sale_amount', 'stock_hour6_22_cnt'] + [col for col in df.columns if 'lag' in col or 'ma' in col]
    df = df.dropna(subset=required_cols)

    return df


# ========== TEST FEATURE ENGINEERING ==========
def prepare_test_features(test_df, train_df):
    test_df = test_df.copy()
    test_df['dt'] = pd.to_datetime(test_df['dt'])

    test_df['hours_sale'] = test_df['hours_sale'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    test_df['hours_stock_status'] = test_df['hours_stock_status'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    test_df = test_df.sort_values(['store_id', 'product_id', 'dt'])

    # Time features
    test_df['day_of_week'] = test_df['dt'].dt.dayofweek
    test_df['month'] = test_df['dt'].dt.month
    test_df['day_of_month'] = test_df['dt'].dt.day
    test_df['quarter'] = test_df['dt'].dt.quarter
    test_df['is_weekend'] = test_df['day_of_week'].isin([5, 6]).astype(int)

    # Label encoding using train mappings
    for col in ['city_id', 'store_id', 'management_group_id',
                'first_category_id', 'second_category_id', 'third_category_id', 'product_id']:
        label_map = {val: idx for idx, val in enumerate(sorted(train_df[col].unique()))}
        test_df[col] = test_df[col].map(label_map).fillna(-1).astype(int)

    test_df['product_category'] = (
        test_df['first_category_id'].astype(str) + '_' +
        test_df['second_category_id'].astype(str) + '_' +
        test_df['third_category_id'].astype(str)
    )

    # Merge last 7 days of train with test for lag/ma
    test_df["__is_test__"] = True
    tail_df = train_df[train_df['dt'] >= (train_df['dt'].max() - pd.Timedelta(days=7))].copy()
    tail_df["__is_test__"] = False

    combined = pd.concat([tail_df, test_df], axis=0).sort_values(['store_id', 'product_id', 'dt'])

    for window in [1, 3, 5, 7]:
        combined[f'sales_lag_{window}'] = combined.groupby(['store_id', 'product_id'])['sale_amount'].shift(window)
        combined[f'stock_lag_{window}'] = combined.groupby(['store_id', 'product_id'])['stock_hour6_22_cnt'].shift(window)

    for window in [3, 5, 7]:
        combined[f'sales_ma_{window}'] = (
            combined.groupby(['store_id', 'product_id'])['sale_amount']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        combined[f'stock_ma_{window}'] = (
            combined.groupby(['store_id', 'product_id'])['stock_hour6_22_cnt']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    test_df_final = combined[combined["__is_test__"] == True].drop(columns="__is_test__").dropna()

    return test_df_final

# --------------------------------------------
# Function to Preprocess the actual data files
# --------------------------------------------

def preprocess_data_files(
    raw_train_path: str = "data/train.parquet",
    raw_eval_path: Optional[str] = "data/eval.parquet",
    processed_train_path: str = "data/train_processed.parquet",
    processed_eval_path: str = "data/test_processed.parquet"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess training and evaluation data.

    Args:
        raw_train_path (str): Path to raw training parquet file.
        raw_eval_path (str): Path to raw evaluation parquet file.
        processed_train_path (str): Path to save/load processed training data.
        processed_eval_path (str): Path to save/load processed evaluation data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, eval_df)
    """

    # --- If processed parquet files exist, load them ---
    if Path(processed_train_path).exists() and Path(processed_eval_path).exists():
        print("✅ Loading cached preprocessed data...")
        train_df = pd.read_parquet(processed_train_path)
        eval_df = pd.read_parquet(processed_eval_path)

    else:
        print("⚙️ Preprocessing raw data from parquet files...")

        # Load raw parquet data
        train_df_raw, eval_df_raw = load_parquet_data(raw_train_path, raw_eval_path, verbose=True)

        # Run feature engineering
        train_df = prepare_train_features(train_df_raw)
        eval_df = prepare_test_features(eval_df_raw, train_df)

        # Save processed data
        train_df.to_parquet(processed_train_path, index=False, compression='snappy')
        eval_df.to_parquet(processed_eval_path, index=False, compression='snappy')

        print("✅ Saved processed data to parquet.")

    return train_df, eval_df