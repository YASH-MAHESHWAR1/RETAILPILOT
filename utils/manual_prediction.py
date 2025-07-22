# utils/manual_prediction.py
import pandas as pd
import numpy as np
import datetime as dt
from typing import Tuple, Any

def create_historical_features(prediction_date: dt.date, historical_data_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([{'dt': pd.to_datetime(prediction_date)}])
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['month'] = df['dt'].dt.month
    df['day_of_month'] = df['dt'].dt.day
    df['quarter'] = df['dt'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    history = historical_data_df.sort_index()
    for w in [1, 3, 5, 7]:
        df[f'sales_lag_{w}'] = history['sale_amount'].iloc[-w] if len(history) >= w else 0
        df[f'stock_lag_{w}'] = history['stock_hour6_22_cnt'].iloc[-w] if len(history) >= w else 0
    for w in [3, 5, 7]:
        df[f'sales_ma_{w}'] = history['sale_amount'].rolling(w, min_periods=1).mean().iloc[-1] if not history.empty else 0
        df[f'stock_ma_{w}'] = history['stock_hour6_22_cnt'].rolling(w, min_periods=1).mean().iloc[-1] if not history.empty else 0
    return df

def run_all_predictions(feature_vector: pd.DataFrame, sale_model: Any, stock_model: Any, oos_model: Any) -> Tuple[float, float, int]:
    pred_sales = max(0, sale_model.predict(feature_vector)[0])
    y_pred_stock = np.clip(stock_model.predict(feature_vector)[0], 0, 16)
    pred_stock=np.round(y_pred_stock).astype(int)
    pred_sev = oos_model.predict(feature_vector)[0]
    return pred_sales, pred_stock, pred_sev