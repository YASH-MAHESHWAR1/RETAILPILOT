import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from datetime import timedelta
import seaborn as sns

def map_severity_class(severity):
    return {
        0: "Fully Stocked",
        1: "Mild",
        2: "Moderate",
        3: "Severe"
    }.get(severity, "Unknown")


def forecast_sales_stock_oos(
    sale_model, stock_model, severity_model,
    data, feature_cols, days=7,
    store_id=None, product_id=None
):
    """
    Forecasts sales, stock-out hours (6–22), and severity for a given store-product combo,
    or overall if not specified.
    Parameters:
    - sale_model: trained regression model for sales
    - stock_model: trained regression model for stock-out hours
    - severity_model: trained classifier for severity class (0–3)
    - data: full DataFrame with 'dt', 'store_id', 'product_id' and input features
    - feature_cols: list of model features
    - days: number of days to forecast
    - store_id: (optional) specific store to forecast
    - product_id: (optional) specific product to forecast
    Returns:
    - forecast_df: DataFrame with predicted results
    - fig: matplotlib figure of forecasted trends
    """
    try:
        df = data.copy()

        # Filter by store_id and/or product_id if specified
        if store_id is not None:
            df = df[df['store_id'] == store_id]
        if product_id is not None:
            df = df[df['product_id'] == product_id]

        if df.empty:
            print(f"❌ No data found for store_id={store_id}, product_id={product_id}")
            return pd.DataFrame(), None

        # Get latest available row
        last_row = df.sort_values("dt").iloc[-1].copy()
        forecasts = []
        current_date = last_row['dt'] + timedelta(days=1)

        # Lag history queues
        sales_history = deque([
            last_row.get('sales_lag_7', 0),
            last_row.get('sales_lag_5', 0),
            last_row.get('sales_lag_3', 0),
            last_row.get('sales_lag_1', 0),
            last_row.get('sale_amount', 0)
        ], maxlen=7)

        stock_history = deque([
            last_row.get('stock_lag_7', 0),
            last_row.get('stock_lag_5', 0),
            last_row.get('stock_lag_3', 0),
            last_row.get('stock_lag_1', 0),
            last_row.get('stock_hour6_22_cnt', 0)
        ], maxlen=7)

        for _ in range(days):
            row = last_row.copy()

            # Time features
            row['dt'] = current_date
            row['day_of_week'] = current_date.weekday()
            row['month'] = current_date.month
            row['day_of_month'] = current_date.day
            row['quarter'] = (current_date.month - 1) // 3 + 1
            row['is_weekend'] = int(current_date.weekday() >= 5)

            # Lag features for sales
            row['sales_lag_1'] = sales_history[-1]
            row['sales_lag_3'] = sales_history[-3] if len(sales_history) >= 3 else sales_history[0]
            row['sales_lag_5'] = sales_history[-5] if len(sales_history) >= 5 else sales_history[0]
            row['sales_lag_7'] = sales_history[0]
            row['sales_ma_3'] = np.mean(list(sales_history)[-3:])
            row['sales_ma_5'] = np.mean(list(sales_history)[-5:])
            row['sales_ma_7'] = np.mean(sales_history)

            # Lag features for stock-out hours
            row['stock_lag_1'] = stock_history[-1]
            row['stock_lag_3'] = stock_history[-3] if len(stock_history) >= 3 else stock_history[0]
            row['stock_lag_5'] = stock_history[-5] if len(stock_history) >= 5 else stock_history[0]
            row['stock_lag_7'] = stock_history[0]
            row['stock_ma_3'] = np.mean(list(stock_history)[-3:])
            row['stock_ma_5'] = np.mean(list(stock_history)[-5:])
            row['stock_ma_7'] = np.mean(stock_history)

            # Predictions
            input_data = row[feature_cols].values.reshape(1, -1)
            predicted_sales = sale_model.predict(input_data)[0]
            predicted_stock = stock_model.predict(input_data)[0]
            predicted_stock = np.round(predicted_stock).astype(int)
            predicted_severity = severity_model.predict(input_data)[0]
            severity_label = map_severity_class(predicted_severity)

            forecasts.append({
                'Date': current_date,
                'Predicted Sales': round(predicted_sales, 2),
                'Predicted Stock-Out Hours (6–22)': round(predicted_stock, 2),
                'Predicted Severity Class': int(predicted_severity),
                'Severity Label': severity_label
            })

            # Update for next iteration
            sales_history.append(predicted_sales)
            stock_history.append(predicted_stock)
            current_date += timedelta(days=1)

        forecast_df = pd.DataFrame(forecasts)

        # ========== 📊 Plot ==========
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax2 = ax1.twinx()

        sns.lineplot(data=forecast_df, x='Date', y='Predicted Sales', marker='o', ax=ax1, color="#1f77b4", label="Sales")
        sns.lineplot(data=forecast_df, x='Date', y='Predicted Stock-Out Hours (6–22)', marker='s', ax=ax1, color="#2ca02c", label="Stock-Out Hours")
        sns.lineplot(data=forecast_df, x='Date', y='Predicted Severity Class', marker='^', ax=ax2, color="#d62728", label="Severity Level")

        ax1.set_ylabel("Sales / Stock-Out Hours", fontsize=10)
        ax2.set_ylabel("Severity Class", fontsize=10, color="#d62728")
        ax1.set_xlabel("Forecast Date", fontsize=10)
        ax1.set_title("🔮 Forecast: Sales, Stock-Out Hours (6–22), and Severity Level", fontsize=14, weight='bold')
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # fig.tight_layout()

        return forecast_df, fig

    except Exception as e:
        print(f"❌ Error in forecast_sales_stock_oos: {e}")
        return pd.DataFrame(), None
