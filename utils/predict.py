import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Optional, Dict
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import ndarray
import traceback
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------
# Function to predict sales
# ------------------------------------

def predict_sales(model, test_df, feature_cols, target_col='sale_amount', store_id=None, product_id=None):
    """
    Predicts sales on filtered test data and returns plot, metrics, and a result table.

    Parameters:
    - model: trained regression model
    - test_df (pd.DataFrame): test dataset
    - feature_cols (list): model features
    - target_col (str): prediction target column
    - store_id (int, optional): filter to specific store
    - product_id (int, optional): filter to specific product

    Returns:
    - fig (plt.Figure): matplotlib figure
    - result_df (pd.DataFrame): prediction + error table
    - metrics (dict): MAE, RMSE, R2, sample size
    """

    try:
        df = test_df.copy()

        # Apply filters
        title_parts = []
        if store_id is not None:
            df = df[df['store_id'] == store_id]
            title_parts.append(f"Store {store_id}")
        if product_id is not None:
            df = df[df['product_id'] == product_id]
            title_parts.append(f"Product {product_id}")

        if df.empty:
            return None, None, {"Info": "No data available for the selected filters."}

        X_test = df[feature_cols]
        y_true = df[target_col]
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "📉 MAE": round(mae, 2),
            "📏 RMSE": round(rmse, 2),
            "🎯 R²": round(r2, 4),
            "🧾 Test Samples": len(df),
            "📍 Scope": " | ".join(title_parts) if title_parts else "Overall"
        }

        df['predicted_' + target_col] = y_pred
        df['error'] = y_true - y_pred
        df['abs_error'] = abs(df['error'])


        
        #  Plot (first 100 for clarity)
        fig, ax = plt.subplots(figsize=(12, 5))
        sample = df.head(100)
        ax.plot(sample[target_col].values, marker='o', label="True", color='#264653')
        ax.plot(sample['predicted_' + target_col].values, marker='x', label="Predicted", color='#e76f51')
        ax.set_title(f"🔍 {target_col} - True vs Predicted ({metrics['📍 Scope']})", fontsize=14, weight='bold')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(target_col.replace("_", " ").title())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        column_rename_map = {
            'dt': 'Date',
            'store_id': 'Store ID',
            'product_id': 'Product ID',
            'sale_amount': 'Actual Sales',
            'predicted_sale_amount': 'Predicted Sales',
            'error': 'Error',
            'abs_error': 'Absolute Error'
        }
        df = df.rename(columns=column_rename_map)

        return fig, df[['Date', 'Store ID', 'Product ID', 'Actual Sales', 'Predicted Sales', 'Error', 'Absolute Error']], metrics

    except Exception as e:
        return None, None, {"❌ Error": str(e)}



# ------------------------------------
# Function to predict stockout hours
# ------------------------------------
def predict_stockout_hours(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    store_id: Optional[int] = None,
    product_id: Optional[int] = None
) -> Tuple[Optional[Figure], pd.DataFrame, Dict]:
    """
    Predict number of out-of-stock hours (6am–10pm) and return plot, table, and metrics.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - model: Trained regression model
    - feature_cols (list): Feature columns for prediction
    - store_id (int, optional): Filter by store_id
    - product_id (int, optional): Filter by product_id

    Returns:
    - fig: Matplotlib figure showing actual vs predicted out-of-stock hours
    - result_df: DataFrame with actual, predicted, and error
    - metrics: Dict of MAE, RMSE, R², and average availability ratio
    """
    try:
        df = df.copy()
        df['dt'] = pd.to_datetime(df['dt'])

        title_parts = []
        if store_id is not None:
            df = df[df["store_id"] == store_id]
            title_parts.append(f"Store {store_id}")
        if product_id is not None:
            df = df[df["product_id"] == product_id]
            title_parts.append(f"Product {product_id}")

        if df.empty:
            print("⚠️ No data available for the selected filters.")
            return None, pd.DataFrame(), {}

        title = " | ".join(["⛔ Predicted Out-of-Stock Hours"] + title_parts)

        X = df[feature_cols]
        y_true = df["stock_hour6_22_cnt"]
        y_pred = model.predict(X)
        y_pred = np.round(y_pred).astype(int)
        abs_error = np.abs(y_true - y_pred)

        # Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        avg_availability = round(1 - np.mean(y_true) / 16, 3)  # 16 hours from 6–22

        metrics = {
            "❌ MAE (Out-of-Stock Hours)": round(mae, 2),
            "📊 RMSE": round(rmse, 2),
            "📈 R²": round(r2, 4),
            "✅ Avg Stock Availability Ratio": f"{avg_availability * 100:.2f}%"
        }

        # Result DataFrame
        result_df = df[["dt", "store_id", "product_id"]].copy()
        result_df["Actual Out-of-Stock Hours"] = y_true.values
        result_df["Predicted Out-of-Stock Hours"] = y_pred
        result_df["Absolute Error"] = abs_error
        result_df.rename(columns={"dt": "Date"}, inplace=True)
        result_df = result_df.sort_values("Date")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=result_df[:100], x="Date", y="Actual Out-of-Stock Hours",
                     label="Actual", marker='o', color="#e76f51", ax=ax)
        sns.lineplot(data=result_df[:100], x="Date", y="Predicted Out-of-Stock Hours",
                     label="Predicted", marker='x', color="#2a9d8f", ax=ax)

        ax.set_title(title, fontsize=15, fontweight='bold', color="#264653")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Out-of-Stock Hours (6–22)", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        return fig, result_df, metrics

    except Exception as e:
        print(f"❌ Error in predict_stockout_hours: {e}")
        return None, pd.DataFrame(), {}




# ----------------------------
# Severity Binning Function
# ----------------------------
def bin_stock_severity(value):
    if value == 0:
        return 0
    elif 1 <= value <= 5:
        return 1
    elif 6 <= value <= 10:
        return 2
    else:
        return 3
    
# ------------------------------------
# Function to predict stock severity
# ------------------------------------

def predict_stock_severity(df: pd.DataFrame, model, feature_cols: list, store_id=None, product_id=None) -> Tuple[Optional[Figure], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Predict stock severity (0–3) using a trained model and generate evaluation visuals.
    """
    try:
        df = df.copy()
        df['dt'] = pd.to_datetime(df['dt'])

        # Filter if needed
        title_parts = []
        if store_id is not None:
            df = df[df["store_id"] == store_id]
            title_parts.append(f"Store {store_id}")
        if product_id is not None:
            df = df[df["product_id"] == product_id]
            title_parts.append(f"Product {product_id}")

        if df.empty:
            print("⚠️ No data after filtering.")
            return None, None, None

        title = " | ".join(["📦 Stock Severity Prediction"] + title_parts)
        # Prepare target
        df['stock_hour6_22_cnt'] = df['stock_hour6_22_cnt'].fillna(0).clip(lower=0, upper=16).astype(int)
        df['severity'] = df['stock_hour6_22_cnt'].apply(bin_stock_severity)

        X = df[feature_cols]
        y_true = df['severity']
        y_classes = sorted(y_true.unique())

        # Predict
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
        else:
            print("⚠️ Model doesn't support probability prediction.")
            y_proba = np.zeros((len(y_true), len(y_classes)))

        # === Evaluation Metrics ===
        metrics: Dict[str, float] = {
            "✅ Accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
            "🎯 Precision": round(float(precision_score(y_true, y_pred, average='macro')), 3),
            "🔁 Recall": round(float(recall_score(y_true, y_pred, average='macro')), 3),
            "📏 F1 Score": round(float(f1_score(y_true, y_pred, average='macro')), 3)
        }

        # === Results Table ===
        result_df = df[['dt', 'store_id', 'product_id']].copy()
        result_df['True Severity'] = y_true
        result_df['Predicted Severity'] = y_pred
        result_df['Max Probability'] = y_proba.max(axis=1) if y_proba.shape[1] > 0 else 0.0
        result_df.rename(columns={'dt': 'Date'}, inplace=True)

        # === Plots ===
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve (if supported)
        if y_proba.shape[1] == len(y_classes):
            y_true_bin =  np.array(label_binarize(y_true, classes=y_classes))
            for i, cls in enumerate(y_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                axs[0].plot(fpr, tpr, label=f"Class {cls} (AUC={auc:.2f})")

            axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
            axs[0].set_title("🔍 ROC Curve (One-vs-Rest)", fontsize=14)
            axs[0].set_xlabel("False Positive Rate")
            axs[0].set_ylabel("True Positive Rate")
            axs[0].legend()
            axs[0].grid(True)
        else:
            axs[0].text(0.5, 0.5, "Probability not available", ha='center', va='center')
            axs[0].set_title("ROC Curve - Not Available")

        # Probability Histogram
        if y_proba.shape[1] == len(y_classes):
            for i, cls in enumerate(y_classes):
                sns.histplot(y_proba[:, i], bins=30, kde=True, ax=axs[1], label=f"Class {cls}", alpha=0.6)
            axs[1].set_title("📊 Probability Distribution", fontsize=14)
            axs[1].set_xlabel("Predicted Probability")
            axs[1].set_ylabel("Frequency")
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].text(0.5, 0.5, "Probability not available", ha='center', va='center')
            axs[1].set_title("Probability Histogram - Not Available")

        fig.suptitle(title, fontsize=16, fontweight='bold', color="#EDF0F6")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return fig, result_df, metrics

    except Exception as e:
        print(f"❌ Error in predict_stock_severity: {e}")
        traceback.print_exc()
        return None, None, None
