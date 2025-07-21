from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

def define_stock_severity(value):
    """
    Binning logic for stock severity based on stock_hour6_22_cnt.
    """
    if value == 0:
        return 0  # Fully Stocked
    elif 1 <= value <= 5:
        return 1  # Mild
    elif 6 <= value <= 10:
        return 2  # Moderate
    else:
        return 3  # Severe


def train_out_of_stock_severity_model(df, feature_cols, target='stock_hour6_22_cnt'):
    df = df.copy()

    # Define severity bins from stock_hour6_22_cnt
    df[target] = df[target].fillna(0).clip(lower=0, upper=16).astype(int)
    df['stock_severity'] = df[target].apply(define_stock_severity)
    y_classes = sorted(df['stock_severity'].unique())

    # Split data
    X = df[feature_cols]
    y = df['stock_severity']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    results = []
    print(" Training and evaluating classifiers...")

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        #  Save model
        joblib.dump(model, f"{name}_severity_model.pkl", compress=3)
        y_pred = model.predict(X_val)

        #  Step 3: Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro')
        rec = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')

        results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4)
        })

        # Classification report
        print(f"\nClassification Report - {name}:\n")
        print(classification_report(y_val, y_pred))

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_val, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=y_classes).plot(ax=ax, cmap="Blues")
        ax.set_title(f" Confusion Matrix - {name}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    #  Final Comparison
    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    print("\n Model Comparison Summary:")
    print(results_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df.melt(id_vars="Model"), x="Model", y="value", hue="variable", palette="Set2")
    plt.title(" Final Model Performance Comparison")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #  Return best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = classifiers[best_model_name]
    print(f"\n Best Model Selected: {best_model_name}")

    return best_model
