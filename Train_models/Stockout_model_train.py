import os
import joblib
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def custom_weights(y):
    return np.where(y == 0, 1.4,
                    np.where(y >= 11, 1.3, 1.0))



def train_and_evaluate_stock_models(df, features, target='stock_hour6_22_cnt', test_size=0.2, random_state=42, save_dir='saved_models'):
    """
    Tunes XGBoost and LightGBM models, evaluates them, and saves immediately after training.
    Returns the best performing model.
    """

    os.makedirs(save_dir, exist_ok=True)

    df = df.copy()
    X = df[features]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Custom weights to deal with skew
    weights = custom_weights(y_train)

    xgb_params = {
        'n_estimators': [300, 400, 500, 800],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [6, 8, 10, 14],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [0.5, 1, 2]
    }

    lgb_params = {
        'n_estimators': [300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [6, 8, 10],
        'num_leaves': [31, 64, 128],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [0.5, 1, 2]
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- XGBoost
    print("\n Tuning XGBoost...")
    xgb = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                       random_state=random_state, n_jobs=-1)

    xgb_search = RandomizedSearchCV(
        xgb, xgb_params, n_iter=25,
        scoring='r2', cv=3, verbose=1,
        random_state=random_state, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train, sample_weight=weights)
    best_xgb = xgb_search.best_estimator_
    xgb_results = pd.DataFrame(xgb_search.cv_results_)
    xgb_results.to_csv(os.path.join(save_dir, f"xgboost_cv_results_{timestamp}.csv"), index=False)
    print(f" XGBoost CV results saved at: xgboost_cv_results_{timestamp}.csv")

    xgb_path = os.path.join(save_dir, f"xgboost_{timestamp}.pkl")
    joblib.dump(best_xgb, xgb_path)
    print(f" XGBoost model saved at: {xgb_path}")

    # --- LightGBM
    print("\n Tuning LightGBM...")
    lgbm = lgb.LGBMRegressor(objective='regression',
                             random_state=random_state, n_jobs=-1)

    lgb_search = RandomizedSearchCV(
        lgbm, lgb_params, n_iter=25,
        scoring='r2', cv=3, verbose=1,
        random_state=random_state, n_jobs=-1
    )
    lgb_search.fit(X_train, y_train, sample_weight=weights)
    best_lgb = lgb_search.best_estimator_
    # Save full search results (LightGBM)
    lgb_results = pd.DataFrame(lgb_search.cv_results_)
    lgb_results.to_csv(os.path.join(save_dir, f"lightgbm_cv_results_{timestamp}.csv"), index=False)
    print(f" LightGBM CV results saved at: lightgbm_cv_results_{timestamp}.csv")


    lgb_path = os.path.join(save_dir, f"lightgbm_{timestamp}.pkl")
    joblib.dump(best_lgb, lgb_path)
    print(f" LightGBM model saved at: {lgb_path}")

    # --- Evaluation
    models = {
        'XGBoost': best_xgb,
        'LightGBM': best_lgb
    }
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 3),
            "R²": round(r2, 4)
        })

        plt.figure(figsize=(10, 4))
        plt.plot(y_val.values[:100], label='Actual', marker='o')
        plt.plot(y_pred[:100], label='Predicted', marker='x')
        plt.title(f"{name} - Prediction vs Actual")
        plt.xlabel("Index")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    results_df = pd.DataFrame(results).sort_values(by='R²', ascending=False)
    print("\n Model Performance Summary:")
    print(results_df)

    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    print(f"\n Best Model: {best_model_name} with R² = {results_df.iloc[0]['R²']}, MAE = {results_df.iloc[0]['MAE']}")
    return best_model
