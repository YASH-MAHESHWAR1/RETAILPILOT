
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb

def train_and_evaluate_sale_models(df, features, target):
    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.03, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, random_state=42)
    }

    results = []  # Store results for all models

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        results.append({
            'Model': name,
            'MAE': mae,
            'R2': r2
        })

        print(f"\n {name} - MAE for {target}: {mae:.2f}, R²: {r2:.4f}")

        # Visualize prediction vs actual
        plt.figure(figsize=(10, 4))
        plt.plot(y_val.values[:100], label='Actual', marker='o')
        plt.plot(y_pred[:100], label='Predicted', marker='x')
        plt.title(f"{name} - Prediction vs Actual ({target})")
        plt.xlabel("Index")
        plt.ylabel(target)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).sort_values(by='MAE')

    # Normalize MAE and R2 for scoring (lower MAE is better, higher R2 is better)
    mae_max = results_df['MAE'].max()
    mae_min = results_df['MAE'].min()
    r2_min = results_df['R2'].min()
    r2_max = results_df['R2'].max()

    results_df['MAE_Score'] = 1 - (results_df['MAE'] - mae_min) / (mae_max - mae_min + 1e-9)
    results_df['R2_Score'] = (results_df['R2'] - r2_min) / (r2_max - r2_min + 1e-9)
    results_df['Combined_Score'] = (results_df['MAE_Score'] + results_df['R2_Score']) / 2

    # Sort by best combined score
    results_df = results_df.sort_values(by='Combined_Score', ascending=False)


    #  Final MAE Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='MAE', data=results_df, palette='Blues_r')
    plt.title(f" Model MAE Comparison for {target}")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #  Final R2 Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='R2', data=results_df, palette='Greens_r')
    plt.title(f" Model R² Comparison for {target}")
    plt.ylabel("R² Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Model', y='Combined_Score', data=results_df, palette='Purples_r')
    plt.title(f" Combined Score (MAE + R²) Comparison for {target}")
    plt.ylabel("Combined Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Return best model (based on MAE)
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    print(f"\n Best Model for {target}: {best_model_name} with MAE = {results_df.iloc[0]['MAE']:.2f} and R² = {results_df.iloc[0]['R2']:.4f}")
    return best_model


