# RETAILPILOT : Smart Retail Inventory Management

## 🧠 Overview

This is a an ML-powered **Smart Inventory Management** application designed for retail businesses to optimize their inventory using predictive analytics. It supports multi-task predictions including:

- 🧾 **Sales Forecasting**  
- 📦 **Stock-Out Hour Prediction**  
- 🚦 **Stock Severity Classification**

Built with **Streamlit**, it combines a rich visual interface with machine learning-driven backend intelligence to enable better inventory decisions through data.

---

## 🧱 System Architecture

### ⚙️ Frontend Architecture
- **Framework**: Streamlit Web Application
- **UI Components**: Interactive controls with custom CSS styling
- **Visualizations**: Plotly, Seaborn, Matplotlib
- **Design**: Wide layout with sidebar navigation, dark mode chart compatibility

### 🧮 Backend Architecture
- **Data Handling**: Pandas for efficient processing
- **Modeling**: Scikit-learn, XGBoost, LightGBM
- **Model Persistence**: Joblib for saving/loading trained models
- **Storage Format**: Parquet files for fast and compressed data access

---

## 🤖 Model Architecture

The system utilizes **three specialized ML models** for end-to-end inventory intelligence:

1. **Sales Prediction Model**  
   Predicts the expected `sale_amount` for given features using regression-based models.

2. **Stock-Out Hour Prediction Model**  
   Predicts the number of hours a product will be out-of-stock (`stock_hour6_22_cnt`) using advanced tree-based regressors.

3. **Stock Severity Classifier**  
   Classifies severity into 4 levels:
   - `0`: Fully Stocked
   - `1`: Mild Shortage
   - `2`: Moderate Shortage
   - `3`: Severe Shortage

These models were selected after extensive testing, tuning, and comparison under file size constraints. Although some ML models performed slightly better, they were excluded due to their model size reaching several gigabytes.

---

## 🧾 Dataset: FreshRetailNet-50K

This project uses the **[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)** dataset—a large-scale benchmark tailored for demand estimation in fresh retail. It includes:

- **50,000** store-product combinations
- **90 days** of detailed **hourly** sales data
- **898** stores across **18 major cities**
- **863** fresh perishable SKUs
- Approximately **20%** of naturally occurring stock-out data
- Contextual covariates such as weather (temperature, humidity, wind), holiday flag, promotions, discounts, and activities

### Key Fields:

| Field                  | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `city_id`              | Encoded city identifier                                                     |
| `store_id`             | Encoded store identifier                                                    |
| `product_id`           | Encoded product identifier                                                  |
| `sale_amount`          | Normalized daily sale amount                                                |
| `hours_sale`           | Hourly sales values (normalized)                                            |
| `stock_hour6_22_cnt`   | Out-of-stock hours during business time (6 AM – 10 PM)                      |
| `hours_stock_status`   | Binary hourly availability (0 = available, 1 = stock-out)                   |
| `discount`             | Discount rate applied                                                       |
| `holiday_flag`         | Holiday indicator                                                           |
| `activity_flag`        | Activity-based flag (e.g., promotional event)                               |
| `precpt`               | Total daily precipitation                                                   |
| `avg_temperature`      | Daily average temperature                                                   |
| `avg_humidity`         | Daily average humidity                                                      |
| `avg_wind_level`       | Average wind level                                                          |

> ⚠️ **Note:** All values are **normalized or preprocessed**. The dataset is already highly structured, reducing the need for extensive raw preprocessing during execution.

---

## 🧪 Why This Dataset?

- It's **rich**, **realistic**, and **well-annotated**.
- Provides **temporal**, **categorical**, and **external context features** required for intelligent forecasting.
- Contains **precomputed out-of-stock hours**, making it uniquely suited for inventory intelligence.
- Stock-out scenarios and retail behavior can be **studied and predicted** across granular time scales.

Because of its **large size (~2GB+)**, this project uses **preprocessed Parquet files** that are already tailored to the models used in this application. This ensures efficient loading and faster execution. Processing the raw file during runtime could lead to crashes due to the dataset’s heavy nature. However, a dedicated method is provided to preprocess the raw Parquet dataset if the preprocessed file is unavailable.

---

## 🔧 Data Processing Pipeline

- **Feature Engineering**: Lag features, temporal features, moving averages
- **Categorical Encoding**: LabelEncoder for city/store/product/category IDs
- **Preprocessing Flow**:
  - Null handling
  - Feature selection based on model type
  - Optional store/product filters
- **Session State** is used extensively to maintain model/data objects without memory overhead

---

## 🧠 ML Component Design

- Multiple ML models (XGBoost, LightGBM, RandomForest, GradientBoosting, CatBoost, VotingRegressor and so on) were tested.
- Final models were selected balancing performance and size (some models >2GB were excluded).
- ML-only approach was intentional to maintain model interpretability and compatibility within file size constraints and make this project ML oriented.
- Hyperparameter tuning was conducted through manual sweeps and performance comparison.
- ✅ **Note**: All predictions are generated using a **dedicated test dataset (`test_processed.parquet`)** that was kept **separate from the training data**, ensuring model generalization and reliable performance evaluation.
---

## 📊 Visualization Features

- 📈 **Sales & Stock Trends**: Time series plots with custom frequency (`Daily`, `Weekly`, `Monthly`)
- 🌡️ **Weather vs Sales Analysis**: Understand how temperature/humidity affect demand
- 🛒 **Top Products/Stores**: Ranked by sales volume or out-of-stock hours
- 🎁 **Discount & Holiday Impact**: Compare sale trends during promotions or holidays
- 📉 **Severity Histograms**: Class-wise distribution of predicted severity

---

## 📦 Functional Highlights

- **Multi-model prediction interface**: One button to predict sales, stock, and severity.
- **Out-of-stock forecasting**: Predict future shortages before they occur.
- **Top-N analysis**: Choose how many top products/stores to display.
- **Interactive filtering**: Filter by `store_id`, `product_id` for focused insights.
- **Confidence-aware severity predictions** with ROC curves and histograms.
- **Trend analysis** for volume, sales, and stock with time frequency selection.

---

## 🛠 Technical Highlights

- **Session State**: Avoids redundant recomputation and reduces memory footprint
- **Modular Utility Files**:
  - `load_data.py` – Load and preprocess datasets
  - `predict.py` – Unified prediction layer
  - `forecast.py` – Multi-day demand forecasting
  - `visualization.py` – All chart-generating functions
- **Streamlit Interface**:
  - Interactive sidebar
  - Dynamic component rendering
  - Spinner and error displays for user clarity

---


