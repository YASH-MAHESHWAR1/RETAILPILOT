# RETAILPILOT : Smart Retail Inventory Management

## 🧠 Overview

RETAILPILOT is an ML-powered **Smart Inventory Management** application designed for retail businesses to optimize their inventory using predictive analytics. The platform enables data-driven decision-making through a suite of powerful features:

- 🧾 **Sales Forecasting**: Predict future sales trends over a given horizon.
- 📦 **Stock-Out Hour Prediction**: Anticipate the number of hours a product will be unavailable.
- 🚦 **Stock Severity Classification**: Classify inventory risk into four distinct levels.
- 🔮 **Manual "What-If" Scenarios**: Run on-demand predictions for custom, user-defined conditions.

Built with **Streamlit**, it combines a rich visual interface with machine learning-driven backend intelligence to transform raw data into actionable insights.

---

## 🧱 System Architecture

### ⚙️ Frontend Architecture
- **Framework**: Streamlit Web Application
- **UI Components**: Interactive controls with custom CSS styling for a professional look and feel.
- **Visualizations**: Matplotlib and Seaborn, styled for dark-mode compatibility.
- **Design**: Wide layout with a clear sidebar for navigation across `Dashboard`, `Analytics`, `Model Evaluation`, `Manual Prediction`, and `Forecasting` sections.

### 🧮 Backend Architecture
- **Data Handling**: Pandas for efficient data processing and manipulation.
- **Modeling**: Scikit-learn, XGBoost, and LightGBM for building robust models.
- **Model Persistence**: Joblib for saving and loading trained models efficiently.
- **Storage Format**: Parquet files for fast, compressed data access, ideal for large datasets.

---

## 🤖 Model Architecture

The system utilizes **three specialized ML models** for end-to-end inventory intelligence:

1.  **Sales Prediction Model**: Predicts the expected `sale_amount` for given features using regression-based models.
2.  **Stock-Out Hour Prediction Model**: Predicts the number of hours a product will be out-of-stock (`stock_hour6_22_cnt`) using advanced tree-based regressors.
3.  **Stock Severity Classifier**: Classifies severity into 4 levels:
    -   `0`: Fully Stocked
    -   `1`: Mild Shortage
    -   `2`: Moderate Shortage
    -   `3`: Severe Shortage

These models were selected after extensive testing and tuning, balancing predictive performance against model file size. Although some models performed slightly better, they were excluded due to their size reaching several gigabytes, making them impractical for web deployment.

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

| Field                | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| `city_id`            | Encoded city identifier                                    |
| `store_id`           | Encoded store identifier                                   |
| `product_id`         | Encoded product identifier                                 |
| `sale_amount`        | Normalized daily sale amount                               |
| `hours_sale`         | Hourly sales values (normalized)                           |
| `stock_hour6_22_cnt` | Out-of-stock hours during business time (6 AM – 10 PM)     |
| `hours_stock_status` | Binary hourly availability (0 = available, 1 = stock-out)  |
| `discount`           | Discount rate applied                                      |
| `holiday_flag`       | Holiday indicator                                          |
| `activity_flag`      | Activity-based flag (e.g., promotional event)              |
| `precpt`             | Total daily precipitation                                  |
| `avg_temperature`    | Daily average temperature                                  |
| `avg_humidity`        | Daily average humidity                                    |
| `avg_wind_level`     | Average wind level                                         |

> ⚠️ **Note:** All values are **normalized or preprocessed**. The dataset is already highly structured, reducing the need for extensive raw preprocessing during execution.

---

## 🧪 Why This Dataset?

-   It's **rich**, **realistic**, and **well-annotated**.
-   Provides **temporal**, **categorical**, and **external context features** required for intelligent forecasting.
-   Contains **precomputed out-of-stock hours**, making it uniquely suited for inventory intelligence.
-   Stock-out scenarios and retail behavior can be **studied and predicted** across granular time scales.

Because of its **large size (~2GB+)**, this project uses **preprocessed Parquet files** tailored to the models. This ensures efficient loading and faster execution. Processing the raw file during runtime could lead to memory issues. However, a dedicated method is provided to preprocess the raw dataset if the preprocessed files are unavailable.

---

## 🔧 Data Processing Pipeline

- **Feature Engineering**: Lag features, temporal features (day of week, month), and rolling moving averages.
- **Categorical Encoding**: LabelEncoder for city/store/product/category IDs.
- **Preprocessing Flow**:
  - Null value handling.
  - Feature selection based on model requirements.
  - Optional store/product filters for focused analysis.
- **Session State** is used extensively to cache models and data, preventing reloads and reducing memory overhead on each interaction.

---

## 🧠 ML Component Design

-   Multiple ML models (XGBoost, LightGBM, RandomForest, etc.) were tested.
-   Final models were selected balancing performance and size (some models >2GB were excluded).
-   ML-only approach was intentional to maintain model interpretability and focus on ML-centric solutions.
-   Hyperparameter tuning was conducted through manual sweeps and performance comparison.
-   ✅ **Note**: All performance metrics in the **Model Evaluation** section are generated using a **dedicated test dataset (`test_processed.parquet`)** that was kept **separate from the training data**, ensuring reliable and unbiased evaluation of model generalization.

---

## 📦 Functional Highlights & Key Sections

- **🏠 Dashboard**: A high-level overview of key performance indicators (KPIs) and platform capabilities.
- **📊 Analytics**: Deep-dive into historical data with interactive charts for trend analysis, top performers, and the impact of external factors like weather and holidays.
- **🧪 Model Evaluation**: Assess the performance of the trained ML models on unseen test data. Visualize prediction accuracy with plots and key metrics like R², MAE, and RMSE.
- **🔮 Manual Prediction**: A powerful "what-if" analysis tool. Manually input features like date, weather conditions, and promotional activities to get instant predictions for a specific store and product.
- **📈 Forecasting**: Generate multi-day forecasts for sales, stock-outs, and severity, providing a forward-looking view of inventory needs.

---

## 🛠 Technical Highlights

- **Session State Management**: Avoids redundant re-computation and reduces memory footprint for a faster user experience.
- **Modular Utility Files**:
  - `preprocessing.py` – Loads and preprocesses the raw datasets.
  - `predict.py` – Contains logic for the **Model Evaluation** section.
  - `manual_prediction.py` - Powers the on-demand **Manual Prediction** feature.
  - `forecast.py` – Implements multi-day demand forecasting logic.
  - `visualization.py` – A centralized module for all chart-generating functions.
- **Professional Streamlit Interface**:
  - Clean sidebar navigation.
  - Custom CSS for styled metric cards, status indicators, and tables.
  - Spinners and user-friendly alerts for clarity during operations.

---

## 🚀 Setup & Deployment

This section helps you get the RETAILPILOT app running on your local machine, and explains how it's hosted online.

### 🖥️ Local Setup Instructions

To run this project locally, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/YASH-MAHESHWAR1/RETAILPILOT.git
cd RETAILPILOT

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # On Windows use: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```
✔️ Make sure:
- Your `data/` folder contains the preprocessed `.parquet` dataset.
- You are using Python 3.8 or higher.
- You have a stable internet connection if external loading is enabled.


### 🚀 Deployment

The RETAILPILOT application is deployed using **Hugging Face Spaces** with a **Streamlit** interface.

#### ☁️ Platform Details

Hugging Face Spaces offers an ideal platform for hosting ML apps like RETAILPILOT, especially under resource constraints.

##### ✅ Why Hugging Face Spaces?

- 💾 **Higher RAM allocation** (up to 16 GB) compared to other free-tier platforms
- 📂 Efficient handling of **large datasets** and **moderately sized models**
- ⚙️ Built-in support for **Streamlit**, **Gradio**, and **Docker**
- 🆓 Completely **free-to-deploy**, no credit card required
- 🔒 Stable and secure for public sharing and collaboration

#### 🌐 Live App

The app is publicly available at:

🔗 **[Launch RETAILPILOT on Hugging Face](https://huggingface.co/spaces/Yash-M1775/RETAILPILOT)**

You can try out all features — sales forecasting, stock prediction, and severity classification — in a fully interactive interface with real data.
Along with that you can also get the deployed code in the files section of this

---


