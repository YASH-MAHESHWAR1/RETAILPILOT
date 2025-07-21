import streamlit as st
from utils.load_models import load_model
from utils.config import feature_cols
from utils import forecast, predict, visualization
from utils.styling import inject_custom_css, create_metric_card, create_status_indicator, create_alert, create_loading_spinner
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Helper function for consistent numeric formatting ---
def format_display_value(value, decimals=4):
    """Formats a numeric value to a specified number of decimal places,
    otherwise returns the value as a string."""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RETAILPILOT", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📦"
)

# --- INJECT CUSTOM CSS ---
inject_custom_css()

# --- TITLE ---
st.markdown('<h1 class="main-title">📦RETAILPILOT : Smart Retail Inventory Management</h1>', unsafe_allow_html=True)

# --- LOAD DATA AND MODELS ---
# Load data with progress indicator
if "train_df" not in st.session_state or "eval_df" not in st.session_state:
    with st.spinner("Loading preprocessed data..."):
        from utils.preprocessing import preprocess_data_files
        train_df, eval_df = preprocess_data_files(
            raw_train_path="data/train.parquet",
            raw_eval_path="data/eval.parquet", 
            processed_train_path="data/train_processed.parquet",
            processed_eval_path="data/test_processed.parquet"
        )
        st.session_state.train_df = train_df
        st.session_state.eval_df = eval_df
else:
    train_df = st.session_state.train_df
    eval_df = st.session_state.eval_df

# Load models with progress indicator
if "sale_model" not in st.session_state or "stock_model" not in st.session_state or "oos_model" not in st.session_state:
    with st.spinner("Loading ML models..."):
        sale_model, stock_model, oos_model = load_model()
        st.session_state.sale_model = sale_model
        st.session_state.stock_model = stock_model
        st.session_state.oos_model = oos_model
else:
    sale_model = st.session_state.sale_model
    stock_model = st.session_state.stock_model
    oos_model = st.session_state.oos_model

# Get options for filters
store_options = sorted(train_df['store_id'].unique())
product_options = sorted(train_df['product_id'].unique())

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="nav-title">🧭 NAVIGATION</div>', unsafe_allow_html=True)
    
    section = st.radio(
        "Select Section",
        ["🏠 Dashboard", "📊 Analytics", "🔮 Predictions", "📈 Forecasting"],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="section-header">🔧 FILTERS</div>', unsafe_allow_html=True)
    
    store_id = st.selectbox(
        "🏪 Store ID", 
        options=[None] + store_options, 
        index=0,
        help="Select a specific store for analysis"
    )
    
    product_id = st.selectbox(
        "📦 Product ID", 
        options=[None] + product_options, 
        index=0,
        help="Select a specific product for analysis"
    )
    
    # Data summary
    st.markdown('<div class="section-header">📊 DATA SUMMARY</div>', unsafe_allow_html=True)
    
    total_stores = len(train_df['store_id'].unique())
    total_products = len(train_df['product_id'].unique())
    total_records = len(train_df)
    date_range = f"{train_df['dt'].min().strftime('%Y-%m-%d')} to {train_df['dt'].max().strftime('%Y-%m-%d')}"
    
    st.markdown(f"""
    - **Stores:** {total_stores:,}
    - **Products:** {total_products:,}
    - **Records:** {total_records:,}
    - **Date Range:** {date_range}
    """)

# ============================
#        🏠 DASHBOARD
# ============================
if section == "🏠 Dashboard":
    st.markdown('<h2 class="section-header">Welcome to Smart Inventory Management</h2>', unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🎯</div>
        <h3>Professional Inventory Intelligence Platform</h3>
        <p>Leverage ML-powered analytics to optimize your retail inventory management with real-time insights, predictive analytics, and intelligent forecasting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = train_df['sale_amount'].sum()
        # Changed to 4 decimal places
        st.markdown(create_metric_card("Total Sales Revenue", f"{total_sales:,.4f}"), unsafe_allow_html=True)
    
    with col2:
        avg_stock_hours = train_df['stock_hour6_22_cnt'].mean()
        # Changed to 4 decimal places
        st.markdown(create_metric_card("Avg Stock-Out Hours", f"{avg_stock_hours:.4f}"), unsafe_allow_html=True)
    
    with col3:
        availability_rate = (1 - avg_stock_hours/16) * 100
        # Changed to 4 decimal places
        st.markdown(create_metric_card("Availability Rate", f"{availability_rate:.4f}%"), unsafe_allow_html=True)
    
    with col4:
        active_products = len(train_df[train_df['sale_amount'] > 0]['product_id'].unique())
        # No change: this is an integer count
        st.markdown(create_metric_card("Active Products", f"{active_products:,}"), unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown('<h3 class="subsection-header">Platform Capabilities</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>Advanced Analytics</h4>
            <p>Comprehensive data exploration with interactive visualizations and trend analysis</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <h4>Predictive Intelligence</h4>
            <p>ML-powered predictions for sales, stock-outs, and inventory severity levels</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h4>Smart Forecasting</h4>
            <p>Advanced forecasting with trend signals and actionable metrics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick insights
    if store_id or product_id:
        st.markdown('<h3 class="subsection-header">Quick Insights</h3>', unsafe_allow_html=True)
        
        filtered_data = train_df.copy()
        if store_id:
            filtered_data = filtered_data[filtered_data['store_id'] == store_id]
        if product_id:
            filtered_data = filtered_data[filtered_data['product_id'] == product_id]
        
        if not filtered_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                avg_sales = filtered_data['sale_amount'].mean()
                # Changed to 4 decimal places
                st.markdown(create_metric_card("Avg Daily Sales", f"{avg_sales:.4f}"), unsafe_allow_html=True)
            
            with col2:
                stock_severity = filtered_data['stock_hour6_22_cnt'].mean()
                if stock_severity == 0:
                    severity_label = "Fully Stocked"
                elif stock_severity <= 5: # Logic remains the same
                    severity_label = "Mild"
                elif stock_severity <= 10: # Logic remains the same
                    severity_label = "Moderate"
                else:
                    severity_label = "Severe"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{create_status_indicator(severity_label)}</div>
                    <div class="metric-label">Stock Status</div>
                </div>
                """, unsafe_allow_html=True)


# ============================
#         📊 ANALYTICS
# ============================
elif section == "📊 Analytics":
    st.markdown('<h2 class="section-header">📊 Data Analytics & Insights</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        eda_task = st.selectbox(
            "Select Analysis Type",
            [
                "Sales Trend Analysis", "Stock-Out Pattern Analysis", "Sales Volume Trend",
                "Top Performing Products", "Top Performing Stores", "Top Out-of-Stock Products",
                "Weather Impact Analysis", "Discount Effectiveness", 
                "Holiday Impact Analysis"
            ]
        )
    with col2:
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.rerun()

    try:
        fig = None
        df_result = None
        stats = {}

        # Optional frequency selector for time-series trends
        if eda_task in ["Sales Trend Analysis", "Stock-Out Pattern Analysis", "Sales Volume Trend"]:
            freq = st.selectbox("Select Aggregation Frequency", options=["D", "W", "M"], index=0, help="D: Daily, W: Weekly, M: Monthly")
        else:
            freq = 'D'  # default

        # Optional Top-N selector for ranking tasks
        if eda_task in ["Top Performing Products", "Top Performing Stores", "Top Out-of-Stock Products"]:
            top_n = st.slider("Select Top N", min_value=5, max_value=50, value=10, step=1)
        else:
            top_n = 10  # default

        if eda_task == "Sales Trend Analysis":
            fig, df_result, stats = visualization.plot_sales_trend(train_df, store_id, product_id, freq)
        elif eda_task == "Stock-Out Pattern Analysis":
            fig, df_result, stats = visualization.plot_out_of_stock_trend(train_df, store_id, product_id, freq)
        elif eda_task == "Sales Volume Trend":
            fig, df_result, stats = visualization.plot_sales_volume_trend(train_df, store_id, product_id, freq)
        elif eda_task == "Top Performing Products":
            fig, df_result, stats = visualization.get_top_products_by_sales(train_df, top_n, store_id)
        elif eda_task == "Top Performing Stores":
            fig, df_result, stats = visualization.get_top_stores_by_sales(train_df, top_n, product_id)
        elif eda_task == "Top Out-of-Stock Products":
            fig, df_result, stats = visualization.top_out_of_stock_products(train_df, top_n=top_n, store_id=store_id)
        elif eda_task == "Weather Impact Analysis":
            weather_feat = st.selectbox("Weather Feature", ["avg_temperature", "avg_humidity", "precpt", "avg_wind_level"])
            fig, df_result, stats = visualization.plot_weather_vs_sales(train_df, weather_feat, store_id, product_id)
        elif eda_task == "Discount Effectiveness":
            fig, df_result, stats = visualization.plot_discount_vs_sales(train_df, store_id, product_id)
        elif eda_task == "Holiday Impact Analysis":
            fig, df_result, stats = visualization.plot_holiday_vs_sales(train_df, store_id, product_id)

        # 🔍 Render Plot with Styling
        if fig:
            # Matplotlib styling for white elements
            fig.patch.set_facecolor('#262730')
            for ax in fig.get_axes():
                ax.set_facecolor('#262730')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
            st.pyplot(fig)

        # 📊 Show Key Metrics
        if stats:
            st.markdown('<h3 class="subsection-header">Key Metrics</h3>', unsafe_allow_html=True)
            metric_cols = st.columns(min(4, len(stats)))
            for i, (key, value) in enumerate(stats.items()):
                with metric_cols[i % 4]:
                    # Using the helper function for formatting
                    st.markdown(create_metric_card(key, format_display_value(value)), unsafe_allow_html=True)

        # 📋 Show Table
        if df_result is not None and not df_result.empty:
            # Round numeric columns in df_result for display
            for col in df_result.select_dtypes(include=['float64', 'int64']).columns:
                # Exclude columns that are likely identifiers or dates
                if col not in ['store_id', 'product_id', 'dt', 'weekday', 'month', 'year']: 
                    df_result[col] = df_result[col].round(4)
            st.markdown('<h3 class="subsection-header">Data Details</h3>', unsafe_allow_html=True)
            st.dataframe(df_result, use_container_width=True)
        elif not fig:
            st.markdown(create_alert("No data available for the selected filters.", "warning"), unsafe_allow_html=True)

    except Exception as e:
        st.markdown(create_alert(f"Error in analysis: {str(e)}", "error"), unsafe_allow_html=True)



# ============================
#       🔮 PREDICTIONS
# ============================
elif section == "🔮 Predictions":
    st.markdown('<h2 class="section-header">AI Predictions & Model Performance</h2>', unsafe_allow_html=True)
    
    # Prediction task selection
    col1, col2 = st.columns([3, 1])
    with col1:
        task = st.selectbox(
            "Select Prediction Model",
            ["Sales Volume Prediction", "Stock-Out Hours Prediction", "Inventory Severity Classification"]
        )
    
    with col2:
        if st.button("🔮 Run Prediction", type="primary"):
            with st.spinner("Running AI predictions..."):
                # Force refresh (actual prediction logic is within predict functions)
                pass 
    
    try:
        fig = None
        df_result = None
        metrics = {}
        
        if task == "Sales Volume Prediction":
            fig, df_result, metrics = predict.predict_sales(sale_model, eval_df, feature_cols, 'sale_amount', store_id, product_id)
        elif task == "Stock-Out Hours Prediction":
            fig, df_result, metrics = predict.predict_stockout_hours(eval_df, stock_model, feature_cols, store_id, product_id)
        elif task == "Inventory Severity Classification":
            fig, df_result, metrics = predict.predict_stock_severity(eval_df, oos_model, feature_cols, store_id, product_id)
        
        if fig is not None:
            # Apply custom styling for Matplotlib plots
            fig.patch.set_facecolor('#262730')
            for ax in fig.get_axes():
                ax.set_facecolor('#262730')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
            
            st.pyplot(fig)
            
            # Model performance metrics
            if metrics:
                st.markdown('<h3 class="subsection-header">Model Performance</h3>', unsafe_allow_html=True)
                metric_cols = st.columns(min(4, len(metrics)))
                for i, (key, value) in enumerate(metrics.items()):
                    with metric_cols[i % 4]:
                        display_value = format_display_value(value) # Using the helper function
                        # Color code based on metric type
                        color = "normal"
                        if "R²" in key and isinstance(value, (int, float)): # Ensure 'value' is numeric before comparison
                            color = "positive" if value > 0.7 else "negative" if value < 0.5 else "normal"
                        
                        st.markdown(create_metric_card(key, display_value), unsafe_allow_html=True)
            
            # Prediction results
            if df_result is not None and not df_result.empty:
                st.markdown('<h3 class="subsection-header">Prediction Results</h3>', unsafe_allow_html=True)
                
                # Round numeric columns in df_result for display, excluding specific columns
                for col in df_result.select_dtypes(include=['float64', 'int64']).columns:
                    if col not in ['store_id', 'product_id', 'dt', 'Severity', 'True Severity', 'Predicted Severity', 'weekday', 'month', 'year']: 
                        df_result[col] = df_result[col].round(4)

                # Add severity indicators for classification tasks
                if "Severity" in df_result.columns or "True Severity" in df_result.columns:
                    df_display = df_result.copy()
                    # Apply status indicators only for columns that hold severity class
                    if 'True Severity' in df_display.columns:
                        df_display['True Severity'] = df_display['True Severity'].apply(
                            lambda x: create_status_indicator({0: "Fully Stocked", 1: "Mild", 2: "Moderate", 3: "Severe"}.get(x, "Unknown"))
                        )
                    if 'Predicted Severity' in df_display.columns:
                        df_display['Predicted Severity'] = df_display['Predicted Severity'].apply(
                            lambda x: create_status_indicator({0: "Fully Stocked", 1: "Mild", 2: "Moderate", 3: "Severe"}.get(x, "Unknown"))
                        )
                    st.markdown(df_display.to_html(escape=False), unsafe_allow_html=True)
                else:
                    st.dataframe(df_result, use_container_width=True)
        else:
            st.markdown(create_alert("No predictions available for the selected filters.", "warning"), unsafe_allow_html=True)
            
    except Exception as e:
        st.markdown(create_alert(f"Error in prediction: {str(e)}", "error"), unsafe_allow_html=True)
    

# ============================
#        📈 FORECASTING
# ============================
elif section == "📈 Forecasting":
    st.markdown('<h2 class="section-header">Intelligent Forecasting</h2>', unsafe_allow_html=True)
    
    # Forecasting controls
    col1, col2= st.columns([2, 1])
    with col1:
        days = st.slider("Forecast Horizon (Days)", 1, 30, 7, help="Number of days to forecast into the future")
    with col2:
        if st.button("📈 Generate Forecast", type="primary"):
            with st.spinner("Generating intelligent forecasts..."):
                pass # The actual forecast generation happens below
    filtered_df = eval_df.copy()
    
    if store_id:
        filtered_df = filtered_df[filtered_df["store_id"] == store_id]
    if product_id:
        filtered_df = filtered_df[filtered_df["product_id"] == product_id]
    
    if filtered_df.empty:
        st.markdown(create_alert("No matching data for selected filters. Please adjust your selection.", "warning"), unsafe_allow_html=True)
    else:
        try:
            if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty and "dt" in filtered_df.columns:
                forecast_df, fig = forecast.forecast_sales_stock_oos(
                    sale_model, stock_model, oos_model, train_df, feature_cols, days, store_id, product_id
                )
                
                if fig:
                    # Apply custom styling for Matplotlib plots
                    fig.patch.set_facecolor('#262730')
                    for ax in fig.get_axes():
                        ax.set_facecolor('#262730')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        ax.spines['bottom'].set_color('white')
                        ax.spines['top'].set_color('white')
                        ax.spines['right'].set_color('white')
                        ax.spines['left'].set_color('white')
                    
                    st.pyplot(fig)
                    
                    # Forecast summary metrics
                    if not forecast_df.empty:
                        st.markdown('<h3 class="subsection-header">Forecast Summary</h3>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_sales = forecast_df['Predicted Sales'].mean()
                            # Changed to 4 decimal places
                            st.markdown(create_metric_card("Avg Predicted Sales", f"{avg_sales:.4f}"), unsafe_allow_html=True)
                        
                        with col2:
                            avg_stock_out = forecast_df['Predicted Stock-Out Hours (6–22)'].mean()
                            # Changed to 4 decimal places
                            st.markdown(create_metric_card("Avg Stock-Out Hours", f"{avg_stock_out:.4f}"), unsafe_allow_html=True)
                        
                        with col3:
                            risk_days = len(forecast_df[forecast_df['Predicted Severity Class'] >= 2])
                            # No change: this is an integer count
                            st.markdown(create_metric_card("High Risk Days", f"{risk_days}/{days}"), unsafe_allow_html=True)
                        
                        with col4:
                            total_revenue = forecast_df['Predicted Sales'].sum()
                            # Changed to 4 decimal places
                            st.markdown(create_metric_card("Total Forecast Revenue", f"{total_revenue:.4f}"), unsafe_allow_html=True)
                        
                        # Detailed forecast table
                        st.markdown('<h3 class="subsection-header">Detailed Forecast</h3>', unsafe_allow_html=True)
                        
                        # Round numeric columns in forecast_df for display
                        # Exclude 'Predicted Severity Class' as it's an integer representing a class
                        for col in forecast_df.select_dtypes(include=['float64', 'int64']).columns:
                            if col not in ['store_id', 'product_id', 'dt', 'Predicted Severity Class', 'weekday', 'month', 'year']: 
                                forecast_df[col] = forecast_df[col].round(4)

                        # Add severity indicators
                        forecast_display = forecast_df.copy()
                        forecast_display['Severity Label'] = forecast_display['Severity Label'].apply(create_status_indicator)
                        
                        st.markdown(forecast_display.to_html(escape=False), unsafe_allow_html=True)
                        
                        # Download forecast data - the DataFrame is already rounded here
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast Data",
                            data=csv,
                            file_name=f"forecast_{days}days_{store_id or 'all_stores'}_{product_id or 'all_products'}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.markdown(create_alert("Unable to generate forecast. Please try different filter settings.", "error"), unsafe_allow_html=True)
            else:
                st.markdown(create_alert("Invalid data format for forecasting.", "error"), unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(create_alert(f"Error in forecasting: {str(e)}", "error"), unsafe_allow_html=True)
    
# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #B8BCC8; margin-top: 2rem;">
    <p>🏢 Smart Inventory Management System | Powered by Machine Learning</p>
    <p style="font-size: 0.8rem;">Built with Streamlit • Enhanced Analytics • Real-time Predictions • Forecasting with Trend Indicators</p>
    <p style="margin-top: 1rem; font-size: 0.75rem; color: #888;">Made by <strong>Yash Maheshwari</strong></p>
</div>
""", unsafe_allow_html=True)
