import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
from matplotlib.figure import Figure

# ------------------------------------
# Function to plot sales volume trend
# ------------------------------------

def plot_sales_volume_trend(df, store_id=None, product_id=None, freq='D'):
    """
    Plots daily/weekly/monthly sales volume trend (stock_hour6_22_cnt).

    Parameters:
    - df (pd.DataFrame): Must contain ['dt', 'stock_hour6_22_cnt', 'store_id', 'product_id']
    - store_id (int, optional): Store to filter
    - product_id (int, optional): Product to filter
    - freq (str): 'D' (daily), 'W' (weekly), 'M' (monthly)

    Returns:
    - fig (plt.Figure): Matplotlib figure object
    - display_df (pd.DataFrame): Aggregated volume data
    - stats (dict): Summary stats
    """
    try:
        df = df.copy()
        df['dt'] = pd.to_datetime(df['dt'])

        # Filter
        if store_id is not None:
            df = df[df['store_id'] == store_id]
        if product_id is not None:
            df = df[df['product_id'] == product_id]

        if df.empty:
            return None, None, None

        label = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}.get(freq.upper(), 'Daily')
        title = f"{label} Sales Volume Trend"
        if store_id is not None:
            title += f" | Store {store_id}"
        if product_id is not None:
            title += f" | Product {product_id}"

        # Aggregate
        agg_df = df.resample(freq, on='dt')['stock_hour6_22_cnt'].sum().reset_index()

        # Stats
        stats = {
            "🗓 Total Periods": len(agg_df),
            "📦 Total Volume": int(agg_df['stock_hour6_22_cnt'].sum()),
            "📈 Max Volume": int(agg_df['stock_hour6_22_cnt'].max()),
            "📉 Min Volume": int(agg_df['stock_hour6_22_cnt'].min()),
            "📊 Avg Volume": round(agg_df['stock_hour6_22_cnt'].mean(), 2)
        }

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.lineplot(data=agg_df, x='dt', y='stock_hour6_22_cnt', marker='o', linewidth=1.5, color='#e76f51', ax=ax)
        ax.set_title(title, fontsize=13, weight='bold')
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Sales Volume", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='x', rotation=0)
        # fig.tight_layout()

        display_df = agg_df.rename(columns={'dt': 'Date', 'stock_hour6_22_cnt': 'Total Sales Volume'})

        return fig, display_df, stats

    except Exception as e:
        print(f"❌ Error in plot_sales_volume_trend: {e}")
        return None, None, None

# -----------------------------
# Function to plot sales trend
# ------------------------------

def plot_sales_trend(df, store_id=None, product_id=None, freq='D'):
    """
    Plots daily/weekly/monthly sales amount trend only.

    Parameters:
    - df (pd.DataFrame): Must contain ['dt', 'sale_amount', 'store_id', 'product_id']
    - store_id (int, optional): Store to filter
    - product_id (int, optional): Product to filter
    - freq (str): 'D' (daily), 'W' (weekly), 'M' (monthly)

    Returns:
    - fig (plt.Figure): Matplotlib figure object
    - display_df (pd.DataFrame): Aggregated sales data
    - stats (dict): Summary stats
    """
    try:
        df = df.copy()
        df['dt'] = pd.to_datetime(df['dt'])

        # Filter
        if store_id is not None:
            df = df[df['store_id'] == store_id]
        if product_id is not None:
            df = df[df['product_id'] == product_id]

        if df.empty:
            return None, None, None

        label = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}.get(freq.upper(), 'Daily')
        title = f"{label} Sales Amount Trend"
        if store_id is not None:
            title += f" | Store {store_id}"
        if product_id is not None:
            title += f" | Product {product_id}"

        # Aggregate
        agg_df = df.resample(freq, on='dt')['sale_amount'].sum().reset_index()

        # Stats
        stats = {
            "🗓 Total Periods": len(agg_df),
            "💰 Total Sales": round(agg_df['sale_amount'].sum(), 2),
            "📈 Max Sales": round(agg_df['sale_amount'].max(), 2),
            "📉 Min Sales": round(agg_df['sale_amount'].min(), 2),
            "📊 Avg Sales": round(agg_df['sale_amount'].mean(), 2)
        }

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.lineplot(data=agg_df, x='dt', y='sale_amount', marker='o', linewidth=1.5, color='#2a9d8f', ax=ax)
        ax.set_title(title, fontsize=13, weight='bold')
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Sales Amount", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='x', rotation=0)
        # fig.tight_layout()

        display_df = agg_df.rename(columns={'dt': 'Date', 'sale_amount': 'Total Sale Amount'})

        return fig, display_df, stats

    except Exception as e:
        print(f"❌ Error in plot_sales_trend: {e}")
        return None, None, None

# --------------------------------------
# Function to get top products by sales
# --------------------------------------

def get_top_products_by_sales(df, top_n=10, store_id=None):
    """
    Returns figure, table and summary stats for top N products by total sales.

    Parameters:
    - df (pd.DataFrame): DataFrame with at least ['product_id', 'sale_amount', 'store_id']
    - top_n (int): Number of top products to return
    - store_id (int, optional): If provided, filters for that store

    Returns:
    - fig (plt.Figure): Bar plot of top N products
    - display_df (pd.DataFrame): Data used in the plot
    - stats (dict): Summary stats like total sales, top product, etc.
    """

    df = df.copy()

    try:
        if store_id is not None:
            df = df[df['store_id'] == store_id]
            title = f"Top {top_n} Products in Store {store_id}"
        else:
            title = f"Top {top_n} Products Overall"

        if df.empty:
            print("⚠️ No data found for the selected filters.")
            return None, pd.DataFrame(), {}

        # Get top products
        top_products = (
            df.groupby('product_id')['sale_amount']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        # --- Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.barplot(
            data=top_products,
            x='product_id',
            y='sale_amount',
            palette='viridis',
            ax=ax
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Product ID", fontsize=10)
        ax.set_ylabel("Total Sale Amount", fontsize=10)
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # fig.tight_layout()

        # --- Summary Stats
        stats = {
            "Total Sale Amount": top_products['sale_amount'].sum(),
            "Max Sale Product": int(top_products.iloc[0]['product_id']),
            "Max Sale Amount": top_products.iloc[0]['sale_amount'],
            "Mean Sale Amount": round(top_products['sale_amount'].mean(), 2),
            "Products Considered": len(top_products)
        }

        return fig, top_products.rename(columns={
            "product_id": "Product ID", "sale_amount": "Total Sale Amount"
        }), stats

    except Exception as e:
        print(f"❌ Error in get_top_products_by_sales: {e}")
        return None, pd.DataFrame(), {}

# --------------------------------------
# Function to get top stores by sales
# --------------------------------------

def get_top_stores_by_sales(df, top_n=10, product_id=None):
    """
    Returns figure, table, and summary stats for top N stores by total sales.

    Parameters:
    - df (pd.DataFrame): DataFrame with at least ['store_id', 'sale_amount', 'product_id']
    - top_n (int): Number of top stores to return
    - product_id (int, optional): If provided, filters by specific product_id

    Returns:
    - fig (plt.Figure): Bar plot of top N stores
    - display_df (pd.DataFrame): Data used in the plot
    - stats (dict): Summary statistics for interpretation
    """
    df = df.copy()

    try:
        # Filter by product_id if provided
        if product_id is not None:
            df = df[df['product_id'] == product_id]
            title = f"Top {top_n} Stores for Product {product_id}"
        else:
            title = f"Top {top_n} Stores Overall"

        if df.empty:
            print("⚠️ No data found for the selected filters.")
            return None, pd.DataFrame(), {}

        # Aggregate sales by store
        top_stores = (
            df.groupby('store_id')['sale_amount']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        # --- Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.barplot(
            data=top_stores,
            x='store_id',
            y='sale_amount',
            palette='crest',
            ax=ax
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Store ID", fontsize=10)
        ax.set_ylabel("Total Sale Amount", fontsize=10)
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # fig.tight_layout()

        # --- Summary stats
        stats = {
            "Total Sale Amount": top_stores['sale_amount'].sum(),
            "Max Sale Store": int(top_stores.iloc[0]['store_id']),
            "Max Sale Amount": top_stores.iloc[0]['sale_amount'],
            "Mean Sale Amount": round(top_stores['sale_amount'].mean(), 2),
            "Stores Considered": len(top_stores)
        }

        return fig, top_stores.rename(columns={
            "store_id": "Store ID", "sale_amount": "Total Sale Amount"
        }), stats

    except Exception as e:
        print(f"❌ Error in get_top_stores_by_sales: {e}")
        return None, pd.DataFrame(), {}

# --------------------------------------
# Function to plot discount vs sales
# --------------------------------------

def plot_discount_vs_sales(df, store_id=None, product_id=None):
    """
    Plots a scatterplot showing the relationship between discount and sale_amount.

    Parameters:
    - df (pd.DataFrame): Must contain ['discount', 'sale_amount', 'store_id', 'product_id']
    - store_id (int, optional): Filter for a specific store
    - product_id (int, optional): Filter for a specific product

    Returns:
    - fig (plt.Figure): Scatter plot
    - display_df (pd.DataFrame): Data used for plotting
    - stats (dict): Correlation and descriptive summary
    """
    try:
        df = df.copy()

        # Filter
        if store_id is not None:
            df = df[df['store_id'] == store_id]
        if product_id is not None:
            df = df[df['product_id'] == product_id]

        if df.empty or 'discount' not in df.columns or 'sale_amount' not in df.columns:
            return None, None, None

        title = "🔻 Discount vs  Sale Amount"
        if store_id is not None:
            title += f" | Store {store_id}"
        if product_id is not None:
            title += f" | Product {product_id}"

        # Limit outliers for readability (optional)
        df = df[df['discount'].between(0, 1)]  # Discounts typically range from 0.0 to 1.0
        df = df[df['sale_amount'] < df['sale_amount'].quantile(0.99)]  # Remove top 1% outliers

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.scatterplot(data=df, x='discount', y='sale_amount', alpha=0.5, s=40, color="#277da1", ax=ax)
        sns.regplot(data=df, x='discount', y='sale_amount', scatter=False, color='red', ax=ax)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Discount", fontsize=10)
        ax.set_ylabel("Sale Amount", fontsize=10)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)
        # fig.tight_layout()

        # Table Data
        display_df = df[['discount', 'sale_amount']].copy()
        display_df = display_df.rename(columns={'discount': 'Discount', 'sale_amount': 'Sale Amount'})

        # Stats
        stats = {
            "🟢 Correlation (r)": round(df['discount'].corr(df['sale_amount']), 4),
            "📊 Avg Discount": round(df['discount'].mean(), 3),
            "💰 Avg Sale Amount": round(df['sale_amount'].mean(), 2),
            "🔻 Max Discount": round(df['discount'].max(), 2),
            "🔺 Max Sale Amount": round(df['sale_amount'].max(), 2),
            "📈 Trend": "Negative" if df['discount'].corr(df['sale_amount']) < 0 else "Positive"
        }

        return fig, display_df, stats

    except Exception as e:
        print(f"❌ Error in plot_discount_vs_sales: {e}")
        return None, None, None

# ----------------------------------
# Function to plot weather vs sales
# -----------------------------------

def plot_weather_vs_sales(
    df: pd.DataFrame,
    weather_feature: str,
    store_id=None,
    product_id=None
):
    """
    Plots relationship between weather feature and sales amount.

    Parameters:
    - df (pd.DataFrame): DataFrame with weather and sales data
    - weather_feature (str): Weather column name (e.g., 'avg_temperature', 'avg_humidity')
    - store_id (int, optional): Filter for specific store
    - product_id (int, optional): Filter for specific product

    Returns:
    - fig (plt.Figure): Scatter plot with regression line
    - display_df (pd.DataFrame): Filtered data used for plotting
    - stats (dict): Correlation and summary statistics
    """
    try:
        df = df.copy()

        # Filter data
        if store_id is not None:
            df = df[df['store_id'] == store_id]
        if product_id is not None:
            df = df[df['product_id'] == product_id]

        if df.empty or weather_feature not in df.columns or 'sale_amount' not in df.columns:
            return None, None, None

        # Remove outliers and null values
        df = df.dropna(subset=[weather_feature, 'sale_amount'])
        df = df[df['sale_amount'] < df['sale_amount'].quantile(0.99)]

        title = f"{weather_feature.replace('_', ' ').title()} vs Sales"
        if store_id is not None:
            title += f" | Store {store_id}"
        if product_id is not None:
            title += f" | Product {product_id}"

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.scatterplot(data=df, x=weather_feature, y='sale_amount', alpha=0.6, s=30, color="#2a9d8f", ax=ax)
        sns.regplot(data=df, x=weather_feature, y='sale_amount', scatter=False, color='#FF6B35', ax=ax)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(weather_feature.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel("Sale Amount", fontsize=10)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)
        # fig.tight_layout()

        # Display data
        display_df = df[[weather_feature, 'sale_amount']].copy()
        display_df.columns = [weather_feature.replace('_', ' ').title(), 'Sale Amount']

        # Statistics
        correlation = df[weather_feature].corr(df['sale_amount'])
        stats = {
            "🟢 Correlation": round(correlation, 4),
            f"📊 {weather_feature.replace('_', ' ').title()}": round(df[weather_feature].mean(), 2),
            "💰 Avg Sale Amount": round(df['sale_amount'].mean(), 2),
            f"📈 {weather_feature.replace('_', ' ').title()} Range": f"{df[weather_feature].min():.1f} - {df[weather_feature].max():.1f}",
            "📈 Relationship": "Positive" if correlation > 0 else "Negative" if correlation < 0 else "Neutral"
        }

        return fig, display_df, stats

    except Exception as e:
        print(f"❌ Error in plot_weather_vs_sales: {e}")
        return None, None, None
    
# ----------------------------------
# Function to plot holiday vs sales
# ----------------------------------

def plot_holiday_vs_sales(df, store_id=None, product_id=None):
    """
    Plots the impact of holidays on sales using box plots.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'holiday_flag' and 'sale_amount'
    - store_id (int, optional): Filter for specific store
    - product_id (int, optional): Filter for specific product

    Returns:
    - fig (plt.Figure): Box plot comparing holiday vs non-holiday sales
    - display_df (pd.DataFrame): Aggregated holiday impact data
    - stats (dict): Holiday impact statistics
    """
    try:
        df = df.copy()

        # Filter data
        if store_id is not None:
            df = df[df['store_id'] == store_id]
        if product_id is not None:
            df = df[df['product_id'] == product_id]

        if df.empty or 'holiday_flag' not in df.columns or 'sale_amount' not in df.columns:
            return None, None, None

        # Remove outliers
        df = df[df['sale_amount'] < df['sale_amount'].quantile(0.99)]
        df['Holiday'] = df['holiday_flag'].map({0: 'Non-Holiday', 1: 'Holiday'})

        title = "Holiday Impact on Sales"
        if store_id is not None:
            title += f" | Store {store_id}"
        if product_id is not None:
            title += f" | Product {product_id}"

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.boxplot(data=df, x='Holiday', y='sale_amount', palette=['#2a9d8f', '#FF6B35'], ax=ax)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Day Type", fontsize=10)
        ax.set_ylabel("Sale Amount", fontsize=10)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)
        # fig.tight_layout()

        # Aggregate data
        holiday_summary = df.groupby('Holiday')['sale_amount'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2).reset_index()
        holiday_summary.columns = ['Day Type', 'Count', 'Mean Sales', 'Median Sales', 'Std Dev']

        # Statistics
        holiday_sales = df[df['holiday_flag'] == 1]['sale_amount'].mean()
        non_holiday_sales = df[df['holiday_flag'] == 0]['sale_amount'].mean()
        impact_pct = ((holiday_sales - non_holiday_sales) / non_holiday_sales * 100) if non_holiday_sales > 0 else 0

        stats = {
            "🎉 Holiday Avg Sales": round(holiday_sales, 2),
            "📅 Non-Holiday Avg Sales": round(non_holiday_sales, 2),
            "📈 Holiday Impact": f"{impact_pct:+.1f}%",
            "🎯 Holiday Days": len(df[df['holiday_flag'] == 1]),
            "📊 Total Days": len(df)
        }

        return fig, holiday_summary, stats

    except Exception as e:
        print(f"❌ Error in plot_holiday_vs_sales: {e}")
        return None, None, None
    
# --------------------------------------
# Function to plot out of stock trend
# --------------------------------------

def plot_out_of_stock_trend(df: pd.DataFrame, store_id: Optional[int] = None,
                            product_id: Optional[int] = None,
                            freq: str = 'D') -> Tuple[Optional[Figure], pd.DataFrame, Dict]:
    """
    Plot out-of-stock trend over time for specified product/store or overall.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with ['dt', 'stock_hour6_22_cnt', 'store_id', 'product_id']
    - store_id (int, optional): Filter by store
    - product_id (int, optional): Filter by product
    - freq (str): Frequency for trend line: 'D', 'W', or 'M'
    
    Returns:
    - fig: matplotlib figure
    - display_df: Aggregated DataFrame with stockout stats
    - stats: Summary statistics including average severity and stock status
    """
    try:
        df = df.copy()
        df['dt'] = pd.to_datetime(df['dt'])

        # Filtering
        title_parts = []
        if store_id is not None:
            df = df[df['store_id'] == store_id]
            title_parts.append(f"Store {store_id}")
        if product_id is not None:
            df = df[df['product_id'] == product_id]
            title_parts.append(f"Product {product_id}")

        if df.empty:
            print("⚠️ No data after filtering.")
            return None, pd.DataFrame(), {}

        title = " | ".join(["Out-of-Stock Severity Trend"] + title_parts)

        # Resample
        df.set_index('dt', inplace=True)
        grouped = df['stock_hour6_22_cnt'].resample(freq).mean().reset_index()
        df.reset_index(inplace=True)

        # Compute severity %
        grouped['stocked_ratio'] = (1 - (grouped['stock_hour6_22_cnt'] / 16)) * 100
        avg_stocked_ratio = grouped['stocked_ratio'].mean()
        avg_oos_hours = grouped['stock_hour6_22_cnt'].mean()

        # Stock health status
        if avg_oos_hours <= 2:
            stock_status = "🟢 Well Stocked"
        elif avg_oos_hours <= 6:
            stock_status = "🟠 Occasionally Out of Stock"
        else:
            stock_status = "🔴 Frequently Out of Stock"

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.lineplot(data=grouped, x='dt', y='stock_hour6_22_cnt', marker='o',
                     linewidth=2, color='#e76f51', ax=ax)
        ax.set_title(title + f" ({'Daily' if freq=='D' else 'Weekly' if freq=='W' else 'Monthly'})",
                     fontsize=13, fontweight='bold')
        ax.set_ylabel("Avg Out-of-Stock Hours (6AM–10PM)", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='x', rotation=0)
        # fig.tight_layout()

        # Rename for table view
        display_df = grouped.rename(columns={
            "dt": "Date",
            "stock_hour6_22_cnt": "Avg OOS Hurs",
            "stocked_ratio": "Stocked % (Inverse of OOS)"
        })

        stats = {
            "📆 Time Periods": len(grouped),
            "🕐 Aggregation": "Daily" if freq == 'D' else "Weekly" if freq == 'W' else "Monthly",
            "⏰ Avg OOS Hours": round(float(avg_oos_hours), 2),
            "📈 Stocked Ratio": f"{round(float(avg_stocked_ratio), 2)}%",
            "📌 Stock Status": stock_status
        }

        return fig, display_df, stats

    except Exception as e:
        print(f"❌ Error in plot_out_of_stock_trend: {e}")
        return None, pd.DataFrame(), {}

# -------------------------------------------
# Function to plot top out of stock products
# -------------------------------------------

def top_out_of_stock_products(df: pd.DataFrame, top_n: int = 10,
                              store_id: Optional[int] = None) -> Tuple[Optional[Figure], pd.DataFrame, Dict]:
    """
    Plots top N products with highest total out-of-stock hours.

    Parameters:
    - df (pd.DataFrame): DataFrame with ['product_id', 'stock_hour6_22_cnt', 'store_id']
    - top_n (int): Number of top products to return
    - store_id (int, optional): Filter by specific store

    Returns:
    - fig: matplotlib figure
    - display_df: top N products with out-of-stock totals
    - stats: Summary statistics
    """
    try:
        df = df.copy()

        if store_id is not None:
            df = df[df['store_id'] == store_id]
            title = f"Top {top_n} Out-of-Stock Products in Store {store_id}"
        else:
            title = f"Top {top_n} Out-of-Stock Products Overall"

        if df.empty:
            print("⚠️ No data available.")
            return None, pd.DataFrame(), {}

        grouped = (
            df.groupby('product_id')['stock_hour6_22_cnt']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3))
        sns.barplot(data=grouped, x='product_id', y='stock_hour6_22_cnt', palette='flare', ax=ax)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Product ID", fontsize=10)
        ax.set_ylabel("Total Out-of-Stock Hours", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=7)
        ax.tick_params(axis='x', rotation=0)
        # fig.tight_layout()

       # Rename for display
        display_df = grouped.rename(columns={
            "product_id": "Product ID", "stock_hour6_22_cnt": "Total Out-of-Stock Hours"
        })

        # Summary stats
        stats = {
            "🔝 Top Products Shown": len(grouped),
            "📦 Max Out-of-Stock Hours": round(grouped['stock_hour6_22_cnt'].max(), 2),
            "📊 Avg (Top N) OOS Hours": round(grouped['stock_hour6_22_cnt'].mean(), 2),
        }

        return fig, display_df, stats

    except Exception as e:
        print(f"❌ Error in top_out_of_stock_products: {e}")
        return None, pd.DataFrame(), {}


