feature_cols = [
    # ID-based categorical features
    'city_id', 'store_id', 'management_group_id',
    'first_category_id', 'second_category_id', 'third_category_id', 'product_id',

    # Time-based features
    'day_of_week', 'month', 'day_of_month', 'quarter', 'is_weekend',

    # External context features
    'discount', 'holiday_flag', 'activity_flag', 'precpt',
    'avg_temperature', 'avg_humidity', 'avg_wind_level',

    # Lag features
    'sales_lag_1', 'sales_lag_3', 'sales_lag_5', 'sales_lag_7',
    'stock_lag_1', 'stock_lag_3', 'stock_lag_5', 'stock_lag_7',

    # Moving average features
    'sales_ma_3', 'sales_ma_5', 'sales_ma_7',
    'stock_ma_3', 'stock_ma_5', 'stock_ma_7',

]
