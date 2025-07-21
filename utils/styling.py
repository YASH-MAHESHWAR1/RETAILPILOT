import streamlit as st

def inject_custom_css():
    """Inject custom CSS for professional dashboard styling"""
    
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    
    /* Main App Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Hide default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Typography */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF6B35; /* fallback solid color */
        text-align: center;
        margin-bottom: 2rem;
        /* Optional: comment out gradient if Hugging Face doesn't support text clipping properly */
        background: linear-gradient(135deg, #FF6B35, #2A9D8F); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text; 
    }
    
    
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 0.5rem;
    }
    
    .subsection-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 500;
        color: #B8BCC8;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1E2936;
        border-right: 1px solid #2A2D3A;
    }
    
    .sidebar .sidebar-content {
        background-color: #1E2936;
    }
    
    /* Remove sidebar container boxes */
    .stSidebar .element-container {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    .stSidebar .stSelectbox {
        background: transparent !important;
    }
    
    .stSidebar .stSelectbox > div {
        background: transparent !important;
        border: none !important;
    }
    
    /* Navigation title styling */
    .nav-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 1.5rem 0 1rem 0;
        padding: 0.75rem 0;
        border-bottom: 3px solid #FF6B35;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Navigation Items */
    .nav-item {
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: #262730;
        border: 1px solid #2A2D3A;
        transition: all 0.3s ease;
    }
    
    .nav-item:hover {
        background-color: #FF6B35;
        transform: translateX(5px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #262730, #1E2936);
        border: 1px solid #2A2D3A;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #FF6B35;
        margin-bottom: 0.5rem;
        word-wrap: break-word;
        line-height: 1.2;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: #B8BCC8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        word-wrap: break-word;
        line-height: 1.4;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .metric-delta.positive {
        color: #4CAF50;
    }
    
    .metric-delta.negative {
        color: #F44336;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-fully-stocked {
        background-color: #4CAF50;
        color: white;
    }
    
    .status-mild {
        background-color: #2196F3;
        color: white;
    }
    
    .status-moderate {
        background-color: #FF9800;
        color: white;
    }
    
    .status-severe {
        background-color: #F44336;
        color: white;
    }
    
    /* Chart Container */
    .chart-container {
        background-color: #262730;
        border: 1px solid #2A2D3A;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Data Table Styling */
    .dataframe {
        background-color: #262730;
        border: 1px solid #2A2D3A;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: #1E2936;
        color: #FFFFFF;
        font-weight: 600;
        padding: 1rem;
        border-bottom: 2px solid #FF6B35;
    }
    
    .dataframe td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #2A2D3A;
        color: #B8BCC8;
    }
    
    .dataframe tr:hover {
        background-color: #1E2936;
    }
    
    /* Buttons */
    .stButton {
        margin: 1rem 0;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #E76F51);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 200px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background-color: #262730;
        border: 1px solid #2A2D3A;
        border-radius: 8px;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #2A2D3A;
        border-top: 4px solid #FF6B35;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Alert Styling */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background-color: rgba(76, 175, 80, 0.1);
        border-left-color: #4CAF50;
        color: #4CAF50;
    }
    
    .alert-warning {
        background-color: rgba(255, 152, 0, 0.1);
        border-left-color: #FF9800;
        color: #FF9800;
    }
    
    .alert-error {
        background-color: rgba(244, 67, 54, 0.1);
        border-left-color: #F44336;
        color: #F44336;
    }
    
    .alert-info {
        background-color: rgba(33, 150, 243, 0.1);
        border-left-color: #2196F3;
        color: #2196F3;
    }
    
    /* Welcome Section */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #262730, #1E2936);
        border-radius: 16px;
        margin: 2rem 0;
        border: 1px solid #2A2D3A;
    }
    
    .welcome-icon {
        font-size: 4rem;
        color: #FF6B35;
        margin-bottom: 1rem;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background-color: #262730;
        border: 1px solid #2A2D3A;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        color: #2A9D8F;
        margin-bottom: 1rem;
    }
    
    /* Filter Section */
    .filter-section {
        background-color: #262730;
        border: 1px solid #2A2D3A;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .filter-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
            word-wrap: break-word;
        }
        
        .section-header {
            font-size: 1.3rem;
        }
        
        .nav-title {
            font-size: 1.2rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .metric-label {
            font-size: 0.8rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.5rem;
        }
        
        .metric-value {
            font-size: 1.2rem;
        }
        
        .metric-label {
            font-size: 0.75rem;
        }
        
        .nav-title {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a professional metric card"""
    delta_class = f"metric-delta {delta_color}" if delta else ""
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {delta_html}
    </div>
    """

def create_status_indicator(status):
    """Create a status indicator badge"""
    status_map = {
        "Fully Stocked": "status-fully-stocked",
        "Mild": "status-mild", 
        "Moderate": "status-moderate",
        "Severe": "status-severe"
    }
    
    css_class = status_map.get(status, "status-mild")
    return f'<span class="status-indicator {css_class}">{status}</span>'

def create_alert(message, alert_type="info"):
    """Create a styled alert message"""
    return f"""
    <div class="alert alert-{alert_type}">
        {message}
    </div>
    """

def create_loading_spinner():
    """Create a loading spinner"""
    return """
    <div class="loading-container">
        <div class="loading-spinner"></div>
    </div>
    """
