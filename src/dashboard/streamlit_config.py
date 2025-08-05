import os
import streamlit as st

# Frontend Configuration for Local Development
def get_api_config():
    """Get API configuration based on environment"""
    
    # Local development (default)
    return {
        'API_BASE_URL': 'http://localhost:8000/api',
        'BACKEND_URL': 'http://localhost:8000',
        'ENVIRONMENT': 'development'
    }

def setup_streamlit_config():
    """Setup Streamlit configuration for deployment"""
    
    # Get configuration
    config = get_api_config()
    
    # Set page config
    st.set_page_config(
        page_title="RetailOps BI Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .big-font {
            font-size: 24px !important;
            font-weight: bold;
        }
        .status-indicator {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-online {
            background-color: #d4edda;
            color: #155724;
        }
        .status-offline {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
    """, unsafe_allow_html=True)
    
    return config

def test_backend_connection(api_url):
    """Test connection to backend API"""
    try:
        import requests
        response = requests.get(f"{api_url.replace('/api', '')}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def show_connection_status(config):
    """Show backend connection status"""
    is_connected = test_backend_connection(config['API_BASE_URL'])
    
    if is_connected:
        st.sidebar.markdown(
            f'<div class="status-indicator status-online">🟢 Backend Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            f'<div class="status-indicator status-offline">🔴 Backend Offline</div>',
            unsafe_allow_html=True
        )
        st.sidebar.warning("⚠️ Using local data fallback")
    
    return is_connected 