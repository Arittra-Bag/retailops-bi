"""
Configuration settings for RetailOps BI project
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
DATA_SCHEMAS_PATH = PROJECT_ROOT / "data" / "schemas"

# Snowflake configuration
SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "RETAIL_BI"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
}

# Spark configuration
SPARK_CONFIG = {
    "app_name": "RetailOps_BI",
    "master": "local[*]",
    "executor_memory": "2g",
    "driver_memory": "1g",
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "port": 8501,
    "host": "0.0.0.0",
}

# GenAI configuration
GENAI_CONFIG = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "langsmith_api_key": os.getenv("LANGSMITH_API_KEY"),
    "model_name": "gemini-2.5-flash",  # Default model
    "temperature": 0.1,
    "max_tokens": 1024,
}

# Data sources
DATA_SOURCES = {
    "online_retail": {
        "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "filename": "online_retail_ii.zip",
        "description": "Online Retail II Dataset from UCI ML Repository"
    },
    "kaggle_retail": {
        "url": "https://www.kaggle.com/datasets/manjeetsingh/retaildataset",
        "filename": "retail_dataset.csv",
        "description": "Retail Dataset from Kaggle (requires Kaggle API)"
    }
}

# Business metrics configuration
BUSINESS_METRICS = {
    "inventory_metrics": [
        "inventory_turnover",
        "stockout_rate", 
        "days_sales_outstanding",
        "inventory_value"
    ],
    "customer_metrics": [
        "customer_lifetime_value",
        "average_order_value",
        "purchase_frequency",
        "customer_retention_rate"
    ],
    "financial_metrics": [
        "gross_margin",
        "net_profit_margin",
        "revenue_growth",
        "cost_of_goods_sold"
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False
        }
    }
}