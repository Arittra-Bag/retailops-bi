"""
Data download utilities for RetailOps BI project
Downloads retail datasets for analysis
"""

import requests
import pandas as pd
import zipfile
import os
from pathlib import Path
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"


def download_file(url: str, filename: str, destination: Path) -> bool:
    """
    Download a file from URL to destination
    
    Args:
        url: URL to download from
        filename: Name to save file as
        destination: Directory to save file in
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        destination.mkdir(parents=True, exist_ok=True)
        file_path = destination / filename
        
        logger.info(f"Downloading {filename} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {filename}: {str(e)}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip file to specified directory
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted {zip_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {zip_path.name}: {str(e)}")
        return False


def download_online_retail_dataset() -> bool:
    """
    Download the Online Retail II dataset from UCI ML Repository
    
    Returns:
        bool: True if successful, False otherwise
    """
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    filename = "online_retail_ii.zip"
    
    # Download the zip file
    if download_file(url, filename, DATA_RAW_PATH):
        # Extract the zip file
        zip_path = DATA_RAW_PATH / filename
        if extract_zip(zip_path, DATA_RAW_PATH):
            # Clean up zip file
            zip_path.unlink()
            logger.info("Online Retail II dataset ready for processing")
            return True
    
    return False


def download_sample_retail_data() -> bool:
    """
    Create a sample retail dataset if other downloads fail
    This generates synthetic data for development/testing
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.info("Generating sample retail dataset...")
        
        # Generate sample data
        np.random.seed(42)
        n_records = 50000
        
        # Sample data generation
        invoice_numbers = [f"INV{100000 + i}" for i in range(n_records)]
        stock_codes = [f"SKU{np.random.randint(10000, 99999)}" for _ in range(n_records)]
        descriptions = [
            "Product A", "Product B", "Product C", "Product D", "Product E",
            "Widget X", "Widget Y", "Gadget Z", "Tool Alpha", "Tool Beta"
        ]
        
        # Generate realistic retail data
        data = {
            'InvoiceNo': np.random.choice([f"INV{i}" for i in range(10000, 60000)], n_records),
            'StockCode': np.random.choice([f"SKU{i}" for i in range(1000, 9999)], n_records),
            'Description': np.random.choice(descriptions, n_records),
            'Quantity': np.random.poisson(3, n_records) + 1,  # Realistic quantities
            'InvoiceDate': [
                datetime.now() - timedelta(days=np.random.randint(0, 365))
                for _ in range(n_records)
            ],
            'UnitPrice': np.round(np.random.exponential(15) + 5, 2),  # Realistic prices
            'CustomerID': np.random.choice(range(10000, 50000), n_records),
            'Country': np.random.choice([
                'United Kingdom', 'France', 'Germany', 'Spain', 'Italy',
                'Netherlands', 'Belgium', 'Switzerland', 'Portugal', 'Austria'
            ], n_records, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
        }
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        
        # Add calculated fields
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df['Month'] = df['InvoiceDate'].dt.to_period('M')
        df['Year'] = df['InvoiceDate'].dt.year
        
        # Save to CSV
        output_path = DATA_RAW_PATH / "sample_retail_data.csv"
        df.to_csv(output_path, index=False)
        
        # Create a smaller sample for quick testing
        sample_df = df.sample(n=5000, random_state=42)
        sample_path = DATA_RAW_PATH / "sample_retail_data_small.csv"
        sample_df.to_csv(sample_path, index=False)
        
        logger.info(f"Generated sample retail dataset with {len(df)} records")
        logger.info(f"Saved to: {output_path}")
        logger.info(f"Small sample saved to: {sample_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return False


def convert_excel_to_csv(excel_path: Path) -> bool:
    """
    Convert Excel file to CSV format
    
    Args:
        excel_path: Path to Excel file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Converting {excel_path.name} to CSV...")
        
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Save as CSV
        csv_path = excel_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Successfully converted to {csv_path.name}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {str(e)}")
        return False


def validate_dataset(file_path: Path) -> bool:
    """
    Validate downloaded dataset
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Handle both CSV and Excel files
        if file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Basic validation
        if len(df) == 0:
            logger.error("Dataset is empty")
            return False
            
        # Check for required columns (flexible)
        required_cols = ['Invoice', 'Quantity', 'Price']  # Adjusted for real dataset
        missing_cols = [col for col in required_cols if not any(col.lower() in c.lower() for c in df.columns)]
        
        if missing_cols:
            logger.warning(f"Some expected columns not found, but proceeding with available data")
            
        logger.info(f"Dataset validation passed: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False


def main():
    """
    Main function to download retail datasets
    """
    logger.info("Starting data download process...")
    
    # Ensure data directory exists
    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
    
    # Try to download real dataset first
    success = download_online_retail_dataset()
    
    if not success:
        logger.info("Real dataset download failed, generating sample data...")
        success = download_sample_retail_data()
    
    if success:
        # Check for Excel files first and convert to CSV
        excel_files = list(DATA_RAW_PATH.glob("*.xlsx"))
        csv_files = list(DATA_RAW_PATH.glob("*.csv"))
        
        if excel_files and not csv_files:
            logger.info("Found Excel file, converting to CSV...")
            for excel_file in excel_files:
                convert_excel_to_csv(excel_file)
            csv_files = list(DATA_RAW_PATH.glob("*.csv"))
        
        # Validate the dataset
        if csv_files:
            validate_dataset(csv_files[0])
            logger.info("Data download process completed successfully!")
        elif excel_files:
            validate_dataset(excel_files[0])
            logger.info("Data download process completed successfully!")
        else:
            logger.error("No data files found after download")
    else:
        logger.error("Data download process failed")


if __name__ == "__main__":
    main()