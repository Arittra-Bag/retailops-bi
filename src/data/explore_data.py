"""
Quick data exploration script for RetailOps BI
Converts Excel to CSV and provides basic analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


def convert_and_explore():
    """Convert Excel to CSV and provide basic exploration"""
    
    excel_file = DATA_RAW_PATH / "online_retail_II.xlsx"
    csv_file = DATA_RAW_PATH / "online_retail_II.csv"
    
    logger.info(f"Excel file exists: {excel_file.exists()}")
    logger.info(f"CSV file exists: {csv_file.exists()}")
    
    # Convert Excel to CSV if needed
    if excel_file.exists() and not csv_file.exists():
        logger.info("Converting Excel to CSV...")
        
        # Read full dataset
        logger.info("Reading Excel file (this may take a moment)...")
        df = pd.read_excel(excel_file)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Save as CSV
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV to: {csv_file}")
        
    elif csv_file.exists():
        logger.info("Loading existing CSV file...")
        df = pd.read_csv(csv_file)
        logger.info(f"CSV dataset shape: {df.shape}")
    else:
        logger.error("No data file found!")
        return None
    
    # Basic exploration
    logger.info("\n" + "="*50)
    logger.info("DATASET OVERVIEW")
    logger.info("="*50)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    logger.info("\nData types:")
    for col in df.columns:
        logger.info(f"  {col}: {df[col].dtype}")
    
    logger.info("\nMemory usage:")
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  Total: {memory_mb:.2f} MB")
    
    logger.info("\nMissing values:")
    missing = df.isnull().sum()
    for col in missing.index:
        if missing[col] > 0:
            pct = (missing[col] / len(df)) * 100
            logger.info(f"  {col}: {missing[col]:,} ({pct:.1f}%)")
    
    logger.info("\nFirst 5 rows:")
    print(df.head())
    
    logger.info("\nNumerical statistics:")
    print(df.describe())
    
    # Save sample for quick testing
    sample_file = DATA_RAW_PATH / "online_retail_II_sample.csv"
    if not sample_file.exists():
        sample_df = df.sample(n=min(10000, len(df)), random_state=42)
        sample_df.to_csv(sample_file, index=False)
        logger.info(f"Saved sample dataset to: {sample_file}")
    
    return df


if __name__ == "__main__":
    convert_and_explore()