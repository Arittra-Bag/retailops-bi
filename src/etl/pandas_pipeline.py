"""
Pandas ETL Pipeline for RetailOps BI
Quick processing for development and smaller datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetailDataProcessor:
    """Pandas-based ETL pipeline for retail data"""
    
    def __init__(self):
        """Initialize paths"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_raw_path = self.project_root / "data" / "raw"
        self.data_processed_path = self.project_root / "data" / "processed"
        
        # Ensure processed directory exists
        self.data_processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filename: str = "online_retail_II_sample.csv") -> pd.DataFrame:
        """Load raw data"""
        try:
            file_path = self.data_raw_path / filename
            logger.info(f"Loading data from: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data"""
        try:
            logger.info("Cleaning data...")
            
            # Handle column names
            df.columns = df.columns.str.strip()
            if 'Customer ID' in df.columns:
                df = df.rename(columns={'Customer ID': 'Customer_ID'})
            
            # Convert date column
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            # Clean text fields
            df['Description'] = df['Description'].fillna('UNKNOWN').str.strip().str.upper()
            df['Country'] = df['Country'].fillna('UNKNOWN').str.strip().str.upper()
            df['StockCode'] = df['StockCode'].fillna('UNKNOWN').str.strip().str.upper()
            
            # Handle returns (negative quantities)
            df['Is_Return'] = df['Quantity'] < 0
            df['Absolute_Quantity'] = df['Quantity'].abs()
            
            # Calculate total amount
            df['Total_Amount'] = df['Absolute_Quantity'] * df['Price']
            
            # Filter invalid records
            initial_count = len(df)
            df = df[
                df['Invoice'].notna() &
                df['StockCode'].notna() &
                df['Quantity'].notna() &
                df['Price'].notna() &
                df['InvoiceDate'].notna() &
                (df['Price'] >= 0) &
                (df['Absolute_Quantity'] > 0)
            ].copy()
            
            logger.info(f"Cleaned data: {initial_count} → {len(df)} rows ({(len(df)/initial_count)*100:.1f}% retained)")
            return df
        
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features"""
        try:
            logger.info("Adding derived features...")
            
            # Date features
            df['Year'] = df['InvoiceDate'].dt.year
            df['Month'] = df['InvoiceDate'].dt.month
            df['Day'] = df['InvoiceDate'].dt.day
            df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
            df['Date'] = df['InvoiceDate'].dt.date
            df['Hour'] = df['InvoiceDate'].dt.hour
            
            # Revenue (considering returns)
            df['Revenue'] = np.where(df['Is_Return'], -df['Total_Amount'], df['Total_Amount'])
            
            # Product categorization
            df['Product_Category'] = 'OTHER'
            df.loc[df['Description'].str.contains('CHRISTMAS', na=False), 'Product_Category'] = 'SEASONAL'
            df.loc[df['Description'].str.contains('GIFT', na=False), 'Product_Category'] = 'GIFTS'
            df.loc[df['Description'].str.contains('BAG', na=False), 'Product_Category'] = 'BAGS'
            df.loc[df['Description'].str.contains('SET', na=False), 'Product_Category'] = 'SETS'
            df.loc[df['Description'].str.contains('LIGHT', na=False), 'Product_Category'] = 'LIGHTING'
            
            # Price ranges
            df['Price_Range'] = pd.cut(df['Price'], 
                                     bins=[0, 5, 20, 50, float('inf')], 
                                     labels=['LOW', 'MEDIUM', 'HIGH', 'PREMIUM'])
            
            # Customer type
            df['Customer_Type'] = np.where(df['Customer_ID'].isna(), 'GUEST', 'REGISTERED')
            
            logger.info("Features added successfully")
            return df
        
        except Exception as e:
            logger.error(f"Error adding features: {str(e)}")
            return df
    
    def create_aggregations(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create aggregated tables for analytics"""
        try:
            logger.info("Creating aggregated tables...")
            
            # Daily sales summary
            daily_sales = df.groupby(['Date', 'Country']).agg({
                'Revenue': 'sum',
                'Absolute_Quantity': 'sum',
                'Invoice': 'count',
                'Customer_ID': 'nunique',
                'Price': 'mean'
            }).round(2).reset_index()
            daily_sales.columns = ['Date', 'Country', 'Total_Revenue', 'Total_Quantity', 
                                  'Total_Transactions', 'Unique_Customers', 'Avg_Unit_Price']
            
            # Product performance
            product_performance = df.groupby(['StockCode', 'Description', 'Product_Category']).agg({
                'Revenue': 'sum',
                'Absolute_Quantity': 'sum',
                'Invoice': 'count',
                'Price': 'mean',
                'Customer_ID': 'nunique'
            }).round(2).reset_index()
            product_performance.columns = ['StockCode', 'Description', 'Product_Category',
                                         'Total_Revenue', 'Total_Quantity_Sold', 'Total_Orders',
                                         'Avg_Price', 'Unique_Customers']
            product_performance = product_performance.sort_values('Total_Revenue', ascending=False)
            
            # Customer analysis (registered customers only)
            customer_df = df[df['Customer_ID'].notna()].copy()
            if len(customer_df) > 0:
                customer_analysis = customer_df.groupby(['Customer_ID', 'Country']).agg({
                    'Revenue': 'sum',
                    'Invoice': 'count',
                    'Absolute_Quantity': 'sum',
                    'Total_Amount': 'mean',
                    'InvoiceDate': ['max', 'min']
                }).round(2).reset_index()
                customer_analysis.columns = ['Customer_ID', 'Country', 'Total_Spent', 'Total_Orders',
                                           'Total_Items', 'Avg_Order_Value', 'Last_Purchase_Date', 'First_Purchase_Date']
                customer_analysis = customer_analysis.sort_values('Total_Spent', ascending=False)
            else:
                customer_analysis = pd.DataFrame()
            
            # Country performance
            country_performance = df.groupby('Country').agg({
                'Revenue': 'sum',
                'Invoice': 'count',
                'Customer_ID': 'nunique',
                'Total_Amount': 'mean'
            }).round(2).reset_index()
            country_performance.columns = ['Country', 'Total_Revenue', 'Total_Transactions',
                                         'Unique_Customers', 'Avg_Transaction_Value']
            country_performance = country_performance.sort_values('Total_Revenue', ascending=False)
            
            # Monthly trends
            monthly_trends = df.groupby(['Year', 'Month']).agg({
                'Revenue': 'sum',
                'Absolute_Quantity': 'sum',
                'Invoice': 'count',
                'Customer_ID': 'nunique'
            }).round(2).reset_index()
            monthly_trends.columns = ['Year', 'Month', 'Total_Revenue', 'Total_Quantity',
                                    'Total_Transactions', 'Unique_Customers']
            
            aggregations = {
                'daily_sales': daily_sales,
                'product_performance': product_performance,
                'customer_analysis': customer_analysis,
                'country_performance': country_performance,
                'monthly_trends': monthly_trends
            }
            
            logger.info("Aggregated tables created successfully")
            return aggregations
        
        except Exception as e:
            logger.error(f"Error creating aggregations: {str(e)}")
            return {}
    
    def save_data(self, df: pd.DataFrame, aggregations: Dict[str, pd.DataFrame]):
        """Save processed data"""
        try:
            logger.info("Saving processed data...")
            
            # Save main dataset
            main_file = self.data_processed_path / "retail_transactions_processed.csv"
            df.to_csv(main_file, index=False)
            logger.info(f"Main dataset saved: {main_file}")
            
            # Save aggregations
            for name, agg_df in aggregations.items():
                if len(agg_df) > 0:
                    file_path = self.data_processed_path / f"{name}.csv"
                    agg_df.to_csv(file_path, index=False)
                    logger.info(f"Saved {name}: {len(agg_df)} rows")
            
            # Create summary file
            summary = {
                'processing_date': datetime.now().isoformat(),
                'total_records': len(df),
                'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
                'total_revenue': df['Revenue'].sum(),
                'unique_products': df['StockCode'].nunique(),
                'unique_customers': df['Customer_ID'].nunique(),
                'countries': df['Country'].nunique()
            }
            
            summary_df = pd.DataFrame([summary])
            summary_file = self.data_processed_path / "processing_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Processing summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def check_if_processing_needed(self) -> bool:
        """Check if data processing is needed by comparing file timestamps"""
        try:
            # Check if processed files exist
            processed_files = [
                'retail_transactions_processed.csv',
                'product_performance.csv', 
                'customer_analysis.csv',
                'country_performance.csv',
                'daily_sales.csv'
            ]
            
            # If any processed file is missing, we need to process
            for file in processed_files:
                if not (self.data_processed_path / file).exists():
                    logger.info(f"Missing processed file: {file} - processing needed")
                    return True
            
            # Check if raw data is newer than processed data
            raw_file = self.data_raw_path / "online_retail_II_sample.csv"
            if not raw_file.exists():
                logger.info("Raw data file not found - processing needed")
                return True
            
            # Get timestamp of oldest processed file
            processed_timestamps = []
            for file in processed_files:
                file_path = self.data_processed_path / file
                if file_path.exists():
                    processed_timestamps.append(file_path.stat().st_mtime)
            
            if not processed_timestamps:
                return True
            
            oldest_processed = min(processed_timestamps)
            raw_timestamp = raw_file.stat().st_mtime
            
            if raw_timestamp > oldest_processed:
                logger.info("Raw data is newer than processed data - processing needed")
                return True
            
            logger.info("✅ Processed data is up-to-date - skipping ETL")
            return False
            
        except Exception as e:
            logger.warning(f"Error checking timestamps: {e} - will process data")
            return True
    
    def run_pipeline(self, filename: str = "online_retail_II_sample.csv") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Run the complete pipeline with smart caching"""
        try:
            logger.info("=" * 50)
            logger.info("STARTING RETAILOPS ETL PIPELINE")
            logger.info("=" * 50)
            
            # Check if processing is needed
            if not self.check_if_processing_needed():
                logger.info("📂 Loading existing processed data...")
                df_processed = pd.read_csv(self.data_processed_path / "retail_transactions_processed.csv")
                
                # Load aggregations
                aggregations = {}
                agg_files = {
                    'product_performance': 'product_performance.csv',
                    'customer_analysis': 'customer_analysis.csv', 
                    'country_performance': 'country_performance.csv',
                    'daily_sales': 'daily_sales.csv'
                }
                
                for key, file in agg_files.items():
                    file_path = self.data_processed_path / file
                    if file_path.exists():
                        aggregations[key] = pd.read_csv(file_path)
                
                logger.info(f"✅ Loaded {len(df_processed):,} processed records from cache")
                return df_processed, aggregations
            
            logger.info("🔄 Processing data from raw files...")
            
            # Load data
            df = self.load_data(filename)
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Add features
            df_processed = self.add_features(df_clean)
            
            # Create aggregations
            aggregations = self.create_aggregations(df_processed)
            
            # Save everything
            self.save_data(df_processed, aggregations)
            
            # Show summary
            logger.info("\n" + "=" * 30)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 30)
            logger.info(f"Total records processed: {len(df_processed):,}")
            logger.info(f"Date range: {df_processed['Date'].min()} to {df_processed['Date'].max()}")
            logger.info(f"Total revenue: £{df_processed['Revenue'].sum():,.2f}")
            logger.info(f"Unique products: {df_processed['StockCode'].nunique():,}")
            logger.info(f"Unique customers: {df_processed['Customer_ID'].nunique():,}")
            logger.info(f"Countries: {df_processed['Country'].nunique()}")
            
            logger.info("\nTop 5 products by revenue:")
            top_products = aggregations['product_performance'].head()
            for _, row in top_products.iterrows():
                logger.info(f"  {row['Description'][:40]:<40} £{row['Total_Revenue']:>8,.2f}")
            
            logger.info("\nPipeline completed successfully! ✅")
            
            return df_processed, aggregations
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function"""
    processor = RetailDataProcessor()
    processor.run_pipeline()


if __name__ == "__main__":
    main()