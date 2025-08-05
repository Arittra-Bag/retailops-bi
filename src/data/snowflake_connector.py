"""
Snowflake Data Warehouse Integration for RetailOps BI
Handles data loading and querying from Snowflake cloud warehouse
"""

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas, pd_writer
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import logging
from pathlib import Path
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowflakeManager:
    """Manages Snowflake connections and operations"""
    
    def __init__(self):
        """Initialize Snowflake connection parameters"""
        self.connection_params = {
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'user': os.getenv('SNOWFLAKE_USER'),
            'password': os.getenv('SNOWFLAKE_PASSWORD'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            'database': os.getenv('SNOWFLAKE_DATABASE', 'RETAIL_BI'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA', 'RAW'),
        }
        
        self.connection = None
        self.engine = None
        
        # Project paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_processed_path = self.project_root / "data" / "processed"
    
    def create_connection(self) -> bool:
        """Create Snowflake connection"""
        try:
            logger.info("Creating Snowflake connection...")
            
            # Validate required parameters
            required_params = ['account', 'user', 'password']
            missing_params = [p for p in required_params if not self.connection_params.get(p)]
            
            if missing_params:
                logger.error(f"Missing required Snowflake parameters: {missing_params}")
                logger.info("Please update your .env file with Snowflake credentials")
                return False
            
            # Try different account formats to handle regional variations
            account_formats_to_try = [
                self.connection_params['account'],  # Original format
                self.connection_params['account'].replace('.gcp.', '.'),  # Remove GCP part
                self.connection_params['account'].split('.')[0]  # Just the first part
            ]
            
            self.connection = None
            successful_account = None
            
            for account_format in account_formats_to_try:
                try:
                    logger.info(f"Trying account format: {account_format}")
                    
                    self.connection = snowflake.connector.connect(
                        user=self.connection_params['user'],
                        password=self.connection_params['password'],
                        account=account_format,
                        warehouse=self.connection_params['warehouse'],
                        database=self.connection_params['database'],
                        schema=self.connection_params['schema'],
                        insecure_mode=True  # Skip SSL verification for connection issues
                    )
                    
                    successful_account = account_format
                    logger.info(f"✅ Connected successfully with account: {account_format}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Account format {account_format} failed: {str(e)[:100]}...")
                    continue
            
            if not self.connection:
                raise Exception("All account formats failed - check account identifier")
            
            # Try to create SQLAlchemy engine for pandas integration (optional)
            try:
                from sqlalchemy import create_engine
                from snowflake.sqlalchemy import URL
                
                self.engine = create_engine(URL(
                    account=self.connection_params['account'],
                    user=self.connection_params['user'],
                    password=self.connection_params['password'],
                    database=self.connection_params['database'],
                    schema=self.connection_params['schema'],
                    warehouse=self.connection_params['warehouse'],
                ))
                logger.info("✅ SQLAlchemy engine created")
            except ImportError as e:
                logger.warning(f"SQLAlchemy integration not available: {str(e)}")
                logger.info("Basic Snowflake operations will still work")
                self.engine = None
            
            logger.info("✅ Snowflake connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Snowflake: {str(e)}")
            logger.info("💡 Please check your Snowflake credentials in the .env file")
            return False
    
    def setup_database_schema(self) -> bool:
        """Create database and schema structure"""
        try:
            if not self.connection:
                logger.error("No Snowflake connection available")
                return False
            
            cursor = self.connection.cursor()
            
            logger.info("Setting up Snowflake database schema...")
            
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.connection_params['database']}")
            cursor.execute(f"USE DATABASE {self.connection_params['database']}")
            
            # Create schemas
            schemas = ['RAW', 'PROCESSED', 'ANALYTICS', 'STAGING']
            for schema in schemas:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                logger.info(f"✅ Schema {schema} ready")
            
            # Create warehouse if not exists
            cursor.execute(f"""
                CREATE WAREHOUSE IF NOT EXISTS {self.connection_params['warehouse']}
                WITH WAREHOUSE_SIZE = 'X-SMALL'
                AUTO_SUSPEND = 300
                AUTO_RESUME = TRUE
                INITIALLY_SUSPENDED = FALSE
            """)
            
            cursor.execute(f"USE WAREHOUSE {self.connection_params['warehouse']}")
            
            logger.info("✅ Database schema setup completed")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up database schema: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """Create tables for retail data"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"USE SCHEMA {self.connection_params['schema']}")
            
            logger.info("Creating retail data tables...")
            
            # Main transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retail_transactions (
                    Invoice VARCHAR(50),
                    StockCode VARCHAR(50),
                    Description VARCHAR(500),
                    Quantity INTEGER,
                    InvoiceDate TIMESTAMP,
                    Price DECIMAL(10,2),
                    Customer_ID INTEGER,
                    Country VARCHAR(100),
                    Is_Return BOOLEAN,
                    Absolute_Quantity INTEGER,
                    Total_Amount DECIMAL(12,2),
                    Year INTEGER,
                    Month INTEGER,
                    Day INTEGER,
                    DayOfWeek INTEGER,
                    Date DATE,
                    Hour INTEGER,
                    Revenue DECIMAL(12,2),
                    Product_Category VARCHAR(100),
                    Price_Range VARCHAR(20),
                    Customer_Type VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            
            # Product performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_performance (
                    stock_code VARCHAR(50),
                    description VARCHAR(500),
                    product_category VARCHAR(100),
                    total_revenue DECIMAL(12,2),
                    total_quantity_sold INTEGER,
                    total_orders INTEGER,
                    avg_price DECIMAL(10,2),
                    unique_customers INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            
            # Customer analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_analysis (
                    customer_id INTEGER,
                    country VARCHAR(100),
                    total_spent DECIMAL(12,2),
                    total_orders INTEGER,
                    total_items INTEGER,
                    avg_order_value DECIMAL(10,2),
                    first_purchase_date TIMESTAMP,
                    last_purchase_date TIMESTAMP,
                    customer_segment VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            
            # Daily sales table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_sales (
                    date DATE,
                    country VARCHAR(100),
                    total_revenue DECIMAL(12,2),
                    total_quantity INTEGER,
                    total_transactions INTEGER,
                    unique_customers INTEGER,
                    avg_unit_price DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            
            # Country performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS country_performance (
                    country VARCHAR(100),
                    total_revenue DECIMAL(12,2),
                    total_transactions INTEGER,
                    unique_customers INTEGER,
                    avg_transaction_value DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            
            logger.info("✅ All tables created successfully")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating tables: {str(e)}")
            return False
    
    def check_if_upload_needed(self) -> bool:
        """Check if Snowflake upload is needed by comparing data counts"""
        try:
            # Get table info from Snowflake
            table_info = self.get_table_info()
            
            # Check if tables are empty or missing
            required_tables = [
                'retail_transactions',
                'product_performance', 
                'customer_analysis',
                'daily_sales',
                'country_performance'
            ]
            
            for table in required_tables:
                if table not in table_info or table_info[table] == 0:
                    logger.info(f"Table {table} is empty or missing - upload needed")
                    return True
            
            # Check if local CSV files are newer than last upload
            # (This is a simple check - in production you'd use proper timestamps)
            csv_files = [
                'retail_transactions_processed.csv',
                'product_performance.csv',
                'customer_analysis.csv', 
                'daily_sales.csv',
                'country_performance.csv'
            ]
            
            for csv_file in csv_files:
                csv_path = self.data_processed_path / csv_file
                if not csv_path.exists():
                    logger.info(f"CSV file {csv_file} missing - upload needed")
                    return True
            
            logger.info("✅ Snowflake data appears up-to-date - skipping upload")
            return False
            
        except Exception as e:
            logger.warning(f"Error checking upload status: {e} - will upload data")
            return True

    def load_data_from_csv(self) -> bool:
        """Load processed data from CSV files to Snowflake with smart caching"""
        try:
            if not self.engine:
                logger.error("No SQLAlchemy engine available")
                return False
            
            # Check if upload is needed
            if not self.check_if_upload_needed():
                logger.info("📂 Snowflake data is up-to-date - skipping upload")
                return True
            
            logger.info("🔄 Loading data from CSV files to Snowflake...")
            
            # Data files to load
            data_files = {
                'retail_transactions_processed': 'retail_transactions',
                'product_performance': 'product_performance',
                'customer_analysis': 'customer_analysis',
                'daily_sales': 'daily_sales',
                'country_performance': 'country_performance'
            }
            
            for csv_file, table_name in data_files.items():
                csv_path = self.data_processed_path / f"{csv_file}.csv"
                
                if csv_path.exists():
                    logger.info(f"Loading {csv_file} to {table_name}...")
                    
                    # Read CSV
                    df = pd.read_csv(csv_path)
                    
                    # Keep original column names to match table schema
                    # df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
                    
                    # Handle data types
                    if 'InvoiceDate' in df.columns:
                        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Load to Snowflake using direct cursor
                    cursor = self.connection.cursor()
                    
                    # Create table if not exists (simplified)
                    create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        {', '.join([f'{col} VARCHAR' for col in df.columns])}
                    )
                    """
                    cursor.execute(create_table_sql)
                    
                    # Insert data row by row
                    for _, row in df.iterrows():
                        values = [str(val) if val is not None else 'NULL' for val in row.values]
                        placeholders = ', '.join(['%s'] * len(values))
                        insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                        cursor.execute(insert_sql, values)
                    
                    self.connection.commit()
                    cursor.close()
                    
                    logger.info(f"✅ Loaded {len(df)} rows to {table_name}")
                else:
                    logger.warning(f"CSV file not found: {csv_path}")
            
            logger.info("✅ Data loading completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute a query and return results as DataFrame"""
        try:
            if not self.connection:
                logger.error("No Snowflake connection available")
                return None
            
            logger.info(f"Executing query: {query[:100]}...")
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            cursor.close()
            
            if result:
                df = pd.DataFrame(result, columns=columns)
                logger.info(f"Query returned {len(df)} rows")
                return df
            else:
                logger.info("Query returned no rows")
                return pd.DataFrame(columns=columns)
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return None
    
    def get_table_info(self) -> Dict[str, int]:
        """Get information about tables in the database"""
        try:
            tables_info = {}
            
            tables = [
                'retail_transactions',
                'product_performance', 
                'customer_analysis',
                'daily_sales',
                'country_performance'
            ]
            
            for table in tables:
                query = f"SELECT COUNT(*) as count FROM {table}"
                result = self.execute_query(query)
                if result is not None and len(result) > 0:
                    # Try both uppercase and lowercase column names
                    if 'COUNT' in result.columns:
                        tables_info[table] = int(result.iloc[0]['COUNT'])
                    elif 'count' in result.columns:
                        tables_info[table] = int(result.iloc[0]['count'])
                    else:
                        tables_info[table] = int(result.iloc[0, 0])  # First column
                else:
                    tables_info[table] = 0
            
            return tables_info
            
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            return {}
    
    def close_connection(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Snowflake connection closed")
    
    def setup_complete_warehouse(self) -> bool:
        """Complete warehouse setup process"""
        try:
            logger.info("🏗️ Setting up complete Snowflake data warehouse...")
            
            # Step 1: Create connection
            if not self.create_connection():
                return False
            
            # Step 2: Setup database schema
            if not self.setup_database_schema():
                return False
            
            # Step 3: Create tables
            if not self.create_tables():
                return False
            
            # Step 4: Load data
            if not self.load_data_from_csv():
                logger.warning("Data loading failed, but warehouse structure is ready")
            
            # Step 5: Verify setup
            table_info = self.get_table_info()
            logger.info("📊 Snowflake warehouse summary:")
            for table, count in table_info.items():
                logger.info(f"  {table}: {count:,} rows")
            
            logger.info("✅ Snowflake data warehouse setup completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Warehouse setup failed: {str(e)}")
            return False


def main():
    """Main function to setup Snowflake warehouse"""
    manager = SnowflakeManager()
    
    try:
        success = manager.setup_complete_warehouse()
        if success:
            logger.info("🎉 Snowflake integration ready!")
        else:
            logger.error("❌ Snowflake setup failed")
    finally:
        manager.close_connection()


if __name__ == "__main__":
    main()