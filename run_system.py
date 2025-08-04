import subprocess
import sys
import time
import logging
from pathlib import Path
from threading import Thread
import webbrowser

# Import ETL classes directly for better integration
try:
    from src.etl.pandas_pipeline import RetailDataProcessor
    from src.data.snowflake_connector import SnowflakeManager
    ETL_CLASSES_AVAILABLE = True
except ImportError:
    ETL_CLASSES_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_etl_pipeline():
    """Run the complete ETL pipeline including Snowflake setup with smart caching"""
    try:
        logger.info("Running Complete ETL Pipeline...")
        
        if ETL_CLASSES_AVAILABLE:
            # Use direct class calls for better integration and logging
            
            # Step 1: Run Pandas ETL Pipeline with caching
            logger.info("Processing data with Pandas...")
            try:
                processor = RetailDataProcessor()
                df_processed, aggregations = processor.run_pipeline()
                logger.info("✅ Pandas ETL Pipeline completed")
                pandas_success = True
            except Exception as e:
                logger.error(f"❌ Pandas ETL Pipeline failed: {str(e)}")
                pandas_success = False
            
            # Step 2: Setup Snowflake Warehouse with caching
            if pandas_success:
                logger.info("Setting up Snowflake warehouse...")
                try:
                    snowflake_manager = SnowflakeManager()
                    snowflake_success = snowflake_manager.setup_complete_warehouse()
                    snowflake_manager.close_connection()
                    
                    if snowflake_success:
                        logger.info("✅ Complete ETL Pipeline completed successfully")
                        logger.info("Data processed and uploaded to Snowflake cloud")
                    else:
                        logger.warning("⚠️ Pandas ETL completed, but Snowflake setup failed")
                        logger.warning("Data available locally for dashboard")
                    
                except Exception as e:
                    logger.error(f"❌ Snowflake setup failed: {str(e)}")
                    logger.warning("Data available locally for dashboard")
            
            return pandas_success
        
        else:
            # Fallback to subprocess if imports fail
            logger.warning("Using fallback subprocess method...")
            
            # Step 1: Run Pandas ETL Pipeline
            logger.info("Processing data with Pandas...")
            result = subprocess.run([
                sys.executable, "src/etl/pandas_pipeline.py"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error("Pandas ETL Pipeline failed")
                logger.error(result.stderr)
                return False
            
            logger.info("Pandas ETL Pipeline completed")
            
            # Step 2: Setup Snowflake Warehouse
            logger.info("Setting up Snowflake warehouse...")
            result = subprocess.run([
                sys.executable, "src/data/snowflake_connector.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Complete ETL Pipeline completed successfully")
                logger.info("Data processed and uploaded to Snowflake cloud")
            else:
                logger.warning("Pandas ETL completed, but Snowflake upload failed")
                logger.warning("Data available locally for dashboard")
                logger.error(result.stderr)
            
            return True
        
    except Exception as e:
        logger.error(f"Error running ETL pipeline: {str(e)}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    try:
        logger.info("Starting FastAPI server...")
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
        logger.info("FastAPI server started on http://localhost:8000")
        time.sleep(3)  # Give server time to start
        return True
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        return False

def start_streamlit_dashboard():
    """Start the Streamlit dashboard"""
    try:
        logger.info("Starting Streamlit dashboard...")
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "src/dashboard/streamlit_app.py", "--server.port", "8501"
        ])
        logger.info("Streamlit dashboard started on http://localhost:8501")
        time.sleep(3)  # Give dashboard time to start
        return True
    except Exception as e:
        logger.error(f"Error starting Streamlit dashboard: {str(e)}")
        return False

def open_browser():
    """Open browser to show the applications"""
    time.sleep(5)  # Wait for servers to start
    
    try:
        webbrowser.open("http://localhost:8000/docs")  # API documentation
        logger.info("API Docs Swagger window opened")
    except Exception as e:
        logger.error(f"Error opening browser: {str(e)}")



def main():
    """Main function to run the entire ML-powered analytics system"""
    logger.info("🎯 Starting RetailOps ML Analytics Platform")
    logger.info("=" * 50)
    # Run ETL pipeline
    if not run_etl_pipeline():
        logger.error("Failed to run ETL pipeline. Exiting.")
        return
    
    # Start API server
    if not start_api_server():
        logger.error("Failed to start API server")
        return
    
    # Start Streamlit dashboard
    if not start_streamlit_dashboard():
        logger.error("Failed to start Streamlit dashboard")
        return
    
    # Open browser
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

if __name__ == "__main__":
    main()