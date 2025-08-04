from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, date
import logging
from src.data.snowflake_connector import SnowflakeManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RetailOps BI API",
    description="Analytics API for retail business intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, be more specific
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage
data_cache = {}

def load_processed_data():
    """Load processed data from CSV files"""
    try:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "processed"
        
        # Load all processed datasets
        datasets = [
            "retail_transactions_processed",
            "daily_sales", 
            "product_performance",
            "customer_analysis",
            "country_performance",
            "monthly_trends"
        ]
        
        for dataset in datasets:
            file_path = data_path / f"{dataset}.csv"
            if file_path.exists():
                data_cache[dataset] = pd.read_csv(file_path)
                logger.info(f"Loaded {dataset}: {len(data_cache[dataset])} rows")
            else:
                logger.warning(f"Dataset not found: {file_path}")
        
        logger.info(f"Data cache loaded with {len(data_cache)} datasets")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    try:
        load_processed_data()
        logger.info("✅ API startup completed successfully")
    except Exception as e:
        logger.error(f"❌ API startup error: {str(e)}")
        # Continue anyway - API will work with empty cache

@app.get("/")
async def root():
    """Root endpoint"""
    try:
        return {
            "message": "RetailOps BI Analytics API", 
            "version": "1.0.0",
            "available_datasets": list(data_cache.keys()) if data_cache else [],
            "docs_url": "/docs",
            "status": "online"
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return {
            "message": "RetailOps BI Analytics API", 
            "version": "1.0.0",
            "status": "online",
            "docs_url": "/docs"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": len(data_cache) > 0,
        "datasets_count": len(data_cache)
    }

@app.get("/api/overview")
async def get_overview():
    """Get high-level business overview"""
    try:
        if "retail_transactions_processed" not in data_cache:
            raise HTTPException(status_code=404, detail="Transaction data not found")
        
        df = data_cache["retail_transactions_processed"]
        
        overview = {
            "total_transactions": int(len(df)),
            "total_revenue": float(df["Revenue"].sum()),
            "unique_customers": int(df["Customer_ID"].nunique()),
            "unique_products": int(df["StockCode"].nunique()),
            "countries": int(df["Country"].nunique()),
            "date_range": {
                "start": str(df["Date"].min()),
                "end": str(df["Date"].max())
            },
            "avg_order_value": float(df["Total_Amount"].mean()),
            "top_country": df["Country"].value_counts().index[0] if len(df) > 0 else None
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"Error in overview endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sales/daily")
async def get_daily_sales(
    country: Optional[str] = Query(None, description="Filter by country"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Number of records to return")
):
    """Get daily sales data"""
    try:
        if "daily_sales" not in data_cache:
            raise HTTPException(status_code=404, detail="Daily sales data not found")
        
        df = data_cache["daily_sales"].copy()
        
        # Apply filters
        if country:
            df = df[df["Country"].str.upper() == country.upper()]
        
        if start_date:
            df = df[pd.to_datetime(df["Date"]) >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[pd.to_datetime(df["Date"]) <= pd.to_datetime(end_date)]
        
        # Sort by date and limit
        df = df.sort_values("Date", ascending=False).head(limit)
        
        # Convert to JSON-serializable format
        result = df.to_dict("records")
        
        return {
            "data": result,
            "total_records": len(result),
            "filters_applied": {
                "country": country,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error in daily sales endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/performance")
async def get_product_performance(
    category: Optional[str] = Query(None, description="Product category"),
    limit: int = Query(20, description="Number of products to return")
):
    """Get product performance data"""
    try:
        if "product_performance" not in data_cache:
            raise HTTPException(status_code=404, detail="Product performance data not found")
        
        df = data_cache["product_performance"].copy()
        
        # Apply category filter
        if category:
            df = df[df["Product_Category"].str.upper() == category.upper()]
        
        # Sort by revenue and limit
        df = df.sort_values("Total_Revenue", ascending=False).head(limit)
        
        result = df.to_dict("records")
        
        return {
            "data": result,
            "total_records": len(result),
            "filters_applied": {
                "category": category,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error in product performance endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/customers/analysis")
async def get_customer_analysis(
    country: Optional[str] = Query(None, description="Filter by country"),
    min_spent: Optional[float] = Query(None, description="Minimum amount spent"),
    limit: int = Query(50, description="Number of customers to return")
):
    """Get customer analysis data"""
    try:
        if "customer_analysis" not in data_cache:
            raise HTTPException(status_code=404, detail="Customer analysis data not found")
        
        df = data_cache["customer_analysis"].copy()
        
        if len(df) == 0:
            return {"data": [], "total_records": 0, "message": "No customer data available"}
        
        # Apply filters
        if country:
            df = df[df["Country"].str.upper() == country.upper()]
        
        if min_spent:
            df = df[df["Total_Spent"] >= min_spent]
        
        # Sort by spending and limit
        df = df.sort_values("Total_Spent", ascending=False).head(limit)
        
        result = df.to_dict("records")
        
        return {
            "data": result,
            "total_records": len(result),
            "filters_applied": {
                "country": country,
                "min_spent": min_spent,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error in customer analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/countries/performance")
async def get_country_performance():
    """Get country performance data"""
    try:
        if "country_performance" not in data_cache:
            raise HTTPException(status_code=404, detail="Country performance data not found")
        
        df = data_cache["country_performance"].copy()
        result = df.to_dict("records")
        
        return {
            "data": result,
            "total_records": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error in country performance endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends/monthly")
async def get_monthly_trends():
    """Get monthly trends data"""
    try:
        if "monthly_trends" not in data_cache:
            raise HTTPException(status_code=404, detail="Monthly trends data not found")
        
        df = data_cache["monthly_trends"].copy()
        
        # Sort by year and month
        df = df.sort_values(["Year", "Month"])
        
        result = df.to_dict("records")
        
        return {
            "data": result,
            "total_records": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error in monthly trends endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/kpi")
async def get_key_metrics():
    """Get key performance indicators"""
    try:
        if "retail_transactions_processed" not in data_cache:
            raise HTTPException(status_code=404, detail="Transaction data not found")
        
        df = data_cache["retail_transactions_processed"]
        
        # Calculate KPIs
        total_revenue = float(df["Revenue"].sum())
        total_transactions = int(len(df))
        avg_order_value = float(df["Total_Amount"].mean())
        total_customers = int(df["Customer_ID"].nunique())
        
        # Calculate growth (mock data for demo)
        revenue_growth = 12.5  # Would calculate from historical data
        transaction_growth = 8.3
        
        metrics = {
            "revenue": {
                "value": total_revenue,
                "growth": revenue_growth,
                "format": "currency"
            },
            "transactions": {
                "value": total_transactions,
                "growth": transaction_growth,
                "format": "number"
            },
            "avg_order_value": {
                "value": avg_order_value,
                "growth": 5.2,
                "format": "currency"
            },
            "customers": {
                "value": total_customers,
                "growth": 15.1,
                "format": "number"
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in KPI endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/refresh")
async def refresh_data():
    """Refresh data cache"""
    try:
        data_cache.clear()
        load_processed_data()
        
        return {
            "message": "Data refreshed successfully",
            "datasets_loaded": len(data_cache),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

'''
@app.post("/api/insights/generate")
async def generate_insights():
    """Generate AI-powered business insights"""
    try:
        from src.genai.insights_generator import RetailInsightsGenerator
        
        generator = RetailInsightsGenerator()
        report = generator.generate_automated_report()
        
        if report.get('status') == 'error':
            raise HTTPException(status_code=500, detail=report.get('error'))
        
        return {
            "message": "Insights generated successfully",
            "report_id": report.get('report_id'),
            "sections": list(report.get('sections', {}).keys()),
            "timestamp": report.get('generated_at')
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
'''

@app.post("/api/insights/generate-pdf")
async def generate_insights_pdf():
    """Generate AI insights and return PDF download"""
    try:
        from src.genai.insights_generator import RetailInsightsGenerator
        from src.utils.pdf_generator import AIReportPDFGenerator
        
        # Generate AI insights
        generator = RetailInsightsGenerator()
        report = generator.generate_automated_report()
        
        if report.get('status') == 'error':
            raise HTTPException(status_code=500, detail=report.get('error'))
        
        # Generate PDF
        pdf_generator = AIReportPDFGenerator()
        pdf_path = pdf_generator.generate_pdf_from_json_report(report)
        
        return {
            "message": "PDF report generated successfully! 📊",
            "report_id": report.get('report_id'),
            "pdf_path": pdf_path,
            "download_url": f"/api/reports/download/{Path(pdf_path).name}",
            "timestamp": report.get('generated_at')
        }
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/download/{filename}")
async def download_pdf_report(filename: str):
    """Download PDF report file"""
    try:
        project_root = Path(__file__).parent.parent.parent
        reports_dir = project_root / "data" / "reports"
        file_path = reports_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/pdf',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# SPARK ANALYTICS ENDPOINTS

@app.get("/api/spark/status")
async def get_spark_status():
    """Get Apache Spark cluster status and performance metrics"""
    try:
        from src.spark.spark_customer_analytics import SparkCustomerAnalytics
        
        spark_analytics = SparkCustomerAnalytics()
        metrics = spark_analytics.get_spark_performance_metrics()
        spark_analytics.close_spark_session()
        
        return {
            "status": "active" if "error" not in metrics else "error",
            "spark_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking Spark status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/spark/customer-segmentation")
async def run_spark_customer_segmentation():
    """Run large-scale customer segmentation using Apache Spark"""
    try:
        from src.spark.spark_customer_analytics import SparkCustomerAnalytics
        
        spark_analytics = SparkCustomerAnalytics()
        result = spark_analytics.spark_customer_segmentation()
        spark_analytics.close_spark_session()
        
        if result.get("status") == "success":
            return {
                "message": "Spark customer segmentation completed! 🔥",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error in Spark customer segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/spark/ml-forecasting")
async def run_spark_ml_forecasting():
    """Run distributed ML forecasting pipeline using Spark ML"""
    try:
        from src.spark.spark_customer_analytics import SparkCustomerAnalytics
        
        spark_analytics = SparkCustomerAnalytics()
        result = spark_analytics.spark_ml_forecasting_pipeline()
        spark_analytics.close_spark_session()
        
        if result.get("status") == "success":
            return {
                "message": "Spark ML forecasting pipeline completed! 🚀",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error in Spark ML forecasting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# DEEP LEARNING ENDPOINTS

@app.get("/api/deeplearning/status")
async def get_deeplearning_status():
    """Get deep learning models status and summary"""
    try:
        from src.deep_learning.customer_behavior_models import DeepLearningCustomerModels
        dl_models = DeepLearningCustomerModels()
        summary = dl_models.get_model_summary()
        # If frameworks are not available, return a clear error but do not raise HTTPException
        tf_ok = summary.get('frameworks', {}).get('tensorflow_available', True)
        pt_ok = summary.get('frameworks', {}).get('pytorch_available', True)
        if not tf_ok or not pt_ok:
            return {
                "status": "unavailable",
                "model_summary": summary,
                "error": "TensorFlow and/or PyTorch not available. See 'frameworks' for details. Please install dependencies and check Windows DLL requirements.",
                "suggestions": [
                    "Install Microsoft Visual C++ Redistributable (Windows)",
                    "Try: pip uninstall tensorflow && pip install tensorflow-cpu",
                    "Use Python 3.8–3.10 for best compatibility"
                ],
                "timestamp": datetime.now().isoformat()
            }
        return {
            "status": "active",
            "model_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # Instead of raising HTTPException, return a JSON error
        return {
            "status": "unavailable",
            "error": f"Deep learning status check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

'''
@app.post("/api/deeplearning/train-tensorflow-churn")
async def train_tensorflow_churn_model():
    """Train TensorFlow neural network for customer churn prediction"""
    try:
        from src.deep_learning.customer_behavior_models import DeepLearningCustomerModels
        
        dl_models = DeepLearningCustomerModels()
        result = dl_models.train_tensorflow_churn_model()
        
        if result.get("status") == "success":
            return {
                "message": "TensorFlow churn model training completed! 🧠",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error training TensorFlow model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
'''

@app.post("/api/deeplearning/train-pytorch-embeddings")
async def train_pytorch_customer_embeddings():
    """Train PyTorch customer embedding model"""
    try:
        from src.deep_learning.customer_behavior_models import DeepLearningCustomerModels
        
        dl_models = DeepLearningCustomerModels()
        result = dl_models.train_pytorch_customer_embedding()
        
        if result.get("status") == "success":
            return {
                "message": "PyTorch customer embeddings training completed! 🔥",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error training PyTorch model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

'''
@app.post("/api/deeplearning/train-lstm-sequence")
async def train_lstm_sequence_model():
    """Train LSTM model for customer purchase sequence prediction"""
    try:
        from src.deep_learning.customer_behavior_models import DeepLearningCustomerModels
        
        dl_models = DeepLearningCustomerModels()
        result = dl_models.train_lstm_customer_sequence()
        
        if result.get("status") == "success":
            return {
                "message": "LSTM sequence model training completed! 📈",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error training LSTM model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
'''

# A/B TESTING & EXPERIMENTATION ENDPOINTS

@app.get("/api/experimentation/status")
async def get_experimentation_status():
    """Get A/B testing framework status and experiment summary"""
    try:
        from src.experimentation.ab_testing_framework import ABTestingFramework
        
        ab_framework = ABTestingFramework()
        summary = ab_framework.get_experiment_summary()
        
        return {
            "status": "active",
            "experiment_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking experimentation status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

'''
@app.post("/api/experimentation/power-analysis")
async def run_power_analysis(
    effect_size: float = Query(..., description="Expected effect size (Cohen's d)"),
    alpha: float = Query(0.05, description="Significance level"),
    power: float = Query(0.8, description="Statistical power")
):
    """Calculate required sample size for A/B testing"""
    try:
        from src.experimentation.ab_testing_framework import ABTestingFramework
        
        ab_framework = ABTestingFramework()
        result = ab_framework.power_analysis(effect_size, alpha, power)
        
        return {
            "message": "Power analysis completed! 📊",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in power analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
'''

@app.post("/api/experimentation/simulate-ab-test")
async def simulate_ab_test(
    experiment_name: str = Query(..., description="Name of the experiment"),
    control_mean: float = Query(..., description="Control group mean"),
    treatment_mean: float = Query(..., description="Treatment group mean"),
    std_dev: float = Query(..., description="Standard deviation"),
    sample_size: int = Query(1000, description="Sample size per group")
):
    """Simulate A/B test with specified parameters"""
    try:
        from src.experimentation.ab_testing_framework import ABTestingFramework
        
        ab_framework = ABTestingFramework()
        result = ab_framework.simulate_ab_test(
            experiment_name, control_mean, treatment_mean, std_dev, sample_size
        )
        
        if result.get("status") == "success":
            return {
                "message": "A/B test simulation completed! 🧪",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error in A/B test simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/experimentation/analyze-customer-segments")
async def analyze_customer_segments_ab():
    """Analyze customer segments using A/B testing statistical methods"""
    try:
        from src.experimentation.ab_testing_framework import ABTestingFramework
        
        ab_framework = ABTestingFramework()
        result = ab_framework.analyze_customer_segments_ab()
        
        if result.get("status") == "success":
            return {
                "message": "Customer segment A/B analysis completed! 📊",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error in customer segment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/experimentation/bayesian-analysis")
async def run_bayesian_ab_analysis(
    control_conversions: int = Query(..., description="Control group conversions"),
    control_total: int = Query(..., description="Control group total"),
    treatment_conversions: int = Query(..., description="Treatment group conversions"),
    treatment_total: int = Query(..., description="Treatment group total")
):
    """Run Bayesian A/B test analysis"""
    try:
        from src.experimentation.ab_testing_framework import ABTestingFramework
        
        ab_framework = ABTestingFramework()
        result = ab_framework.bayesian_ab_analysis(
            control_conversions, control_total, treatment_conversions, treatment_total
        )
        
        if result.get("status") == "success":
            return {
                "message": "Bayesian A/B analysis completed! 🎯",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Error in Bayesian analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/insights/quick-pdf")
async def generate_quick_insights_pdf(
    query: str = Query(..., description="Natural language query about the data")
):
    """Generate quick insights and return PDF download"""
    try:
        from src.genai.insights_generator import RetailInsightsGenerator
        from src.utils.pdf_generator import AIReportPDFGenerator
        
        # Generate AI insights
        generator = RetailInsightsGenerator()
        insights_text = generator.get_quick_insights(query)
        
        if not insights_text or "not available" in insights_text.lower():
            raise HTTPException(status_code=500, detail="Failed to generate insights")
        
        # Generate PDF
        pdf_generator = AIReportPDFGenerator()
        pdf_path = pdf_generator.generate_pdf_from_quick_insights(insights_text, query)
        
        return {
            "message": "Quick insights PDF generated! 🚀",
            "query": query,
            "pdf_path": pdf_path,
            "download_url": f"/api/reports/download/{Path(pdf_path).name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating quick insights PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/insights/quick")
async def get_quick_insights(
    query: str = Query(..., description="Natural language query about the data")
):
    """Get quick insights using RAG-enhanced natural language query"""
    try:
        from src.genai.rag_insights_generator import RAGInsightsGenerator
        
        generator = RAGInsightsGenerator()
        result = generator.get_insights_with_data_exploration(query)
        
        return {
            "query": query,
            "insights": result.get('insights', ''),
            "data_exploration": result.get('data_exploration', {}),
            "retrieval_method": result.get('retrieval_method', 'RAG'),
            "chunks_used": result.get('chunks_used', 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in RAG quick insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/status")
async def get_rag_status():
    """Check RAG embeddings system status"""
    try:
        from src.genai.rag_insights_generator import RAGInsightsGenerator
        
        generator = RAGInsightsGenerator()
        stats = generator.rag_manager.get_embeddings_stats()
        
        return {
            "status": "active" if stats.get('total', 0) > 0 else "no_embeddings",
            "embeddings_stats": stats,
            "total_chunks": stats.get('total', 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking RAG status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/rebuild")
async def rebuild_rag_embeddings():
    """Rebuild RAG embeddings from current data"""
    try:
        from src.genai.rag_insights_generator import RAGInsightsGenerator
        
        generator = RAGInsightsGenerator()
        result = generator.rebuild_embeddings()
        
        if result.get('status') == 'success':
            return {
                "message": "RAG embeddings rebuilt successfully! 🎉",
                "total_chunks": result.get('total_chunks', 0),
                "stats": result.get('stats', {}),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
    except Exception as e:
        logger.error(f"Error rebuilding RAG embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/snowflake/status")
async def get_snowflake_status():
    """Check Snowflake connection and table status"""
    try:
        from src.data.snowflake_connector import SnowflakeManager
        
        manager = SnowflakeManager()
        
        if manager.create_connection():
            table_info = manager.get_table_info()
            manager.close_connection()
            
            return {
                "status": "connected",
                "tables": table_info,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "disconnected",
                "message": "Unable to connect to Snowflake",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Snowflake status check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/etl/run")
async def run_etl_pipeline():
    """Run the complete ETL pipeline including Snowflake upload"""
    try:
        from src.etl.pandas_pipeline import RetailDataProcessor
        from src.data.snowflake_connector import SnowflakeManager
        
        # Step 1: Run Pandas ETL Pipeline
        logger.info("🔄 Running Pandas ETL pipeline...")
        processor = RetailDataProcessor()
        df_processed, aggregations = processor.run_pipeline()
        
        # Step 2: Setup and Load to Snowflake
        logger.info("☁️ Setting up Snowflake warehouse and loading data...")
        snowflake_manager = SnowflakeManager()
        snowflake_success = snowflake_manager.setup_complete_warehouse()
        
        # Get table counts from Snowflake
        table_info = {}
        if snowflake_success:
            table_info = snowflake_manager.get_table_info()
        
        snowflake_manager.close_connection()
        
        # Step 3: Refresh local data cache
        data_cache.clear()
        load_processed_data()
        
        return {
            "message": "Complete ETL pipeline executed successfully",
            "pandas_pipeline": {
                "processed_rows": len(df_processed),
                "aggregation_tables": len(aggregations),
                "total_revenue": float(df_processed['Revenue'].sum())
            },
            "snowflake_pipeline": {
                "success": snowflake_success,
                "tables_loaded": table_info
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# SQL Analytics Endpoint
@app.post("/api/analytics/sql")
async def run_sql_query(query: str = Query(..., description="SQL query to execute")):
    """Execute custom SQL queries on Snowflake data warehouse"""
    try:
        logger.info(f"Executing SQL query: {query[:100]}...")
        
        # Initialize Snowflake manager
        snowflake_manager = SnowflakeManager()
        
        if not snowflake_manager.create_connection():
            raise HTTPException(status_code=500, detail="Failed to connect to Snowflake")
        
        # Execute query
        result_df = snowflake_manager.execute_query(query)
        snowflake_manager.close_connection()
        
        if result_df is None:
            raise HTTPException(status_code=400, detail="Query execution failed")
        
        # Convert to JSON-friendly format
        result_json = result_df.to_dict('records')
        
        return {
            "status": "success",
            "query": query,
            "rows_returned": len(result_df),
            "columns": list(result_df.columns),
            "data": result_json,
            "execution_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"SQL query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL execution error: {str(e)}")

# Business Alerts Endpoint
@app.get("/api/alerts/business")
async def get_business_alerts():
    """Get automated business alerts based on current data"""
    try:
        if "retail_transactions_processed" not in data_cache:
            raise HTTPException(status_code=404, detail="No data available for alerts")
        
        df = data_cache["retail_transactions_processed"]
        alerts = []
        
        # Revenue drop alert
        if len(df) > 1:
            recent_revenue = df['Revenue'].tail(100).mean()
            historical_revenue = df['Revenue'].head(len(df)-100).mean() if len(df) > 100 else recent_revenue
            
            if recent_revenue < historical_revenue * 0.8:  # 20% drop
                alerts.append({
                    "type": "revenue_drop",
                    "severity": "high",
                    "message": f"Revenue dropped {((historical_revenue - recent_revenue) / historical_revenue * 100):.1f}% recently",
                    "current_avg": round(recent_revenue, 2),
                    "historical_avg": round(historical_revenue, 2),
                    "action": "Review pricing strategy and customer retention"
                })
        
        # Customer concentration risk
        if "customer_analysis" in data_cache:
            customer_df = data_cache["customer_analysis"]
            top_customer_revenue = customer_df.nlargest(5, 'Total_Spent')['Total_Spent'].sum()
            total_revenue = customer_df['Total_Spent'].sum()
            concentration = (top_customer_revenue / total_revenue) * 100
            
            if concentration > 50:  # Top 5 customers >50% of revenue
                alerts.append({
                    "type": "customer_concentration",
                    "severity": "medium", 
                    "message": f"Top 5 customers represent {concentration:.1f}% of revenue",
                    "concentration_percentage": round(concentration, 1),
                    "action": "Diversify customer base to reduce risk"
                })
        
        # Product performance anomaly
        if "product_performance" in data_cache:
            product_df = data_cache["product_performance"]
            zero_revenue_products = len(product_df[product_df['Total_Revenue'] == 0])
            total_products = len(product_df)
            dead_stock_pct = (zero_revenue_products / total_products) * 100
            
            if dead_stock_pct > 30:  # >30% products with no revenue
                alerts.append({
                    "type": "inventory_anomaly",
                    "severity": "medium",
                    "message": f"{dead_stock_pct:.1f}% of products have zero revenue",
                    "dead_stock_count": zero_revenue_products,
                    "total_products": total_products,
                    "action": "Review product portfolio and discontinue dead stock"
                })
        
        return {
            "status": "success",
            "alerts_count": len(alerts),
            "alerts": alerts,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Business alerts failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)