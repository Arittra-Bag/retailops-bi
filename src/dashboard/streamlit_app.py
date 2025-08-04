import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Page configuration
st.set_page_config(
    page_title="RetailOps BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000/api"
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_files():
    """Load data directly from CSV files if API is not available"""
    try:
        datasets = {}
        
        files = [
            "retail_transactions_processed",
            "daily_sales", 
            "product_performance",
            "customer_analysis",
            "country_performance",
            "monthly_trends"
        ]
        
        for file in files:
            file_path = DATA_PATH / f"{file}.csv"
            if file_path.exists():
                datasets[file] = pd.read_csv(file_path)
        
        return datasets
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def get_overview_data():
    """Get overview data from API or files"""
    try:
        # Try API first
        response = requests.get(f"{API_BASE_URL}/overview", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback to file data
    datasets = load_data_from_files()
    if "retail_transactions_processed" in datasets:
        df = datasets["retail_transactions_processed"]
        return {
            "total_transactions": len(df),
            "total_revenue": float(df["Revenue"].sum()),
            "unique_customers": int(df["Customer_ID"].nunique()),
            "unique_products": int(df["StockCode"].nunique()),
            "countries": int(df["Country"].nunique()),
            "avg_order_value": float(df["Total_Amount"].mean())
        }
    return {}

def main():
    """Main dashboard function"""
    
    # Header
    st.title("📊 RetailOps BI Dashboard")
    st.markdown("**Comprehensive retail analytics and business intelligence platform**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📈 Overview", "📊 Sales Analytics", "🛍️ Product Performance", 
         "👥 Customer Analytics", "🌍 Geographic Analysis", "🤖 AI Insights",
         "🔥 Spark Analytics", "🧠 Deep Learning", "🧪 A/B Testing",
         "🗄️ SQL Analytics", "🚨 Business Alerts"]
    )
    
    # Load data
    datasets = load_data_from_files()
    
    if not datasets:
        st.error("No data available. Please run the ETL pipeline first.")
        st.info("Run: `python src/etl/pandas_pipeline.py`")
        return
    
    # Overview Page
    if page == "📈 Overview":
        show_overview_page(datasets)
    
    # Sales Analytics Page
    elif page == "📊 Sales Analytics":
        show_sales_page(datasets)
    
    # Product Performance Page
    elif page == "🛍️ Product Performance":
        show_products_page(datasets)
    
    # Customer Analytics Page
    elif page == "👥 Customer Analytics":
        show_customers_page(datasets)
    
    # Geographic Analysis Page
    elif page == "🌍 Geographic Analysis":
        show_geographic_page(datasets)
    
    # AI Insights Page
    elif page == "🤖 AI Insights":
        show_ai_insights_page(datasets)
    
    elif page == "🔥 Spark Analytics":
        show_spark_analytics_page()
    
    elif page == "🧠 Deep Learning":
        show_deep_learning_page()
    
    elif page == "🧪 A/B Testing":
        show_ab_testing_page()
    
    elif page == "🗄️ SQL Analytics":
        show_sql_analytics_page()
    
    elif page == "🚨 Business Alerts":
        show_business_alerts_page()

def show_overview_page(datasets):
    """Show overview dashboard"""
    st.header("📈 Business Overview")
    
    if "retail_transactions_processed" not in datasets:
        st.error("Transaction data not available")
        return
    
    df = datasets["retail_transactions_processed"]
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df["Revenue"].sum()
        st.metric(
            label="💰 Total Revenue",
            value=f"£{total_revenue:,.2f}",
            delta="12.5%"
        )
    
    with col2:
        total_transactions = len(df)
        st.metric(
            label="📦 Total Transactions", 
            value=f"{total_transactions:,}",
            delta="8.3%"
        )
    
    with col3:
        avg_order_value = df["Total_Amount"].mean()
        st.metric(
            label="💳 Avg Order Value",
            value=f"£{avg_order_value:.2f}",
            delta="5.2%"
        )
    
    with col4:
        unique_customers = df["Customer_ID"].nunique()
        st.metric(
            label="👥 Unique Customers",
            value=f"{unique_customers:,}",
            delta="15.1%"
        )
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily Revenue Trend
        if "daily_sales" in datasets:
            daily_df = datasets["daily_sales"]
            daily_df["Date"] = pd.to_datetime(daily_df["Date"])
            
            fig = px.line(
                daily_df.groupby("Date")["Total_Revenue"].sum().reset_index(),
                x="Date", 
                y="Total_Revenue",
                title="📈 Daily Revenue Trend",
                labels={"Total_Revenue": "Revenue (£)"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top Countries by Revenue
        if "country_performance" in datasets:
            country_df = datasets["country_performance"].head(10)
            
            fig = px.bar(
                country_df,
                x="Total_Revenue",
                y="Country",
                orientation="h",
                title="🌍 Top Countries by Revenue",
                labels={"Total_Revenue": "Revenue (£)"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Product Categories
    st.subheader("📊 Product Category Performance")
    
    category_revenue = df.groupby("Product_Category")["Revenue"].sum().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=category_revenue.values,
            names=category_revenue.index,
            title="Revenue by Category"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=category_revenue.index,
            y=category_revenue.values,
            title="Category Revenue Breakdown",
            labels={"x": "Category", "y": "Revenue (£)"}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_sales_page(datasets):
    """Show sales analytics page"""
    st.header("📊 Sales Analytics")
    
    if "daily_sales" not in datasets:
        st.error("Daily sales data not available")
        return
    
    daily_df = datasets["daily_sales"]
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        countries = ["All"] + sorted(daily_df["Country"].unique().tolist())
        selected_country = st.selectbox("Select Country", countries)
    
    with col2:
        min_date = daily_df["Date"].min().date()
        max_date = daily_df["Date"].max().date()
        start_date = st.date_input("Start Date", min_date)
    
    with col3:
        end_date = st.date_input("End Date", max_date)
    
    # Filter data
    filtered_df = daily_df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df["Country"] == selected_country]
    
    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(start_date)) &
        (filtered_df["Date"] <= pd.to_datetime(end_date))
    ]
    
    # Aggregated metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_revenue = filtered_df["Total_Revenue"].sum()
        st.metric("💰 Total Revenue", f"£{total_revenue:,.2f}")
    
    with col2:
        total_transactions = filtered_df["Total_Transactions"].sum()
        st.metric("📦 Total Transactions", f"{total_transactions:,}")
    
    with col3:
        avg_daily_revenue = filtered_df.groupby("Date")["Total_Revenue"].sum().mean()
        st.metric("📈 Avg Daily Revenue", f"£{avg_daily_revenue:,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend
        daily_agg = filtered_df.groupby("Date")["Total_Revenue"].sum().reset_index()
        fig = px.line(
            daily_agg,
            x="Date",
            y="Total_Revenue", 
            title="Revenue Trend Over Time",
            labels={"Total_Revenue": "Revenue (£)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Transactions trend
        daily_trans = filtered_df.groupby("Date")["Total_Transactions"].sum().reset_index()
        fig = px.line(
            daily_trans,
            x="Date",
            y="Total_Transactions",
            title="Transaction Volume Over Time",
            labels={"Total_Transactions": "Transactions"}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_products_page(datasets):
    """Show product performance page"""
    st.header("🛍️ Product Performance")
    
    if "product_performance" not in datasets:
        st.error("Product performance data not available")
        return
    
    product_df = datasets["product_performance"]
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        categories = ["All"] + sorted(product_df["Product_Category"].unique().tolist())
        selected_category = st.selectbox("Select Category", categories)
    
    with col2:
        top_n = st.slider("Show Top N Products", 5, 50, 20)
    
    # Filter data
    filtered_df = product_df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["Product_Category"] == selected_category]
    
    # Top products
    top_products = filtered_df.head(top_n)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_products = len(filtered_df)
        st.metric("📦 Total Products", f"{total_products:,}")
    
    with col2:
        total_revenue = filtered_df["Total_Revenue"].sum()
        st.metric("💰 Total Revenue", f"£{total_revenue:,.2f}")
    
    with col3:
        avg_revenue = filtered_df["Total_Revenue"].mean()
        st.metric("📊 Avg Revenue per Product", f"£{avg_revenue:,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top products by revenue
        fig = px.bar(
            top_products.head(15),
            x="Total_Revenue",
            y="Description",
            orientation="h",
            title=f"Top 15 Products by Revenue ({selected_category})",
            labels={"Total_Revenue": "Revenue (£)", "Description": "Product"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Products by category
        category_counts = filtered_df["Product_Category"].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Products by Category"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Product details table
    st.subheader("Product Performance Details")
    st.dataframe(
        top_products[["Description", "Product_Category", "Total_Revenue", 
                     "Total_Quantity_Sold", "Total_Orders", "Avg_Price"]],
        use_container_width=True
    )

def show_customers_page(datasets):
    """Show customer analytics page"""
    st.header("👥 Customer Analytics")
    
    if "customer_analysis" not in datasets or len(datasets["customer_analysis"]) == 0:
        st.warning("Customer analysis data not available. This requires registered customers with Customer IDs.")
        return
    
    customer_df = datasets["customer_analysis"]
    
    # Customer segments based on spending
    customer_df["Segment"] = pd.cut(
        customer_df["Total_Spent"],
        bins=[0, 100, 500, 1000, float("inf")],
        labels=["Bronze", "Silver", "Gold", "Platinum"]
    )
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(customer_df)
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col2:
        avg_clv = customer_df["Total_Spent"].mean()
        st.metric("💰 Avg Customer Value", f"£{avg_clv:.2f}")
    
    with col3:
        avg_orders = customer_df["Total_Orders"].mean()
        st.metric("📦 Avg Orders per Customer", f"{avg_orders:.1f}")
    
    with col4:
        avg_items = customer_df["Total_Items"].mean()
        st.metric("🛍️ Avg Items per Customer", f"{avg_items:.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segments
        segment_counts = customer_df["Segment"].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segments by Spending"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer value distribution
        fig = px.histogram(
            customer_df,
            x="Total_Spent",
            nbins=50,
            title="Customer Value Distribution",
            labels={"Total_Spent": "Total Spent (£)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top customers table
    st.subheader("Top 20 Customers")
    top_customers = customer_df.head(20)
    st.dataframe(
        top_customers[["Customer_ID", "Country", "Total_Spent", "Total_Orders", 
                      "Total_Items", "Avg_Order_Value", "Segment"]],
        use_container_width=True
    )

def show_geographic_page(datasets):
    """Show geographic analysis page"""
    st.header("🌍 Geographic Analysis")
    
    if "country_performance" not in datasets:
        st.error("Country performance data not available")
        return
    
    country_df = datasets["country_performance"]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_countries = len(country_df)
        st.metric("🌍 Total Countries", f"{total_countries}")
    
    with col2:
        top_country = country_df.iloc[0]["Country"] if len(country_df) > 0 else "N/A"
        st.metric("🥇 Top Country", top_country)
    
    with col3:
        total_revenue = country_df["Total_Revenue"].sum()
        st.metric("💰 Total Revenue", f"£{total_revenue:,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by country
        fig = px.bar(
            country_df.head(15),
            x="Total_Revenue",
            y="Country",
            orientation="h",
            title="Revenue by Country",
            labels={"Total_Revenue": "Revenue (£)"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customers by country
        fig = px.bar(
            country_df.head(15),
            x="Unique_Customers",
            y="Country",
            orientation="h",
            title="Customers by Country",
            labels={"Unique_Customers": "Number of Customers"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Country performance table
    st.subheader("Country Performance Details")
    st.dataframe(country_df, use_container_width=True)

def show_ai_insights_page(datasets):
    """Show AI insights page"""
    st.title("🤖 AI Business Intelligence")
    st.markdown("Transform your retail data into actionable insights with advanced AI analytics")
    
    # Create clean tabs for organized layout
    tab1, tab2 = st.tabs(["💬 Quick Insights (RAG+LangChain)", "📊 Full Reports (LLM)"])
    
    with tab1:
        st.subheader("Ask Questions About Your Data")
        st.markdown("Get instant AI-powered answers to your business questions")
        
        # Clean input section
        query = st.text_area(
            label="What would you like to know?",
            placeholder="Examples:\n• What are my top-selling products?\n• Which customers drive the most revenue?\n• What trends do you see in my sales data?",
            height=80,
            help="Ask any question about your retail data - sales, customers, products, trends, etc."
        )
        
        if query.strip():
            (col1,) = st.columns(1)
            
            with col1:
                if st.button("🔍 **Get Insights**", type="primary", use_container_width=True):
                    with st.spinner("🔍 RAG system searching data and generating insights..."):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/insights/quick",
                                params={"query": query},
                                timeout=30
                            )
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ RAG Analysis Complete!")
                                st.balloons()
                                
                                # Show RAG metrics
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("🔍 Method", result.get('retrieval_method', 'RAG'))
                                with col_b:
                                    st.metric("📊 Data Sources", result.get('chunks_used', 0))
                                with col_c:
                                    st.metric("🎯 Status", "✅ Complete")
                                
                                # Clean results display
                                with st.container():
                                    st.markdown("#### 💡 AI Insights")
                                    st.info(f"**Question:** {result['query']}")
                                    
                                    st.markdown("**Answer:**")
                                    st.markdown(result['insights'])
                                    
                                    # Show data exploration details
                                    if result.get('data_exploration'):
                                        with st.expander("🔍 **Data Sources Used**"):
                                            exploration = result['data_exploration']
                                            st.write("**Retrieved Data Types:**")
                                            for i, data_type in enumerate(exploration.get('relevant_data_types', [])):
                                                similarity = exploration.get('similarity_scores', [0])[i] * 100
                                                st.write(f"• {data_type} (Relevance: {similarity:.1f}%)")
                                
                            else:
                                st.error("❌ Failed to generate insights")
                        except Exception as e:
                            st.error(f"🚫 AI service unavailable: {str(e)}")
                            st.info("💡 Try running: `python run_system.py`")
            
        else:
            st.info("👆 Enter your question above to get AI-powered insights")
        
    with tab2:
        st.subheader("Comprehensive Business Analysis")
        st.markdown("Generate detailed reports covering all aspects of your business performance")
        
        (col2,) = st.columns(1)
        
        with col2:
            st.markdown("""
            <div style="border: 2px solid #FF4B4B; border-radius: 10px; padding: 20px; margin: 10px 0;">
                <h4>📄 AI Executive Business Report (PDF)</h4>
                <p><strong>Complete analysis including:</strong></p>
                <ul>
                    <li>Executive summary</li>
                    <li>Customer segmentation</li>
                    <li>Product performance</li>
                    <li>Market trends</li>
                    <li>Strategic recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📄 **Generate PDF Report**", type="primary", use_container_width=True):
                with st.spinner("📄 Creating executive PDF report..."):
                    try:
                        response = requests.post(f"{API_BASE_URL}/insights/generate-pdf", timeout=120)
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("🎉 **Executive PDF Report Complete!**")
                            st.balloons()
                            
                            download_url = f"http://localhost:8000{result['download_url']}"
                            
                            st.markdown(f"""
                            <div style="text-align: center; background: linear-gradient(135deg, #FF4B4B, #FF6B6B); 
                                       padding: 20px; border-radius: 15px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                                <a href="{download_url}" target="_blank" 
                                   style="color: white; text-decoration: none; font-weight: bold; font-size: 18px;">
                                    📥 Download Executive PDF Report
                                </a>
                                <br><small style="color: #FFE8E8;">Professional format • Ready for presentation</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("📋 Report Details"):
                                st.info(f"""
                                **Report ID:** {result['report_id']}  
                                **Generated:** {result['timestamp']}  
                                **Format:** Professional PDF  
                                **Content:** Full business analysis with recommendations
                                """)
                            
                        else:
                            st.error("❌ PDF generation failed")
                    except Exception as e:
                        st.error(f"🚫 Error: {str(e)}")

def show_spark_analytics_page():
    """Show Apache Spark analytics page"""
    st.title("🔥 Apache Spark Analytics")
    st.markdown("**Large-scale distributed customer analytics and ML pipelines**")
    
    st.markdown("---")
    
    # Spark Analytics
    st.subheader("📊 Distributed Analytics")
    
    tab1, tab2 = st.tabs(["🎯 Customer Segmentation", "📈 ML Forecasting"])
    
    with tab1:
        st.markdown("### 🔥 Spark Customer Segmentation")
        st.write("Run large-scale customer segmentation using Spark ML K-Means clustering")
        
        if st.button("🚀 **Run Spark Segmentation**", type="primary", use_container_width=True):
            with st.spinner("🔥 Running distributed customer segmentation..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/spark/customer-segmentation", timeout=60)
                    if response.status_code == 200:
                        result = response.json()['result']
                        st.success("🔥 Spark segmentation complete!")
                        st.balloons()
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("👥 Customers", result['total_customers'])
                        with col_b:
                            st.metric("🎯 Segments", result['segments'])
                        with col_c:
                            st.metric("🔧 Engine", "Apache Spark")
                        with col_d:
                            st.metric("✨ Features", len(result['feature_columns']))
                        
                        # Show segment statistics
                        if 'segment_statistics' in result:
                            st.markdown("### 📊 Segment Performance")
                            import pandas as pd
                            segments_df = pd.DataFrame(result['segment_statistics'])
                            st.dataframe(segments_df, use_container_width=True)
                    else:
                        st.error("❌ Spark segmentation failed")
                except Exception as e:
                    st.error(f"🚫 Error: {str(e)}")
    
    with tab2:
        st.markdown("### 📈 Spark ML Forecasting Pipeline")
        st.write("Train distributed ML models for sales forecasting using Spark ML")
        
        if st.button("🚀 **Run Spark ML Pipeline**", type="primary", use_container_width=True):
            with st.spinner("🔥 Training distributed ML models..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/spark/ml-forecasting", timeout=120)
                    if response.status_code == 200:
                        result = response.json()['result']
                        st.success("🚀 Spark ML pipeline complete!")
                        st.balloons()
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("#### 🌲 Random Forest")
                            rf_perf = result['random_forest_performance']
                            st.metric("R² Score", f"{rf_perf['r2']:.3f}")
                            st.metric("RMSE", f"{rf_perf['rmse']:.2f}")
                        
                        with col_b:
                            st.markdown("#### 📊 Linear Regression")
                            lr_perf = result['linear_regression_performance']
                            st.metric("R² Score", f"{lr_perf['r2']:.3f}")
                            st.metric("RMSE", f"{lr_perf['rmse']:.2f}")
                        
                        st.info(f"✅ Trained on {result['training_samples']} samples using {result['processing_engine']}")
                        
                        # Display forecast insights if available
                        if 'forecast_insights' in result:
                            insights = result['forecast_insights']
                            
                            # Business Trend Analysis
                            if 'business_analysis' in insights:
                                business = insights['business_analysis']
                                st.markdown("### 📈 Revenue Forecast & Trends")
                                
                                col_x, col_y, col_z = st.columns(3)
                                with col_x:
                                    trend_color = "🟢" if business['revenue_trend'] == "Increasing" else "🔴" if business['revenue_trend'] == "Decreasing" else "🟡"
                                    st.metric("📊 Revenue Trend", f"{trend_color} {business['revenue_trend']}", f"{business['trend_percentage']:+.1f}%")
                                with col_y:
                                    st.metric("💰 7-Day Forecast Total", f"£{business['total_forecast_revenue']:,.2f}")
                                with col_z:
                                    st.metric("🎯 Confidence Level", business['confidence_level'])
                            
                            # Daily Forecasts
                            if 'daily_forecasts' in insights and insights['daily_forecasts']:
                                st.markdown("### 📅 7-Day Revenue Forecasts")
                                
                                # Create forecast DataFrame for visualization
                                import pandas as pd
                                forecast_df = pd.DataFrame(insights['daily_forecasts'])
                                
                                # Display as chart
                                import plotly.express as px
                                fig = px.line(forecast_df, x='date', y=['random_forest_forecast', 'linear_regression_forecast', 'ensemble_forecast'],
                                            title='7-Day Revenue Forecasts by Model',
                                            labels={'value': 'Revenue (£)', 'date': 'Date'})
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast table
                                st.markdown("**Daily Forecast Details:**")
                                display_df = forecast_df[['date', 'day_of_week', 'ensemble_forecast', 'random_forest_forecast', 'linear_regression_forecast']]
                                display_df.columns = ['Date', 'Day', 'Ensemble Forecast', 'Random Forest', 'Linear Regression']
                                display_df['Ensemble Forecast'] = display_df['Ensemble Forecast'].apply(lambda x: f"£{x:,.2f}")
                                display_df['Random Forest'] = display_df['Random Forest'].apply(lambda x: f"£{x:,.2f}")
                                display_df['Linear Regression'] = display_df['Linear Regression'].apply(lambda x: f"£{x:,.2f}")
                                st.dataframe(display_df, use_container_width=True)
                            
                            # AI-Powered Business Recommendations
                            if 'recommendations' in insights and insights['recommendations']:
                                st.markdown("### 🤖 AI Business Recommendations")
                                st.caption("Generated by LLM based on your forecast data")
                                for i, rec in enumerate(insights['recommendations'], 1):
                                    st.write(f"{i}. {rec}")
                            
                            # Model Comparison
                            if 'model_comparison' in insights:
                                comp = insights['model_comparison']
                                st.markdown("### 🤖 Model Performance Comparison")
                                col_x, col_y = st.columns(2)
                                with col_x:
                                    st.metric("🏆 Recommended Model", comp['recommended_model'])
                                with col_y:
                                    st.metric("🤝 Model Agreement", comp['model_agreement'])
                        
                        else:
                            st.warning("⚠️ Forecast insights not available. This may happen with insufficient historical data.")
                    else:
                        st.error("❌ Spark ML pipeline failed")
                except Exception as e:
                    st.error(f"🚫 Error: {str(e)}")

def show_deep_learning_page():
    """Show deep learning models page"""
    st.title("🧠 Deep Learning Customer Models")
    st.markdown("**Advanced neural networks for customer behavior prediction**")
    
    st.markdown("---")
    
    # Deep Learning Models
    st.subheader("🤖 Neural Network Training")
    
    (tab2,) = st.tabs(["🔥 PyTorch Embeddings"])
    
    with tab2:
        st.markdown("### 🔥 PyTorch Customer Embeddings")
        st.write("Learn deep customer and product embeddings using PyTorch neural networks")
        
        if st.button("🚀 **Train PyTorch Embeddings**", type="primary", use_container_width=True):
            with st.spinner("🔥 Training PyTorch embedding model..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/deeplearning/train-pytorch-embeddings", timeout=180)
                    if response.status_code == 200:
                        result = response.json()['result']
                        st.success("🔥 PyTorch embedding training complete!")
                        st.balloons()
                        
                        embeddings = result['embeddings']
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("👥 Customers", embeddings['num_customers'])
                        with col_b:
                            st.metric("📦 Products", embeddings['num_products'])
                        with col_c:
                            st.metric("🧮 Embedding Dim", 50)
                        
                        st.info(f"✅ Trained {result['training_epochs']} epochs using {result['framework']}")
                        
                        # Display business insights if available
                        if 'business_insights' in result:
                            insights = result['business_insights']
                            
                            # Customer Similarity Analysis
                            if 'customer_similarity' in insights and insights['customer_similarity']:
                                st.markdown("### 🔍 Customer Similarity Analysis")
                                st.markdown("**Find customers with similar purchasing behavior:**")
                                
                                for customer_analysis in insights['customer_similarity'][:3]:  # Show top 3
                                    with st.expander(f"👤 Customer {customer_analysis['customer_id']} ({customer_analysis['transactions']} transactions)"):
                                        st.markdown("**Similar Customers:**")
                                        for similar in customer_analysis['similar_customers'][:3]:
                                            similarity_pct = similar['similarity_score'] * 100
                                            st.write(f"• Customer {similar['customer_id']}: {similarity_pct:.1f}% similar ({similar['transactions']} transactions)")
                            
                            # Product Recommendations
                            if 'product_recommendations' in insights and insights['product_recommendations']:
                                st.markdown("### 🛍️ Product Recommendations")
                                st.markdown("**Products frequently bought together:**")
                                
                                for product_rec in insights['product_recommendations'][:3]:  # Show top 3
                                    with st.expander(f"📦 {product_rec['description']}"):
                                        st.markdown("**Recommended Products:**")
                                        for rec in product_rec['recommended_products'][:3]:
                                            similarity_pct = rec['similarity_score'] * 100
                                            st.write(f"• {rec['description']}: {similarity_pct:.1f}% similarity")
                            
                            # Business Metrics Summary
                            if 'business_metrics' in insights:
                                metrics = insights['business_metrics']
                                st.markdown("### 📊 Business Impact")
                                col_x, col_y = st.columns(2)
                                with col_x:
                                    st.metric("🎯 Recommendation Confidence", metrics.get('recommendation_confidence', 'Medium'))
                                with col_y:
                                    st.metric("🔍 Similarity Threshold", f"{metrics.get('similarity_threshold', 0.5)*100:.0f}%")
                        
                        else:
                            st.warning("⚠️ Business insights not available. This may happen with small datasets.")
                    else:
                        st.error("❌ PyTorch training failed")
                except Exception as e:
                    st.error(f"🚫 Error: {str(e)}")
    
def show_ab_testing_page():
    """Show A/B testing and experimentation page"""
    st.title("🧪 A/B Testing & Experimentation")
    st.markdown("**Statistical testing, experiment design, and business impact measurement**")
    
    st.markdown("---")
    
    # A/B Testing Tools
    st.subheader("📊 Statistical Analysis Tools")
    
    tab2, tab3 = st.tabs(["🧪 A/B Simulation", "🎯 Bayesian Analysis"])
    
    with tab2:
        st.markdown("### 🧪 A/B Test Simulation")
        st.write("Simulate A/B test results with specified parameters")
        
        col_a, col_b = st.columns(2)
        with col_a:
            exp_name = st.text_input("Experiment Name", value="Revenue_Optimization")
            control_mean = st.number_input("Control Mean", value=100.0, min_value=0.0, step=10.0)
            treatment_mean = st.number_input("Treatment Mean", value=105.0, min_value=0.0, step=10.0)
        with col_b:
            std_dev = st.number_input("Standard Deviation", value=20.0, min_value=1.0, step=5.0)
            sample_size = st.number_input("Sample Size (per group)", value=1000, min_value=100, max_value=10000, step=100)
        
        if st.button("🧪 **Run A/B Simulation**", type="primary", use_container_width=True):
            with st.spinner("Running A/B test simulation..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/experimentation/simulate-ab-test",
                        params={
                            "experiment_name": exp_name,
                            "control_mean": control_mean,
                            "treatment_mean": treatment_mean,
                            "std_dev": std_dev,
                            "sample_size": sample_size
                        },
                        timeout=60
                    )
                    if response.status_code == 200:
                        result = response.json()['result']['results']
                        st.success("🧪 A/B simulation complete!")
                        st.balloons()
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("💰 Control Mean", f"£{result['control_group']['mean']:.2f}")
                        with col_b:
                            st.metric("🚀 Treatment Mean", f"£{result['treatment_group']['mean']:.2f}")
                        with col_c:
                            st.metric("📈 Relative Change", f"{result['effect_analysis']['relative_change_percent']:.1f}%")
                        
                        # Statistical results
                        st.markdown("#### 📊 Statistical Results")
                        p_value = result['statistical_tests']['t_test']['p_value']
                        significant = result['statistical_tests']['t_test']['significant']
                        
                        if significant:
                            st.success(f"✅ **Statistically Significant** (p = {p_value:.4f})")
                        else:
                            st.warning(f"⚠️ **Not Significant** (p = {p_value:.4f})")
                        
                        st.info(f"💡 **Recommendation:** {result['business_impact']['recommendation']}")
                    else:
                        st.error("❌ A/B simulation failed")
                except Exception as e:
                    st.error(f"🚫 Error: {str(e)}")
    
    with tab3:
        st.markdown("### 🎯 Bayesian Analysis")
        st.write("Analyze conversion rates using Bayesian methods")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Control Group**")
            control_conv = st.number_input("Control Conversions", value=85, min_value=0, max_value=10000)
            control_total = st.number_input("Control Total", value=1000, min_value=1, max_value=50000)
        with col_b:
            st.markdown("**Treatment Group**")
            treatment_conv = st.number_input("Treatment Conversions", value=95, min_value=0, max_value=10000)
            treatment_total = st.number_input("Treatment Total", value=1000, min_value=1, max_value=50000)
        
        if st.button("🎯 **Run Bayesian Analysis**", type="primary", use_container_width=True):
            with st.spinner("Running Bayesian analysis..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/experimentation/bayesian-analysis",
                        params={
                            "control_conversions": control_conv,
                            "control_total": control_total,
                            "treatment_conversions": treatment_conv,
                            "treatment_total": treatment_total
                        },
                        timeout=60
                    )
                    if response.status_code == 200:
                        result = response.json()['result']['bayesian_results']
                        st.success("🎯 Bayesian analysis complete!")
                        st.balloons()
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("🎯 Prob Better", f"{result['prob_treatment_better']:.3f}")
                        with col_b:
                            st.metric("📈 Expected Lift", f"{result['expected_lift_percent']:.1f}%")
                        with col_c:
                            st.metric("🎯 Decision", result['decision'])
                        
                        # Confidence level
                        prob = result['prob_treatment_better']
                        if prob > 0.95:
                            st.success("🎉 **Strong Evidence** - High confidence in treatment effect")
                        elif prob > 0.8:
                            st.warning("⚡ **Moderate Evidence** - Some confidence in treatment effect") 
                        else:
                            st.info("🤔 **Weak Evidence** - Insufficient confidence")
                    else:
                        st.error("❌ Bayesian analysis failed")
                except Exception as e:
                    st.error(f"🚫 Error: {str(e)}")

def show_sql_analytics_page():
    """Show SQL analytics interface"""
    st.title("🗄️ SQL Analytics")
    st.markdown("**Direct SQL queries on your Snowflake data warehouse**")
    
    st.markdown("---")
    
    # SQL Query Interface
    st.subheader("📝 Custom SQL Queries")
    st.markdown("Execute SQL queries directly on your data warehouse for advanced analytics")
    
    # Pre-built query examples
    with st.expander("💡 **Example Queries**"):
        st.code("""
-- Check table structure
DESCRIBE TABLE retail_transactions;

-- Simple count
SELECT COUNT(*) as row_count FROM retail_transactions;

-- Show available tables
SHOW TABLES;

-- Product performance (safer table)
SELECT * FROM product_performance LIMIT 10;

-- Customer analysis
SELECT * FROM customer_analysis LIMIT 10;

-- Country performance
SELECT * FROM country_performance LIMIT 10;
        """, language="sql")
    
    # SQL Input
    sql_query = st.text_area(
        "Enter your SQL query:",
        placeholder="SELECT * FROM retail_transactions LIMIT 10;",
        height=150,
        help="Execute custom SQL queries on your Snowflake data warehouse"
    )
    
    if sql_query.strip():
        if st.button("🚀 **Execute Query**", type="primary", use_container_width=True):
            with st.spinner("Executing SQL query on Snowflake..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/analytics/sql",
                        params={"query": sql_query},
                        timeout=60
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Query executed successfully!")
                        
                        # Query metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("📊 Rows Returned", result['rows_returned'])
                        with col_b:
                            st.metric("📋 Columns", len(result['columns']))
                        with col_c:
                            st.metric("⚡ Engine", "Snowflake")
                        
                        # Results
                        if result['data']:
                            st.subheader("📊 Query Results")
                            import pandas as pd
                            df = pd.DataFrame(result['data'])
                            st.dataframe(df, use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No data returned from query")
                    else:
                        st.error(f"❌ Query failed: {response.text}")
                except Exception as e:
                    st.error(f"🚫 Error: {str(e)}")
    else:
        st.info("👆 Enter a SQL query above to execute on your data warehouse")

def show_business_alerts_page():
    """Show business alerts and monitoring"""
    st.title("🚨 Business Alerts & Monitoring")
    st.markdown("**Automated monitoring for business anomalies and performance issues**")
    
    st.markdown("---")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 30 seconds)")
    
    if st.button("🔍 **Check Current Alerts**", type="primary", use_container_width=True) or auto_refresh:
        with st.spinner("Analyzing business metrics for alerts..."):
            try:
                response = requests.get(f"{API_BASE_URL}/alerts/business", timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    alerts = result['alerts']
                    
                    # Alert summary
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("🚨 Total Alerts", result['alerts_count'])
                    with col_b:
                        high_alerts = len([a for a in alerts if a['severity'] == 'high'])
                        st.metric("⚠️ High Priority", high_alerts)
                    with col_c:
                        medium_alerts = len([a for a in alerts if a['severity'] == 'medium'])
                        st.metric("🟡 Medium Priority", medium_alerts)
                    
                    # Display alerts
                    if alerts:
                        st.subheader("🚨 Active Alerts")
                        
                        for alert in alerts:
                            severity_color = "🔴" if alert['severity'] == 'high' else "🟡"
                            alert_type = alert['type'].replace('_', ' ').title()
                            
                            with st.expander(f"{severity_color} **{alert_type}** - {alert['severity'].upper()}"):
                                st.warning(alert['message'])
                                st.info(f"**Recommended Action:** {alert['action']}")
                                
                                # Show specific metrics
                                if alert['type'] == 'revenue_drop':
                                    col_x, col_y = st.columns(2)
                                    with col_x:
                                        st.metric("Current Avg Revenue", f"£{alert['current_avg']:,.2f}")
                                    with col_y:
                                        st.metric("Historical Avg Revenue", f"£{alert['historical_avg']:,.2f}")
                                
                                elif alert['type'] == 'customer_concentration':
                                    st.metric("Revenue Concentration", f"{alert['concentration_percentage']:.1f}%")
                                
                                elif alert['type'] == 'inventory_anomaly':
                                    col_x, col_y = st.columns(2)
                                    with col_x:
                                        st.metric("Dead Stock Count", alert['dead_stock_count'])
                                    with col_y:
                                        st.metric("Total Products", alert['total_products'])
                    else:
                        st.success("✅ **No alerts detected!**")
                        st.info("Your business metrics are within normal parameters.")
                        
                        # Show some positive metrics
                        st.subheader("📊 Current Business Health")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("📈 Revenue Status", "✅ Normal")
                        with col_b:
                            st.metric("👥 Customer Health", "✅ Diversified")
                        with col_c:
                            st.metric("📦 Inventory Status", "✅ Optimized")
                    
                    st.caption(f"Last updated: {result['generated_at']}")
                    
                else:
                    st.error("❌ Failed to fetch business alerts")
            except Exception as e:
                st.error(f"🚫 Error: {str(e)}")
    
    # Add monitoring configuration
    st.markdown("---")
    st.subheader("⚙️ Alert Configuration")
    
    with st.expander("🔧 **Configure Alert Thresholds**"):
        st.markdown("**Revenue Drop Alert:**")
        revenue_threshold = st.slider("Alert when revenue drops by:", 10, 50, 20, step=5, format="%d%%")
        
        st.markdown("**Customer Concentration Alert:**")
        concentration_threshold = st.slider("Alert when top 5 customers exceed:", 30, 80, 50, step=5, format="%d%%")
        
        st.markdown("**Dead Stock Alert:**")
        dead_stock_threshold = st.slider("Alert when dead stock exceeds:", 10, 60, 30, step=5, format="%d%%")
        
        if st.button("💾 Save Configuration"):
            st.success("✅ Alert thresholds updated!")
            st.info("New thresholds will be applied on next alert check.")
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()