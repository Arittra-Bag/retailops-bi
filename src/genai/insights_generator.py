"""
GenAI Insights Generator for RetailOps BI
Uses LangChain and Gemini/OpenAI to generate automated business insights
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain.agents.agent_types import AgentType
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

# LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RetailOps-BI"

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetailInsightsGenerator:
    """Generates automated business insights using AI"""
    
    def __init__(self):
        """Initialize the insights generator"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_processed_path = self.project_root / "data" / "processed"
        
        # Set up API keys
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
        
        # Set LangSmith API key
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        
        # Initialize LLM
        self.llm = None
        self.chat_model = None
        self._initialize_models()
        
        # Load data
        self.datasets = {}
        self._load_datasets()
    
    def _initialize_models(self):
        """Initialize LangChain models"""
        try:
            if not LANGCHAIN_AVAILABLE:
                logger.error("LangChain packages not available")
                return
                
            if not self.gemini_api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                return
            
            # Initialize Google Generative AI model
            logger.info("Initializing Gemini AI model...")
            
            # Try different model names for compatibility  
            model_names_to_try = ["gemini-1.5-flash","gemini-2.5-pro"]
            
            self.llm = None
            self.chat_model = None
            
            for model_name in model_names_to_try:
                try:
                    logger.info(f"Trying model: {model_name}")
                    
                    self.llm = GoogleGenerativeAI(
                        model=model_name,
                        google_api_key=self.gemini_api_key,
                        temperature=0.1,
                        max_output_tokens=1024
                    )
                    
                    self.chat_model = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=self.gemini_api_key,
                        temperature=0.1,
                        max_output_tokens=1024
                    )
                    
                    # Test the model with a simple query
                    test_response = self.llm.invoke("Hello")
                    logger.info(f"✅ Model {model_name} working: {test_response[:50]}...")
                    break
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {str(e)[:100]}...")
                    continue
            
            if not self.llm:
                raise Exception("No working Gemini model found")
            
            logger.info("✅ Gemini AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing AI models: {str(e)}")
            logger.info("💡 Please check your GEMINI_API_KEY in the .env file")
            logger.info("💡 Install required packages: pip install langchain-google-genai")
    
    def _load_datasets(self):
        """Load processed datasets"""
        try:
            logger.info("Loading datasets for analysis...")
            
            dataset_files = [
                'retail_transactions_processed',
                'product_performance',
                'customer_analysis', 
                'country_performance',
                'daily_sales',
                'monthly_trends'
            ]
            
            for dataset in dataset_files:
                file_path = self.data_processed_path / f"{dataset}.csv"
                if file_path.exists():
                    self.datasets[dataset] = pd.read_csv(file_path)
                    logger.info(f"Loaded {dataset}: {len(self.datasets[dataset])} rows")
                else:
                    logger.warning(f"Dataset not found: {file_path}")
            
            logger.info(f"✅ Loaded {len(self.datasets)} datasets")
            
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
    
    def generate_business_summary(self) -> Dict[str, Any]:
        """Generate high-level business summary insights"""
        try:
            if not self.llm or 'retail_transactions_processed' not in self.datasets:
                logger.error("AI model or data not available")
                return {}
            
            logger.info("Generating business summary insights...")
            
            # Prepare data summary
            df = self.datasets['retail_transactions_processed']
            
            data_summary = {
                'total_revenue': float(df['Revenue'].sum()),
                'total_transactions': len(df),
                'unique_customers': int(df['Customer_ID'].nunique()),
                'unique_products': int(df['StockCode'].nunique()),
                'avg_order_value': float(df['Total_Amount'].mean()),
                'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
                'top_countries': df['Country'].value_counts().head(5).to_dict(),
                'revenue_by_category': df.groupby('Product_Category')['Revenue'].sum().to_dict()
            }
            
            # Create prompt for business summary
            prompt_template = PromptTemplate(
                input_variables=["data_summary"],
                template="""
                As a senior retail business analyst, analyze the following retail business data and provide executive-level insights:

                Business Metrics:
                - Total Revenue: £{total_revenue:,.2f}
                - Total Transactions: {total_transactions:,}
                - Unique Customers: {unique_customers:,}
                - Unique Products: {unique_products:,}
                - Average Order Value: £{avg_order_value:.2f}
                - Date Range: {date_range}
                
                Top Countries: {top_countries}
                Revenue by Category: {revenue_by_category}

                Please provide:
                1. **Executive Summary** (2-3 sentences)
                2. **Key Performance Highlights** (3-4 bullet points)
                3. **Strategic Recommendations** (3-4 actionable items)
                4. **Risk Areas to Monitor** (2-3 concerns)
                5. **Growth Opportunities** (2-3 opportunities)

                Keep the analysis professional, data-driven, and actionable for retail executives.
                """
            )
            
            # Generate insights
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(**data_summary)
            
            return {
                'type': 'business_summary',
                'generated_at': datetime.now().isoformat(),
                'data_summary': data_summary,
                'insights': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error generating business summary: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def analyze_customer_segments(self) -> Dict[str, Any]:
        """Generate customer segmentation insights"""
        try:
            if 'customer_analysis' not in self.datasets:
                logger.warning("Customer analysis data not available")
                return {}
            
            logger.info("Analyzing customer segments...")
            
            df = self.datasets['customer_analysis']
            if len(df) == 0:
                return {'status': 'no_data', 'message': 'No customer data available for analysis'}
            
            # Customer segmentation
            df['Customer_Segment'] = pd.cut(
                df['Total_Spent'],
                bins=[0, 100, 500, 1000, float('inf')],
                labels=['Bronze', 'Silver', 'Gold', 'Platinum']
            )
            
            segment_analysis = {
                'segment_distribution': df['Customer_Segment'].value_counts().to_dict(),
                'avg_spent_by_segment': df.groupby('Customer_Segment')['Total_Spent'].mean().to_dict(),
                'avg_orders_by_segment': df.groupby('Customer_Segment')['Total_Orders'].mean().to_dict(),
                'total_customers': len(df),
                'high_value_customers': len(df[df['Total_Spent'] > 1000]),
                'repeat_customers': len(df[df['Total_Orders'] > 1])
            }
            
            prompt_template = PromptTemplate(
                input_variables=["segment_analysis"],
                template="""
                Analyze this customer segmentation data for a retail business:

                Customer Segments:
                - Distribution: {segment_distribution}
                - Average Spending by Segment: {avg_spent_by_segment}
                - Average Orders by Segment: {avg_orders_by_segment}
                - Total Customers: {total_customers}
                - High-Value Customers (>£1000): {high_value_customers}
                - Repeat Customers: {repeat_customers}

                Provide insights on:
                1. **Customer Segment Analysis** - What do these segments tell us?
                2. **Customer Lifetime Value Insights** - Which segments drive most value?
                3. **Retention Strategies** - How to move customers up segments?
                4. **Marketing Recommendations** - Targeted approaches for each segment
                5. **Revenue Optimization** - Focus areas for growth

                Keep analysis focused on actionable customer strategies.
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(**segment_analysis)
            
            return {
                'type': 'customer_segmentation',
                'generated_at': datetime.now().isoformat(),
                'segment_analysis': segment_analysis,
                'insights': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing customer segments: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def analyze_product_performance(self) -> Dict[str, Any]:
        """Generate product performance insights"""
        try:
            if 'product_performance' not in self.datasets:
                return {}
            
            logger.info("Analyzing product performance...")
            
            df = self.datasets['product_performance']
            
            product_analysis = {
                'total_products': len(df),
                'top_10_products': df.head(10)[['Description', 'Total_Revenue']].to_dict('records'),
                'category_performance': df.groupby('Product_Category')['Total_Revenue'].sum().sort_values(ascending=False).to_dict(),
                'avg_revenue_per_product': float(df['Total_Revenue'].mean()),
                'low_performing_products': len(df[df['Total_Revenue'] < df['Total_Revenue'].quantile(0.25)]),
                'high_performing_products': len(df[df['Total_Revenue'] > df['Total_Revenue'].quantile(0.75)])
            }
            
            prompt_template = PromptTemplate(
                input_variables=["product_analysis"],
                template="""
                Analyze this product performance data for retail optimization:

                Product Metrics:
                - Total Products: {total_products}
                - Top 10 Products: {top_10_products}
                - Category Performance: {category_performance}
                - Average Revenue per Product: £{avg_revenue_per_product:.2f}
                - Low Performing Products: {low_performing_products}
                - High Performing Products: {high_performing_products}

                Provide insights on:
                1. **Product Portfolio Analysis** - Overall product mix health
                2. **Category Insights** - Which categories drive most revenue?
                3. **Inventory Optimization** - Products to focus/reduce
                4. **Cross-selling Opportunities** - Product bundling recommendations
                5. **Pricing Strategy** - Products with pricing potential

                Focus on actionable product management strategies.
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(**product_analysis)
            
            return {
                'type': 'product_performance',
                'generated_at': datetime.now().isoformat(),
                'product_analysis': product_analysis,
                'insights': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing products: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def analyze_market_trends(self) -> Dict[str, Any]:
        """Generate market and geographic trends insights"""
        try:
            if 'country_performance' not in self.datasets:
                return {}
            
            logger.info("Analyzing market trends...")
            
            country_df = self.datasets['country_performance']
            
            # Calculate market concentration
            total_revenue = country_df['Total_Revenue'].sum()
            country_df['Revenue_Share'] = (country_df['Total_Revenue'] / total_revenue * 100).round(2)
            
            market_analysis = {
                'total_markets': len(country_df),
                'top_5_markets': country_df.head(5)[['Country', 'Total_Revenue', 'Revenue_Share']].to_dict('records'),
                'market_concentration': float(country_df.head(3)['Revenue_Share'].sum()),  # Top 3 countries
                'avg_transaction_value_by_country': country_df[['Country', 'Avg_Transaction_Value']].head(5).to_dict('records'),
                'emerging_markets': country_df.tail(5)[['Country', 'Total_Revenue']].to_dict('records')
            }
            
            prompt_template = PromptTemplate(
                input_variables=["market_analysis"],
                template="""
                Analyze this geographic market performance data:

                Market Overview:
                - Total Markets: {total_markets}
                - Top 5 Markets: {top_5_markets}
                - Market Concentration (Top 3): {market_concentration:.1f}%
                - Avg Transaction Value by Country: {avg_transaction_value_by_country}
                - Emerging Markets: {emerging_markets}

                Provide insights on:
                1. **Market Diversification** - How concentrated is the business?
                2. **Geographic Expansion** - Which markets show potential?
                3. **Market-Specific Strategies** - Tailored approaches by region
                4. **Risk Assessment** - Over-dependence on specific markets
                5. **Growth Opportunities** - Markets for investment focus

                Focus on international expansion and market development strategies.
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(**market_analysis)
            
            return {
                'type': 'market_trends',
                'generated_at': datetime.now().isoformat(),
                'market_analysis': market_analysis,
                'insights': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def generate_automated_report(self) -> Dict[str, Any]:
        """Generate comprehensive automated business report"""
        try:
            logger.info("🤖 Generating comprehensive automated business report...")
            
            report = {
                'report_id': f"retail_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'generated_at': datetime.now().isoformat(),
                'report_type': 'automated_business_insights',
                'sections': {}
            }
            
            # Generate all insight sections
            sections = [
                ('business_summary', self.generate_business_summary),
                ('customer_segmentation', self.analyze_customer_segments),
                ('product_performance', self.analyze_product_performance),
                ('market_trends', self.analyze_market_trends)
            ]
            
            for section_name, section_func in sections:
                logger.info(f"Generating {section_name} insights...")
                section_result = section_func()
                if section_result.get('status') == 'success':
                    report['sections'][section_name] = section_result
                    logger.info(f"✅ {section_name} completed")
                else:
                    logger.warning(f"⚠️ {section_name} failed: {section_result.get('error', 'Unknown error')}")
            
            # Save report
            self._save_report(report)
            
            logger.info("✅ Automated business report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating automated report: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _save_report(self, report: Dict[str, Any]):
        """Save generated report to file"""
        try:
            reports_dir = self.project_root / "data" / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            filename = f"{report['report_id']}.json"
            filepath = reports_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
    
    def get_quick_insights(self, query: str) -> str:
        """Get quick insights based on natural language query"""
        try:
            if not self.llm:
                return "AI model not available - check GEMINI_API_KEY"
            
            # Use the main transactions dataset for quick queries
            if 'retail_transactions_processed' not in self.datasets:
                return "Transaction data not available"
            
            df = self.datasets['retail_transactions_processed']
            
            # Generate basic data summary for context
            data_context = f"""
            Dataset: Retail Transactions
            Shape: {df.shape}
            Columns: {list(df.columns)}
            Revenue Range: £{df['Revenue'].min():.2f} to £{df['Revenue'].max():.2f}
            Date Range: {df['Date'].min()} to {df['Date'].max()}
            Top Countries: {df['Country'].value_counts().head(3).to_dict()}
            """
            
            # Create prompt for answering the query
            prompt_template = PromptTemplate(
                input_variables=["query", "data_context"],
                template="""
                You are a retail business analyst. Answer the following question about retail transaction data:
                
                Question: {query}
                
                Data Context: {data_context}
                
                Provide a clear, specific answer based on the data context provided. If you need specific calculations, mention what analysis would be helpful.
                """
            )
            
            # Generate insights using LLM
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(query=query, data_context=data_context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in quick insights: {str(e)}")
            return f"Error generating insights: {str(e)}. Please check your Gemini API key configuration."


def main():
    """Main function to generate insights"""
    generator = RetailInsightsGenerator()
    
    # Generate comprehensive report
    report = generator.generate_automated_report()
    
    if report.get('status') != 'error':
        logger.info("🎉 AI-powered insights generation completed!")
        logger.info(f"Report sections generated: {list(report.get('sections', {}).keys())}")
    else:
        logger.error("❌ Insights generation failed")


if __name__ == "__main__":
    main()