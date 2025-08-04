import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticSalesDataGenerator:
    """Generate synthetic but realistic sales data for forecasting demo"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data" / "processed"
    
    def generate_synthetic_sales_data(self, start_date: str = "2023-01-01", 
                                    end_date: str = "2024-12-31",
                                    base_daily_revenue: float = 2000) -> pd.DataFrame:
        """
        Generate synthetic daily sales data with realistic patterns
        
        Features:
        - Seasonal trends (higher sales in Q4, lower in Q1)
        - Weekly patterns (weekend vs weekday differences) 
        - Monthly patterns (month-end spikes)
        - Random noise and growth trend
        - Holiday effects
        """
        try:
            logger.info(f"Generating synthetic sales data from {start_date} to {end_date}")
            
            # Create date range
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Initialize data
            np.random.seed(42)  # For reproducible results
            
            synthetic_data = []
            
            for i, date in enumerate(dates):
                # Base revenue with growth trend
                days_from_start = i
                growth_factor = 1 + (days_from_start / len(dates)) * 0.3  # 30% growth over period
                base_revenue = base_daily_revenue * growth_factor
                
                # Seasonal patterns
                month = date.month
                if month in [11, 12]:  # Black Friday, Christmas
                    seasonal_multiplier = 1.8
                elif month in [1, 2]:  # Post-holiday slump
                    seasonal_multiplier = 0.7
                elif month in [6, 7, 8]:  # Summer boost
                    seasonal_multiplier = 1.2
                else:
                    seasonal_multiplier = 1.0
                
                # Weekly patterns
                day_of_week = date.dayofweek
                if day_of_week in [5, 6]:  # Weekend (Sat, Sun)
                    weekly_multiplier = 1.3
                elif day_of_week in [0, 4]:  # Monday, Friday
                    weekly_multiplier = 1.1
                else:  # Tue, Wed, Thu
                    weekly_multiplier = 0.9
                
                # Month-end effect (higher sales end of month)
                if date.day >= 28:
                    month_end_multiplier = 1.2
                elif date.day <= 3:
                    month_end_multiplier = 0.8
                else:
                    month_end_multiplier = 1.0
                
                # Holiday effects (simplified)
                holiday_multiplier = 1.0
                if month == 12 and date.day in [24, 25, 26]:  # Christmas
                    holiday_multiplier = 2.5
                elif month == 11 and 22 <= date.day <= 24:  # Black Friday weekend
                    holiday_multiplier = 3.0
                elif month == 1 and date.day == 1:  # New Year
                    holiday_multiplier = 0.3
                
                # Calculate final revenue
                daily_revenue = (base_revenue * seasonal_multiplier * 
                               weekly_multiplier * month_end_multiplier * 
                               holiday_multiplier)
                
                # Add random noise (±20%)
                noise_factor = np.random.normal(1.0, 0.15)
                daily_revenue *= max(0.1, noise_factor)  # Ensure positive
                
                # Generate correlated metrics
                # Orders are somewhat correlated with revenue but with noise
                daily_orders = max(1, int(daily_revenue / np.random.normal(45, 10)))
                
                # Customers per order ratio
                customer_ratio = np.random.normal(0.7, 0.1)  # 70% of orders are from unique customers
                daily_customers = max(1, int(daily_orders * max(0.3, customer_ratio)))
                
                # Items per order
                items_per_order = np.random.normal(2.5, 0.8)
                daily_items = max(1, int(daily_orders * max(1.0, items_per_order)))
                
                # Product variety (more variety on higher revenue days)
                daily_products = max(1, int(20 + (daily_revenue / base_daily_revenue) * 15 + np.random.normal(0, 5)))
                
                synthetic_data.append({
                    'Date': date,
                    'Daily_Revenue': round(daily_revenue, 2),
                    'Daily_Orders': daily_orders,
                    'Daily_Customers': daily_customers,
                    'Daily_Items': daily_items,
                    'Daily_Products': daily_products
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(synthetic_data)
            
            # Add time-based features (same as in forecasting model)
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfYear'] = df['Date'].dt.dayofyear
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week
            df['Quarter'] = df['Date'].dt.quarter
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
            df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
            
            logger.info(f"Generated {len(df)} days of synthetic sales data")
            logger.info(f"Total synthetic revenue: £{df['Daily_Revenue'].sum():,.2f}")
            logger.info(f"Average daily revenue: £{df['Daily_Revenue'].mean():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise
    
    def save_synthetic_data(self, df: pd.DataFrame, filename: str = "synthetic_sales_data.csv"):
        """Save synthetic data for use in forecasting models"""
        try:
            output_file = self.data_path / filename
            df.to_csv(output_file, index=False)
            logger.info(f"Synthetic sales data saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving synthetic data: {str(e)}")
            raise


def main():
    """Generate synthetic sales data for forecasting demo"""
    try:
        generator = SyntheticSalesDataGenerator()
        
        # Generate 2 years of data for comprehensive forecasting demo
        synthetic_df = generator.generate_synthetic_sales_data(
            start_date="2023-01-01",
            end_date="2024-12-31",
            base_daily_revenue=1500  # Realistic base for demo
        )
        
        # Save the data
        output_file = generator.save_synthetic_data(synthetic_df)
        
        print("\n🎉 Synthetic Sales Data Generated Successfully!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 {len(synthetic_df)} days of realistic sales data")
        print(f"💰 Total Revenue: £{synthetic_df['Daily_Revenue'].sum():,.2f}")
        print(f"📈 Daily Average: £{synthetic_df['Daily_Revenue'].mean():.2f}")
        print("\n✅ Ready for time series forecasting analysis!")
        
        return synthetic_df
        
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {str(e)}")
        raise


if __name__ == "__main__":
    main()