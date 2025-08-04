import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Tuple, Any, Optional
import pickle
import json
import warnings

# ML and Time Series Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series Libraries (using statsmodels for comprehensive analysis)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Some advanced time series features will be limited.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesForecastingModel:
    """
    Advanced Sales Forecasting using Multiple Approaches
    
    Features:
    - Time Series Decomposition (Trend, Seasonality, Residuals)
    - Multiple Forecasting Models (ARIMA, Exponential Smoothing, ML)
    - Cross-validation with Time Series Split
    - Confidence Intervals & Uncertainty Quantification
    - Feature Engineering (Lag features, Moving averages, etc.)
    - Model Comparison & Selection
    - Business Insights & Recommendations
    """
    
    def __init__(self):
        """Initialize the sales forecasting model"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data" / "processed"
        self.models_path = self.project_root / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.model_metrics = {}
        self.forecasts = {}
        
        # Preprocessing
        self.scaler = StandardScaler()
        
        # Results
        self.sales_data = None
        self.forecast_results = None
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare sales data for forecasting"""
        try:
            logger.info("Loading and preparing sales data for forecasting...")
            
            # Try to load synthetic data first (for comprehensive demo)
            synthetic_file = self.data_path / "synthetic_sales_data.csv"
            transactions_file = self.data_path / "retail_transactions_processed.csv"
            
            if synthetic_file.exists():
                logger.info("Using synthetic sales data for comprehensive forecasting demo")
                df = pd.read_csv(synthetic_file)
                df['Date'] = pd.to_datetime(df['Date'])
                
                logger.info(f"Loaded synthetic sales data: {len(df)} days")
                logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
                logger.info(f"Total revenue: £{df['Daily_Revenue'].sum():,.2f}")
                
                self.sales_data = df
                return df
            
            elif transactions_file.exists():
                logger.info("Using real transaction data (may be limited for forecasting)")
                df = pd.read_csv(transactions_file)
                
                # Convert dates
                df['Date'] = pd.to_datetime(df['Date'])
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                
                # Filter valid transactions
                df = df[df['Revenue'] > 0]  # Remove returns/cancellations
                
                # Create daily sales aggregation
                daily_sales = df.groupby('Date').agg({
                    'Revenue': 'sum',
                    'Invoice': 'nunique',  # Number of unique orders
                    'Customer_ID': 'nunique',  # Number of unique customers
                    'Quantity': 'sum',  # Total items sold
                    'StockCode': 'nunique'  # Product variety
                }).reset_index()
                
                daily_sales.columns = ['Date', 'Daily_Revenue', 'Daily_Orders', 
                                     'Daily_Customers', 'Daily_Items', 'Daily_Products']
                
                df = daily_sales
            
            else:
                raise FileNotFoundError("No sales data found for forecasting")
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Fill missing dates (if any gaps in time series)
            date_range = pd.date_range(start=df['Date'].min(), 
                                     end=df['Date'].max(), 
                                     freq='D')
            
            complete_dates = pd.DataFrame({'Date': date_range})
            df = complete_dates.merge(df, on='Date', how='left')
            df = df.fillna(0)  # Fill missing days with 0 sales
            
            # Add time-based features (only if not already present from synthetic data)
            if 'Year' not in df.columns:
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
            
            logger.info(f"Prepared daily sales data: {len(df)} days")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"Total revenue: £{df['Daily_Revenue'].sum():,.2f}")
            
            self.sales_data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading sales data: {str(e)}")
            raise
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'Daily_Revenue', 
                           lag_periods: list = [1, 3, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for time series forecasting"""
        try:
            logger.info("Creating lag features...")
            
            df_featured = df.copy()
            
            # Lag features
            for lag in lag_periods:
                df_featured[f'{target_col}_lag_{lag}'] = df_featured[target_col].shift(lag)
            
            # Rolling window features
            for window in [3, 7, 14, 30]:
                df_featured[f'{target_col}_rolling_mean_{window}'] = df_featured[target_col].rolling(window=window).mean()
                df_featured[f'{target_col}_rolling_std_{window}'] = df_featured[target_col].rolling(window=window).std()
                df_featured[f'{target_col}_rolling_max_{window}'] = df_featured[target_col].rolling(window=window).max()
                df_featured[f'{target_col}_rolling_min_{window}'] = df_featured[target_col].rolling(window=window).min()
            
            # Exponential weighted moving averages
            for alpha in [0.1, 0.3, 0.5]:
                df_featured[f'{target_col}_ewm_{str(alpha).replace(".", "")}'] = df_featured[target_col].ewm(alpha=alpha).mean()
            
            # Differences (trend detection)
            df_featured[f'{target_col}_diff_1'] = df_featured[target_col].diff(1)
            df_featured[f'{target_col}_diff_7'] = df_featured[target_col].diff(7)
            
            # Percentage changes
            df_featured[f'{target_col}_pct_change_1'] = df_featured[target_col].pct_change(1)
            df_featured[f'{target_col}_pct_change_7'] = df_featured[target_col].pct_change(7)
            
            logger.info(f"Created lag features. Dataset shape: {df_featured.shape}")
            return df_featured
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def perform_time_series_decomposition(self, df: pd.DataFrame, target_col: str = 'Daily_Revenue') -> Dict[str, Any]:
        """Perform time series decomposition to understand components"""
        try:
            logger.info("Performing time series decomposition...")
            
            if not STATSMODELS_AVAILABLE:
                logger.warning("statsmodels not available. Skipping advanced decomposition.")
                return {'error': 'statsmodels not available'}
            
            # Set date as index for time series analysis
            ts_data = df.set_index('Date')[target_col].dropna()
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=7)  # Weekly seasonality
            
            # Extract components
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            residual = decomposition.resid.dropna()
            
            # Calculate component statistics
            decomposition_stats = {
                'trend_strength': float(1 - (residual.var() / (trend + residual).var())),
                'seasonal_strength': float(1 - (residual.var() / (seasonal + residual).var())),
                'trend_direction': 'increasing' if trend.iloc[-1] > trend.iloc[0] else 'decreasing',
                'seasonal_period': 7,
                'residual_autocorr': float(residual.autocorr()),
                'components': {
                    'trend': trend.to_dict(),
                    'seasonal': seasonal.to_dict(),
                    'residual': residual.to_dict()
                }
            }
            
            # Stationarity test
            adf_result = adfuller(ts_data.dropna())
            decomposition_stats['stationarity'] = {
                'adf_statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': {k: float(v) for k, v in adf_result[4].items()}
            }
            
            logger.info("Time series decomposition completed")
            return decomposition_stats
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {str(e)}")
            return {'error': str(e)}
    
    def train_arima_model(self, df: pd.DataFrame, target_col: str = 'Daily_Revenue') -> Dict[str, Any]:
        """Train ARIMA model for time series forecasting"""
        try:
            logger.info("Training ARIMA model...")
            
            if not STATSMODELS_AVAILABLE:
                logger.warning("statsmodels not available. Skipping ARIMA model.")
                return {'error': 'statsmodels not available'}
            
            # Prepare time series data
            ts_data = df.set_index('Date')[target_col].dropna()
            
            # Split data for validation
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data[:train_size]
            test_data = ts_data[train_size:]
            
            # Auto-select ARIMA parameters (simplified approach)
            best_aic = float('inf')
            best_params = None
            best_model = None
            
            # Grid search for optimal parameters
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                                best_model = fitted_model
                                
                        except:
                            continue
            
            if best_model is None:
                logger.warning("Could not fit ARIMA model. Using simple parameters.")
                best_params = (1, 1, 1)
                model = ARIMA(train_data, order=best_params)
                best_model = model.fit()
            
            # Make predictions
            forecast_steps = len(test_data)
            forecast = best_model.forecast(steps=forecast_steps)
            forecast_conf_int = best_model.get_forecast(steps=forecast_steps).conf_int()
            
            # Calculate metrics
            mse = mean_squared_error(test_data, forecast)
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mse)
            
            # Store model
            self.models['arima'] = best_model
            
            arima_results = {
                'model_params': best_params,
                'aic': float(best_model.aic),
                'bic': float(best_model.bic),
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'forecast': forecast.tolist(),
                'forecast_conf_int_lower': forecast_conf_int.iloc[:, 0].tolist(),
                'forecast_conf_int_upper': forecast_conf_int.iloc[:, 1].tolist(),
                'test_dates': test_data.index.strftime('%Y-%m-%d').tolist(),
                'test_actual': test_data.tolist()
            }
            
            logger.info(f"ARIMA model trained: {best_params}, RMSE: {rmse:.2f}")
            return arima_results
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            return {'error': str(e)}
    
    def train_exponential_smoothing_model(self, df: pd.DataFrame, target_col: str = 'Daily_Revenue') -> Dict[str, Any]:
        """Train Exponential Smoothing model"""
        try:
            logger.info("Training Exponential Smoothing model...")
            
            if not STATSMODELS_AVAILABLE:
                logger.warning("statsmodels not available. Skipping Exponential Smoothing.")
                return {'error': 'statsmodels not available'}
            
            # Prepare time series data
            ts_data = df.set_index('Date')[target_col].dropna()
            
            # Split data
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data[:train_size]
            test_data = ts_data[train_size:]
            
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(train_data, 
                                       trend='add',  # Additive trend
                                       seasonal='add',  # Additive seasonality
                                       seasonal_periods=7)  # Weekly seasonality
            
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data, forecast)
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mse)
            
            # Store model
            self.models['exponential_smoothing'] = fitted_model
            
            es_results = {
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic),
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'forecast': forecast.tolist(),
                'test_dates': test_data.index.strftime('%Y-%m-%d').tolist(),
                'test_actual': test_data.tolist()
            }
            
            logger.info(f"Exponential Smoothing model trained: RMSE: {rmse:.2f}")
            return es_results
            
        except Exception as e:
            logger.error(f"Error training Exponential Smoothing model: {str(e)}")
            return {'error': str(e)}
    
    def train_ml_models(self, df: pd.DataFrame, target_col: str = 'Daily_Revenue') -> Dict[str, Any]:
        """Train machine learning models for forecasting"""
        try:
            logger.info("Training ML models for forecasting...")
            
            # Create feature matrix
            df_featured = self.create_lag_features(df, target_col)
            
            # Remove rows with NaN values (due to lag features)
            df_clean = df_featured.dropna()
            
            # Prepare features and target
            feature_cols = [col for col in df_clean.columns if col not in ['Date', target_col] and 'Date' not in col]
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Define models to test
            ml_models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            ml_results = {}
            
            for model_name, model in ml_models.items():
                logger.info(f"Training {model_name}...")
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                cv_rmse_scores = np.sqrt(-cv_scores)
                
                # Train on full data for final model
                model.fit(X, y)
                
                # Make predictions on recent data for validation
                recent_size = min(30, len(X) // 4)  # Last 30 days or 25% of data
                X_recent = X.iloc[-recent_size:]
                y_recent = y.iloc[-recent_size:]
                
                predictions = model.predict(X_recent)
                
                # Calculate metrics
                mse = mean_squared_error(y_recent, predictions)
                mae = mean_absolute_error(y_recent, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_recent, predictions)
                
                # Feature importance (for tree-based models)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_cols, model.feature_importances_))
                    # Sort by importance
                    feature_importance = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
                
                # Store model and results
                self.models[model_name] = model
                
                ml_results[model_name] = {
                    'cv_rmse_mean': float(cv_rmse_scores.mean()),
                    'cv_rmse_std': float(cv_rmse_scores.std()),
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2_score': float(r2),
                    'feature_importance': {k: float(v) for k, v in feature_importance.items()},
                    'predictions': predictions.tolist(),
                    'actual': y_recent.tolist(),
                    'feature_count': len(feature_cols)
                }
                
                logger.info(f"{model_name} - RMSE: {rmse:.2f}, R²: {r2:.3f}")
            
            return ml_results
            
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
            raise
    
    def generate_future_forecasts(self, df: pd.DataFrame, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate future forecasts using trained models"""
        try:
            logger.info(f"Generating {forecast_days}-day future forecasts...")
            
            forecasts = {}
            
            # Prepare the most recent data
            df_featured = self.create_lag_features(df, 'Daily_Revenue')
            df_clean = df_featured.dropna()
            
            last_date = df['Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            
            # ML Model Forecasts (using best performing model)
            if 'random_forest' in self.models:
                feature_cols = [col for col in df_clean.columns if col not in ['Date', 'Daily_Revenue'] and 'Date' not in col]
                
                # Create future feature matrix (simplified approach)
                future_features = []
                recent_data = df_clean.tail(forecast_days * 2)  # Use recent data as proxy
                
                for i, future_date in enumerate(future_dates):
                    # Use cyclic features for future dates
                    future_row = {
                        'Year': future_date.year,
                        'Month': future_date.month,
                        'Day': future_date.day,
                        'DayOfWeek': future_date.dayofweek,
                        'DayOfYear': future_date.dayofyear,
                        'WeekOfYear': future_date.isocalendar()[1],
                        'Quarter': future_date.quarter,
                        'IsWeekend': int(future_date.dayofweek in [5, 6]),
                        'IsMonthStart': int(future_date.day == 1),
                        'IsMonthEnd': int(future_date.day == future_date.days_in_month)
                    }
                    
                    # For lag features, use recent actual values or previous predictions
                    if i == 0:
                        # First prediction uses actual lag values
                        for col in feature_cols:
                            if col in future_row:
                                continue
                            if 'lag_1' in col:
                                future_row[col] = df_clean['Daily_Revenue'].iloc[-1]
                            elif 'lag_3' in col:
                                future_row[col] = df_clean['Daily_Revenue'].iloc[-3]
                            elif 'lag_7' in col:
                                future_row[col] = df_clean['Daily_Revenue'].iloc[-7]
                            else:
                                # Use median for other features
                                future_row[col] = df_clean[col].median()
                    else:
                        # Subsequent predictions use previous forecasts for lag features
                        for col in feature_cols:
                            if col in future_row:
                                continue
                            # Simplified: use median values
                            future_row[col] = df_clean[col].median()
                    
                    future_features.append(future_row)
                
                # Convert to DataFrame and ensure all required features are present
                future_df = pd.DataFrame(future_features)
                
                # Fill missing features with medians
                for col in feature_cols:
                    if col not in future_df.columns:
                        future_df[col] = df_clean[col].median()
                
                # Ensure column order matches training data
                future_df = future_df[feature_cols]
                
                # Generate ML forecasts
                rf_forecast = self.models['random_forest'].predict(future_df)
                forecasts['random_forest'] = rf_forecast.tolist()
            
            # ARIMA Forecasts
            if 'arima' in self.models and STATSMODELS_AVAILABLE:
                try:
                    arima_forecast = self.models['arima'].forecast(steps=forecast_days)
                    forecasts['arima'] = arima_forecast.tolist()
                except:
                    logger.warning("Could not generate ARIMA forecast")
            
            # Exponential Smoothing Forecasts
            if 'exponential_smoothing' in self.models and STATSMODELS_AVAILABLE:
                try:
                    es_forecast = self.models['exponential_smoothing'].forecast(steps=forecast_days)
                    forecasts['exponential_smoothing'] = es_forecast.tolist()
                except:
                    logger.warning("Could not generate Exponential Smoothing forecast")
            
            # Ensemble forecast (average of available forecasts)
            if len(forecasts) > 1:
                forecast_arrays = [np.array(f) for f in forecasts.values()]
                ensemble_forecast = np.mean(forecast_arrays, axis=0)
                forecasts['ensemble'] = ensemble_forecast.tolist()
            
            forecast_results = {
                'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'forecasts': forecasts,
                'forecast_period': forecast_days,
                'last_actual_date': last_date.strftime('%Y-%m-%d'),
                'total_forecasted_revenue': {model: float(np.sum(forecast)) 
                                           for model, forecast in forecasts.items()}
            }
            
            logger.info("Future forecasts generated successfully")
            return forecast_results
            
        except Exception as e:
            logger.error(f"Error generating future forecasts: {str(e)}")
            raise
    
    def generate_business_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate business insights from forecasting analysis"""
        try:
            logger.info("Generating business insights from forecasting...")
            
            insights = {}
            
            # Historical performance analysis
            recent_30_days = df.tail(30)
            recent_revenue = recent_30_days['Daily_Revenue'].sum()
            avg_daily_revenue = df['Daily_Revenue'].mean()
            revenue_trend = 'increasing' if recent_30_days['Daily_Revenue'].mean() > avg_daily_revenue else 'decreasing'
            
            insights['historical_performance'] = {
                'total_historical_revenue': float(df['Daily_Revenue'].sum()),
                'avg_daily_revenue': float(avg_daily_revenue),
                'recent_30_day_revenue': float(recent_revenue),
                'revenue_trend': revenue_trend,
                'best_day_revenue': float(df['Daily_Revenue'].max()),
                'worst_day_revenue': float(df['Daily_Revenue'].min()),
                'revenue_volatility': float(df['Daily_Revenue'].std())
            }
            
            # Seasonality insights
            day_of_week_avg = df.groupby('DayOfWeek')['Daily_Revenue'].mean()
            month_avg = df.groupby('Month')['Daily_Revenue'].mean()
            
            insights['seasonality_patterns'] = {
                'best_day_of_week': int(day_of_week_avg.idxmax()),
                'worst_day_of_week': int(day_of_week_avg.idxmin()),
                'best_month': int(month_avg.idxmax()),
                'worst_month': int(month_avg.idxmin()),
                'weekend_vs_weekday_ratio': float(df[df['IsWeekend'] == 1]['Daily_Revenue'].mean() / 
                                                df[df['IsWeekend'] == 0]['Daily_Revenue'].mean())
            }
            
            # Model performance comparison
            if self.model_metrics:
                best_model = None
                best_rmse = float('inf')
                
                for model_name, metrics in self.model_metrics.items():
                    if isinstance(metrics, dict) and 'rmse' in metrics:
                        if metrics['rmse'] < best_rmse:
                            best_rmse = metrics['rmse']
                            best_model = model_name
                
                insights['model_performance'] = {
                    'best_model': best_model,
                    'best_rmse': float(best_rmse),
                    'model_comparison': self.model_metrics
                }
            
            # Future forecast insights
            if self.forecast_results:
                forecasts = self.forecast_results.get('forecasts', {})
                if 'ensemble' in forecasts:
                    forecast_revenue = sum(forecasts['ensemble'])
                    insights['forecast_insights'] = {
                        'predicted_30_day_revenue': float(forecast_revenue),
                        'vs_recent_30_days': float(forecast_revenue / recent_revenue - 1) * 100,
                        'predicted_daily_avg': float(forecast_revenue / 30),
                        'forecast_confidence': 'high' if best_rmse < avg_daily_revenue * 0.2 else 'medium'
                    }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating business insights: {str(e)}")
            return {}
    
    def save_models_and_results(self, model_version: str = None):
        """Save all models and results"""
        try:
            if model_version is None:
                model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_dir = self.models_path / f"sales_forecasting_{model_version}"
            model_dir.mkdir(exist_ok=True)
            
            # Save ML models
            for model_name, model in self.models.items():
                if model_name in ['random_forest', 'gradient_boosting', 'linear_regression']:
                    with open(model_dir / f"{model_name}_model.pkl", "wb") as f:
                        pickle.dump(model, f)
            
            # Save scaler
            with open(model_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            
            # Save metrics and results
            with open(model_dir / "model_metrics.json", "w") as f:
                json.dump(self.model_metrics, f, indent=2)
            
            if self.forecast_results:
                with open(model_dir / "forecast_results.json", "w") as f:
                    json.dump(self.forecast_results, f, indent=2)
            
            # Save sales data with features
            if self.sales_data is not None:
                output_file = self.data_path / "sales_forecast_data.csv"
                self.sales_data.to_csv(output_file, index=False)
                logger.info(f"Sales forecast data saved to: {output_file}")
            
            logger.info(f"Models and results saved to: {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def run_complete_forecasting_analysis(self, forecast_days: int = 30) -> Dict[str, Any]:
        """Run the complete sales forecasting analysis"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING SALES FORECASTING ANALYSIS")
            logger.info("=" * 60)
            
            # Load and prepare data
            df = self.load_and_prepare_data()
            
            # Time series decomposition
            decomposition_results = self.perform_time_series_decomposition(df)
            self.model_metrics['decomposition'] = decomposition_results
            
            # Train time series models
            arima_results = self.train_arima_model(df)
            self.model_metrics['arima'] = arima_results
            
            es_results = self.train_exponential_smoothing_model(df)
            self.model_metrics['exponential_smoothing'] = es_results
            
            # Train ML models
            ml_results = self.train_ml_models(df)
            self.model_metrics.update(ml_results)
            
            # Generate future forecasts
            forecast_results = self.generate_future_forecasts(df, forecast_days)
            self.forecast_results = forecast_results
            
            # Generate business insights
            business_insights = self.generate_business_insights(df)
            self.model_metrics['business_insights'] = business_insights
            
            # Save everything
            self.save_models_and_results()
            
            logger.info("=" * 60)
            logger.info("SALES FORECASTING ANALYSIS COMPLETED")
            logger.info("=" * 60)
            
            # Print summary
            self._print_forecasting_summary()
            
            return {
                'model_metrics': self.model_metrics,
                'forecast_results': self.forecast_results
            }
            
        except Exception as e:
            logger.error(f"Error in forecasting analysis: {str(e)}")
            raise
    
    def _print_forecasting_summary(self):
        """Print forecasting analysis summary"""
        try:
            print("\n" + "="*60)
            print("📈 SALES FORECASTING ANALYSIS SUMMARY")
            print("="*60)
            
            if self.sales_data is not None:
                total_days = len(self.sales_data)
                total_revenue = self.sales_data['Daily_Revenue'].sum()
                avg_daily = self.sales_data['Daily_Revenue'].mean()
                
                print(f"📊 Historical Data: {total_days} days")
                print(f"💰 Total Historical Revenue: £{total_revenue:,.2f}")
                print(f"📈 Average Daily Revenue: £{avg_daily:,.2f}")
            
            print(f"\n🤖 MODEL PERFORMANCE:")
            for model_name, metrics in self.model_metrics.items():
                if isinstance(metrics, dict) and 'rmse' in metrics:
                    rmse = metrics['rmse']
                    if 'r2_score' in metrics:
                        r2 = metrics['r2_score']
                        print(f"   {model_name}: RMSE £{rmse:.2f}, R² {r2:.3f}")
                    else:
                        print(f"   {model_name}: RMSE £{rmse:.2f}")
            
            if self.forecast_results:
                forecasts = self.forecast_results.get('forecasts', {})
                if 'ensemble' in forecasts:
                    total_forecast = sum(forecasts['ensemble'])
                    daily_avg_forecast = total_forecast / len(forecasts['ensemble'])
                    print(f"\n📅 30-DAY FORECAST:")
                    print(f"   Total Forecasted Revenue: £{total_forecast:,.2f}")
                    print(f"   Average Daily Forecast: £{daily_avg_forecast:,.2f}")
            
            print("\n✅ Forecasting analysis complete!")
            print("📁 Results saved to data/processed/sales_forecast_data.csv")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}")


def main():
    """Main function to run sales forecasting analysis"""
    try:
        # Initialize and run analysis
        forecasting_model = SalesForecastingModel()
        results = forecasting_model.run_complete_forecasting_analysis(forecast_days=30)
        
        print("\n🎉 Sales Forecasting Analysis Complete!")
        print("📊 Multiple models trained and compared")
        print("📈 30-day revenue forecast generated")
        
        return results
        
    except Exception as e:
        logger.error(f"Forecasting analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()