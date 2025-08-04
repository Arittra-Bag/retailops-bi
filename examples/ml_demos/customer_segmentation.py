import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Tuple, Any, Optional
import pickle
import json

# ML Libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerSegmentationModel:
    """
    Advanced Customer Segmentation using RFM Analysis + ML
    
    Features:
    - RFM (Recency, Frequency, Monetary) Analysis
    - K-Means Clustering with optimal cluster selection
    - Customer Lifetime Value (CLV) Prediction
    - Churn Risk Classification
    - Model Validation & Cross-validation
    - Business Insights Generation
    """
    
    def __init__(self):
        """Initialize the customer segmentation model"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data" / "processed"
        self.models_path = self.project_root / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Model components
        self.rfm_scaler = StandardScaler()
        self.kmeans_model = None
        self.clv_model = None
        self.churn_model = None
        
        # Analysis date (for recency calculation)
        self.analysis_date = datetime.now()
        
        # Results storage
        self.customer_segments = None
        self.model_metrics = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare customer transaction data"""
        try:
            logger.info("Loading customer transaction data...")
            
            # Load main transactions
            transactions_file = self.data_path / "retail_transactions_processed.csv"
            if not transactions_file.exists():
                raise FileNotFoundError(f"Transactions file not found: {transactions_file}")
            
            df = pd.read_csv(transactions_file)
            
            # Convert date columns
            df['Date'] = pd.to_datetime(df['Date'])
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            # Filter out invalid customers and returns
            df = df[df['Customer_ID'].notna()]
            df = df[df['Revenue'] > 0]  # Remove returns for segmentation
            
            logger.info(f"Loaded {len(df):,} valid transactions for {df['Customer_ID'].nunique():,} customers")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) features
        Core of customer segmentation analysis
        """
        try:
            logger.info("Calculating RFM features...")
            
            # Use the most recent date in data as analysis date
            self.analysis_date = df['Date'].max()
            
            # Calculate RFM metrics per customer
            rfm = df.groupby('Customer_ID').agg({
                'Date': 'max',           # Most recent purchase date
                'Invoice': 'nunique',    # Frequency (number of unique orders)
                'Revenue': 'sum'         # Monetary (total spent)
            }).reset_index()
            
            # Calculate Recency (days since last purchase)
            rfm['Recency'] = (self.analysis_date - rfm['Date']).dt.days
            rfm['Frequency'] = rfm['Invoice']
            rfm['Monetary'] = rfm['Revenue']
            
            # Additional customer features
            customer_features = df.groupby('Customer_ID').agg({
                'Quantity': 'sum',       # Total items purchased
                'Date': ['min', 'count'], # First purchase date, transaction count
                'StockCode': 'nunique',  # Product variety
                'Country': 'first'       # Customer location
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = ['Customer_ID', 'Total_Items', 'First_Purchase', 
                                       'Transaction_Count', 'Product_Variety', 'Country']
            
            # Calculate customer tenure
            customer_features['First_Purchase'] = pd.to_datetime(customer_features['First_Purchase'])
            customer_features['Tenure_Days'] = (self.analysis_date - customer_features['First_Purchase']).dt.days
            
            # Merge RFM with additional features
            rfm_enhanced = rfm.merge(customer_features, on='Customer_ID', how='left')
            
            # Calculate additional business metrics
            rfm_enhanced['Avg_Order_Value'] = rfm_enhanced['Monetary'] / rfm_enhanced['Frequency']
            rfm_enhanced['Purchase_Frequency'] = rfm_enhanced['Frequency'] / (rfm_enhanced['Tenure_Days'] + 1) * 30  # Monthly frequency
            rfm_enhanced['Customer_Value_Score'] = rfm_enhanced['Monetary'] * rfm_enhanced['Frequency']
            
            logger.info(f"Calculated RFM features for {len(rfm_enhanced):,} customers")
            return rfm_enhanced
            
        except Exception as e:
            logger.error(f"Error calculating RFM features: {str(e)}")
            raise
    
    def create_rfm_scores(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM scores (1-5 scale) for traditional RFM analysis
        """
        try:
            logger.info("Creating RFM scores...")
            
            rfm_scored = rfm_df.copy()
            
            # Create quintile-based scores (1-5)
            # Recency: Lower is better (recent customers score higher)
            rfm_scored['R_Score'] = pd.qcut(rfm_scored['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
            
            # Frequency: Higher is better
            rfm_scored['F_Score'] = pd.qcut(rfm_scored['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
            
            # Monetary: Higher is better
            rfm_scored['M_Score'] = pd.qcut(rfm_scored['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
            
            # Combine scores
            rfm_scored['RFM_Score'] = rfm_scored['R_Score'].astype(str) + rfm_scored['F_Score'].astype(str) + rfm_scored['M_Score'].astype(str)
            rfm_scored['RFM_Score_Numeric'] = rfm_scored['R_Score'].astype(int) + rfm_scored['F_Score'].astype(int) + rfm_scored['M_Score'].astype(int)
            
            # Create traditional RFM segments
            def rfm_segment(row):
                if row['RFM_Score_Numeric'] >= 13:
                    return 'Champions'
                elif row['RFM_Score_Numeric'] >= 11:
                    return 'Loyal Customers'
                elif row['RFM_Score_Numeric'] >= 9:
                    return 'Potential Loyalists'
                elif row['RFM_Score_Numeric'] >= 7:
                    return 'New Customers'
                elif row['RFM_Score_Numeric'] >= 5:
                    return 'Promising'
                elif row['RFM_Score_Numeric'] >= 3:
                    return 'Need Attention'
                else:
                    return 'At Risk'
            
            rfm_scored['RFM_Segment'] = rfm_scored.apply(rfm_segment, axis=1)
            
            logger.info("RFM scores created successfully")
            return rfm_scored
            
        except Exception as e:
            logger.error(f"Error creating RFM scores: {str(e)}")
            raise
    
    def perform_clustering(self, rfm_df: pd.DataFrame, n_clusters_range: Tuple[int, int] = (2, 10)) -> pd.DataFrame:
        """
        Perform K-Means clustering with optimal cluster selection
        Uses multiple metrics for robust cluster validation
        """
        try:
            logger.info("Performing K-Means clustering...")
            
            # Select features for clustering
            clustering_features = ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value', 
                                 'Purchase_Frequency', 'Product_Variety', 'Tenure_Days']
            
            X = rfm_df[clustering_features].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Scale features for clustering
            X_scaled = self.rfm_scaler.fit_transform(X)
            
            # Find optimal number of clusters
            cluster_metrics = []
            min_clusters, max_clusters = n_clusters_range
            
            for n_clusters in range(min_clusters, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Calculate clustering metrics
                silhouette = silhouette_score(X_scaled, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                inertia = kmeans.inertia_
                
                cluster_metrics.append({
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'davies_bouldin_score': davies_bouldin,
                    'inertia': inertia
                })
            
            # Select optimal clusters (highest silhouette score)
            best_clusters = max(cluster_metrics, key=lambda x: x['silhouette_score'])
            optimal_k = best_clusters['n_clusters']
            
            logger.info(f"Optimal number of clusters: {optimal_k} (Silhouette Score: {best_clusters['silhouette_score']:.3f})")
            
            # Train final model with optimal clusters
            self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(X_scaled)
            
            # Add cluster labels to data
            rfm_clustered = rfm_df.copy()
            rfm_clustered['ML_Cluster'] = cluster_labels
            
            # Create cluster profiles
            cluster_profiles = self.create_cluster_profiles(rfm_clustered, clustering_features)
            
            # Store metrics
            self.model_metrics['clustering'] = {
                'optimal_clusters': optimal_k,
                'metrics': cluster_metrics,
                'best_metrics': best_clusters,
                'cluster_profiles': cluster_profiles
            }
            
            logger.info("Clustering completed successfully")
            return rfm_clustered
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            raise
    
    def create_cluster_profiles(self, df: pd.DataFrame, features: list) -> Dict:
        """Create detailed profiles for each cluster"""
        try:
            profiles = {}
            
            for cluster in sorted(df['ML_Cluster'].unique()):
                cluster_data = df[df['ML_Cluster'] == cluster]
                
                profile = {
                    'cluster_id': int(cluster),
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100,
                    'characteristics': {}
                }
                
                # Calculate mean values for each feature
                for feature in features:
                    profile['characteristics'][feature] = {
                        'mean': float(cluster_data[feature].mean()),
                        'median': float(cluster_data[feature].median()),
                        'std': float(cluster_data[feature].std())
                    }
                
                # Add business insights
                profile['business_insights'] = {
                    'avg_clv': float(cluster_data['Monetary'].mean()),
                    'avg_recency': float(cluster_data['Recency'].mean()),
                    'avg_frequency': float(cluster_data['Frequency'].mean()),
                    'total_revenue': float(cluster_data['Monetary'].sum())
                }
                
                profiles[f'cluster_{cluster}'] = profile
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error creating cluster profiles: {str(e)}")
            raise
    
    def train_clv_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Customer Lifetime Value prediction model
        """
        try:
            logger.info("Training Customer Lifetime Value prediction model...")
            
            # Prepare features for CLV prediction
            features = ['Recency', 'Frequency', 'Avg_Order_Value', 'Purchase_Frequency', 
                       'Product_Variety', 'Tenure_Days', 'Transaction_Count']
            
            X = df[features].copy()
            y = df['Monetary']  # Current customer value as target
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest model
            self.clv_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.clv_model.fit(X_train, y_train)
            
            # Predictions and evaluation
            y_pred = self.clv_model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(self.clv_model, X, y, cv=5, scoring='r2')
            
            # Feature importance
            feature_importance = dict(zip(features, self.clv_model.feature_importances_))
            
            # Add CLV predictions to dataframe
            df['Predicted_CLV'] = self.clv_model.predict(X)
            
            # Store metrics
            clv_metrics = {
                'rmse': float(rmse),
                'r2_score': float(r2),
                'cv_mean_r2': float(cv_scores.mean()),
                'cv_std_r2': float(cv_scores.std()),
                'feature_importance': {k: float(v) for k, v in feature_importance.items()}
            }
            
            logger.info(f"CLV Model - R² Score: {r2:.3f}, RMSE: {rmse:.2f}, CV R² Mean: {cv_scores.mean():.3f}")
            return clv_metrics
            
        except Exception as e:
            logger.error(f"Error training CLV model: {str(e)}")
            raise
    
    def train_churn_model(self, df: pd.DataFrame, churn_threshold_days: int = 30) -> Dict[str, Any]:
        """
        Train churn prediction model
        Identifies customers at risk of churning
        """
        try:
            logger.info("Training churn prediction model...")
            
            # Define churn (customers who haven't purchased in X days)
            df['Is_Churned'] = (df['Recency'] > churn_threshold_days).astype(int)
            
            # Check if we have both classes
            unique_classes = df['Is_Churned'].nunique()
            if unique_classes < 2:
                logger.warning(f"Only {unique_classes} churn class found. Adjusting threshold...")
                # Try different thresholds to get class balance
                for threshold in [15, 45, 60, 120]:
                    df['Is_Churned'] = (df['Recency'] > threshold).astype(int)
                    if df['Is_Churned'].nunique() >= 2:
                        churn_threshold_days = threshold
                        logger.info(f"Using churn threshold: {threshold} days")
                        break
                else:
                    # If still only one class, create synthetic churn based on quantiles
                    logger.warning("Creating synthetic churn labels based on customer behavior")
                    # Customers in bottom 20% of frequency and monetary are "churned"
                    freq_threshold = df['Frequency'].quantile(0.2)
                    monetary_threshold = df['Monetary'].quantile(0.2)
                    df['Is_Churned'] = ((df['Frequency'] <= freq_threshold) & 
                                      (df['Monetary'] <= monetary_threshold)).astype(int)
            
            # Prepare features
            features = ['Frequency', 'Monetary', 'Avg_Order_Value', 'Purchase_Frequency',
                       'Product_Variety', 'Tenure_Days', 'Transaction_Count']
            
            X = df[features].copy()
            y = df['Is_Churned']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Final check for class balance
            if y.nunique() < 2:
                logger.error("Unable to create balanced churn classes. Skipping churn model.")
                return {'error': 'Insufficient class variation for churn prediction'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train Gradient Boosting model
            self.churn_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.churn_model.fit(X_train, y_train)
            
            # Predictions and evaluation
            y_pred = self.churn_model.predict(X_test)
            y_pred_proba = self.churn_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # Cross-validation
            cv_scores = cross_val_score(self.churn_model, X, y, cv=5, scoring='roc_auc')
            
            # Feature importance
            feature_importance = dict(zip(features, self.churn_model.feature_importances_))
            
            # Add churn predictions to dataframe
            df['Churn_Probability'] = self.churn_model.predict_proba(X)[:, 1]
            df['Churn_Risk'] = pd.cut(df['Churn_Probability'], 
                                    bins=[0, 0.3, 0.7, 1.0], 
                                    labels=['Low', 'Medium', 'High'])
            
            # Store metrics
            churn_metrics = {
                'auc_score': float(auc_score),
                'cv_mean_auc': float(cv_scores.mean()),
                'cv_std_auc': float(cv_scores.std()),
                'classification_report': classification_rep,
                'feature_importance': {k: float(v) for k, v in feature_importance.items()},
                'churn_threshold_days': churn_threshold_days
            }
            
            logger.info(f"Churn Model - AUC Score: {auc_score:.3f}, CV AUC Mean: {cv_scores.mean():.3f}")
            return churn_metrics
            
        except Exception as e:
            logger.error(f"Error training churn model: {str(e)}")
            raise
    
    def generate_business_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate actionable business insights from segmentation
        """
        try:
            logger.info("Generating business insights...")
            
            insights = {}
            
            # Overall customer metrics
            insights['customer_overview'] = {
                'total_customers': int(df['Customer_ID'].nunique()),
                'total_revenue': float(df['Monetary'].sum()),
                'avg_clv': float(df['Monetary'].mean()),
                'avg_recency': float(df['Recency'].mean()),
                'avg_frequency': float(df['Frequency'].mean())
            }
            
            # RFM Segment Analysis
            agg_dict = {
                'Customer_ID': 'count',
                'Monetary': ['sum', 'mean'],
                'Frequency': 'mean',
                'Recency': 'mean'
            }
            
            # Add churn probability if available
            if 'Churn_Probability' in df.columns:
                agg_dict['Churn_Probability'] = 'mean'
            
            rfm_analysis = df.groupby('RFM_Segment').agg(agg_dict).round(2)
            
            # Set column names based on available data
            col_names = ['Customer_Count', 'Total_Revenue', 'Avg_CLV', 'Avg_Frequency', 'Avg_Recency']
            if 'Churn_Probability' in df.columns:
                col_names.append('Avg_Churn_Risk')
            
            rfm_analysis.columns = col_names
            
            insights['rfm_segments'] = rfm_analysis.to_dict('index')
            
            # ML Cluster Analysis
            if 'ML_Cluster' in df.columns:
                cluster_agg_dict = {
                    'Customer_ID': 'count',
                    'Monetary': ['sum', 'mean'],
                    'Predicted_CLV': 'mean'
                }
                
                if 'Churn_Probability' in df.columns:
                    cluster_agg_dict['Churn_Probability'] = 'mean'
                
                cluster_analysis = df.groupby('ML_Cluster').agg(cluster_agg_dict).round(2)
                
                cluster_col_names = ['Customer_Count', 'Total_Revenue', 'Avg_CLV', 'Predicted_CLV']
                if 'Churn_Probability' in df.columns:
                    cluster_col_names.append('Avg_Churn_Risk')
                
                cluster_analysis.columns = cluster_col_names
                
                insights['ml_clusters'] = cluster_analysis.to_dict('index')
            
            # High-value customer identification
            high_value_threshold = df['Monetary'].quantile(0.8)
            high_value_customers = df[df['Monetary'] >= high_value_threshold]
            
            high_value_insight = {
                'count': len(high_value_customers),
                'percentage': len(high_value_customers) / len(df) * 100,
                'revenue_contribution': high_value_customers['Monetary'].sum() / df['Monetary'].sum() * 100,
                'avg_clv': float(high_value_customers['Monetary'].mean())
            }
            
            if 'Churn_Probability' in df.columns:
                high_value_insight['avg_churn_risk'] = float(high_value_customers['Churn_Probability'].mean())
            
            insights['high_value_customers'] = high_value_insight
            
            # Churn risk analysis (only if churn prediction is available)
            if 'Churn_Risk' in df.columns:
                churn_analysis = df.groupby('Churn_Risk').agg({
                    'Customer_ID': 'count',
                    'Monetary': 'sum',
                    'Predicted_CLV': 'mean'
                }).round(2)
                
                insights['churn_risk_analysis'] = churn_analysis.to_dict('index')
            
            # Actionable recommendations
            insights['recommendations'] = self._generate_recommendations(df)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating business insights: {str(e)}")
            raise
    
    def _generate_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate actionable business recommendations"""
        try:
            recommendations = {}
            
            # High churn risk customers
            high_churn = df[df['Churn_Risk'] == 'High']
            if len(high_churn) > 0:
                recommendations['churn_prevention'] = {
                    'customers_at_risk': len(high_churn),
                    'revenue_at_risk': float(high_churn['Predicted_CLV'].sum()),
                    'action': 'Implement targeted retention campaigns for high-value, high-churn-risk customers'
                }
            
            # Low frequency, high value customers
            low_freq_high_value = df[(df['Frequency'] <= df['Frequency'].quantile(0.3)) & 
                                   (df['Monetary'] >= df['Monetary'].quantile(0.7))]
            
            if len(low_freq_high_value) > 0:
                recommendations['engagement_opportunity'] = {
                    'customer_count': len(low_freq_high_value),
                    'potential_revenue': float(low_freq_high_value['Predicted_CLV'].sum()),
                    'action': 'Increase purchase frequency through personalized recommendations and loyalty programs'
                }
            
            # Champions segment optimization
            champions = df[df['RFM_Segment'] == 'Champions']
            if len(champions) > 0:
                recommendations['vip_program'] = {
                    'customer_count': len(champions),
                    'revenue_contribution': float(champions['Monetary'].sum()),
                    'action': 'Create VIP program with exclusive benefits and early access to new products'
                }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {}
    
    def save_models(self, model_version: str = None):
        """Save trained models and results"""
        try:
            if model_version is None:
                model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_dir = self.models_path / f"customer_segmentation_{model_version}"
            model_dir.mkdir(exist_ok=True)
            
            # Save models
            if self.kmeans_model:
                with open(model_dir / "kmeans_model.pkl", "wb") as f:
                    pickle.dump(self.kmeans_model, f)
            
            if self.clv_model:
                with open(model_dir / "clv_model.pkl", "wb") as f:
                    pickle.dump(self.clv_model, f)
            
            if self.churn_model:
                with open(model_dir / "churn_model.pkl", "wb") as f:
                    pickle.dump(self.churn_model, f)
            
            # Save scaler
            with open(model_dir / "rfm_scaler.pkl", "wb") as f:
                pickle.dump(self.rfm_scaler, f)
            
            # Save metrics
            with open(model_dir / "model_metrics.json", "w") as f:
                json.dump(self.model_metrics, f, indent=2)
            
            # Save customer segments
            if self.customer_segments is not None:
                self.customer_segments.to_csv(model_dir / "customer_segments.csv", index=False)
            
            logger.info(f"Models saved to: {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete customer segmentation analysis
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING CUSTOMER SEGMENTATION ML ANALYSIS")
            logger.info("=" * 60)
            
            # Load data
            df = self.load_data()
            
            # Calculate RFM features
            rfm_df = self.calculate_rfm_features(df)
            
            # Create RFM scores
            rfm_scored = self.create_rfm_scores(rfm_df)
            
            # Perform ML clustering
            clustered_df = self.perform_clustering(rfm_scored)
            
            # Train predictive models
            clv_metrics = self.train_clv_model(clustered_df)
            
            # Try churn model (may fail with small datasets)
            try:
                churn_metrics = self.train_churn_model(clustered_df)
                self.model_metrics['churn_model'] = churn_metrics
            except Exception as e:
                logger.warning(f"Churn model training failed: {str(e)}")
                logger.warning("Continuing analysis without churn prediction...")
                churn_metrics = {'error': str(e)}
                self.model_metrics['churn_model'] = churn_metrics
            
            # Store metrics
            self.model_metrics['clv_model'] = clv_metrics
            
            # Generate business insights
            business_insights = self.generate_business_insights(clustered_df)
            self.model_metrics['business_insights'] = business_insights
            
            # Store results
            self.customer_segments = clustered_df
            
            # Save everything
            self.save_models()
            
            # Save customer segments to processed data
            output_file = self.data_path / "customer_segments.csv"
            clustered_df.to_csv(output_file, index=False)
            logger.info(f"Customer segments saved to: {output_file}")
            
            logger.info("=" * 60)
            logger.info("CUSTOMER SEGMENTATION ANALYSIS COMPLETED")
            logger.info("=" * 60)
            
            # Print summary
            self._print_analysis_summary()
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            raise
    
    def _print_analysis_summary(self):
        """Print analysis summary for demo purposes"""
        try:
            if self.customer_segments is None:
                return
            
            print("\n" + "="*50)
            print("📊 CUSTOMER SEGMENTATION ANALYSIS SUMMARY")
            print("="*50)
            
            # Overall metrics
            total_customers = len(self.customer_segments)
            total_revenue = self.customer_segments['Monetary'].sum()
            
            print(f"📈 Total Customers Analyzed: {total_customers:,}")
            print(f"💰 Total Revenue: £{total_revenue:,.2f}")
            
            # RFM Segments
            print(f"\n🎯 RFM SEGMENTS:")
            rfm_summary = self.customer_segments['RFM_Segment'].value_counts()
            for segment, count in rfm_summary.items():
                pct = count / total_customers * 100
                print(f"   {segment}: {count:,} customers ({pct:.1f}%)")
            
            # ML Clusters
            if 'ML_Cluster' in self.customer_segments.columns:
                print(f"\n🤖 ML CLUSTERS:")
                cluster_summary = self.customer_segments['ML_Cluster'].value_counts().sort_index()
                for cluster, count in cluster_summary.items():
                    pct = count / total_customers * 100
                    avg_clv = self.customer_segments[self.customer_segments['ML_Cluster'] == cluster]['Monetary'].mean()
                    print(f"   Cluster {cluster}: {count:,} customers ({pct:.1f}%) - Avg CLV: £{avg_clv:.2f}")
            
            # Model Performance
            if 'clv_model' in self.model_metrics:
                clv_r2 = self.model_metrics['clv_model']['r2_score']
                print(f"\n📊 MODEL PERFORMANCE:")
                print(f"   CLV Prediction R² Score: {clv_r2:.3f}")
            
            if 'churn_model' in self.model_metrics:
                churn_auc = self.model_metrics['churn_model']['auc_score']
                print(f"   Churn Prediction AUC Score: {churn_auc:.3f}")
            
            print("\n✅ Analysis complete! Check data/processed/customer_segments.csv for results")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}")


def main():
    """Main function to run customer segmentation analysis"""
    try:
        # Initialize and run analysis
        segmentation_model = CustomerSegmentationModel()
        results = segmentation_model.run_complete_analysis()
        
        print("\n🎉 Customer Segmentation Analysis Complete!")
        print("📁 Results saved to data/processed/customer_segments.csv")
        print("📊 Model metrics and insights generated")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()