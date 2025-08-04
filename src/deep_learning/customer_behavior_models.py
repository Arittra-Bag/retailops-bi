import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
#import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorFlow imports with improved error handling
TENSORFLOW_AVAILABLE = False
TENSORFLOW_ERROR = None
try:
    # Set environment variables to help with Windows DLL issues
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    import tensorflow as tf
    #from tensorflow.keras.models import Sequential, Model
    #from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, LSTM, Input
    #from tensorflow.keras.optimizers import Adam
    #from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    #from tensorflow.keras.utils import to_categorical
    
    # Test TensorFlow functionality immediately
    test_tensor = tf.constant([1.0, 2.0, 3.0])
    TENSORFLOW_AVAILABLE = True
    logger.info("✅ TensorFlow loaded successfully")
    
except ImportError as e:
    TENSORFLOW_ERROR = f"TensorFlow import failed: {str(e)}"
    logger.warning(f"⚠️ {TENSORFLOW_ERROR}")
except Exception as e:
    TENSORFLOW_ERROR = f"TensorFlow runtime error: {str(e)}"
    logger.warning(f"⚠️ {TENSORFLOW_ERROR}")

# PyTorch imports
PYTORCH_AVAILABLE = False
PYTORCH_ERROR = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    logger.info("✅ PyTorch loaded successfully")
except ImportError as e:
    PYTORCH_ERROR = f"PyTorch not available: {str(e)}"
    logger.warning(f"⚠️ {PYTORCH_ERROR}")

# Scikit-learn imports (always available fallback)
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
    logger.info("✅ Scikit-learn available as fallback")
except ImportError:
    SKLEARN_AVAILABLE = False

class CustomerBehaviorDataset(Dataset):
    """PyTorch Dataset for customer behavior data"""
    
    def __init__(self, features, targets):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CustomerChurnNet(nn.Module):
    """PyTorch Neural Network for Customer Churn Prediction"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        super(CustomerChurnNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CustomerEmbeddingNet(nn.Module):
    """PyTorch Neural Network for Customer Embeddings"""
    
    def __init__(self, num_customers, num_products, embedding_dim=50):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        super(CustomerEmbeddingNet, self).__init__()
        
        self.customer_embedding = nn.Embedding(num_customers, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, customer_ids, product_ids):
        customer_emb = self.customer_embedding(customer_ids)
        product_emb = self.product_embedding(product_ids)
        
        combined = torch.cat([customer_emb, product_emb], dim=1)
        output = self.fc_layers(combined)
        
        return output

class DeepLearningCustomerModels:
    """Deep Learning models for customer behavior analysis with robust error handling"""
    
    def __init__(self):
        """Initialize deep learning models"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data" / "processed"
        self.models_path = self.project_root / "data" / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Model storage
        self.tf_models = {}
        self.pytorch_models = {}
        self.sklearn_models = {}
        self.scalers = {}
        self.encoders = {}
        
        logger.info("✅ Deep Learning Customer Models initialized")
        #logger.info(f"📊 TensorFlow Available: {TENSORFLOW_AVAILABLE}")
        logger.info(f"🔥 PyTorch Available: {PYTORCH_AVAILABLE}")
        logger.info(f"🤖 Scikit-learn Available: {SKLEARN_AVAILABLE}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for deep learning models"""
        try:
            # Load customer and transaction data
            customer_file = self.data_path / "customer_analysis.csv"
            transaction_file = self.data_path / "retail_transactions_processed.csv"
            
            if not customer_file.exists() or not transaction_file.exists():
                raise FileNotFoundError("Required data files not found")
            
            customers_df = pd.read_csv(customer_file)
            transactions_df = pd.read_csv(transaction_file)
            
            logger.info(f"📊 Loaded {len(customers_df)} customers, {len(transactions_df)} transactions")
            
            return customers_df, transactions_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    '''
    def create_tensorflow_churn_model(self, input_dim: int):
        """Create TensorFlow model for churn prediction"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(f"TensorFlow not available: {TENSORFLOW_ERROR}")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_sklearn_churn_model(self) -> Dict[str, Any]:
        """Train scikit-learn churn prediction model as fallback"""
        try:
            if not SKLEARN_AVAILABLE:
                return {
                    "status": "error",
                    "error": "Scikit-learn not available",
                    "suggestion": "Install scikit-learn: pip install scikit-learn"
                }
            logger.info("🤖 Training Scikit-learn churn prediction model (fallback)...")
            customers_df, transactions_df = self.load_and_prepare_data()
            # Prepare features
            feature_columns = ['Total_Spent', 'Transaction_Count', 'Avg_Order_Value']
            missing = [col for col in feature_columns if col not in customers_df.columns]
            if missing:
                return {
                    "status": "error",
                    "error": f"Required feature columns not found: {missing}. Please ensure your ETL pipeline creates these columns in customer_analysis.csv.",
                    "suggestion": "Check your data processing scripts or re-run ETL."
                }
            
            # Create churn labels
            transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
            last_purchase = transactions_df.groupby('Customer_ID')['Date'].max()
            
            max_date = transactions_df['Date'].max()
            days_since_last = (max_date - last_purchase).dt.days
            
            customers_df = customers_df.merge(
                days_since_last.rename('Days_Since_Last'), 
                left_on='Customer_ID', 
                right_index=True, 
                how='left'
            )
            
            churn_threshold = customers_df['Days_Since_Last'].quantile(0.7)
            customers_df['Is_Churned'] = (customers_df['Days_Since_Last'] > churn_threshold).astype(int)
            
            # Prepare features and targets
            X = customers_df[feature_columns].fillna(0)
            y = customers_df['Is_Churned']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create and train model
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Store model and scaler
            self.sklearn_models['churn_prediction'] = model
            self.scalers['churn_features'] = scaler
            
            # Save model
            model_path = self.models_path / "sklearn_churn_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info("✅ Scikit-learn churn model training complete")
            
            return {
                "status": "success",
                "model_type": "Scikit-learn MLPClassifier",
                "framework": "Scikit-learn",
                "architecture": "Multi-layer Perceptron (128, 64, 32)",
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "performance": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                },
                "feature_columns": feature_columns,
                "churn_threshold_days": float(churn_threshold),
                "model_saved": str(model_path),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training Scikit-learn churn model: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def train_tensorflow_churn_model(self) -> Dict[str, Any]:
        """Train TensorFlow churn prediction model"""
        if not TENSORFLOW_AVAILABLE:
            logger.info("🔄 TensorFlow not available, falling back to Scikit-learn...")
            fallback_result = self.train_sklearn_churn_model()
            if fallback_result.get("status") == "success":
                fallback_result["fallback_used"] = "Scikit-learn MLPClassifier"
                fallback_result["original_framework_error"] = TENSORFLOW_ERROR
            return fallback_result
        try:
            logger.info("🧠 Training TensorFlow churn prediction model...")
            customers_df, transactions_df = self.load_and_prepare_data()
            # Prepare features
            feature_columns = ['Total_Spent', 'Transaction_Count', 'Avg_Order_Value']
            missing = [col for col in feature_columns if col not in customers_df.columns]
            if missing:
                return {
                    "status": "error",
                    "error": f"Required feature columns not found: {missing}. Please ensure your ETL pipeline creates these columns in customer_analysis.csv.",
                    "suggestion": "Check your data processing scripts or re-run ETL."
                }
            
            # Create churn labels
            transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
            last_purchase = transactions_df.groupby('Customer_ID')['Date'].max()
            
            max_date = transactions_df['Date'].max()
            days_since_last = (max_date - last_purchase).dt.days
            
            customers_df = customers_df.merge(
                days_since_last.rename('Days_Since_Last'), 
                left_on='Customer_ID', 
                right_index=True, 
                how='left'
            )
            
            churn_threshold = customers_df['Days_Since_Last'].quantile(0.7)
            customers_df['Is_Churned'] = (customers_df['Days_Since_Last'] > churn_threshold).astype(int)
            
            # Prepare features and targets
            X = customers_df[feature_columns].fillna(0)
            y = customers_df['Is_Churned']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create and train model
            model = self.create_tensorflow_churn_model(X_train.shape[1])
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
            
            # Calculate F1 score
            f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-8)
            
            # Store model and scaler
            self.tf_models['churn_prediction'] = model
            self.scalers['churn_features'] = scaler
            
            # Save model
            model_path = self.models_path / "tf_churn_model.h5"
            model.save(model_path)
            
            logger.info("✅ TensorFlow churn model training complete")
            
            return {
                "status": "success",
                "model_type": "TensorFlow Neural Network",
                "framework": "TensorFlow/Keras",
                "architecture": "Dense layers with BatchNorm and Dropout",
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "performance": {
                    "accuracy": float(test_accuracy),
                    "precision": float(test_precision),
                    "recall": float(test_recall),
                    "f1_score": float(f1_score),
                    "loss": float(test_loss)
                },
                "feature_columns": feature_columns,
                "churn_threshold_days": float(churn_threshold),
                "training_epochs": len(history.history['loss']),
                "model_saved": str(model_path),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training TensorFlow churn model: {str(e)}")
            # Fall back to scikit-learn
            logger.info("🔄 TensorFlow training failed, falling back to Scikit-learn...")
            fallback_result = self.train_sklearn_churn_model()
            if fallback_result.get("status") == "success":
                fallback_result["fallback_used"] = "Scikit-learn MLPClassifier"
                fallback_result["tensorflow_error"] = str(e)
            return fallback_result
    '''
    def train_pytorch_customer_embedding(self) -> Dict[str, Any]:
        """Train PyTorch customer embedding model"""
        if not PYTORCH_AVAILABLE:
            return {
                "status": "error", 
                "error": f"PyTorch not available: {PYTORCH_ERROR}",
                "suggestion": "Install PyTorch: pip install torch torchvision",
                "fallback_suggestion": "Use scikit-learn matrix factorization instead"
            }
        
        try:
            logger.info("🔥 Training PyTorch customer embedding model...")
            
            customers_df, transactions_df = self.load_and_prepare_data()
            
            # Prepare data for embedding learning
            customer_encoder = LabelEncoder()
            product_encoder = LabelEncoder()
            
            transactions_df['Customer_Index'] = customer_encoder.fit_transform(transactions_df['Customer_ID'])
            transactions_df['Product_Index'] = product_encoder.fit_transform(transactions_df['StockCode'])
            
            # Target: Revenue
            X_customers = transactions_df['Customer_Index'].values
            X_products = transactions_df['Product_Index'].values
            y = transactions_df['Revenue'].values
            
            # Normalize target
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Train/test split
            indices = np.arange(len(X_customers))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            # Create datasets
            train_dataset = CustomerBehaviorDataset(
                np.column_stack([X_customers[train_idx], X_products[train_idx]]),
                y_scaled[train_idx]
            )
            test_dataset = CustomerBehaviorDataset(
                np.column_stack([X_customers[test_idx], X_products[test_idx]]),
                y_scaled[test_idx]
            )
            
            # Data loaders
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            # Create model
            num_customers = len(customer_encoder.classes_)
            num_products = len(product_encoder.classes_)
            
            model = CustomerEmbeddingNet(num_customers, num_products, embedding_dim=50)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            num_epochs = 50
            train_losses = []
            
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for batch_features, batch_targets in train_loader:
                    customer_ids = batch_features[:, 0].long()
                    product_ids = batch_features[:, 1].long()
                    
                    optimizer.zero_grad()
                    outputs = model(customer_ids, product_ids).squeeze()
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Evaluate model
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    customer_ids = batch_features[:, 0].long()
                    product_ids = batch_features[:, 1].long()
                    outputs = model(customer_ids, product_ids).squeeze()
                    loss = criterion(outputs, batch_targets)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            
            # Store model and encoders
            self.pytorch_models['customer_embedding'] = model
            self.encoders['customer_encoder'] = customer_encoder
            self.encoders['product_encoder'] = product_encoder
            self.scalers['revenue_scaler'] = scaler
            
            # Save model
            model_path = self.models_path / "pytorch_customer_embedding.pth"
            torch.save(model.state_dict(), model_path)
            
            # Generate practical business insights using embeddings
            logger.info("🔍 Generating customer similarity and product recommendations...")
            business_insights = self._generate_embedding_insights(
                model, customer_encoder, product_encoder, customers_df, transactions_df
            )
            
            logger.info("✅ PyTorch customer embedding training complete")
            
            return {
                "status": "success",
                "model_type": "PyTorch Customer Embedding Network",
                "framework": "PyTorch",
                "architecture": "Embedding layers + Dense network",
                "training_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "performance": {
                    "final_train_loss": float(train_losses[-1]),
                    "test_loss": float(avg_test_loss),
                    "embedding_dimension": 50
                },
                "embeddings": {
                    "num_customers": num_customers,
                    "num_products": num_products,
                    "customer_embedding_shape": [num_customers, 50],
                    "product_embedding_shape": [num_products, 50]
                },
                "business_insights": business_insights,
                "training_epochs": num_epochs,
                "model_saved": str(model_path),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training PyTorch embedding model: {str(e)}")
            return {"status": "error", "error": str(e)}
    '''
    def train_lstm_customer_sequence(self) -> Dict[str, Any]:
        """Train LSTM for customer purchase sequence prediction"""
        if not TENSORFLOW_AVAILABLE:
            return {
                "status": "error", 
                "error": f"TensorFlow not available: {TENSORFLOW_ERROR}",
                "suggestion": "Install TensorFlow: pip install tensorflow",
                "fallback_suggestion": "Use ARIMA or Prophet for time series modeling"
            }
        
        try:
            logger.info("🔄 Training LSTM customer sequence model...")
            
            customers_df, transactions_df = self.load_and_prepare_data()
            
            # Sort by customer and date
            transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
            transactions_df = transactions_df.sort_values(['Customer_ID', 'Date'])
            
            # Create sequences for top customers
            customer_counts = transactions_df['Customer_ID'].value_counts()
            top_customers = customer_counts[customer_counts >= 10].index[:100]
            
            sequences = []
            targets = []
            
            for customer_id in top_customers:
                customer_data = transactions_df[transactions_df['Customer_ID'] == customer_id]
                revenue_sequence = customer_data['Revenue'].values
                
                seq_length = 5
                for i in range(len(revenue_sequence) - seq_length):
                    sequences.append(revenue_sequence[i:i+seq_length])
                    targets.append(revenue_sequence[i+seq_length])
            
            if len(sequences) < 50:
                return {"status": "error", "error": "Insufficient sequence data for LSTM training"}
            
            # Convert to arrays
            X = np.array(sequences).reshape(-1, seq_length, 1)
            y = np.array(targets)
            
            # Normalize data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X.reshape(-1, seq_length)).reshape(-1, seq_length, 1)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            # Create LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1)
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5)
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            
            # Store model
            self.tf_models['customer_sequence_lstm'] = model
            self.scalers['sequence_X'] = scaler_X
            self.scalers['sequence_y'] = scaler_y
            
            # Save model
            model_path = self.models_path / "lstm_customer_sequence.h5"
            model.save(model_path)
            
            logger.info("✅ LSTM customer sequence training complete")
            
            return {
                "status": "success",
                "model_type": "LSTM Customer Purchase Sequence",
                "framework": "TensorFlow/Keras",
                "architecture": "LSTM layers with Dropout",
                "sequence_length": seq_length,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "performance": {
                    "test_loss": float(test_loss),
                    "test_mae": float(test_mae)
                },
                "customers_analyzed": len(top_customers),
                "total_sequences": len(sequences),
                "training_epochs": len(history.history['loss']),
                "model_saved": str(model_path),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return {"status": "error", "error": str(e)}
    '''
    
    def _generate_embedding_insights(self, model, customer_encoder, product_encoder, 
                                   customers_df, transactions_df) -> Dict[str, Any]:
        """Generate practical business insights from trained embeddings"""
        try:
            if not PYTORCH_AVAILABLE:
                return {"error": "PyTorch not available for insights generation"}
                
            model.eval()
            insights = {}
            
            # Extract embeddings
            with torch.no_grad():
                # Get all customer embeddings
                customer_ids = torch.arange(len(customer_encoder.classes_))
                customer_embeddings = model.customer_embedding(customer_ids).numpy()
                
                # Get all product embeddings  
                product_ids = torch.arange(len(product_encoder.classes_))
                product_embeddings = model.product_embedding(product_ids).numpy()
            
            # 1. Customer Similarity Analysis
            logger.info("🔍 Analyzing customer similarities...")
            
            # Calculate customer similarity matrix (using first 10 customers for demo)
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Focus on top customers by transaction count
            top_customers = transactions_df['Customer_ID'].value_counts().head(10)
            customer_similarities = []
            
            for customer_id in top_customers.index[:5]:  # Top 5 customers
                try:
                    customer_idx = customer_encoder.transform([customer_id])[0]
                    customer_emb = customer_embeddings[customer_idx:customer_idx+1]
                    
                    # Find most similar customers
                    similarities = cosine_similarity(customer_emb, customer_embeddings)[0]
                    top_similar_indices = np.argsort(similarities)[-6:-1][::-1]  # Exclude self
                    
                    similar_customers = []
                    for idx in top_similar_indices:
                        similar_id = customer_encoder.inverse_transform([idx])[0]
                        similar_customers.append({
                            "customer_id": str(similar_id),
                            "similarity_score": float(similarities[idx]),
                            "transactions": int(transactions_df[transactions_df['Customer_ID'] == similar_id].shape[0])
                        })
                    
                    customer_similarities.append({
                        "customer_id": str(customer_id),
                        "transactions": int(transactions_df[transactions_df['Customer_ID'] == customer_id].shape[0]),
                        "similar_customers": similar_customers
                    })
                except Exception as e:
                    logger.warning(f"Skipping customer {customer_id}: {str(e)}")
                    continue
            
            insights["customer_similarity"] = customer_similarities
            
            # 2. Product Recommendations
            logger.info("🛍️ Generating product recommendations...")
            
            # Find products frequently bought together
            product_recommendations = []
            top_products = transactions_df['StockCode'].value_counts().head(10)
            
            for product_code in top_products.index[:5]:  # Top 5 products
                try:
                    product_idx = product_encoder.transform([product_code])[0]
                    product_emb = product_embeddings[product_idx:product_idx+1]
                    
                    # Find most similar products
                    similarities = cosine_similarity(product_emb, product_embeddings)[0]
                    top_similar_indices = np.argsort(similarities)[-6:-1][::-1]  # Exclude self
                    
                    similar_products = []
                    for idx in top_similar_indices:
                        similar_code = product_encoder.inverse_transform([idx])[0]
                        # Get product description
                        product_desc = transactions_df[transactions_df['StockCode'] == similar_code]['Description'].iloc[0] if len(transactions_df[transactions_df['StockCode'] == similar_code]) > 0 else "Unknown"
                        
                        similar_products.append({
                            "product_code": str(similar_code),
                            "description": str(product_desc)[:50] + "..." if len(str(product_desc)) > 50 else str(product_desc),
                            "similarity_score": float(similarities[idx])
                        })
                    
                    # Get original product description
                    original_desc = transactions_df[transactions_df['StockCode'] == product_code]['Description'].iloc[0] if len(transactions_df[transactions_df['StockCode'] == product_code]) > 0 else "Unknown"
                    
                    product_recommendations.append({
                        "product_code": str(product_code),
                        "description": str(original_desc)[:50] + "..." if len(str(original_desc)) > 50 else str(original_desc),
                        "recommended_products": similar_products
                    })
                except Exception as e:
                    logger.warning(f"Skipping product {product_code}: {str(e)}")
                    continue
                    
            insights["product_recommendations"] = product_recommendations
            
            # 3. Business Metrics
            insights["business_metrics"] = {
                "total_customers_analyzed": len(customer_encoder.classes_),
                "total_products_analyzed": len(product_encoder.classes_),
                "similarity_threshold": 0.5,
                "recommendation_confidence": "High" if len(product_recommendations) > 3 else "Medium"
            }
            
            logger.info(f"✅ Generated insights for {len(customer_similarities)} customers and {len(product_recommendations)} products")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating embedding insights: {str(e)}")
            return {
                "error": str(e),
                "fallback_insights": {
                    "message": "Unable to generate detailed insights, but embeddings were trained successfully",
                    "suggestion": "Try with larger dataset for better similarity analysis"
                }
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        try:
            summary = {
                "tensorflow_models": {
                    name: {
                        "type": type(model).__name__,
                        "parameters": model.count_params() if hasattr(model, 'count_params') else "Unknown"
                    }
                    for name, model in self.tf_models.items()
                },
                "pytorch_models": {
                    name: {
                        "type": type(model).__name__,
                        "parameters": sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else "Unknown"
                    }
                    for name, model in self.pytorch_models.items()
                },
                "sklearn_models": {
                    name: {
                        "type": type(model).__name__,
                        "parameters": getattr(model, 'n_iter_', 'Unknown')
                    }
                    for name, model in self.sklearn_models.items()
                },
                "scalers": list(self.scalers.keys()),
                "encoders": list(self.encoders.keys()),
                "frameworks": {
                    "tensorflow_available": TENSORFLOW_AVAILABLE,
                    "tensorflow_error": TENSORFLOW_ERROR if not TENSORFLOW_AVAILABLE else None,
                    "pytorch_available": PYTORCH_AVAILABLE,
                    "pytorch_error": PYTORCH_ERROR if not PYTORCH_AVAILABLE else None,
                    "sklearn_available": SKLEARN_AVAILABLE
                },
                "last_error": getattr(self, 'last_error', None),
                "timestamp": datetime.now().isoformat()
            }
            return summary
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return {"error": str(e)}


def main():
    """Main function to test deep learning models with robust error handling"""
    try:
        # Initialize deep learning models
        dl_models = DeepLearningCustomerModels()
        
        print("🧠 Testing Deep Learning Customer Models (FIXED VERSION)")
        print("=" * 60)
        
        # Framework Status
        print(f"\n📊 Framework Status:")
        #print(f"   TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Unavailable'}")
        #if not TENSORFLOW_AVAILABLE:
        #    print(f"   Error: {TENSORFLOW_ERROR}")
        print(f"   PyTorch: {'✅ Available' if PYTORCH_AVAILABLE else '❌ Unavailable'}")
        if not PYTORCH_AVAILABLE:
            print(f"   Error: {PYTORCH_ERROR}")
        #print(f"   Scikit-learn: {'✅ Available' if SKLEARN_AVAILABLE else '❌ Unavailable'}")
        '''
        # Test 1: TensorFlow/Scikit-learn Churn Model
        print("\n1️⃣ Testing Churn Prediction Model...")
        churn_result = dl_models.train_tensorflow_churn_model()
        
        if churn_result.get("status") == "success":
            print(f"✅ Churn model complete:")
            print(f"   🎯 Framework: {churn_result.get('framework', 'Unknown')}")
            print(f"   📊 Accuracy: {churn_result['performance']['accuracy']:.3f}")
            print(f"   🔄 F1 Score: {churn_result['performance']['f1_score']:.3f}")
            if churn_result.get('fallback_used'):
                print(f"   ⚠️ Fallback: {churn_result['fallback_used']}")
        else:
            print(f"❌ Churn model failed: {churn_result.get('error')}")
        '''
        # Test 2: PyTorch Embedding Model
        print("\n2️⃣ Testing PyTorch Customer Embeddings...")
        if PYTORCH_AVAILABLE:
            pytorch_result = dl_models.train_pytorch_customer_embedding()
            
            if pytorch_result.get("status") == "success":
                print(f"✅ PyTorch model complete:")
                print(f"   🔢 Customers: {pytorch_result['embeddings']['num_customers']}")
                print(f"   📦 Products: {pytorch_result['embeddings']['num_products']}")
                print(f"   🏗️ Framework: {pytorch_result['framework']}")
            else:
                print(f"❌ PyTorch model failed: {pytorch_result.get('error')}")
        else:
            print("⚠️ PyTorch not available - skipping embedding model")
        '''
        # Test 3: LSTM Sequence Model
        print("\n3️⃣ Testing LSTM Sequence Prediction...")
        if TENSORFLOW_AVAILABLE:
            lstm_result = dl_models.train_lstm_customer_sequence()
            
            if lstm_result.get("status") == "success":
                print(f"✅ LSTM model complete:")
                print(f"   📈 Sequences: {lstm_result['total_sequences']}")
                print(f"   🎯 Test MAE: {lstm_result['performance']['test_mae']:.3f}")
                print(f"   🏗️ Architecture: {lstm_result['architecture']}")
            else:
                print(f"❌ LSTM model failed: {lstm_result.get('error')}")
        else:
            print("⚠️ TensorFlow not available - skipping LSTM model")
        '''
        # Test 4: Model Summary
        print("\n4️⃣ Model Summary...")
        summary = dl_models.get_model_summary()
        #print(f"✅ TensorFlow models: {len(summary.get('tensorflow_models', {}))}")
        print(f"✅ PyTorch models: {len(summary.get('pytorch_models', {}))}")
        #print(f"✅ Scikit-learn models: {len(summary.get('sklearn_models', {}))}")
        print(f"✅ Preprocessing tools: {len(summary.get('scalers', [])) + len(summary.get('encoders', []))}")
        
        print("\n" + "=" * 60)
        print("🎉 Deep Learning Models Test Complete!")
        '''
        # Windows-specific troubleshooting tips
        if not TENSORFLOW_AVAILABLE:
            print("\n🔧 TensorFlow Troubleshooting Tips:")
            print("   1. Install Microsoft Visual C++ Redistributable")
            print("   2. Try: pip uninstall tensorflow && pip install tensorflow-cpu")
            print("   3. Use Python 3.8-3.10 for better compatibility")
            print("   4. Fallback to Scikit-learn models works perfectly!")
        '''
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()