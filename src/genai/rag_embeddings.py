"""
RAG (Retrieval-Augmented Generation) Embeddings System
Uses Gemini API for embeddings and vector similarity search
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sqlite3
from dotenv import load_dotenv

# Google GenAI for embeddings
try:
    import google.generativeai as genai
    from google.generativeai import embed_content
    GENAI_AVAILABLE = True
except ImportError as e:
    print(f"Google GenAI import error: {e}")
    GENAI_AVAILABLE = False

# For vector similarity
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
logger = logging.getLogger(__name__)

class RAGEmbeddingsManager:
    """Manages RAG embeddings using Gemini API and vector search"""
    
    def __init__(self):
        """Initialize RAG embeddings manager"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data" / "processed"
        self.embeddings_db_path = self.project_root / "data" / "embeddings.db"
        
        # Initialize Google GenAI
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.embedding_model = "models/embedding-001"  # Gemini embedding model
        
        # Initialize embeddings database
        self._init_embeddings_db()
        
        logger.info("✅ RAG Embeddings Manager initialized")
    
    def _init_embeddings_db(self):
        """Initialize SQLite database for storing embeddings"""
        try:
            conn = sqlite3.connect(self.embeddings_db_path)
            cursor = conn.cursor()
            
            # Create embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_type TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chunk_type, chunk_id)
                )
            """)
            
            # Create index for faster searches
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_type 
                ON data_embeddings(chunk_type)
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Embeddings database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings database: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini API"""
        try:
            # Use Gemini embedding model
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []
    
    def chunk_dataframe(self, df: pd.DataFrame, chunk_type: str, chunk_size: int = 50) -> List[Dict[str, Any]]:
        """Convert DataFrame into chunks for embedding"""
        chunks = []
        
        if chunk_type == "transactions":
            # Group transactions by meaningful segments
            for i in range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i+chunk_size]
                
                # Create meaningful text representation
                revenue = chunk_df['Revenue'].sum()
                customers = chunk_df['Customer_ID'].nunique()
                products = chunk_df['StockCode'].nunique()
                countries = chunk_df['Country'].value_counts().head(3).to_dict()
                date_range = f"{chunk_df['Date'].min()} to {chunk_df['Date'].max()}"
                
                content = f"""
                Transaction Segment {i//chunk_size + 1}:
                - Date Range: {date_range}
                - Total Revenue: £{revenue:,.2f}
                - Unique Customers: {customers}
                - Unique Products: {products}
                - Top Countries: {countries}
                - Transaction Count: {len(chunk_df)}
                - Average Order Value: £{chunk_df['Total_Amount'].mean():.2f}
                """
                
                chunks.append({
                    'chunk_id': f"transactions_{i//chunk_size + 1}",
                    'content': content.strip(),
                    'metadata': {
                        'start_idx': i,
                        'end_idx': min(i + chunk_size, len(df)),
                        'revenue': revenue,
                        'date_range': date_range,
                        'customers': customers,
                        'products': products
                    }
                })
        
        elif chunk_type == "customer_analysis":
            # Group customers by segments
            for i in range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i+chunk_size]
                
                total_spent = chunk_df['Total_Spent'].sum()
                avg_spent = chunk_df['Total_Spent'].mean()
                countries = chunk_df['Country'].value_counts().head(3).to_dict()
                
                content = f"""
                Customer Segment {i//chunk_size + 1}:
                - Customer Count: {len(chunk_df)}
                - Total Spending: £{total_spent:,.2f}
                - Average Spending: £{avg_spent:.2f}
                - Countries: {countries}
                - Spending Range: £{chunk_df['Total_Spent'].min():.2f} - £{chunk_df['Total_Spent'].max():.2f}
                """
                
                chunks.append({
                    'chunk_id': f"customers_{i//chunk_size + 1}",
                    'content': content.strip(),
                    'metadata': {
                        'start_idx': i,
                        'end_idx': min(i + chunk_size, len(df)),
                        'total_spent': total_spent,
                        'customer_count': len(chunk_df)
                    }
                })
        
        elif chunk_type == "product_performance":
            # Group products by categories or performance tiers
            for i in range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i+chunk_size]
                
                total_revenue = chunk_df['Total_Revenue'].sum()
                categories = chunk_df['Product_Category'].value_counts().to_dict()
                
                content = f"""
                Product Performance Segment {i//chunk_size + 1}:
                - Product Count: {len(chunk_df)}
                - Total Revenue: £{total_revenue:,.2f}
                - Categories: {categories}
                - Top Product: {chunk_df.iloc[0]['StockCode']} (£{chunk_df.iloc[0]['Total_Revenue']:,.2f})
                - Revenue Range: £{chunk_df['Total_Revenue'].min():.2f} - £{chunk_df['Total_Revenue'].max():.2f}
                """
                
                chunks.append({
                    'chunk_id': f"products_{i//chunk_size + 1}",
                    'content': content.strip(),
                    'metadata': {
                        'start_idx': i,
                        'end_idx': min(i + chunk_size, len(df)),
                        'total_revenue': total_revenue,
                        'product_count': len(chunk_df)
                    }
                })
        
        return chunks
    
    def store_embeddings(self, chunk_type: str, chunks: List[Dict[str, Any]]):
        """Store chunks and their embeddings in database"""
        try:
            conn = sqlite3.connect(self.embeddings_db_path)
            cursor = conn.cursor()
            
            stored_count = 0
            for chunk in chunks:
                # Generate embedding
                embedding = self.generate_embedding(chunk['content'])
                if not embedding:
                    continue
                
                # Convert embedding to bytes for storage
                embedding_bytes = np.array(embedding).tobytes()
                
                # Store in database (replace if exists)
                cursor.execute("""
                    INSERT OR REPLACE INTO data_embeddings 
                    (chunk_type, chunk_id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk_type,
                    chunk['chunk_id'],
                    chunk['content'],
                    json.dumps(chunk['metadata']),
                    embedding_bytes
                ))
                stored_count += 1
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Stored {stored_count} embeddings for {chunk_type}")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return 0
    
    def search_similar_chunks(self, query: str, chunk_types: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)
            if not query_embedding:
                return []
            
            # Retrieve stored embeddings
            conn = sqlite3.connect(self.embeddings_db_path)
            cursor = conn.cursor()
            
            # Build query
            sql = "SELECT chunk_type, chunk_id, content, metadata, embedding FROM data_embeddings"
            params = []
            
            if chunk_types:
                placeholders = ','.join(['?' for _ in chunk_types])
                sql += f" WHERE chunk_type IN ({placeholders})"
                params.extend(chunk_types)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning("No embeddings found in database")
                return []
            
            # Calculate similarities
            similarities = []
            query_vector = np.array(query_embedding).reshape(1, -1)
            
            for row in rows:
                chunk_type, chunk_id, content, metadata, embedding_bytes = row
                stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float64).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(query_vector, stored_embedding)[0][0]
                
                similarities.append({
                    'chunk_type': chunk_type,
                    'chunk_id': chunk_id,
                    'content': content,
                    'metadata': json.loads(metadata),
                    'similarity': float(similarity)
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Found {len(similarities)} chunks, returning top {top_k}")
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def build_embeddings_from_data(self):
        """Build embeddings for all available datasets"""
        try:
            logger.info("🔄 Building RAG embeddings from datasets...")
            
            # Dataset mappings
            datasets = {
                'transactions': 'retail_transactions_processed.csv',
                'customer_analysis': 'customer_analysis.csv',
                'product_performance': 'product_performance.csv',
                'country_performance': 'country_performance.csv'
            }
            
            total_chunks = 0
            
            for chunk_type, filename in datasets.items():
                file_path = self.data_path / filename
                
                if not file_path.exists():
                    logger.warning(f"Dataset not found: {filename}")
                    continue
                
                # Load dataset
                df = pd.read_csv(file_path)
                logger.info(f"Processing {chunk_type}: {len(df)} rows")
                
                # Create chunks
                chunks = self.chunk_dataframe(df, chunk_type)
                
                if chunks:
                    # Store embeddings
                    stored = self.store_embeddings(chunk_type, chunks)
                    total_chunks += stored
                    logger.info(f"✅ {chunk_type}: {stored} chunks embedded")
            
            logger.info(f"🎉 RAG embeddings complete! Total chunks: {total_chunks}")
            return total_chunks
            
        except Exception as e:
            logger.error(f"Error building embeddings: {str(e)}")
            return 0
    
    def get_embeddings_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        try:
            conn = sqlite3.connect(self.embeddings_db_path)
            cursor = conn.cursor()
            
            # Get counts by chunk type
            cursor.execute("""
                SELECT chunk_type, COUNT(*) as count
                FROM data_embeddings
                GROUP BY chunk_type
            """)
            
            stats = {}
            for row in cursor.fetchall():
                chunk_type, count = row
                stats[chunk_type] = count
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM data_embeddings")
            total = cursor.fetchone()[0]
            stats['total'] = total
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting embeddings stats: {str(e)}")
            return {}


def main():
    """Main function to build embeddings"""
    if not GENAI_AVAILABLE:
        print("❌ Google GenAI not available. Install: pip install google-generativeai")
        return
    
    try:
        # Initialize RAG manager
        rag_manager = RAGEmbeddingsManager()
        
        # Build embeddings
        total_chunks = rag_manager.build_embeddings_from_data()
        
        if total_chunks > 0:
            # Show stats
            stats = rag_manager.get_embeddings_stats()
            print(f"\n🎉 RAG Embeddings Built Successfully!")
            print(f"📊 Total Chunks: {stats.get('total', 0)}")
            for chunk_type, count in stats.items():
                if chunk_type != 'total':
                    print(f"   - {chunk_type}: {count} chunks")
            
            # Test search
            print(f"\n🔍 Testing search...")
            results = rag_manager.search_similar_chunks("customer revenue analysis", top_k=3)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['chunk_type']} (similarity: {result['similarity']:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()