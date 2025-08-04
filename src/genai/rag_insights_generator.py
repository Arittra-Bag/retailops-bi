"""
RAG-Enhanced Insights Generator for RetailOps BI
Uses vector embeddings and retrieval for contextual AI responses
"""

import os
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Import RAG embeddings manager
from .rag_embeddings import RAGEmbeddingsManager

# LangChain imports for chat
try:
    from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

# Direct Google GenAI for backup
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RetailOps-BI"

load_dotenv()
logger = logging.getLogger(__name__)

class RAGInsightsGenerator:
    """RAG-enhanced insights generator using vector embeddings"""
    
    def __init__(self):
        """Initialize RAG insights generator"""
        self.project_root = Path(__file__).parent.parent.parent
        
        # Initialize API keys
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Set LangSmith API key for tracing
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        
        # Initialize RAG embeddings manager
        self.rag_manager = RAGEmbeddingsManager()
        
        # Initialize AI models
        self.llm = None
        self.chat_model = None
        self._initialize_models()
        
        logger.info("✅ RAG Insights Generator initialized")
    
    def _initialize_models(self):
        """Initialize AI models for generation"""
        try:
            if LANGCHAIN_AVAILABLE:
                # Try LangChain models first
                model_names = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
                
                for model_name in model_names:
                    try:
                        self.llm = GoogleGenerativeAI(
                            model=model_name,
                            google_api_key=self.gemini_api_key,
                            temperature=0.1,
                            max_output_tokens=2048
                        )
                        
                        self.chat_model = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=self.gemini_api_key,
                            temperature=0.1,
                            max_output_tokens=2048
                        )
                        
                        # Test the model
                        test_response = self.llm.invoke("Hello")
                        logger.info(f"✅ LangChain model {model_name} working")
                        break
                        
                    except Exception as e:
                        logger.warning(f"LangChain model {model_name} failed: {str(e)[:100]}...")
                        continue
            
            # Fallback to direct GenAI
            if not self.llm and GENAI_AVAILABLE:
                genai.configure(api_key=self.gemini_api_key)
                self.direct_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Direct GenAI model initialized")
            
            if not self.llm and not hasattr(self, 'direct_model'):
                raise Exception("No working AI model found")
                
        except Exception as e:
            logger.error(f"Error initializing AI models: {str(e)}")
            raise
    
    def ensure_embeddings_ready(self) -> bool:
        """Ensure embeddings are built and ready"""
        try:
            stats = self.rag_manager.get_embeddings_stats()
            total_chunks = stats.get('total', 0)
            
            if total_chunks == 0:
                logger.info("🔄 No embeddings found, building from data...")
                total_chunks = self.rag_manager.build_embeddings_from_data()
            
            if total_chunks > 0:
                logger.info(f"✅ RAG embeddings ready: {total_chunks} chunks")
                return True
            else:
                logger.error("❌ Failed to build embeddings")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring embeddings: {str(e)}")
            return False
    
    def get_rag_insights(self, query: str, top_k: int = 5) -> str:
        """Get insights using RAG retrieval and generation"""
        try:
            # Ensure embeddings are ready
            if not self.ensure_embeddings_ready():
                return "❌ RAG embeddings not available. Please check your data and try again."
            
            logger.info(f"🔍 RAG Query: {query}")
            
            # Step 1: Retrieve relevant chunks using vector similarity
            relevant_chunks = self.rag_manager.search_similar_chunks(query, top_k=top_k)
            
            if not relevant_chunks:
                return "❌ No relevant data found for your query. Please try a different question."
            
            # Step 2: Build context from retrieved chunks
            context_parts = []
            context_parts.append("📊 **RELEVANT DATA CONTEXT:**\n")
            
            for i, chunk in enumerate(relevant_chunks, 1):
                similarity_score = chunk['similarity'] * 100
                context_parts.append(f"\n**Context {i}** (Relevance: {similarity_score:.1f}%):")
                context_parts.append(f"Type: {chunk['chunk_type']}")
                context_parts.append(f"Content:\n{chunk['content']}")
                context_parts.append("---")
            
            retrieved_context = "\n".join(context_parts)
            
            # Step 3: Generate insights using LLM with retrieved context
            if self.llm:
                # Use LangChain
                prompt_template = PromptTemplate(
                    input_variables=["query", "context"],
                    template="""
                    You are an expert retail business analyst with access to comprehensive data.
                    
                    USER QUESTION: {query}
                    
                    RETRIEVED DATA CONTEXT:
                    {context}
                    
                    Based on the specific data context above, provide a detailed, actionable analysis that:
                    
                    1. **Direct Answer**: Directly answers the user's question using the retrieved data
                    2. **Key Insights**: Extract 3-4 specific insights from the data
                    3. **Business Implications**: Explain what these findings mean for the business
                    4. **Recommendations**: Provide 2-3 actionable recommendations
                    5. **Supporting Data**: Reference specific numbers and trends from the context
                    
                    Make your response data-driven, specific, and actionable. Use the exact figures and patterns from the retrieved context.
                    """
                )
                
                chain = LLMChain(llm=self.llm, prompt=prompt_template)
                response = chain.run(query=query, context=retrieved_context)
                
            elif hasattr(self, 'direct_model'):
                # Use direct GenAI
                prompt = f"""
                You are an expert retail business analyst with access to comprehensive data.
                
                USER QUESTION: {query}
                
                RETRIEVED DATA CONTEXT:
                {retrieved_context}
                
                Based on the specific data context above, provide a detailed, actionable analysis that directly answers the question using the retrieved data.
                
                Include key insights, business implications, and actionable recommendations based on the specific numbers and trends shown.
                """
                
                response = self.direct_model.generate_content(prompt).text
            
            else:
                return "❌ No AI model available for generating insights."
            
            # Step 4: Add retrieval metadata
            retrieval_info = f"\n\n---\n**🔍 Data Sources Used:**\n"
            for chunk in relevant_chunks:
                retrieval_info += f"• {chunk['chunk_type']} (Relevance: {chunk['similarity']*100:.1f}%)\n"
            
            final_response = response + retrieval_info
            
            logger.info(f"✅ RAG insights generated using {len(relevant_chunks)} data sources")
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating RAG insights: {str(e)}")
            return f"❌ Error generating insights: {str(e)}"
    
    def get_insights_with_data_exploration(self, query: str) -> Dict[str, Any]:
        """Get insights with detailed data exploration"""
        try:
            # Get RAG insights
            insights = self.get_rag_insights(query)
            
            # Get relevant chunks for data exploration
            relevant_chunks = self.rag_manager.search_similar_chunks(query, top_k=3)
            
            # Extract metadata for charts/exploration
            data_exploration = {
                'relevant_data_types': [chunk['chunk_type'] for chunk in relevant_chunks],
                'similarity_scores': [chunk['similarity'] for chunk in relevant_chunks],
                'data_summaries': []
            }
            
            for chunk in relevant_chunks:
                metadata = chunk.get('metadata', {})
                data_exploration['data_summaries'].append({
                    'type': chunk['chunk_type'],
                    'summary': metadata,
                    'relevance': chunk['similarity']
                })
            
            return {
                'query': query,
                'insights': insights,
                'timestamp': datetime.now().isoformat(),
                'data_exploration': data_exploration,
                'retrieval_method': 'RAG_vector_similarity',
                'chunks_used': len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in data exploration: {str(e)}")
            return {
                'query': query,
                'insights': f"❌ Error: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def rebuild_embeddings(self) -> Dict[str, Any]:
        """Rebuild embeddings from current data"""
        try:
            logger.info("🔄 Rebuilding RAG embeddings...")
            
            total_chunks = self.rag_manager.build_embeddings_from_data()
            stats = self.rag_manager.get_embeddings_stats()
            
            return {
                'status': 'success',
                'total_chunks': total_chunks,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding embeddings: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def test_rag_system(self) -> Dict[str, Any]:
        """Test the RAG system with sample queries"""
        try:
            test_queries = [
                "What are my top customer segments by revenue?",
                "How is my product performance trending?",
                "Which countries drive the most sales?",
                "What are my key business metrics?"
            ]
            
            results = {}
            for query in test_queries:
                logger.info(f"Testing: {query}")
                insights = self.get_rag_insights(query, top_k=3)
                results[query] = {
                    'insights': insights[:200] + "..." if len(insights) > 200 else insights,
                    'success': "❌" not in insights
                }
            
            return {
                'status': 'completed',
                'test_results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error testing RAG system: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main function to test RAG insights"""
    try:
        # Initialize RAG insights generator
        generator = RAGInsightsGenerator()
        
        # Test queries
        test_queries = [
            "Analyze my customer segments and tell me which ones drive the most revenue",
            "What are my top-performing products and categories?",
            "How do my sales vary by country?",
            "What trends do you see in my transaction data?"
        ]
        
        print("🤖 Testing RAG-Enhanced Insights Generator\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"🔍 Test {i}: {query}")
            print("-" * 60)
            
            result = generator.get_insights_with_data_exploration(query)
            print(f"✅ Insights: {result['insights'][:300]}...")
            print(f"📊 Data sources: {result.get('chunks_used', 0)} chunks")
            print("\n" + "="*80 + "\n")
        
        # Show system stats
        stats = generator.rag_manager.get_embeddings_stats()
        print(f"📈 RAG System Stats:")
        print(f"   Total embeddings: {stats.get('total', 0)}")
        for chunk_type, count in stats.items():
            if chunk_type != 'total':
                print(f"   - {chunk_type}: {count}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()