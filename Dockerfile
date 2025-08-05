# RetailOps BI - Docker Configuration
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment for Spark
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/processed data/raw data/models data/reports

# Set environment variables
ENV PYTHONPATH=/app
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Snowflake environment variables (for Docker deployment)
ENV SNOWFLAKE_ACCOUNT=your_account
ENV SNOWFLAKE_USER=your_username
ENV SNOWFLAKE_PASSWORD=your_password
ENV SNOWFLAKE_WAREHOUSE=COMPUTE_WH
ENV SNOWFLAKE_DATABASE=RETAIL_BI
ENV SNOWFLAKE_SCHEMA=RAW
ENV SNOWFLAKE_ROLE=ACCOUNTADMIN

# GenAI API Keys (for enhanced features)
ENV GEMINI_API_KEY=your_gemini_api_key
ENV LANGSMITH_API_KEY=your_langsmith_api_key

# Development settings
ENV DEBUG=True
ENV ENVIRONMENT=development

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run ETL first, then both FastAPI and Streamlit
CMD ["sh", "-c", "python run_system.py & sleep 30 && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"] 