# ML Demonstration Code

This folder contains standalone ML demonstration scripts that showcase advanced analytics capabilities but are **NOT integrated** into the main application.

## Contents

- `ml_demos/` - Standalone ML model implementations:
  - `customer_segmentation.py` - RFM analysis, K-means clustering, CLV prediction
  - `sales_forecasting.py` - ARIMA, Exponential Smoothing, ML forecasting  
  - `synthetic_data_generator.py` - Realistic sales data generation

- `saved_models_demos/` - Saved model artifacts from demonstration runs (not used in production)

## Usage

These are **demonstration scripts only**. Run them independently:

```bash
cd examples/ml_demos
python customer_segmentation.py
python sales_forecasting.py
```

## Note

The main application uses:
- **Spark ML** for distributed analytics (`src/spark/`)
- **Deep Learning** models (`src/deep_learning/`) 
- **A/B Testing** framework (`src/experimentation/`)

These examples showcase additional ML capabilities but are not part of the core system.