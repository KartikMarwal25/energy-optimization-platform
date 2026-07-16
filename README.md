# Autonomous Energy Optimization Platform for Smart Grids

This project provides a modular Streamlit application for energy analytics, forecasting, anomaly detection, clustering, optimization recommendations, explainability, and reporting.

## Features
- ETL-style data loading for block-based energy datasets
- Data preprocessing and feature engineering
- EDA with statistical plots and correlation analysis
- Forecasting with multiple baseline and boosted models
- Anomaly detection and segmentation
- Recommendation engine and explainability helpers
- SQLite-backed persistence and report export

## Run locally
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Docker
```bash
docker build -t energy-platform .
docker run -p 8501:8501 energy-platform
```
