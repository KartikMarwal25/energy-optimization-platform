import streamlit as st
from src.database.db import Database
from src.preprocessing.data_loader import load_raw_data, ensure_data_present
from src.preprocessing.processor import preprocess_pipeline
from src.eda.plots import generate_all_plots
from src.forecasting.models import ModelManager
from src.anomaly_detection.detectors import AnomalyDetector
from src.clustering.cluster import ClusterEngine
from src.recommendation.engine import RecommendationEngine
from src.utils.logging_config import setup_logging
from src.explainability.explain import Explainability
from src.reports.generator import export_csv, export_excel

setup_logging()
DB_PATH = "data/processed/database.sqlite"


def main():
    st.set_page_config(page_title="Autonomous Energy Optimization", layout="wide")
    st.title("Autonomous Energy Optimization Platform for Smart Grids")
    db = Database(DB_PATH)

    menu = [
        "Home",
        "Dataset",
        "EDA",
        "Forecasting",
        "Anomaly Detection",
        "Customer Segmentation",
        "Optimization Engine",
        "Explainability",
        "Model Performance",
        "Reports",
        "Settings",
    ]
    choice = st.sidebar.selectbox("Navigation", menu)

    ensure_data_present()
    raw = load_raw_data()
    processed = preprocess_pipeline(raw)

    if choice == "Home":
        st.header("Overview")
        st.write("Dataset preview and summary metrics")
        st.dataframe(processed.head(50))

    if choice == "Dataset":
        st.header("Dataset")
        st.dataframe(raw.head(100))
        st.download_button("Download processed CSV", processed.to_csv(index=False), file_name="processed.csv")

    if choice == "EDA":
        st.header("Exploratory Data Analysis")
        generate_all_plots(processed)
        st.write("Plots saved to reports/figs/")
        if st.button("Export EDA summary as CSV"):
            path = export_csv(processed.describe().reset_index(), name="eda_summary")
            st.write(f"Saved: {path}")

    if choice == "Forecasting":
        st.header("Forecasting Models")
        mm = ModelManager(processed)
        if st.button("Train models and compare"):
            results = mm.train_and_compare(target_col="consumption")
            st.write(results)
            best = mm.select_best()
            st.success(f"Best model: {best}")

    if choice == "Anomaly Detection":
        st.header("Anomaly Detection")
        ad = AnomalyDetector(processed)
        anomalies = ad.run_all()
        st.write(anomalies.head(50))

    if choice == "Customer Segmentation":
        st.header("Customer Segmentation")
        ce = ClusterEngine(processed)
        clusters = ce.run_kmeans()
        st.write(clusters.head(50))

    if choice == "Optimization Engine":
        st.header("Recommendations")
        re = RecommendationEngine(processed)
        recs = re.generate_all()
        st.write(recs.head(50))

    if choice == "Explainability":
        st.header("Explainability")
        expl = Explainability(processed)
        st.write("Explainability helpers are ready for SHAP-based inspection.")

    if choice == "Model Performance":
        st.header("Model Performance")
        st.write("Saved model metrics and outputs are stored in the SQLite database and reports folder.")

    if choice == "Reports":
        st.header("Generated Reports")
        st.write("See reports/ for generated summaries and charts.")

    if choice == "Settings":
        st.header("Settings")
        st.write("Application settings and model retraining controls are ready to expand.")


if __name__ == "__main__":
    main()
