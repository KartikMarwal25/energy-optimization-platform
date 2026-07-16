import streamlit as st
import os
import glob
from PIL import Image
import streamlit.components.v1 as components
import pandas as pd
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
    with st.spinner("Loading raw data..."):
        raw = load_raw_data()
    st.success(f"Loaded raw data: {raw.shape[0]} rows, {raw.shape[1]} cols")

    with st.spinner("Running preprocessing pipeline..."):
        processed = preprocess_pipeline(raw)
    st.success(f"Processed data: {processed.shape[0]} rows, {processed.shape[1]} cols")

    def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        # convert object and category columns to pandas string dtype to avoid pyarrow numeric inference issues
        for c in df2.select_dtypes(include=["object", "category"]).columns:
            try:
                df2[c] = df2[c].astype("string")
            except Exception:
                df2[c] = df2[c].astype(str)
        # ensure datetimes are proper dtype
        for c in df2.columns:
            if c.lower().startswith("ts") or c.lower().endswith("time") or c == "timestamp":
                try:
                    df2[c] = pd.to_datetime(df2[c], errors="coerce")
                except Exception:
                    pass
        return df2

    if choice == "Home":
        st.header("Overview")
        st.write("Dataset preview and summary metrics")
        st.dataframe(_arrow_safe(processed.head(50)))
        st.markdown("**Basic summary**")
        st.write(processed.describe(include='all'))

    if choice == "Dataset":
        st.header("Dataset")
        st.subheader("Raw sample")
        st.dataframe(_arrow_safe(raw.head(100)))
        st.subheader("Processed sample")
        st.dataframe(_arrow_safe(processed.head(100)))
        st.download_button("Download processed CSV", processed.to_csv(index=False), file_name="processed.csv")

    if choice == "EDA":
        st.header("Exploratory Data Analysis")
        generate_all_plots(processed)
        st.write("Plots saved to reports/figs/")
        fig_dir = os.path.join(os.getcwd(), "reports", "figs")
        imgs = sorted(glob.glob(os.path.join(fig_dir, "*.png")))
        htmls = sorted(glob.glob(os.path.join(fig_dir, "*.html")))
        for img in imgs:
            try:
                st.image(Image.open(img), caption=os.path.basename(img))
            except Exception:
                st.write(f"Could not display {img}")
        for h in htmls:
            try:
                with open(h, 'r', encoding='utf-8') as f:
                    html = f.read()
                components.html(html, height=400)
            except Exception:
                st.write(f"Could not display {h}")
        if st.button("Export EDA summary as CSV"):
            path = export_csv(processed.describe().reset_index(), name="eda_summary")
            st.write(f"Saved: {path}")

    if choice == "Forecasting":
        st.header("Forecasting Models")
        mm = ModelManager(processed)
        st.subheader("Saved model metadata")
        try:
            db_models = mm.db.query_table("models")
        except Exception:
            db_models = pd.DataFrame()
        if not db_models.empty:
            st.dataframe(_arrow_safe(db_models.sort_values("trained_at", ascending=False).head(50)))
        else:
            st.info("No saved model metadata found in the database.")

        if st.button("Train models and compare"):
            with st.spinner("Training candidate models — this may take several minutes"):
                try:
                    results = mm.train_and_compare(target_col="consumption")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    results = pd.DataFrame()
            if results.empty:
                st.warning("No models were successfully trained. Check logs and ensure numeric features are available.")
                if not db_models.empty:
                    st.info("You can inspect previously saved model metadata in the table above.")
            else:
                st.success("Training complete")
                st.dataframe(_arrow_safe(results))
                best = mm.select_best()
                if best:
                    st.success(f"Best model: {best}")
                    path = results.loc[results['model'] == best, 'path'].values
                    if len(path) and os.path.exists(path[0]):
                        with open(path[0], 'rb') as f:
                            st.download_button(f"Download best model ({best})", f, file_name=os.path.basename(path[0]))

    if choice == "Anomaly Detection":
        st.header("Anomaly Detection")
        st.write("Anomaly detection can be expensive on large datasets, so this app runs analytics on a sample of the data.")
        ad = AnomalyDetector(processed)
        if st.button("Run anomaly detection"):
            with st.spinner("Running anomaly detection on a sampled dataset..."):
                anomalies = ad.run_all()
            st.success(f"Found {len(anomalies)} anomaly rows")
            if anomalies.empty:
                st.warning("No anomalies were detected in the sampled dataset.")
            else:
                st.dataframe(_arrow_safe(anomalies.head(50)))
        else:
            st.info("Click the button above to start anomaly detection.")

    if choice == "Customer Segmentation":
        st.header("Customer Segmentation")
        ce = ClusterEngine(processed)
        with st.spinner("Running clustering..."):
            clusters = ce.run_kmeans()
        st.success("Clustering complete")
        st.write(clusters[["cluster"]].value_counts().rename("count"))
        st.dataframe(_arrow_safe(clusters.head(50)))

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
