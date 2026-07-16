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
from src.clustering.visuals import segment_pie, pca_scatter, radar_chart
from src.recommendation.engine import RecommendationEngine
from src.utils.logging_config import setup_logging
from src.explainability.explain import Explainability
from src.reports.generator import export_csv, export_excel, export_pptx, export_html, export_pdf

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

    # Cache raw and processed data so widget interactions don't re-run heavy pipelines
    @st.cache_data(show_spinner=False)
    def _cached_load_raw():
        ensure_data_present()
        return load_raw_data()

    @st.cache_data(show_spinner=False)
    def _cached_preprocess(df_raw):
        return preprocess_pipeline(df_raw)

    # Load raw data once and keep in session_state to avoid spinner on every rerun
    if 'raw' not in st.session_state:
        with st.spinner("Loading raw data..."):
            raw = _cached_load_raw()
            st.session_state['raw'] = raw
        st.success(f"Loaded raw data: {st.session_state['raw'].shape[0]} rows, {st.session_state['raw'].shape[1]} cols")
    else:
        raw = st.session_state['raw']

    # Preprocess once and keep in session_state to avoid retriggering on widget changes
    if 'processed' not in st.session_state:
        with st.spinner("Running preprocessing pipeline..."):
            processed = _cached_preprocess(raw)
            st.session_state['processed'] = processed
        st.success(f"Processed data: {st.session_state['processed'].shape[0]} rows, {st.session_state['processed'].shape[1]} cols")
    else:
        processed = st.session_state['processed']

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
        st.write("Interactive charts (Plotly). Charts are saved under `reports/figs/` as HTML files.")
        # generate_all_plots can be expensive; require explicit run and persist in session_state
        @st.cache_data(show_spinner=False)
        def _cached_generate_all_plots(df):
            return generate_all_plots(df)

        if 'eda_figs' not in st.session_state:
            st.info("Click 'Generate EDA' to produce interactive plots (cached).")
        if st.button("Generate EDA"):
            with st.spinner("Generating EDA plots..."):
                figs = _cached_generate_all_plots(processed)
                st.session_state['eda_figs'] = figs
            st.success("EDA plots generated and cached")

        figs = st.session_state.get('eda_figs', {})
        fig_dir = os.path.join(os.getcwd(), "reports", "figs")
        for fig_name, fig in figs.items():
            if fig is None:
                continue
            st.subheader(fig_name.replace('_',' ').title())
            try:
                st.plotly_chart(fig, use_container_width=True, key=f"eda_{fig_name}")
            except Exception:
                st.write("Could not render interactive chart; saved HTML may still be available.")
            html_path = os.path.join(fig_dir, f"{fig_name}.html")
            if os.path.exists(html_path):
                with open(html_path, 'rb') as f:
                    st.download_button(f"Download {fig_name} (HTML)", f, file_name=os.path.basename(html_path), key=f"download_eda_{fig_name}")
        if st.button("Export EDA summary as CSV"):
            path = export_csv(processed.describe().reset_index(), name="eda_summary")
            st.write(f"Saved: {path}")

    if choice == "Forecasting":
        st.header("Forecasting")
        mm = None
        # instantiate ModelManager only when running the forecast
        if 'forecasts' not in st.session_state:
            st.session_state['forecasts'] = {}
        st.write("Select a consumer and forecast horizon, then run a quick forecast (simple linear trend).")
        # consumer selector
        consumers = processed['meter_id'].astype(str).unique().tolist() if 'meter_id' in processed.columns else []
        consumer = st.selectbox("Consumer (meter_id)", options=consumers[:200] if consumers else [], index=0 if consumers else None)
        horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=365, value=30)
        method = st.selectbox("Method", options=["linear_trend", "mean"], index=0)

        key = f"{consumer}_{horizon}_{method}"
        if st.button("Run forecast"):
            if not consumer:
                st.error("No consumer selected")
            else:
                mm = ModelManager(processed)
                with st.spinner("Running forecast..."):
                    try:
                        df_fc = mm.forecast_consumer(consumer, horizon_days=horizon)
                        st.session_state['forecasts'][key] = df_fc
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")
                        df_fc = pd.DataFrame()
                if df_fc.empty:
                    st.warning("No forecast produced for this consumer (insufficient data).")
                else:
                    import plotly.graph_objects as go
                    # historical daily
                    df_hist = processed[processed['meter_id'].astype(str) == str(consumer)].dropna(subset=['consumption','timestamp']).copy()
                    if not df_hist.empty:
                        df_hist['date'] = pd.to_datetime(df_hist['timestamp']).dt.date
                        hist_daily = df_hist.groupby('date')['consumption'].sum().reset_index()
                        hist_daily['date'] = pd.to_datetime(hist_daily['date'])
                    else:
                        hist_daily = None

                    fig = go.Figure()
                    if hist_daily is not None and not hist_daily.empty:
                        fig.add_trace(go.Scatter(x=hist_daily['date'], y=hist_daily['consumption'], mode='lines+markers', name='Historical'))
                    fig.add_trace(go.Scatter(x=df_fc['date'], y=df_fc['forecast'], mode='lines+markers', name='Forecast'))
                    if 'lower' in df_fc.columns and 'upper' in df_fc.columns:
                        fig.add_trace(go.Scatter(x=df_fc['date'].tolist()+df_fc['date'].tolist()[::-1],
                                                 y=df_fc['upper'].tolist()+df_fc['lower'].tolist()[::-1],
                                                 fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False, name='95% CI'))
                    fig.update_layout(title=f'Historical vs Forecast for {consumer}', xaxis_title='Date', yaxis_title='Consumption')
                    st.plotly_chart(fig, use_container_width=True, key=f"forecast_run_{consumer}_{horizon}_{method}")

                    # stats
                    fc_sum = float(df_fc['forecast'].sum())
                    peak_idx = df_fc['forecast'].idxmax()
                    min_idx = df_fc['forecast'].idxmin()
                    peak_day = df_fc.loc[peak_idx, 'date']
                    min_day = df_fc.loc[min_idx, 'date']
                    st.write(f"Expected total over next {horizon} days: {fc_sum:.2f}")
                    st.write(f"Peak forecast day: {peak_day}")
                    st.write(f"Minimum forecast day: {min_day}")

                    # downloads
                    csv_bytes = df_fc.to_csv(index=False).encode('utf-8')
                    st.download_button("Download prediction CSV", data=csv_bytes, file_name=f"forecast_{consumer}.csv")
                    # export chart as html
                    import plotly.io as pio
                    html = pio.to_html(fig, include_plotlyjs='cdn')
                    st.download_button("Download forecast graph (HTML)", data=html, file_name=f"forecast_{consumer}.html")
        # display cached forecast if available
        if key in st.session_state.get('forecasts', {}):
            df_fc = st.session_state['forecasts'][key]
            if not df_fc.empty:
                import plotly.graph_objects as go
                df_hist = processed[processed['meter_id'].astype(str) == str(consumer)].dropna(subset=['consumption','timestamp']).copy()
                if not df_hist.empty:
                    df_hist['date'] = pd.to_datetime(df_hist['timestamp']).dt.date
                    hist_daily = df_hist.groupby('date')['consumption'].sum().reset_index()
                    hist_daily['date'] = pd.to_datetime(hist_daily['date'])
                else:
                    hist_daily = None

                fig = go.Figure()
                if hist_daily is not None and not hist_daily.empty:
                    fig.add_trace(go.Scatter(x=hist_daily['date'], y=hist_daily['consumption'], mode='lines+markers', name='Historical'))
                fig.add_trace(go.Scatter(x=df_fc['date'], y=df_fc['forecast'], mode='lines+markers', name='Forecast'))
                if 'lower' in df_fc.columns and 'upper' in df_fc.columns:
                    fig.add_trace(go.Scatter(x=df_fc['date'].tolist()+df_fc['date'].tolist()[::-1],
                                             y=df_fc['upper'].tolist()+df_fc['lower'].tolist()[::-1],
                                             fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False, name='95% CI'))
                fig.update_layout(title=f'Historical vs Forecast for {consumer}', xaxis_title='Date', yaxis_title='Consumption')
                st.plotly_chart(fig, use_container_width=True, key=f"forecast_cached_{consumer}_{horizon}_{method}")

                fc_sum = float(df_fc['forecast'].sum())
                peak_idx = df_fc['forecast'].idxmax()
                min_idx = df_fc['forecast'].idxmin()
                peak_day = df_fc.loc[peak_idx, 'date']
                min_day = df_fc.loc[min_idx, 'date']
                st.write(f"Expected total over next {horizon} days: {fc_sum:.2f}")
                st.write(f"Peak forecast day: {peak_day}")
                st.write(f"Minimum forecast day: {min_day}")

    if choice == "Anomaly Detection":
        st.header("Anomaly Detection")
        st.write("Anomaly detection can be expensive on large datasets, so this app runs analytics on a sample of the data.")
        ad = AnomalyDetector(processed)
        if 'anomalies' not in st.session_state:
            st.session_state['anomalies'] = pd.DataFrame()
        if st.button("Run anomaly detection"):
            with st.spinner("Running anomaly detection on a sampled dataset..."):
                anomalies = ad.run_all()
                st.session_state['anomalies'] = anomalies
            st.success(f"Found {len(anomalies)} anomaly rows")
        else:
            if st.session_state['anomalies'].empty:
                st.info("Click the button above to start anomaly detection.")
        if not st.session_state['anomalies'].empty:
            anomalies = st.session_state['anomalies']
            if anomalies.empty:
                st.warning("No anomalies were detected in the sampled dataset.")
            else:
                st.dataframe(_arrow_safe(anomalies.head(50)))

    if choice == "Customer Segmentation":
        st.header("Customer Segmentation")
        ce = ClusterEngine(processed)
        if st.button("Run clustering"):
            with st.spinner("Running clustering..."):
                clusters = ce.run_kmeans()
                st.session_state['clusters'] = clusters
            st.success("Clustering complete")
        else:
            clusters = st.session_state.get('clusters', pd.DataFrame())
            if clusters.empty:
                st.info("Click 'Run clustering' to compute customer segments.")
        if not clusters.empty:
            # add human readable segment names
            if 'segments_named' not in st.session_state:
                named = ce.name_clusters(clusters)
                st.session_state['segments_named'] = named
            else:
                named = st.session_state['segments_named']

            st.subheader('Segment summary')
            st.write(named[["segment"]].value_counts().rename("count"))
            # segmentation visuals: generate on demand and persist
            if 'segmentation_figs' not in st.session_state:
                st.info("Click 'Generate segmentation visuals' to create charts.")
            if st.button("Generate segmentation visuals"):
                with st.spinner("Generating segmentation visuals..."):
                    figs = {}
                    figs['pie'] = segment_pie(named)
                    figs['pca'] = pca_scatter(named)
                    figs['radar'] = radar_chart(named)
                    st.session_state['segmentation_figs'] = figs
                st.success("Segmentation visuals ready")

            seg_figs = st.session_state.get('segmentation_figs', {})
            if 'pie' in seg_figs and seg_figs['pie'] is not None:
                st.subheader('Segment distribution')
                st.plotly_chart(seg_figs['pie'], use_container_width=True, key='seg_pie')
            if 'pca' in seg_figs and seg_figs['pca'] is not None:
                st.subheader('PCA scatter (sample)')
                st.plotly_chart(seg_figs['pca'], use_container_width=True, key='seg_pca')
            if 'radar' in seg_figs and seg_figs['radar'] is not None:
                st.subheader('Cluster centroid radar')
                st.plotly_chart(seg_figs['radar'], use_container_width=True, key='seg_radar')

            st.dataframe(_arrow_safe(named.head(50)))
            

    if choice == "Optimization Engine":
        st.header("Recommendations")
        re = RecommendationEngine(processed)
        if 'recommendations' not in st.session_state:
            st.session_state['recommendations'] = pd.DataFrame()

        if st.button("Generate recommendations"):
            with st.spinner("Generating recommendations from forecasts, anomalies, and segments..."):
                forecasts = st.session_state.get('forecasts', {})
                anomalies = st.session_state.get('anomalies', pd.DataFrame())
                segments = st.session_state.get('segments_named', pd.DataFrame())
                recs = re.generate_from_forecast_and_anomalies(forecasts=forecasts, anomalies=anomalies, segments=segments)
                st.session_state['recommendations'] = recs
            st.success("Recommendations generated")

        recs = st.session_state.get('recommendations', pd.DataFrame())
        if recs.empty:
            st.info("No recommendations yet — click 'Generate recommendations' to analyze forecasts, anomalies, and segments.")
        else:
            st.dataframe(recs)
            csv_bytes = recs.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations CSV", data=csv_bytes, file_name='recommendations.csv')

    if choice == "Explainability":
        st.header("Explainability")
        expl = Explainability(processed)
        model_dir = os.path.join(os.getcwd(), 'models')
        model_files = []
        if os.path.exists(model_dir):
            model_files = [os.path.basename(p) for p in glob.glob(os.path.join(model_dir, '*.pkl')) + glob.glob(os.path.join(model_dir, '*.joblib')) + glob.glob(os.path.join(model_dir, '*.sav'))]

        if not model_files:
            st.info('No saved models found in the models/ directory. Save a model (joblib) to run SHAP explanations.')
        else:
            sel = st.selectbox('Select model file', options=model_files)
            nsamples = st.slider('SHAP sample size', min_value=10, max_value=1000, value=100, step=10)
            if st.button('Run explainability'):
                model_path = os.path.join(model_dir, sel)
                with st.spinner('Running SHAP explainability (this may take a while)...'):
                    try:
                        res = expl.explain_model(model_path=model_path, nsamples=nsamples)
                        st.session_state['explain_result'] = res
                    except Exception as e:
                        st.error(f'Explainability failed: {e}')

        if 'explain_result' in st.session_state:
            res = st.session_state['explain_result']
            if res.get('notes'):
                st.info('Note: ' + ' '.join(res.get('notes', [])))
            try:
                import shap
                import matplotlib.pyplot as plt
                shap_values = res['shap_values']

                st.subheader('SHAP summary (bar)')
                plt.clf()
                shap.plots.bar(shap_values, show=False)
                fig = plt.gcf()
                try:
                    st.pyplot(fig)
                except Exception:
                    st.write('Could not render SHAP bar plot directly.')

                st.subheader('SHAP beeswarm')
                try:
                    plt.clf()
                    shap.plots.beeswarm(shap_values, show=False)
                    beeswarm_fig = plt.gcf()
                    st.pyplot(beeswarm_fig)
                except Exception:
                    try:
                        f_html = shap.plots.force(shap_values).html()
                        components.html(f_html, height=600)
                    except Exception as e:
                        st.write('Could not render SHAP beeswarm/force plot:', e)
            except Exception as e:
                st.error(f'Error rendering SHAP plots: {e}')

    if choice == "Model Performance":
        st.header("Model Performance")
        st.write("Saved model metrics, metadata, and model analysis are shown below.")

        models_df = db.query_table('models')
        if not models_df.empty:
            st.subheader('Stored Model Metadata')
            st.dataframe(_arrow_safe(models_df))
            if 'model' in models_df.columns:
                st.write('Loaded model names:', models_df['model'].unique().tolist())

            summary_cols = [c for c in ['model', 'mae', 'mse', 'rmse', 'r2', 'trained_at'] if c in models_df.columns]
            if summary_cols:
                st.subheader('Model statistics summary')
                st.dataframe(_arrow_safe(models_df[summary_cols].sort_values('rmse', ascending=True).head(10)))
            if 'rmse' in models_df.columns:
                best_idx = models_df['rmse'].idxmin()
                if best_idx is not None and best_idx in models_df.index:
                    best_model = models_df.loc[best_idx]
                    st.markdown(f"**Best model:** {best_model.get('model', 'N/A')} with RMSE {best_model.get('rmse', 'N/A'):.3f} and R2 {best_model.get('r2', 'N/A'):.3f}")

            if st.button('Export model performance summary as HTML'):
                performance_text = "Model performance metrics for saved models."
                try:
                    html_path = export_html(performance_text, tables={'Model Metrics': models_df[summary_cols]}, name='model_performance')
                    st.success(f"Model performance HTML saved: {html_path}")
                    with open(html_path, 'rb') as f:
                        st.download_button('Download model performance HTML', data=f, file_name=os.path.basename(html_path))
                except Exception as e:
                    st.error(f"Could not export performance summary: {e}")
        else:
            st.info('No model metadata found in the database yet.')

        model_dir = os.path.join(os.getcwd(), 'models')
        model_files = []
        if os.path.exists(model_dir):
            model_files = [os.path.basename(p) for p in glob.glob(os.path.join(model_dir, '*.pkl')) + glob.glob(os.path.join(model_dir, '*.joblib')) + glob.glob(os.path.join(model_dir, '*.sav'))]

        if model_files:
            st.subheader('Saved Model Files')
            st.write(model_files)
            if st.button('Analyze saved models'):
                analysis_rows = []
                for mf in model_files:
                    model_path = os.path.join(model_dir, mf)
                    try:
                        import joblib
                        mdl = joblib.load(model_path)
                        model_type = type(mdl).__name__
                        feature_count = None
                        has_importances = False
                        has_coef = False
                        top_features = None

                        if hasattr(mdl, 'feature_importances_'):
                            has_importances = True
                            importances = getattr(mdl, 'feature_importances_')
                            try:
                                feature_names = list(getattr(mdl, 'feature_names_in_', []))
                                if not feature_names:
                                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                                top_features = pd.DataFrame({'feature': feature_names, 'importance': importances})
                                top_features = top_features.sort_values('importance', ascending=False).head(10)
                                feature_count = len(feature_names)
                            except Exception:
                                pass
                        if hasattr(mdl, 'coef_'):
                            has_coef = True
                            coef = getattr(mdl, 'coef_')
                            try:
                                coef = coef.ravel()
                                feature_names = list(getattr(mdl, 'feature_names_in_', []))
                                if not feature_names:
                                    feature_names = [f'feature_{i}' for i in range(len(coef))]
                                top_features = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
                                top_features = top_features.iloc[:10]
                                feature_count = len(feature_names)
                            except Exception:
                                pass

                        analysis_rows.append({
                            'model_file': mf,
                            'model_type': model_type,
                            'feature_count': feature_count,
                            'has_importances': has_importances,
                            'has_coef': has_coef,
                        })

                        if top_features is not None:
                            st.subheader(f'Top features for {mf}')
                            st.dataframe(_arrow_safe(top_features))
                            try:
                                import plotly.express as px
                                fig = px.bar(top_features, x=top_features.columns[0], y=top_features.columns[1], title=f'Top features: {mf}')
                                st.plotly_chart(fig, use_container_width=True, key=f'model_feat_{mf}')
                            except Exception:
                                pass
                    except Exception as e:
                        analysis_rows.append({'model_file': mf, 'model_type': 'load_error', 'feature_count': None, 'has_importances': False, 'has_coef': False})
                        st.warning(f'Could not analyze model {mf}: {e}')

                if analysis_rows:
                    analysis_df = pd.DataFrame(analysis_rows)
                    st.subheader('Model Analysis Summary')
                    st.dataframe(_arrow_safe(analysis_df))
        else:
            st.info('No saved model files found in models/ directory.')

    if choice == "Reports":
        st.header("Generated Reports")
        st.write("See reports/ for generated summaries, charts, and executive export files.")

        if st.button("Generate Executive Report"):
            with st.spinner("Compiling executive report..."):
                num_rows = processed.shape[0]
                num_consumers = processed['meter_id'].nunique() if 'meter_id' in processed.columns else 0
                num_segments = 0
                if 'segments_named' in st.session_state and not st.session_state['segments_named'].empty:
                    num_segments = st.session_state['segments_named']['segment'].nunique()

                recs = st.session_state.get('recommendations', pd.DataFrame())
                segments = st.session_state.get('segments_named', pd.DataFrame())
                anomalies = st.session_state.get('anomalies', pd.DataFrame())
                models_df = db.query_table('models')

                top_recs = recs.head(5).to_dict(orient='records') if not recs.empty else []
                summary_lines = ["Executive Energy Optimization Report", "", f"Dataset rows: {num_rows}", f"Consumers: {num_consumers}", f"Segments: {num_segments}", f"Recommendations generated: {len(top_recs)}"]
                for r in top_recs:
                    summary_lines.append(f"- {r.get('issue')}: {r.get('suggestion')}")
                if models_df is not None and not models_df.empty:
                    best_models = models_df.sort_values('rmse', ascending=True).head(3)
                    summary_lines.append("")
                    summary_lines.append("Best models:")
                    for _, row in best_models.iterrows():
                        summary_lines.append(f"- {row.get('model')} RMSE={row.get('rmse'):.3f} R2={row.get('r2', 0):.3f}")

                summary_text = '\n'.join(summary_lines)

                os.makedirs(os.path.join('reports', 'figs'), exist_ok=True)
                chart_paths = []
                eda_figs = st.session_state.get('eda_figs', {})
                try:
                    import plotly.io as pio
                    for name, fig in eda_figs.items():
                        if fig is None:
                            continue
                        out_path = os.path.join('reports', 'figs', f"eda_{name}.png")
                        try:
                            pio.to_image(fig, file=out_path, format='png')
                            chart_paths.append(out_path)
                        except Exception:
                            pass
                except Exception:
                    pass

                seg_figs = st.session_state.get('segmentation_figs', {})
                try:
                    import plotly.io as pio
                    for name, fig in seg_figs.items():
                        if fig is None:
                            continue
                        out_path = os.path.join('reports', 'figs', f"seg_{name}.png")
                        try:
                            pio.to_image(fig, file=out_path, format='png')
                            chart_paths.append(out_path)
                        except Exception:
                            pass
                except Exception:
                    pass

                report_tables = {}
                if not recs.empty:
                    report_tables['Top recommendations'] = recs.head(20)
                if not segments.empty:
                    report_tables['Segments'] = segments.head(20)
                if not anomalies.empty:
                    report_tables['Anomaly sample'] = anomalies.head(20)
                if not models_df.empty:
                    report_tables['Model metadata'] = models_df.head(20)

                try:
                    html_path = export_html(summary_text, tables=report_tables, charts=chart_paths, name='executive_report')
                except Exception as e:
                    html_path = None
                    st.warning(f"HTML report generation failed: {e}")

                try:
                    pdf_path = export_pdf(summary_text, tables=report_tables, charts=chart_paths, name='executive_report')
                except Exception as e:
                    pdf_path = None
                    st.warning(f"PDF export skipped: {e}")

                try:
                    pptx_path = export_pptx(summary_text, charts=chart_paths, dataframes=report_tables, name='executive_report')
                except Exception as e:
                    st.error(f"PowerPoint export failed: {e}")
                    pptx_path = None

            if html_path:
                st.success(f"Executive HTML report saved: {html_path}")
                with open(html_path, 'rb') as f:
                    st.download_button('Download executive HTML', data=f, file_name=os.path.basename(html_path))
            if pdf_path:
                st.success(f"Executive PDF report saved: {pdf_path}")
                with open(pdf_path, 'rb') as f:
                    st.download_button('Download executive PDF', data=f, file_name=os.path.basename(pdf_path))
            if pptx_path:
                st.success(f"Executive PPTX report saved: {pptx_path}")
                with open(pptx_path, 'rb') as f:
                    st.download_button('Download executive PPTX', data=f, file_name=os.path.basename(pptx_path))
            if not any([html_path, pdf_path, pptx_path]):
                st.error('No executive report could be created. Check logs above.')

    if choice == "Settings":
        st.header("Settings")
        st.write("Application settings and model retraining controls are ready to expand.")


if __name__ == "__main__":
    main()
