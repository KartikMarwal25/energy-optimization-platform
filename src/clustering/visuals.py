import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def segment_pie(clusters_df: pd.DataFrame):
    if 'segment' in clusters_df.columns:
        counts = clusters_df['segment'].value_counts().reset_index()
        counts.columns = ['segment', 'count']
        fig = px.pie(counts, names='segment', values='count', title='Segment distribution')
        return fig
    else:
        counts = clusters_df['cluster'].value_counts().reset_index()
        counts.columns = ['cluster', 'count']
        fig = px.pie(counts, names='cluster', values='count', title='Cluster distribution')
        return fig


def pca_scatter(clusters_df: pd.DataFrame, sample_size: int = 5000):
    X = clusters_df.select_dtypes(include=['number']).drop(columns=['cluster'], errors='ignore')
    if X.shape[0] == 0:
        return None
    if X.shape[0] > sample_size:
        sample = X.sample(n=sample_size, random_state=42)
        sample_idx = sample.index
    else:
        sample = X
        sample_idx = X.index

    pca = PCA(n_components=2)
    coords = pca.fit_transform(sample.fillna(0))
    labels = clusters_df.loc[sample_idx, 'cluster'].astype(str).values if 'cluster' in clusters_df.columns else ['0'] * len(coords)
    df2 = pd.DataFrame({'pc1': coords[:, 0], 'pc2': coords[:, 1], 'cluster': labels})
    fig = px.scatter(df2, x='pc1', y='pc2', color='cluster', title='PCA scatter of clusters', width=800, height=500)
    return fig


def radar_chart(clusters_df: pd.DataFrame, features: list = None):
    num = clusters_df.select_dtypes(include=['number'])
    if features is None:
        # pick up to 6 informative numeric features excluding cluster id
        features = [c for c in num.columns if c != 'cluster'][:6]
    if not features:
        return None
    centroids = clusters_df.groupby('cluster')[features].mean()
    # normalize per feature for radar plotting
    scaled = (centroids - centroids.min()) / (centroids.max() - centroids.min() + 1e-9)

    fig = go.Figure()
    for cid, row in scaled.iterrows():
        values = row.values.tolist()
        fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=features + [features[0]], fill='toself', name=f'Cluster {cid}'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Cluster centroid comparison (normalized)')
    return fig
