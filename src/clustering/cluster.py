import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np

class ClusterEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _numeric(self):
        return self.df.select_dtypes(include=['number']).fillna(0)

    def run_kmeans(self, max_k: int = 6, sample_size: int = 10000) -> pd.DataFrame:
        X = self._numeric()
        n = len(X)
        if n == 0:
            return self.df.assign(cluster=pd.Series(dtype="int64"))
        if n < 3:
            res = self.df.copy()
            res['cluster'] = 0
            return res
        # choose a sample for silhouette scoring to avoid OOM on large data
        if n > sample_size:
            sample = X.sample(n=sample_size, random_state=42)
        else:
            sample = X

        best_k = 2
        best_score = -1
        upper_k = min(max_k, len(sample) - 1)
        for k in range(2, upper_k + 1):
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
            km.fit(sample)
            labels = km.predict(sample)
            try:
                score = silhouette_score(sample, labels)
            except Exception:
                score = -1
            if score > best_score:
                best_score = score
                best_k = k

        # fit on full data using MiniBatchKMeans for memory efficiency
        final_km = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=4096)
        final_km.fit(X)
        res = self.df.copy()
        res['cluster'] = final_km.predict(X)
        return res

    def gaussian_mixture(self, n_components: int = 3) -> pd.DataFrame:
        X = self._numeric()
        gm = GaussianMixture(n_components=n_components, random_state=42).fit(X)
        res = self.df.copy()
        res['gmm_cluster'] = gm.predict(X)
        return res

    def name_clusters(self, clusters_df: pd.DataFrame, consumption_col: str = 'consumption') -> pd.DataFrame:
        """Map numeric cluster ids to human-readable segment names and descriptions.

        Strategy:
        - If `consumption_col` exists, use cluster mean consumption to order clusters and label
          them Low/Medium/High usage.
        - Otherwise, use the sum of numeric feature means as a proxy.

        Adds `segment` and `segment_description` columns to a copy of `clusters_df`.
        """
        df = clusters_df.copy()
        if 'cluster' not in df.columns:
            return df

        numeric = df.select_dtypes(include=['number'])
        # compute centroid stats per cluster
        # compute centroids using numeric columns only to avoid aggregation on categorical dtypes
        numeric = df.select_dtypes(include=['number'])
        if numeric.shape[1] == 0:
            # nothing numeric to aggregate; fall back to cluster size ordering
            counts = df.groupby('cluster').size()
            score = counts
            centroids = pd.DataFrame({'size': counts})
        else:
            centroids = numeric.groupby(df['cluster']).mean()
            # choose a score for ordering
            if consumption_col in centroids.columns:
                score = centroids[consumption_col]
            else:
                score = centroids.sum(axis=1)

        # rank clusters by score
        ordered = score.sort_values()
        mapping = {}
        # assign simple human-friendly names
        labels = ["Low usage", "Medium-low usage", "Medium usage", "Medium-high usage", "High usage", "Very high usage"]
        descriptions = {
            "Low usage": "Consistently low electricity consumption.",
            "Medium-low usage": "Lower-than-average consumption with occasional peaks.",
            "Medium usage": "Average consumption patterns.",
            "Medium-high usage": "Above-average consumption, potential for optimization.",
            "High usage": "High consumption; candidate for demand-side measures.",
            "Very high usage": "Very high and/or highly variable consumption; priority for investigation."
        }

        for i, cid in enumerate(ordered.index.tolist()):
            label = labels[min(i, len(labels)-1)]
            mapping[cid] = (label, descriptions.get(label, ''))

        df['segment'] = df['cluster'].map(lambda x: mapping.get(x, (f'Cluster {x}', ''))[0])
        df['segment_description'] = df['cluster'].map(lambda x: mapping.get(x, (f'Cluster {x}', ''))[1])
        return df
