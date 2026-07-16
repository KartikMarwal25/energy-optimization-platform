import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class ClusterEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _numeric(self):
        return self.df.select_dtypes(include=['number']).fillna(0)

    def run_kmeans(self, max_k: int = 6, sample_size: int = 10000) -> pd.DataFrame:
        X = self._numeric()
        n = len(X)
        # choose a sample for silhouette scoring to avoid OOM on large data
        if n > sample_size:
            sample = X.sample(n=sample_size, random_state=42)
        else:
            sample = X

        best_k = 2
        best_score = -1
        for k in range(2, max_k+1):
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
