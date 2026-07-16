import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class ClusterEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _numeric(self):
        return self.df.select_dtypes(include=['number']).fillna(0)

    def run_kmeans(self, max_k: int = 6) -> pd.DataFrame:
        X = self._numeric()
        best_k = 2
        best_score = -1
        for k in range(2, max_k+1):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        km = KMeans(n_clusters=best_k, random_state=42).fit(X)
        res = self.df.copy()
        res['cluster'] = km.labels_
        return res

    def gaussian_mixture(self, n_components: int = 3) -> pd.DataFrame:
        X = self._numeric()
        gm = GaussianMixture(n_components=n_components, random_state=42).fit(X)
        res = self.df.copy()
        res['gmm_cluster'] = gm.predict(X)
        return res
