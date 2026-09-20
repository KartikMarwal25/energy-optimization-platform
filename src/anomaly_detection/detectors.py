import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

class AnomalyDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _numeric(self) -> pd.DataFrame:
        return self.df.select_dtypes(include=['number']).fillna(0)

    def _sample(self, X: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
        if len(X) > max_rows:
            return X.sample(n=max_rows, random_state=42)
        return X

    def isolation_forest(self, contamination: float = 0.01, max_rows: int = 50000) -> pd.DataFrame:
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        X = self._sample(self._numeric(), max_rows=max_rows)
        preds = model.fit_predict(X)
        res = self.df.loc[X.index].copy()
        res['if_anomaly'] = (preds == -1).astype(int)
        return res[res['if_anomaly'] == 1]

    def local_outlier_factor(self, n_neighbors: int = 20, max_rows: int = 50000) -> pd.DataFrame:
        X = self._sample(self._numeric(), max_rows=max_rows)
        if len(X) < 3:
            return self.df.iloc[0:0].copy()
        lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(X) - 1))
        preds = lof.fit_predict(X)
        res = self.df.loc[X.index].copy()
        res['lof_anomaly'] = (preds == -1).astype(int)
        return res[res['lof_anomaly'] == 1]

    def dbscan(self, eps: float = 0.5, min_samples: int = 5, max_rows: int = 10000) -> pd.DataFrame:
        X = self._sample(self._numeric(), max_rows=max_rows)
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        preds = db.fit_predict(X)
        res = self.df.loc[X.index].copy()
        res['dbscan_label'] = preds
        return res[res['dbscan_label'] == -1]

    def one_class_svm(self, max_rows: int = 20000) -> pd.DataFrame:
        model = OneClassSVM(gamma='auto')
        X = self._sample(self._numeric(), max_rows=max_rows)
        preds = model.fit_predict(X)
        res = self.df.loc[X.index].copy()
        res['ocsvm_anomaly'] = (preds == -1).astype(int)
        return res[res['ocsvm_anomaly'] == 1]

    def run_all(self, max_rows: int = 50000) -> pd.DataFrame:
        iso = self.isolation_forest(max_rows=max_rows)
        lof = self.local_outlier_factor(max_rows=max_rows)
        db = self.dbscan(max_rows=min(max_rows, 10000))
        oc = self.one_class_svm(max_rows=min(max_rows, 20000))
        all_anoms = pd.concat([iso, lof, db, oc]).drop_duplicates()
        return all_anoms
