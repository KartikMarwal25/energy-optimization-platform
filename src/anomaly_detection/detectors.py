import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

class AnomalyDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def isolation_forest(self, contamination: float = 0.01) -> pd.DataFrame:
        model = IsolationForest(contamination=contamination, random_state=42)
        X = self.df.select_dtypes(include=['number']).fillna(0)
        preds = model.fit_predict(X)
        res = self.df.copy()
        res['if_anomaly'] = (preds == -1).astype(int)
        return res[res['if_anomaly'] == 1]

    def local_outlier_factor(self, n_neighbors: int = 20) -> pd.DataFrame:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        X = self.df.select_dtypes(include=['number']).fillna(0)
        preds = lof.fit_predict(X)
        res = self.df.copy()
        res['lof_anomaly'] = (preds == -1).astype(int)
        return res[res['lof_anomaly'] == 1]

    def dbscan(self, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
        X = self.df.select_dtypes(include=['number']).fillna(0)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        preds = db.fit_predict(X)
        res = self.df.copy()
        res['dbscan_label'] = preds
        return res[res['dbscan_label'] == -1]

    def one_class_svm(self) -> pd.DataFrame:
        model = OneClassSVM(gamma='auto')
        X = self.df.select_dtypes(include=['number']).fillna(0)
        preds = model.fit_predict(X)
        res = self.df.copy()
        res['ocsvm_anomaly'] = (preds == -1).astype(int)
        return res[res['ocsvm_anomaly'] == 1]

    def run_all(self) -> pd.DataFrame:
        iso = self.isolation_forest()
        lof = self.local_outlier_factor()
        db = self.dbscan()
        oc = self.one_class_svm()
        all_anoms = pd.concat([iso, lof, db, oc]).drop_duplicates()
        return all_anoms
