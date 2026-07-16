from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Table, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os

class Database:
    def __init__(self, path: str = "data/processed/database.sqlite"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", echo=False)
        self.meta = MetaData()
        self.Session = sessionmaker(bind=self.engine)

    def save_predictions(self, df: pd.DataFrame, table_name: str = "predictions"):
        df.to_sql(table_name, self.engine, if_exists="append", index=False)

    def save_recommendations(self, df: pd.DataFrame, table_name: str = "recommendations"):
        df.to_sql(table_name, self.engine, if_exists="append", index=False)

    def query_table(self, table_name: str) -> pd.DataFrame:
        try:
            return pd.read_sql_table(table_name, self.engine)
        except Exception:
            return pd.DataFrame()

    def save_model_metadata(self, metadata: dict, table_name: str = "models"):
        df = pd.DataFrame([metadata])
        df.to_sql(table_name, self.engine, if_exists="append", index=False)

    def save_log(self, entry: dict, table_name: str = "logs"):
        df = pd.DataFrame([entry])
        df.to_sql(table_name, self.engine, if_exists="append", index=False)
