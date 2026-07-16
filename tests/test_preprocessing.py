import pandas as pd
from src.preprocessing.processor import preprocess_pipeline

def test_preprocess_pipeline_minimal():
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2021-01-01', periods=10, freq='H'),
        'consumption': [1,2,3,4,5,6,7,8,9,10]
    })
    processed = preprocess_pipeline(df)
    assert 'hour' in processed.columns
    assert 'lag_1' in processed.columns
    assert processed['consumption'].isnull().sum() == 0
