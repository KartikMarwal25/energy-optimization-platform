import pandas as pd
from src.preprocessing.processor import preprocess_pipeline

def test_preprocess_pipeline_minimal():
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2021-01-01', periods=10, freq='h'),
        'consumption': [1,2,3,4,5,6,7,8,9,10]
    })
    processed = preprocess_pipeline(df, persist=False)
    assert 'hour' in processed.columns
    assert 'lag_1' in processed.columns
    assert processed['consumption'].isnull().sum() == 0
    assert processed['consumption'].tolist() == [1,2,3,4,5,6,7,8,9,10]


def test_lag_features_do_not_cross_meter_boundaries():
    df = pd.DataFrame({
        'meter_id': ['a', 'a', 'b', 'b'],
        'timestamp': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-01', '2021-01-02']),
        'consumption': [10, 20, 100, 200],
    })

    processed = preprocess_pipeline(df, persist=False)
    first_reading_per_meter = processed.groupby('meter_id')['lag_1'].first()
    assert first_reading_per_meter.to_dict() == {'a': 0.0, 'b': 0.0}
