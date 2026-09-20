import pandas as pd

from src.anomaly_detection.detectors import AnomalyDetector
from src.clustering.cluster import ClusterEngine
from src.forecasting.models import ModelManager
from src.recommendation.engine import RecommendationEngine


def test_small_datasets_are_supported_by_analysis_engines():
    df = pd.DataFrame({
        'meter_id': ['a', 'b'],
        'timestamp': pd.to_datetime(['2021-01-01', '2021-01-02']),
        'consumption': [1.0, 2.0],
    })

    clusters = ClusterEngine(df).run_kmeans()
    assert clusters['cluster'].tolist() == [0, 0]
    assert AnomalyDetector(df).local_outlier_factor().empty


def test_forecast_returns_requested_horizon():
    df = pd.DataFrame({
        'meter_id': ['a'] * 4,
        'timestamp': pd.date_range('2021-01-01', periods=4, freq='D'),
        'consumption': [1.0, 2.0, 3.0, 4.0],
    })

    forecast = ModelManager(df).forecast_consumer('a', horizon_days=7)
    assert len(forecast) == 7
    assert {'date', 'forecast', 'lower', 'upper'} <= set(forecast.columns)

    mean_forecast = ModelManager(df).forecast_consumer('a', horizon_days=3, method='mean')
    assert mean_forecast['forecast'].nunique() == 1


def test_recommendations_compare_like_for_like_forecast_windows():
    df = pd.DataFrame({
        'meter_id': ['a'] * 4,
        'timestamp': pd.date_range('2021-01-01', periods=4, freq='D'),
        'consumption': [10.0, 10.0, 10.0, 10.0],
    })
    forecasts = {'a_2_linear_trend': pd.DataFrame({'date': pd.date_range('2021-01-05', periods=2), 'forecast': [20.0, 20.0]})}

    recommendations = RecommendationEngine(df).generate_from_forecast_and_anomalies(forecasts=forecasts)
    assert 'High forecasted consumption for a' in recommendations['issue'].tolist()
    assert recommendations.loc[0, 'confidence'] == 'Medium'
