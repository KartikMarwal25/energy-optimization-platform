import pandas as pd
import numpy as np

class RecommendationEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def generate_rule_recommendations(self) -> pd.DataFrame:
        df = self.df.copy()
        recs = []
        # Simple rules
        # High night usage
        night = df[df['hour'].isin([0,1,2,3,4,5,6])]
        if not night.empty and night['consumption'].mean() > df['consumption'].mean()*1.2:
            recs.append({
                'issue':'High night usage',
                'suggestion':'Consider shifting heavy loads to daytime or schedule off-peak charging',
                'estimated_monthly_savings': round((night['consumption'].mean()-df['consumption'].mean())*30,2)
            })

        # Peak load
        peak = df[df['is_peak']==1]
        if not peak.empty and peak['consumption'].mean() > df['consumption'].mean()*1.1:
            recs.append({
                'issue':'Peak hour load',
                'suggestion':'Enable demand response or shift appliances outside peak window',
                'estimated_monthly_savings': round((peak['consumption'].mean()-df['consumption'].mean())*30,2)
            })

        if not recs:
            recs.append({'issue':'No immediate issues detected','suggestion':'Maintain current usage patterns','estimated_monthly_savings':0.0})

        return pd.DataFrame(recs)

    def generate_all(self) -> pd.DataFrame:
        return self.generate_rule_recommendations()

    def generate_from_forecast_and_anomalies(self, forecasts: dict = None, anomalies: pd.DataFrame = None, segments: pd.DataFrame = None) -> pd.DataFrame:
        """Generate recommendations using forecasts (dict of key->df), anomalies dataframe, and optional segments.

        - forecasts: dict where values are forecast DataFrames with 'forecast' column and 'date'.
        - anomalies: DataFrame of anomaly rows (may include meter_id/timestamp/score)
        - segments: DataFrame with `meter_id` and `segment` columns.
        """
        recs = []
        # From forecasts: detect large expected increases
        if forecasts:
            for k, df_fc in forecasts.items():
                try:
                    # compute forecast total and compare to historical per-meter mean if available
                    total_fc = float(df_fc['forecast'].sum())
                except Exception:
                    continue
                # try to extract consumer id from key or df
                consumer = None
                if 'meter_id' in df_fc.columns:
                    consumer = df_fc['meter_id'].astype(str).unique().tolist()
                    consumer = consumer[0] if consumer else None
                else:
                    parts = str(k).split('_')
                    consumer = parts[0] if parts else None

                # estimate historical avg for this consumer if available in self.df
                hist_avg = None
                if consumer is not None and 'meter_id' in self.df.columns:
                    hist = self.df[self.df['meter_id'].astype(str) == str(consumer)]
                    if not hist.empty and 'consumption' in hist.columns:
                        hist_avg = hist['consumption'].sum()
                if hist_avg and hist_avg > 0 and total_fc > hist_avg * 1.15:
                    recs.append({
                        'issue': f'High forecasted consumption for {consumer}',
                        'suggestion': 'Investigate load shifting or temporary demand response during forecasted peak period',
                        'estimated_monthly_savings': round((total_fc - hist_avg) * 0.1, 2),
                        'meter_id': consumer
                    })

        # From anomalies: create tailored suggestions per anomaly row
        if anomalies is not None and not anomalies.empty:
            # compute historical medians per meter for context
            medians = None
            if 'meter_id' in self.df.columns and 'consumption' in self.df.columns:
                medians = self.df.groupby(self.df['meter_id'].astype(str))['consumption'].median()

            for idx, row in anomalies.iterrows():
                mid = row.get('meter_id', None)
                cons = row.get('consumption', None)
                ts = row.get('timestamp', None)
                suggestion = 'Inspect meter/installation for faults or unusual behavior'
                issue = 'Anomaly detected'

                # Heuristic: large spike
                try:
                    if cons is not None and medians is not None and mid is not None:
                        hist_med = medians.get(str(mid), None)
                        if hist_med is not None and cons > hist_med * 1.5:
                            issue = f'Consumption spike for {mid}'
                            suggestion = 'High spike observed; check for short-term high-load events or mis-metering.'
                        elif hist_med is not None and cons < hist_med * 0.5:
                            issue = f'Unusually low consumption for {mid}'
                            suggestion = 'Possible outage or meter reporting issue; verify connectivity and recent maintenance.'
                except Exception:
                    pass

                # Time-based hint
                try:
                    if ts is not None:
                        # try to parse hour
                        import pandas as _pd
                        h = None
                        if not pd.isna(ts):
                            t = _pd.to_datetime(ts, errors='coerce')
                            if not pd.isna(t):
                                h = t.hour
                        if h is not None and h in [0,1,2,3,4,5,6]:
                            suggestion += ' Occurs during night hours — consider checking overnight processes or unauthorized consumption.'
                except Exception:
                    pass

                recs.append({
                    'issue': issue,
                    'suggestion': suggestion,
                    'estimated_monthly_savings': 0.0,
                    'meter_id': mid,
                    'anomaly_index': idx
                })

        # From segments: suggest targeted actions for High usage segments
        if segments is not None and not segments.empty and 'segment' in segments.columns:
            seg_counts = segments['segment'].value_counts()
            # find segments labelled with 'High' in name
            high_segs = [s for s in seg_counts.index if 'High' in str(s) or 'Very' in str(s)]
            for s in high_segs:
                recs.append({
                    'issue': f'Segment {s} shows high consumption',
                    'suggestion': 'Targeted efficiency campaign and time-of-use incentives for this segment',
                    'estimated_monthly_savings': 0.0,
                    'segment': s
                })

        if not recs:
            recs.append({'issue': 'No significant recommendations', 'suggestion': 'No actions identified from forecasts/anomalies/segments', 'estimated_monthly_savings': 0.0})

        return pd.DataFrame(recs)
