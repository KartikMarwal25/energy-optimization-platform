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

                # Compare the forecast to the equivalent recent historical
                # window, not to the entire customer history.
                historical_window_total = None
                if consumer is not None and 'meter_id' in self.df.columns:
                    hist = self.df[self.df['meter_id'].astype(str) == str(consumer)].copy()
                    if not hist.empty and {'consumption', 'timestamp'} <= set(hist.columns):
                        hist['date'] = pd.to_datetime(hist['timestamp'], errors='coerce').dt.date
                        daily = hist.dropna(subset=['date']).groupby('date')['consumption'].sum().sort_index()
                        historical_window_total = float(daily.tail(len(df_fc)).sum())
                if historical_window_total and historical_window_total > 0 and total_fc > historical_window_total * 1.15:
                    recs.append({
                        'issue': f'High forecasted consumption for {consumer}',
                        'suggestion': 'Investigate load shifting or temporary demand response during forecasted peak period',
                        'estimated_monthly_savings': round((total_fc - historical_window_total) * 0.1, 2),
                        'meter_id': consumer,
                        'evidence': f'Forecast is {(total_fc / historical_window_total - 1):.0%} above the previous {len(df_fc)}-day total.',
                        'confidence': 'Medium',
                        'next_step': 'Validate the forecast against planned operations before scheduling demand response.',
                    })

        # From anomalies: one recommendation per meter and issue type, so a meter
        # with many flagged readings appears once, with the count as evidence.
        if anomalies is not None and not anomalies.empty:
            a = anomalies.copy()
            if 'meter_id' not in a.columns:
                a['meter_id'] = 'unknown'
            a['meter_id'] = a['meter_id'].astype(str)
            a['ts'] = pd.to_datetime(a['timestamp'], errors='coerce') if 'timestamp' in a.columns else pd.NaT
            medians = pd.Series(dtype=float)
            if 'meter_id' in self.df.columns and 'consumption' in self.df.columns:
                medians = self.df.groupby(self.df['meter_id'].astype(str), observed=True)['consumption'].median()
            a['typical'] = a['meter_id'].map(medians)
            cons = pd.to_numeric(a['consumption'], errors='coerce') if 'consumption' in a.columns else pd.Series(np.nan, index=a.index)
            a['cons'] = cons
            ratio = cons / a['typical'].replace(0, np.nan)
            a['kind'] = np.where(ratio > 1.5, 'spike', np.where(ratio < 0.5, 'low', 'other'))
            # Daily data has one midnight timestamp per reading, so hour-of-day says nothing there.
            subdaily = 'hour' in self.df.columns and self.df['hour'].nunique() > 1
            texts = {
                'spike': ('Consumption spike for {m}', 'High spike observed; check for short-term high-load events or mis-metering.'),
                'low': ('Unusually low consumption for {m}', 'Possible outage or meter reporting issue; verify connectivity and recent maintenance.'),
                'other': ('Anomaly detected for {m}', 'Inspect meter/installation for faults or unusual behavior.'),
            }
            anomaly_recs = []
            for (mid, kind), g in a.groupby(['meter_id', 'kind']):
                issue, suggestion = texts[kind]
                n = len(g)
                parts = [f'{n} flagged reading{"s" if n != 1 else ""}']
                if g['ts'].notna().any():
                    parts[0] += f' between {g["ts"].min():%d %b %Y} and {g["ts"].max():%d %b %Y}'
                typical = g['typical'].iloc[0]
                if g['cons'].notna().any() and pd.notna(typical) and typical > 0:
                    extreme = g['cons'].max() if kind != 'low' else g['cons'].min()
                    parts.append(f'{"peak" if kind != "low" else "lowest"} {extreme:.2f} vs typical {typical:.2f} ({extreme / typical:.1f}x)')
                if subdaily and g['ts'].notna().any() and (g['ts'].dt.hour <= 6).mean() > 0.5:
                    suggestion += ' Most flagged readings fall at night; check overnight processes or unauthorized consumption.'
                anomaly_recs.append({
                    'issue': issue.format(m=mid),
                    'suggestion': suggestion,
                    'estimated_monthly_savings': 0.0,
                    'meter_id': mid,
                    'flagged_readings': n,
                    'evidence': '; '.join(parts) + '. Flagged by anomaly detection; this is not a fault diagnosis.',
                    'confidence': 'Review required',
                    'next_step': 'Compare against maintenance records and meter telemetry before taking action.',
                })
            anomaly_recs.sort(key=lambda r: -r['flagged_readings'])
            recs.extend(anomaly_recs[:20])

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
                    'segment': s,
                    'evidence': f'{seg_counts[s]:,} consumers belong to this usage segment.',
                    'confidence': 'Medium',
                    'next_step': 'Validate segment economics and customer eligibility before launching an incentive.',
                })

        if not recs:
            recs.append({'issue': 'No significant recommendations', 'suggestion': 'No actions identified from forecasts, anomalies, or segments.', 'estimated_monthly_savings': 0.0, 'evidence': 'No configured rule threshold was exceeded.', 'confidence': 'High', 'next_step': 'Continue monitoring and rerun after new readings arrive.'})

        return pd.DataFrame(recs)
