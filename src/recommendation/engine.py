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
