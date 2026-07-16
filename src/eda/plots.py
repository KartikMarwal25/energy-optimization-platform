import pandas as pd
import os
from config import settings

def _ensure_fig_dir():
    os.makedirs(settings.FIG_DIR, exist_ok=True)

def _save_html(fig, name: str):
    _ensure_fig_dir()
    try:
        import plotly.io as pio
        pio.write_html(fig, os.path.join(settings.FIG_DIR, f"{name}.html"), auto_open=False)
    except Exception:
        # ignore if plotly not available
        pass

def _safe_plotly(fig, name: str):
    _save_html(fig, name)
    return fig

def correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"]) if not df.empty else pd.DataFrame()
    if numeric_df.empty:
        return None
    try:
        import plotly.express as px
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        return _safe_plotly(fig, "corr_matrix")
    except Exception:
        return None

def daily_trend(df: pd.DataFrame):
    if "timestamp" not in df.columns:
        return None
    try:
        import plotly.express as px
        s = df.groupby(df['timestamp'].dt.date)['consumption'].sum().reset_index()
        fig = px.line(s, x='timestamp', y='consumption', title='Daily Consumption')
        return _safe_plotly(fig, 'daily_consumption')
    except Exception:
        return None

def monthly_consumption(df: pd.DataFrame):
    if "timestamp" not in df.columns:
        return None
    try:
        import plotly.express as px
        df2 = df.copy()
        df2['month_year'] = df2['timestamp'].dt.to_period('M').astype(str)
        s = df2.groupby('month_year')['consumption'].sum().reset_index()
        fig = px.bar(s, x='month_year', y='consumption', title='Monthly Consumption')
        return _safe_plotly(fig, 'monthly_consumption')
    except Exception:
        return None

def seasonal_consumption(df: pd.DataFrame):
    if "timestamp" not in df.columns:
        return None
    try:
        import plotly.express as px
        df2 = df.copy()
        df2['month'] = df2['timestamp'].dt.month
        s = df2.groupby('month')['consumption'].mean().reset_index()
        s['month_name'] = s['month'].apply(lambda x: pd.Timestamp(month=x, day=1, year=2000).strftime('%b'))
        fig = px.line(s, x='month_name', y='consumption', title='Seasonal (Monthly) Average Consumption')
        return _safe_plotly(fig, 'seasonal_consumption')
    except Exception:
        return None

def weekday_consumption(df: pd.DataFrame):
    if "timestamp" not in df.columns:
        return None
    try:
        import plotly.express as px
        df2 = df.copy()
        df2['weekday'] = df2['timestamp'].dt.day_name()
        s = df2.groupby('weekday')['consumption'].mean().reindex([
            'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'
        ]).reset_index()
        fig = px.bar(s, x='weekday', y='consumption', title='Average Consumption by Weekday')
        return _safe_plotly(fig, 'weekday_consumption')
    except Exception:
        return None

def holiday_vs_nonholiday(df: pd.DataFrame, holidays: pd.Series = None):
    if "timestamp" not in df.columns:
        return None
    try:
        import plotly.express as px
        df2 = df.copy()
        if holidays is None:
            # try to load from data/raw/uk_bank_holidays.csv
            try:
                hol = pd.read_csv(os.path.join('data','raw','uk_bank_holidays.csv'))
                holidays = pd.to_datetime(hol.iloc[:,0]).dt.date
            except Exception:
                holidays = pd.Series([], dtype='datetime64[ns]')
        df2['date'] = df2['timestamp'].dt.date
        df2['is_holiday'] = df2['date'].isin(holidays)
        s = df2.groupby('is_holiday')['consumption'].mean().reset_index()
        s['label'] = s['is_holiday'].map({True: 'Holiday', False: 'Non-holiday'})
        fig = px.bar(s, x='label', y='consumption', title='Holiday vs Non-holiday Average Consumption')
        return _safe_plotly(fig, 'holiday_vs_nonholiday')
    except Exception:
        return None

def acorn_analysis(df: pd.DataFrame):
    # ACORN info expected in data/raw/acorn_details.csv mapping meter_id or postcode to ACORN
    try:
        import plotly.express as px
        acorn_path = os.path.join('data','raw','acorn_details.csv')
        if not os.path.exists(acorn_path):
            return None
        ac = pd.read_csv(acorn_path)
        # assume ac has columns ['LCLid','acorn_category'] or ['meter_id','acorn']
        cols = ac.columns.tolist()
        id_col = None
        cat_col = None
        for c in cols:
            if c.lower() in ('lclid','meter_id','meter','id'):
                id_col = c
            if 'acorn' in c.lower():
                cat_col = c
        if id_col is None or cat_col is None:
            return None
        df2 = df.merge(ac[[id_col, cat_col]], left_on='meter_id', right_on=id_col, how='left')
        s = df2.groupby(cat_col)['consumption'].mean().reset_index().sort_values('consumption', ascending=False)
        fig = px.bar(s, x=cat_col, y='consumption', title='ACORN Category Average Consumption')
        return _safe_plotly(fig, 'acorn_consumption')
    except Exception:
        return None

def top_consumers(df: pd.DataFrame, top_n: int = 20):
    if 'meter_id' not in df.columns:
        return None
    try:
        import plotly.express as px
        s = df.groupby('meter_id')['consumption'].sum().nlargest(top_n).reset_index()
        fig = px.bar(s, x='meter_id', y='consumption', title=f'Top {top_n} Consumers')
        return _safe_plotly(fig, f'top_{top_n}_consumers')
    except Exception:
        return None

def peak_consumption_dates(df: pd.DataFrame, top_n: int = 10):
    if 'timestamp' not in df.columns:
        return None
    try:
        import plotly.express as px
        s = df.groupby(df['timestamp'].dt.date)['consumption'].sum().nlargest(top_n).reset_index()
        fig = px.bar(s, x='timestamp', y='consumption', title=f'Top {top_n} Peak Consumption Dates')
        return _safe_plotly(fig, 'peak_consumption_dates')
    except Exception:
        return None

def generate_all_plots(df: pd.DataFrame):
    """Generate interactive Plotly figures for EDA and save HTML outputs. Returns dict of figures."""
    required = ["consumption", "timestamp"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    figs = {}
    figs['correlation'] = correlation_heatmap(df)
    figs['daily_trend'] = daily_trend(df)
    figs['monthly'] = monthly_consumption(df)
    figs['seasonal'] = seasonal_consumption(df)
    figs['weekday'] = weekday_consumption(df)
    figs['holiday'] = holiday_vs_nonholiday(df)
    figs['acorn'] = acorn_analysis(df)
    figs['top_consumers'] = top_consumers(df)
    figs['peak_dates'] = peak_consumption_dates(df)
    return figs
