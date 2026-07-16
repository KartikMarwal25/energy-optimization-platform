import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import settings

def _save_fig(fig, name: str):
    path = os.path.join(settings.FIG_DIR)
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, f"{name}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

def distribution_plot(df: pd.DataFrame, col: str = "consumption"):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df[col].dropna(), bins=50, ax=ax)
    ax.set_title(f"Distribution of {col}")
    _save_fig(fig, f"dist_{col}")

def correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    corr = numeric_df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax,
    )

    ax.set_title("Correlation Matrix")

    _save_fig(fig, "corr_matrix")

def hourly_consumption(df: pd.DataFrame):
    if "hour" not in df.columns:
        return

    s = df.groupby("hour")["consumption"].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    s.plot(kind='bar', ax=ax)
    ax.set_title('Average Hourly Consumption')
    _save_fig(fig, 'hourly_consumption')

def daily_consumption(df: pd.DataFrame):
    try:
        import plotly.express as px
        import plotly.io as pio
    except Exception:
        # fallback to saved png
        s = df.groupby(df['timestamp'].dt.date)['consumption'].sum()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(s.index, s.values)
        ax.set_title('Daily Consumption')
        _save_fig(fig, 'daily_consumption')
        return
    s = df.groupby(df['timestamp'].dt.date)['consumption'].sum()
    fig = px.line(x=s.index, y=s.values, title='Daily Consumption')
    os.makedirs(settings.FIG_DIR, exist_ok=True)
    pio.write_html(fig, os.path.join(settings.FIG_DIR, 'daily_consumption.html'), auto_open=False)

def monthly_consumption(df: pd.DataFrame):
    try:
        import plotly.express as px
        import plotly.io as pio
    except Exception:
        df2 = df.copy()
        df2['month_year'] = df2['timestamp'].dt.to_period('M').astype(str)
        s = df2.groupby('month_year')['consumption'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(s['month_year'], s['consumption'])
        ax.set_title('Monthly Consumption')
        plt.xticks(rotation=45)
        _save_fig(fig, 'monthly_consumption')
        return
    df2 = df.copy()
    df2['month_year'] = df2['timestamp'].dt.to_period('M').astype(str)
    s = df2.groupby('month_year')['consumption'].sum().reset_index()
    fig = px.bar(s, x='month_year', y='consumption', title='Monthly Consumption')
    pio.write_html(fig, os.path.join(settings.FIG_DIR, 'monthly_consumption.html'), auto_open=False)

def box_violin_plots(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12,6))
    if {"hour", "consumption"}.issubset(df.columns):
        sns.boxplot(
            x="hour",
            y="consumption",
            data=df,
            ax=ax
        )
    ax.set_title('Hourly Consumption Boxplot')
    _save_fig(fig, 'hourly_boxplot')
    try:
        import plotly.express as px
        import plotly.io as pio
        fig2 = px.violin(df, x='weekday', y='consumption', box=True, title='Weekday Consumption Violin')
        pio.write_html(fig2, os.path.join(settings.FIG_DIR, 'weekday_violin.html'), auto_open=False)
    except Exception:
        pass

def generate_all_plots(df: pd.DataFrame):
    # Check required columns
    required = ["consumption"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Generate plots
    distribution_plot(df)
    correlation_heatmap(df)
    hourly_consumption(df)
    daily_consumption(df)
    monthly_consumption(df)
    box_violin_plots(df)
