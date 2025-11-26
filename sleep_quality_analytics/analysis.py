# analysis.py
# This module handles visualization and correlation analysis
# 1. Plot correlation heatmap for numerical columns
# 2. Display descriptive statistics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sleep_quality_analytics.data_loader import load_data

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_PATH = _PACKAGE_ROOT.parent / "data" / "Sleep_health_and_lifestyle_dataset.csv"
df = load_data(_DATA_PATH)

def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for numerical columns.
    Params:
        df: pandas DataFrame
    """
    plt.figure(figsize=(10,8))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

def descriptive_statistics(df):
    """
    Display descriptive statistics for all columns.
    Params:
        df: pandas DataFrame
    """
    print("\nDescriptive Statistics:")
    print(df.describe(include='all').T)


