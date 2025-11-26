# dashboard.py
# This module handles visualizations and personal recommendations for Sleep Analytics
# Includes:
# 1. Histograms for numeric columns
# 2. Boxplots for numeric columns
# 4. Simple recommendation system based on user's sleep data

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sleep_quality_analytics.data_loader import load_data

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_PATH = _PACKAGE_ROOT.parent / "data" / "Sleep_health_and_lifestyle_dataset.csv"
df = load_data(_DATA_PATH)


# 1️⃣ Histogram function
def plot_histogram(df, column):
    """
    Plot histogram for a numeric column.
    Params:
        df: pandas DataFrame
        column: column name (string)
    """
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 3️⃣ Pie chart function for categorical columns
def plot_pie(df, column):
    """
    Plot pie chart for a categorical column.
    Params:
        df: pandas DataFrame
        column: column name (string)
    """
    counts = df[column].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of {column}')
    plt.show()

# 4️⃣ Recommendation function
def sleep_recommendation(user_data):
    """
    Generate simple sleep recommendations based on user's data.
    Params:
        user_data: dict with keys:
            'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level'
    Returns:
        List of recommendations (strings)
    """
    recommendations = []

    # Sleep Duration
    if user_data['Sleep Duration'] < 7:
        recommendations.append("Try to sleep at least 7 hours per night.")
    elif user_data['Sleep Duration'] > 9:
        recommendations.append("Avoid oversleeping; maintain a consistent schedule.")

    # Quality of Sleep
    if user_data['Quality of Sleep'] < 6:
        recommendations.append("Consider improving sleep environment and reduce caffeine or screen time before bed.")

    # Physical Activity
    if user_data['Physical Activity Level'] < 50:
        recommendations.append("Increase daily physical activity to improve sleep quality.")

    # Stress Level
    if user_data['Stress Level'] > 6:
        recommendations.append("Practice relaxation techniques to reduce stress before bedtime.")

    if not recommendations:
        recommendations.append("Your sleep and lifestyle habits are good. Keep maintaining them!")

    return recommendations


