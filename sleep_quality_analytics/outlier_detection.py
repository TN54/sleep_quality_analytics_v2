# outlier_detection.py
# This module handles detecting outliers using the IQR method.
# It includes:
# 1. Function to detect outliers for one column
# 2. Function to detect outliers for all numeric columns
# 3. Function to visualize the outliers (boxplot + histogram)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a specific numeric column using the IQR method.
    Params:
        df: pandas DataFrame
        column: column name (string)
    Returns:
        outliers: DataFrame containing the outlier rows
        lower_bound: lower limit for outliers
        upper_bound: upper limit for outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def detect_outliers_for_all(df, numeric_cols):
    """
    Detect outliers for all numeric columns.
    Returns a dictionary with column names and number of outliers.
    """
    outlier_summary = {}
    for col in numeric_cols:
        outliers, low, high = detect_outliers_iqr(df, col)
        outlier_summary[col] = len(outliers)
    return outlier_summary


def show_outlier_details(df, column):
    """
    Print detailed outlier information and plot visualizations.
    Shows:
        - Lower and upper thresholds
        - Number of outliers
        - The actual outlier rows
        - Boxplot
        - Histogram
    """
    outliers, low, high = detect_outliers_iqr(df, column)

    print(f"\n📌 Column: {column}")
    print(f"Lower Bound: {low}")
    print(f"Upper Bound: {high}")
    print(f"Number of Outliers: {len(outliers)}")
    print("\nOutlier Rows:")
    #display(outliers)

    # Plot Histogram
    plt.figure(figsize=(7,4))
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f'Histogram of {column}')
    plt.show()


# # See number of outliers in all columns
# detect_outliers_for_all(df, numeric_cols)

# # See details for a single column
# show_outlier_details(df, 'Sleep Duration')
