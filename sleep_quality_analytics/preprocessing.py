# preprocessing.py
# This module handles safe preprocessing (NO SCALING here to avoid data leakage)
# Includes:
# 1. Filling missing values
# 2. Encoding categorical columns
# 3. Splitting Blood Pressure

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fill_missing_sleep_disorder(df):
    """
    Fill missing values in 'Sleep Disorder' with 'None'.
    Safe because it does not introduce leakage.
    """
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    return df


def encode_categorical(df):
    """
    Encode categorical columns by creating a separate LabelEncoder for each column.
    Returns (df_encoded, encoders_dict).
    Note: Prefer fitting encoders on training data only. This helper can be used
    for quick encoding but callers should consider fitting on train split.
    """
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def process_blood_pressure(df):
    """
    Split Blood Pressure into systolic and diastolic values.
    """
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df = df.drop('Blood Pressure', axis=1)
    return df

