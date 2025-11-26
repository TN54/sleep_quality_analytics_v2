
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from sleep_quality_analytics.preprocessing import fill_missing_sleep_disorder, process_blood_pressure
from sleep_quality_analytics.models import get_or_train_quality_model, predict_quality
from sleep_quality_analytics.dashboard import sleep_recommendation

app = FastAPI(title="Sleep Quality Analytics API")

# ----------------------------
# Load and preprocess dataset
# ----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
dataset_path = _PROJECT_ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"

df = pd.read_csv(dataset_path)
df = fill_missing_sleep_disorder(df)
df = process_blood_pressure(df)

# Create a copy for the quality model
processed_df = df.copy()

# Drop 'Sleep Disorder' to avoid data leakage
processed_df = processed_df.drop(columns=['Sleep Disorder'], errors='ignore')

# ----------------------------
# Train or load sleep quality model
# ----------------------------
model, scaler, encoders = get_or_train_quality_model(
    processed_df,
    categorical_cols=['Gender', 'Occupation', 'BMI Category'],
    force_retrain=True,
)

# Default values for categorical columns
default_gender = df['Gender'].mode().iloc[0] if 'Gender' in df.columns else 'Male'
default_occupation = df['Occupation'].mode().iloc[0] if 'Occupation' in df.columns else 'Other'

# ----------------------------
# Define user input schema
# ----------------------------
class UserInput(BaseModel):
    Age: int
    Sleep_Duration: float
    #Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str

# ----------------------------
# Endpoint: Predict Sleep Quality
# ----------------------------
@app.post("/predict_quality")
def predict_sleep_quality(data: UserInput):
    new_user = pd.DataFrame([{
        'Person ID': int(df['Person ID'].max()) + 1 if 'Person ID' in df.columns else 0,
        'Age': data.Age,
        'Sleep Duration': data.Sleep_Duration,
        'Physical Activity Level': data.Physical_Activity_Level,
        'Stress Level': data.Stress_Level,
        'BMI Category': data.BMI_Category,
        'Gender': default_gender,
        'Occupation': default_occupation,
        'Heart Rate': 70,
        'Daily Steps': 7000,
        'Systolic_BP': 130,
        'Diastolic_BP': 85
    }])

    feature_order = [
        col for col in processed_df.columns if col != 'Quality of Sleep'
    ]
    new_user = new_user[feature_order]
    predicted_quality = float(predict_quality(model, scaler, new_user, encoders=encoders)[0])
    return {"predicted_quality": predicted_quality}
# ----------------------------
# Endpoint: Get Recommendations
# ----------------------------
from pydantic import BaseModel

class RecommendationInput(BaseModel):
    predicted_quality: float
    sleep_duration: float
    activity_level: int
    stress_level: int

@app.post("/recommendations")
def get_recommendations(data: RecommendationInput):
    user_dict = {
        "Sleep Duration": data.sleep_duration,
        "Quality of Sleep": data.predicted_quality,
        "Physical Activity Level": data.activity_level,
        "Stress Level": data.stress_level
    }
    recs = sleep_recommendation(user_dict)
    return {"recommendations": recs}
