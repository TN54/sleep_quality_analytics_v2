# Sleep Quality Analytics

A small Python project for analyzing sleep data and predicting "Quality of Sleep" using machine learning. This repository includes a Streamlit UI for interactive exploration and a FastAPI backend that exposes a prediction endpoint.

## Project Overview

- Data: `data/Sleep_health_and_lifestyle_dataset.csv` — contains demographic, lifestyle, and sleep-related measurements.
- Models:
  - `sleep_quality_model` (regressor) — predicts the numeric Quality of Sleep (1-10).
  - `sleep_rf_model` (classifier) — historically used for Sleep Disorder classification (may be unused depending on current setup).
- Artifacts: persisted models, scalers, and encoders are stored in `sleep_quality_analytics/artifacts/`.
- UI: `streamlit_app.py` provides an interactive dashboard to explore data, run predictions, and view charts.
- API: `fastapi_app.py` exposes `/predict` and `/recommend` endpoints for programmatic access.

## Goals & Design Principles

- Keep training separate from inference where possible.
- Avoid data leakage: fits (scalers/encoders) must be done only on training data, not on the full dataset prior to splitting.
- Persist model artifacts (model, scaler, encoders) via `joblib` for consistent inference.

## Quick Setup

1. Create a virtual environment (Python 3.8+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Verify dataset exists at `data/Sleep_health_and_lifestyle_dataset.csv`.

## Running the Streamlit UI

This app provides the simplest way to retrain the model interactively and inspect predictions.

```bash
# From project root
streamlit run streamlit_app.py
```

- Open the browser at the URL printed by Streamlit (typically `http://localhost:8501`).
- Use "Sleep Disorder Module" (or "Predict Sleep Quality") to run the model training/prediction flow and interactive charts.

Notes:
- The Streamlit flow may trigger model training in interactive mode. This is intended for exploration and not recommended for production.

## Running the FastAPI Server

Start the API locally:

```bash
uvicorn fastapi_app:app --reload
```

Example request to `/predict` (replace values as needed):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 30,
    "Sleep_Duration": 7.5,
    "Physical_Activity_Level": 50,
    "Stress_Level": 4,
    "BMI_Category": "Normal"
  }'
```

Response (example):

```json
{"predicted_quality": 7.12}
```

Important: depending on the current project configuration, the API may require model artifacts to exist before `/predict` accepts requests (no automatic training at import). If the server returns a 503 stating artifacts are missing, follow the "Training / Producing artifacts" instructions below.

## Training / Producing Artifacts

Two common ways to produce model artifacts:

1. Use the Streamlit UI: interactively run the training flow using the UI's training action. The training functions are designed to persist model, scaler, and encoders in `sleep_quality_analytics/artifacts/` when called with the `save_artifacts=True` flag.

2. From a Python shell or script (non-interactive). Example:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
from sleep_quality_analytics.preprocessing import fill_missing_sleep_disorder, process_blood_pressure
from sleep_quality_analytics.models import get_or_train_quality_model

root = Path(__file__).resolve().parent
csv = root / 'data' / 'Sleep_health_and_lifestyle_dataset.csv'

# Load and apply minimal preprocessing (same as used for training)
df = pd.read_csv(csv)
df = fill_missing_sleep_disorder(df)
df = process_blood_pressure(df)

# Train and save artifacts (force_retrain=True ensures re-fitting)
get_or_train_quality_model(df, categorical_cols=['Gender', 'Occupation', 'BMI Category'], force_retrain=True)
print('Training complete — artifacts saved to sleep_quality_analytics/artifacts/')
PY
```

This will create or overwrite the files in `sleep_quality_analytics/artifacts/` such as:

- `sleep_quality_model.joblib`
- `sleep_quality_scaler.joblib`
- `sleep_quality_encoders.joblib`

Once artifacts are present, the FastAPI `/predict` endpoint will be able to load them for inference.

## Project Structure

```
fastapi_app.py
streamlit_app.py
requirements.txt
sleep_quality_analytics/
  ├─ __init__.py
  ├─ models.py    # model training / loading / prediction helpers
  ├─ preprocessing.py    # safe preprocessing helpers
  ├─ data_loader.py
  ├─ dashboard.py        # plotting + recommendations
  ├─ outlier_detection.py
  └─ artifacts/          # persisted model artifacts (joblib files)
```

## Avoiding Data Leakage (important)

- Do NOT fit scalers or encoders on the entire dataset before splitting into train/test. The correct sequence is:
  1. Split raw data into train/test (e.g., `train_test_split`).
  2. Fit LabelEncoders on `X_train` categorical columns and persist those encoders.
  3. Fit `StandardScaler` on `X_train` numeric features only.
  4. Transform `X_test` using the fitted encoders and scaler.

- The training functions in `models.py` follow this pattern. Avoid using helper functions that call `fit` on the entire dataframe for any production training path.

## Testing

- Basic test: ensure that predictions work after generating artifacts (see Training section). Use `curl` or `httpie` to POST to `/predict`.
- Unit tests: consider adding tests to ensure no `encoder.fit()` or `scaler.fit()` calls are executed on full dataset paths used in production.

## Troubleshooting

- ValueError about feature names mismatch: this means the DataFrame passed to the scaler/transform step does not have the same columns as the DataFrame used at fit time. Ensure your prediction rows include the same features (names and order) as used during training.
- If `/predict` returns 503: artifacts are missing — run the training flow (Streamlit or the non-interactive script above) to produce artifacts.

## Notes & Next Improvements

- Move training out of interactive or import-time code into a dedicated training script (`scripts/train_models.py`) and call it explicitly.
- Add unit tests and CI checks to guard against accidental leakage.
- Add a small management endpoint (protected) to trigger retraining in controlled environments if desired.

## License & Credits

This project is provided as-is for demonstration and learning. Cite or reuse components as needed.

---

If you want, I can also:
- Add a `scripts/train_models.py` convenience script to produce artifacts,
- Add example `curl` or Python test scripts that call the API,
- Create a short checklist for deployment (dockerfile / uvicorn setup).

Tell me which of those you'd like next and I'll implement it.
