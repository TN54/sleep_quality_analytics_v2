# classification.py
# This module trains a classification model to predict Sleep Disorder
# with NO DATA LEAKAGE.
#
# Steps:
# 1. Prepare X and y
# 2. Split data into train/test sets
# 3. Scale ONLY X_train (fit), THEN transform X_test
# 4. Train RandomForest model
# 5. Evaluate model (Accuracy, F1 Score, Confusion Matrix)
# 6. Provide a safe prediction function for new data

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
# _MODEL_PATH = _ARTIFACT_DIR / "sleep_rf_model.joblib"
# _SCALER_PATH = _ARTIFACT_DIR / "sleep_scaler.joblib"
# _ENCODERS_PATH = _ARTIFACT_DIR / "sleep_label_encoders.joblib"
_MODEL_QUALITY_PATH = _ARTIFACT_DIR / "sleep_quality_model.joblib"
_SCALER_QUALITY_PATH = _ARTIFACT_DIR / "sleep_quality_scaler.joblib"
_QUALITY_ENCODERS_PATH = _ARTIFACT_DIR / "sleep_quality_encoders.joblib"


# def prepare_classification_data(df, target_column='Sleep Disorder'):
#     """
#     Separate features (X) and target label (y).
#     Safe because it only splits columns, no fitting involved.
#     """
#     X = df.drop(target_column, axis=1)
#     y = df[target_column]
#     return X, y


# def train_sleep_disorder_model(df, save_artifacts=False):
#     """
#     Train a RandomForest model to predict Sleep Disorder.
#     NO DATA LEAKAGE:
#         - Train/Test split BEFORE scaling
#         - Fit scaler ONLY on X_train
#         - Transform X_test ONLY using the trained scaler
#     Returns:
#         model: trained classifier
#         scaler: fitted StandardScaler
#         X_test_scaled: transformed test set
#         y_test: ground truth labels
#     """

#     # 1. Prepare features and target
#     X, y = prepare_classification_data(df)

#     # 2. Train/Test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.30,
#         random_state=42
#     )

#     # 3. Scaling AFTER splitting (to prevent data leakage)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)   # Learn from training data ONLY
#     X_test_scaled = scaler.transform(X_test)         # Transform test using fitted scaler

#     # 4. Initialize the RandomForest model
#     model = RandomForestClassifier(
#         n_estimators=150,
#         random_state=42
#     )

#     # Train the model
#     model.fit(X_train_scaled, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test_scaled)

#     # 5. Evaluation
#     print("\nMODEL PERFORMANCE")
#     print("----------------------------------")
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))

#     if save_artifacts:
#         save_model_artifacts(model, scaler)

#     return model, scaler, X_test_scaled, y_test


# def predict_new(model, scaler, new_data_df):
#     """
#     Predict Sleep Disorder for new unseen data.
#     Scaling must use the SAME scaler learned from training.
#     Params:
#         model: trained RandomForestClassifier
#         scaler: fitted StandardScaler
#         new_data_df: 1-row dataframe of features
#     Returns:
#         prediction: predicted class label
#     """
#     new_data_scaled = scaler.transform(new_data_df)
#     prediction = model.predict(new_data_scaled)
#     return prediction


def train_sleep_quality_model(df, target_column='Quality of Sleep', categorical_cols=None, save_artifacts=False):
    """
    Train a RandomForestRegressor to predict Quality of Sleep.
    Returns (model, scaler, X_test_scaled, y_test)
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split first to avoid any leakage from fitting encoders on full data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Fit LabelEncoders on X_train categorical columns (if provided)
    encoders = {}
    if categorical_cols is None:
        categorical_cols = []
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            # transform X_test values using same encoder; unknown values will raise
            X_test[col] = le.transform(X_test[col].astype(str))
            encoders[col] = le

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\nQUALITY MODEL PERFORMANCE")
    print("----------------------------------")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    if save_artifacts:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, _MODEL_QUALITY_PATH)
        joblib.dump(scaler, _SCALER_QUALITY_PATH)
        joblib.dump(encoders, _QUALITY_ENCODERS_PATH)

    return model, scaler, X_test_scaled, y_test, encoders


def predict_quality(model, scaler, new_data_df, encoders=None):
    """Predict quality score for new data rows. Applies encoders to categorical cols if provided."""
    df = new_data_df.copy()
    if encoders is not None:
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

    new_data_scaled = scaler.transform(df)
    prediction = model.predict(new_data_scaled)
    return prediction


def load_quality_artifacts(model_path=_MODEL_QUALITY_PATH, scaler_path=_SCALER_QUALITY_PATH, encoders_path=_QUALITY_ENCODERS_PATH):
    """Load persisted quality model, scaler and encoders (if present)."""
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Saved quality model artifacts not found. Train the quality model first.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = None
    try:
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)
    except Exception:
        encoders = None
    return model, scaler, encoders


def get_or_train_quality_model(df, categorical_cols=None, force_retrain=False):
    """
    Load persisted quality model if available, otherwise train and save one.
    Returns (model, scaler)
    """
    if not force_retrain:
        try:
            model, scaler, encoders = load_quality_artifacts()
            # If encoders were not persisted but we have the dataframe and categorical_cols,
            # create and persist encoders so future loads are consistent.
            if encoders is None and categorical_cols is not None:
                encoders = {}
                for col in categorical_cols:
                    if col in df.columns:
                        le = LabelEncoder()
                        le.fit(df[col].astype(str))
                        encoders[col] = le
                try:
                    joblib.dump(encoders, _QUALITY_ENCODERS_PATH)
                except Exception:
                    pass
            return model, scaler, encoders
        except FileNotFoundError:
            pass

    model, scaler, X_test_scaled, y_test, encoders = train_sleep_quality_model(df, categorical_cols=categorical_cols, save_artifacts=True)
    return model, scaler, encoders
# # # Preprocessing steps
# def save_model_artifacts(model, scaler, model_path=_MODEL_PATH, scaler_path=_SCALER_PATH):
#     """Persist trained model and scaler for later reuse."""
#     _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, model_path)
#     joblib.dump(scaler, scaler_path)


# def save_model_with_encoders(model, scaler, label_encoders, model_path=_MODEL_PATH, scaler_path=_SCALER_PATH, encoders_path=_ENCODERS_PATH):
#     """Persist model, scaler and label encoders together."""
#     _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, model_path)
#     joblib.dump(scaler, scaler_path)
#     joblib.dump(label_encoders, encoders_path)


# def load_model_artifacts(model_path=_MODEL_PATH, scaler_path=_SCALER_PATH):
#     """Load previously trained model/scaler pair."""
#     if not model_path.exists() or not scaler_path.exists():
#         raise FileNotFoundError("Saved model artifacts not found. Train the model first.")
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     # Try to also load encoders if present
#     encoders = None
#     try:
#         if _ENCODERS_PATH.exists():
#             encoders = joblib.load(_ENCODERS_PATH)
#     except Exception:
#         encoders = None
#     return model, scaler, encoders


# def get_or_train_model(df, label_encoders=None, force_retrain=False):
#     """
#     Load a persisted model if available, otherwise train a new one and cache it.
#     """
#     if not force_retrain:
#         try:
#             model, scaler, saved_encoders = load_model_artifacts()
#             # If encoders weren't persisted but we have them from the current run,
#             # persist them so inverse mappings match the loaded model on future runs.
#             if saved_encoders is None and label_encoders is not None:
#                 save_model_with_encoders(model, scaler, label_encoders)
#                 saved_encoders = label_encoders
#             return model, scaler, saved_encoders
#         except FileNotFoundError:
#             pass

#     # Train a new model. If label_encoders provided, save them alongside artifacts.
#     model, scaler, X_test_scaled, y_test = train_sleep_disorder_model(df, save_artifacts=False)

#     if label_encoders is not None:
#         save_model_with_encoders(model, scaler, label_encoders)
#     else:
#         # fallback to saving model/scaler only
#         save_model_artifacts(model, scaler)

#     return model, scaler
