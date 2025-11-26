"""Learn about your sleep quality1 – interactive Streamlit experience."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

#from sleep_quality_analytics.models import get_or_train_model, predict_new
from sleep_quality_analytics.data_loader import load_data
from sleep_quality_analytics.dashboard import sleep_recommendation
from sleep_quality_analytics.preprocessing import fill_missing_sleep_disorder, process_blood_pressure


# -----------------------------------------------------------------------------
# Streamlit configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Discover your sleep health at a glance",
    page_icon="💤",
    layout="wide",


)


# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------
DATASET_PATH = Path(__file__).resolve().parent / "data" / "Sleep_health_and_lifestyle_dataset.csv"
CATEGORICAL_COLUMNS = ["Gender", "Occupation", "BMI Category"]
LANDING_TITLE = "Discover your sleep health at a glance"
MODULE_TITLE = "Sleep quality Module"
TIPS_TITLE = "Sleep Tips & Recipes"


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------
def inject_custom_styles() -> None:
    """Apply cohesive Nouf-inspired palette and typography."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #F4F4F4;
                --card: #F4F4F4;
                --card-2: #ffffff;
                --accent: #1D546C;
                --accent-2: #1A3D64;
                --text: #0C2B4E;
                --muted: #1A3D64;
                --white: #ffffff;
                --page-max-width: 1200px;
                --card-radius: 18px;
                --soft-shadow: 0 12px 24px rgba(12,43,78,0.06);
            }
            html, body, .main, .stApp {
                background-color: var(--bg) !important;
                color: var(--text);
                font-family: 'Poppins', 'Segoe UI', 'Helvetica', sans-serif;
            }
            /* center page content and limit width */
            .block-container {
                padding-top: 28px;
                padding-bottom: 40px;
                max-width: var(--page-max-width) !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            h1, h2, h3, h4, h5, h6, p, label, span {
                color: var(--text) !important;
            }
            section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #F4F4F4,
                #1D546C,
                #1A3D64,
                #0C2B4E
            );
            color: #F4F4F4;
            border-right: none;
            padding-top: 30px;
            padding-left: 20px;
            padding-right: 20px;
            box-shadow: 6px 0 20px rgba(28,42,68,0.15);
            min-width: 250px;
            border-radius: 0 20px 20px 0;
        }
            section[data-testid="stSidebar"] .stRadio label {
                font-size: 1.05rem;
                font-weight: 600;
                color: var(--accent);
            }
            .hero-title, .hero-subtitle {
                color: var(--text);
                margin: 0;
            }
            .hero-subtitle { margin-bottom: 0.75rem; display: block; }
            .hero-gap { height: 2.5rem; }
            .option-card {
                background: linear-gradient(180deg, var(--card), var(--card-2));
                border-radius: 26px;
                padding: 28px 26px;
                text-align: center;
                box-shadow: var(--soft-shadow);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }
            .option-card:hover { transform: translateY(-6px); box-shadow: 0 12px 24px rgba(31,42,68,0.06); }
            .recipe-card, .analysis-card, .result-card {
                background: linear-gradient(180deg, #d9e5ff, #ffffff);
                border-radius: var(--card-radius);
                padding: 20px;
                box-shadow: var(--soft-shadow);
                margin-bottom: 20px;

            }
            .result-card {
                border: 1px solid rgba(12,43,78,0.08);
                text-align: center;
                max-width: 380px;
                margin-left: auto;
                margin-right: auto;
            }
            .result-card .label { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
            .result-card .value { font-size: 1.2rem; font-weight: 700; margin-top: 6px; color: var(--accent); }
            .stButton>button {
                background: var(--accent);
                color: var(--white);
                border-radius: 999px;
                border: none;
                padding: 8px 30px;
                font-weight: 600;
                font-size: 0.95rem;
                box-shadow: 0 10px 20px rgba(12,43,78,0.12);
            }
            .stButton>button:hover { background: #3c68a3; }
            .footer-note {
                text-align: center;
                color: var(--muted);
                font-size: 0.9rem;
                margin-top: 2.5rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(31,42,68,0.05);
            }
            .tips-card select {
                background: var(--card);
                border: none;
                padding: 6px 10px;
                border-radius: 12px;
                color: var(--accent);
                font-weight: 600;
            }
            .tips-card, .tips-card p, .tips-card h4, .tips-card label {
                color: var(--accent) !important;
            }
            . .stNumberInput input, .stTextInput input {
                border: 1px solid rgba(47,75,143,0.4);
                background: rgba(217,229,255,0.5);
                color: var(--accent);
            }
            .stSelectbox label, .stSlider label, .stNumberInput label {
                color: var(--accent) !important;
            }
            .stSelectbox div[data-baseweb="select"] {
                border: 1px solid rgba(47,75,143,0.4);
                border-radius: 14px;
                color: var(--accent);
            }
            .stSlider div[data-baseweb="slider"] > div {
                background: var(--card);
            }
            .stSlider div[data-baseweb="slider"] .st-af {
                background: var(--accent);
            }
            .stSlider span[data-baseweb="slider-handle"] {
                background: var(--accent);
                border: 2px solid var(--white);
            }
            /* analysis card specific spacing and responsive grid */
            .analysis-card { display: block; padding: 18px; }
            @media (max-width: 900px) {
                .block-container { padding-left: 12px; padding-right: 12px; }
                .analysis-card { padding: 14px; }
            }
            .footer-note {
            text-align: center;
            color: #1A3D64;
            font-size: 0.9rem;
            margin-top: 2.5rem;
            padding: 1rem 0;
            border-top: 1px solid rgba(28,42,68,0.1);
            position: relative;
            bottom: 0;
            width: 100%;
            background: transparent;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data preparation utilities
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the mandated CSV file using shared data_loader helper."""
    return load_data(DATASET_PATH)


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical columns while preserving encoders for inverse mapping."""
    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders


def preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Run fill and blood-pressure splitting in sequence and drop Sleep Disorder.

    Note: Do NOT fit LabelEncoders here to avoid leakage. Encoders are
    fitted on training data inside the quality-model training function.
    """
    working_df = fill_missing_sleep_disorder(df.copy())
    # Drop Sleep Disorder entirely (focus on Quality of Sleep)
    if 'Sleep Disorder' in working_df.columns:
        working_df = working_df.drop(columns=['Sleep Disorder'])
    working_df = process_blood_pressure(working_df)
    # Return unencoded dataframe; encoders will be created during training
    return working_df, {}


# def train_model(df: pd.DataFrame):
#     """Train RandomForestClassifier with proper scaling order."""
#     # Pass encoders through to ensure the same label mappings are saved with the model
#     # Note: caller should provide encoders via outer scope when needed
#     model, scaler = get_or_train_model(df)
#     return model, scaler


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def plot_histogram(raw_df: pd.DataFrame, column: str, user_value: float) -> None:
    """Plot histogram for a column and overlay the user's value."""
    # Interactive histogram with Plotly
    data = raw_df[column].dropna()
    fig = px.histogram(data_frame=raw_df, x=column, nbins=15, template='simple_white',
                       color_discrete_sequence=['#1D546C'])
    # highlight user value with a vertical line
    fig.add_vline(x=user_value, line=dict(color='#1A3D64', width=3, dash='dash'))
    fig.update_layout(
        plot_bgcolor='#F4F4F4',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=column,
        height=520,
        title={'text': f"{column} distribution", 'font': {'size': 18}},
        xaxis={'title': {'font': {'size': 14}}, 'tickfont': {'size': 12}},
        yaxis={'tickfont': {'size': 12}},
        margin={'t': 60, 'b': 40, 'l': 40, 'r': 20},
    )
    st.plotly_chart(fig, use_container_width=True)
    try:
        pct = (data <= user_value).mean() * 100
        st.caption(f"You are at the {pct:.0f}th percentile — higher means more than most users.")
    except Exception:
        st.caption("Distribution of users for this metric.")


# plot_boxplot removed per user request (Plotly boxplot function deleted)


def plot_pie(raw_df: pd.DataFrame, column: str, user_value=None) -> None:
    """Display an easy-to-read distribution chart.

    - For numeric columns (like Quality of Sleep): show bar counts per score (1-10)
      and highlight the user's score.
    - For categorical columns: show horizontal bar chart with percentages and
      highlight user's category if provided.
    """
    data = raw_df[column].dropna()
    left, right = st.columns([4, 1])
    if pd.api.types.is_numeric_dtype(data):
        counts = data.value_counts().sort_index()
        idx = list(range(int(data.min()), int(data.max()) + 1))
        counts = counts.reindex(idx, fill_value=0)
        perc = counts / counts.sum() * 100
        colors = []
        for v in counts.index:
            if v <= 4:
                colors.append("#ff6b6b")
            elif v <= 7:
                colors.append("#1A3D64")
            else:
                colors.append("#1D546C")
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index.astype(str), color_discrete_sequence=colors)
        fig.update_layout(
            plot_bgcolor='#F4F4F4',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis_title=column,
            height=520,
            title={'text': f"{column} counts", 'font': {'size': 18}},
            xaxis={'tickfont': {'size': 12}, 'title': {'font': {'size': 14}}},
            yaxis={'tickfont': {'size': 12}},
            margin={'t': 60, 'b': 40, 'l': 40, 'r': 20},
        )
        if user_value is not None:
            fig.add_vline(x=user_value, line=dict(color='#1A3D64', width=3, dash='dash'))
            if user_value <= 4:
                caption_text = "Your predicted quality is <strong>Poor</strong> — consider sleep hygiene improvements."
            elif user_value <= 7:
                caption_text = "Your predicted quality is <strong>Average</strong> — some improvements may help."
            else:
                caption_text = "Your predicted quality is <strong>Good</strong> — keep what you do!"
        else:
            caption_text = "Distribution of quality scores in the dataset."
        with left:
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown(f"<div style='font-size:14px; line-height:1.4'>{caption_text}</div>", unsafe_allow_html=True)
    else:
        counts = data.value_counts()
        fig = px.bar(x=counts.values, y=counts.index.astype(str), orientation='h', color_discrete_sequence=['#1D546C'])
        fig.update_layout(
            plot_bgcolor='#F4F4F4',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=520,
            title={'text': f"{column} breakdown", 'font': {'size': 18}},
            xaxis={'tickfont': {'size': 12}},
            yaxis={'tickfont': {'size': 12}},
            margin={'t': 60, 'b': 40, 'l': 40, 'r': 20},
        )
        with left:
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("<div style='font-size:14px; line-height:1.4'>Category breakdown.</div>", unsafe_allow_html=True)


def plot_quality_gauge(predicted_quality: float) -> None:
    """Render a compact horizontal gauge indicating predicted sleep quality.

    Colors match the site palette. Shows a single marker for the predicted value.
    """
    fig, ax = plt.subplots(figsize=(13, 3), dpi=110)
    # segments using site palette
    ax.barh(0, 4, left=0, height=1, color="#ff6b6b", edgecolor="#ffffff")
    ax.barh(0, 3, left=4, height=1, color="#1A3D64", edgecolor="#ffffff")
    ax.barh(0, 3, left=7, height=1, color="#1D546C", edgecolor="#ffffff")
    # marker
    x = max(0.0, min(10.0, predicted_quality))
    ax.plot([x], [0.5], marker="o", color="#1A3D64", markersize=22)
    ax.set_xlim(0, 10)
    ax.set_yticks([])
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Predicted Quality (1-10)", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfcf")
    # render gauge full-width then show description underneath
    st.pyplot(fig)
    if predicted_quality <= 4.0:
        desc = "Predicted quality: <strong>Poor</strong> — consider sleep hygiene improvements."
    elif predicted_quality <= 7.0:
        desc = "Predicted quality: <strong>Average</strong> — some improvements may help."
    else:
        desc = "Predicted quality: <strong>Good</strong> — keep your healthy routines!"
    # display description centered below the figure
    st.markdown(f"<div style='font-size:13px; line-height:1.4; text-align:left'>{desc}</div>", unsafe_allow_html=True)





def plot_correlation_heatmap(df: pd.DataFrame, predicted_quality: float = None) -> None:
    """Show a correlation heatmap between numeric features and Quality of Sleep."""
    numeric_cols = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Systolic_BP",
        "Diastolic_BP",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(16, 12), dpi=120)
    fig.patch.set_facecolor('#F4F4F4')
    ax.set_facecolor('#F4F4F4')
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.7},
        annot_kws={"fontsize": 14},
    )
    ax.set_title("Feature correlations (Quality of Sleep highlighted)", fontsize=20)
    ax.tick_params(axis='x', labelsize=14, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
    fig.tight_layout()
    # render full-width heatmap and put a simple caption below it
    st.pyplot(fig)
    # If predicted_quality is provided, show it under the heatmap using the simpler caption style
    if predicted_quality is not None:
        if predicted_quality <= 4.0:
            q_cat = "Poor"
        elif predicted_quality <= 7.0:
            q_cat = "Average"
        else:
            q_cat = "Good"
        st.caption(f"Predicted quality: {predicted_quality:.2f} — {q_cat}")
    else:
        # fallback: short caption about the heatmap
        st.caption("Correlation matrix for numeric features.")


def plot_scatter_with_user(df: pd.DataFrame, x_col: str, y_col: str, user_x: float, user_y: float) -> None:
    """Scatter plot disabled per user request.

    This function is kept as a no-op to avoid breaking callers elsewhere
    while ensuring no scatter chart is rendered.
    """
    return None


# -----------------------------------------------------------------------------
# Layout sections
# -----------------------------------------------------------------------------
def render_landing_page() -> None:
    """Landing view with two infographic cards."""
    st.markdown(f"<h1 class='hero-title' style='text-align:center;'>Discover your sleep health at a glance</h1>", unsafe_allow_html=True)
    st.markdown(
    "<p class='hero-subtitle' style='text-align:center;'>Guides, insights, and tips — made simple.</p>",
    unsafe_allow_html=True,
)

    st.markdown("<div class='hero-gap'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        if st.button("😴 Sleep quality Module", use_container_width=True):
            st.session_state.active_view = MODULE_TITLE
        st.markdown(
            """
            <div class="option-card">
                <div style="font-size:70px;">😴</div>
                <h3>Sleep quality Module</h3>
                <p>Analyze your sleep patterns See what affects your rest and energy</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("🍵 Sleep Tips & Recipes", use_container_width=True):
            st.session_state.active_view = TIPS_TITLE
        st.markdown(
            """
            <div class="option-card">
                <div style="font-size:70px;">🍵</div>
                <h3>Sleep Tips & Recipes</h3>
                <p>Sleep-friendly drinks, exercises, and scents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
    """
    <style>
    /* Footer fixed at the bottom of the viewport */
    .sticky-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F4F4F4;
        color: #1A3D64;
        text-align: center;
        padding: 15px 0;
        border-top: 1px solid rgba(28,42,68,0.1);
        z-index: 9999; /* on top of everything */
    }
    /* Add bottom padding to the page content so it doesn't overlap the footer */
    .footer-padding {
        padding-bottom: 60px; /* same or slightly larger than footer height */
    }
    </style>

    <div class="footer-padding"></div>  <!-- space to avoid overlap -->
    <div class="sticky-footer">
        Created with care by Nouf
    </div>
    """,
    unsafe_allow_html=True
)



def render_prediction_module() -> None:
    """Sleep quality Module: preprocessing, training, prediction, and charts."""
    st.markdown(
        """
        <h2 style="
            text-align: center;
            margin-bottom: 40px;
            color: #1D546C;
        ">
            Sleep Quality Module
        </h2>
        """,
    unsafe_allow_html=True
)

    try:
        raw_df = load_dataset()
    except FileNotFoundError:
        st.error(f"Dataset file not found on disk. Expected at {DATASET_PATH}.")
        return

    processed_df, _ = preprocess_dataset(raw_df)

    # Train or load the Quality-of-Sleep model (regressor). We use a separate
    # artifacts path so it doesn't collide with the Sleep Disorder classifier.
    from sleep_quality_analytics.models import get_or_train_quality_model, predict_quality

    # get_or_train_quality_model will fit encoders on the training split and
    # return them so we can encode user inputs consistently.
    model, scaler, encoders = get_or_train_quality_model(processed_df, categorical_cols=CATEGORICAL_COLUMNS)

    bmi_options = sorted(raw_df["BMI Category"].dropna().unique())

    st.markdown("### Tell us about yourself")
    c1, c2, c3 = st.columns(3, gap="large")
    c4, c5, c6 = st.columns(3, gap="large")

    with c1:
        age = st.number_input("Age", min_value=10, max_value=90, value=int(raw_df["Age"].median()))
    with c2:
        sleep_duration = st.slider("Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.25)

    with c4:
        activity_level = st.number_input("Physical Activity Level (0-100)", min_value=0, max_value=300, value=50)
    with c5:
        stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    with c6:
        bmi_choice = st.selectbox("BMI Category", bmi_options)

    button_col = st.columns([2, 1, 2])[1]
    with button_col:
        submitted = st.button("Predict Sleep Quality")
    if not submitted:
        return

    # Build template features (processed_df includes the target 'Quality of Sleep', so drop it)
    features_only = processed_df.drop(columns=["Quality of Sleep"]) if "Quality of Sleep" in processed_df.columns else processed_df
    template = features_only.mode().iloc[0].to_dict()
    template.update(
        {
            "Age": age,
            "Sleep Duration": sleep_duration,
            # we don't set 'Quality of Sleep' here — that's the target we predict
            "Physical Activity Level": activity_level,
            "Stress Level": stress_level,
            # encode BMI Category using encoders returned from the training step
            # fallback: if encoders missing, fit a temporary one on raw data
        }
    )

    # Leave categorical values as raw strings; the prediction function will
    # apply encoders (fitted on training data) to user inputs.
    template["BMI Category"] = bmi_choice

    prediction_df = pd.DataFrame([template])[features_only.columns]
    # Use encoders when predicting so categorical columns are handled
    predicted_quality = float(predict_quality(model, scaler, prediction_df, encoders=encoders)[0])

    # Map numeric quality to easy categories
    if predicted_quality <= 4.0:
        quality_cat = "Poor"
    elif predicted_quality <= 7.0:
        quality_cat = "Average"
    else:
        quality_cat = "Good"

    st.markdown(
        f"""
        <div class="result-card">
            <p class="label">Predicted Sleep Quality (1-10)</p>
            <p class="value">{predicted_quality:.2f} — {quality_cat}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("### Personalized Recommendations")
    recs = sleep_recommendation(
        {
            "Sleep Duration": sleep_duration,
            "Quality of Sleep": predicted_quality,
            "Physical Activity Level": activity_level,
            "Stress Level": stress_level,
        }
    )
    for rec in recs:
        st.markdown(f"- ✅ {rec}")

    st.write("### Your analysis in charts")
    # Show two charts side-by-side with captions underneath each
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        # Gauge showing predicted quality
        plot_quality_gauge(predicted_quality)
        st.caption("Gauge: predicted sleep quality (1-10) with category guidance.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        # Correlation heatmap for numeric features
        plot_correlation_heatmap(raw_df, predicted_quality)
        st.caption("Correlation matrix: shows relationships between numeric features and predicted quality.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        /* Footer fixed at the bottom of the viewport */
        .sticky-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #F4F4F4;
            color: #1A3D64;
            text-align: center;
            padding: 15px 0;
            border-top: 1px solid rgba(28,42,68,0.1);
            z-index: 9999; /* on top of everything */
        }
        /* Add bottom padding to the page content so it doesn't overlap the footer */
        .footer-padding {
            padding-bottom: 60px; /* same or slightly larger than footer height */
        }
        </style>

        <div class="footer-padding"></div>  <!-- space to avoid overlap -->
        <div class="sticky-footer">
            Created with care by Nouf
        </div>
        """,
        unsafe_allow_html=True
    )

def render_tips_page() -> None:
    """Sleep tips & recipes cards."""
    st.markdown("## Sleep Tips & Recipes")
    st.caption("Choose a card to reveal curated recipes, rituals, and sensory cues.")

    drink_recipes = {
        "Lavender Moon Milk": {
            "ingredients": ["1 cup almond milk", "1 tsp honey", "1/4 tsp culinary lavender", "Pinch of cinnamon"],
            "instructions": "Heat milk gently, whisk in lavender and cinnamon, sweeten with honey, strain, then sip slowly.",
        },
        "Golden Chamomile Latte": {
            "ingredients": ["1 cup oat milk", "1 chamomile tea bag", "1/4 tsp turmeric", "Fresh ginger slice"],
            "instructions": "Steep chamomile and ginger, stir in turmeric, froth lightly, and enjoy before bedtime.",
        },
        "Cherry Dream Smoothie": {
            "ingredients": ["1 cup tart cherries", "1/2 banana", "1 tbsp oats", "Splash of kefir"],
            "instructions": "Blend all ingredients until silky; cherries boost melatonin while oats balance blood sugar.",
        },
    }

    exercises = {
        "4-7-8 Breathing": {
            "benefits": "Activates the parasympathetic system, slows heart rate, and eases anxious thoughts.",
            "how_to": "Inhale 4 seconds, hold 7, exhale 8. Repeat 4 rounds while seated with an upright spine.",
        },
        "Legs-Up-the-Wall": {
            "benefits": "Improves circulation, drains lymph, and tells your body it’s safe to rest.",
            "how_to": "Lie on your back, extend legs onto a wall for 5 minutes, breathe deeply through the nose.",
        },
        "Slow Flow Stretch": {
            "benefits": "Relieves tech-neck and lower-back tightness from long workdays.",
            "how_to": "Cycle through cat-cow, child’s pose, and seated twists—30 seconds each for two rounds.",
        },
    }

    scents = {
        "Lavender + Sage": {
            "notes": "Classic calming blend that eases nervous system chatter.",
            "usage": "Light a soy candle 30 minutes before bed; pair with journaling or a gratitude list.",
        },
        "Cedarwood + Vanilla": {
            "notes": "Grounding forest notes with cozy vanilla to signal safety.",
            "usage": "Diffuse 3 drops cedarwood + 2 drops vanilla while reading an easy novel.",
        },
        "Chamomile + Bergamot": {
            "notes": "Floral citrus duo ideal for winding down without feeling drowsy.",
            "usage": "Add 5 drops to a bedside reed diffuser or dab onto wool dryer balls for linens.",
        },
    }

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="recipe-card tips-card">🍵 <strong>Sleep-friendly drinks</strong></div>', unsafe_allow_html=True)
        drink_choice = st.selectbox("Choose a drink", list(drink_recipes.keys()), key="drink_select")
        recipe = drink_recipes[drink_choice]
        st.markdown("**Ingredients**")
        for item in recipe["ingredients"]:
            st.markdown(f"- {item}")
        st.markdown("**Instructions**")
        st.markdown(recipe["instructions"])

    with col2:
        st.markdown('<div class="recipe-card tips-card">🧘 <strong>Gentle exercises</strong></div>', unsafe_allow_html=True)
        exercise_choice = st.selectbox("Pick a exercises", list(exercises.keys()), key="exercise_select")
        flow = exercises[exercise_choice]
        st.markdown("**Benefits**")
        st.markdown(flow["benefits"])
        st.markdown("**How to practice**")
        st.markdown(flow["how_to"])

    with col3:
        st.markdown('<div class="recipe-card tips-card">🕯️ <strong>Candles & scents</strong></div>', unsafe_allow_html=True)
        scent_choice = st.selectbox("Select a scent story", list(scents.keys()), key="scent_select")
        scent = scents[scent_choice]
        st.markdown("**Why it helps**")
        st.markdown(scent["notes"])
        st.markdown("**How to use**")
        st.markdown(scent["usage"])
st.markdown(
    """
    <style>
    /* Footer fixed at the bottom of the viewport */
    .sticky-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F4F4F4;
        color: #1A3D64;
        text-align: center;
        padding: 15px 0;
        border-top: 1px solid rgba(28,42,68,0.1);
        z-index: 9999; /* on top of everything */
    }
    /* Add bottom padding to the page content so it doesn't overlap the footer */
    .footer-padding {
        padding-bottom: 60px; /* same or slightly larger than footer height */
    }
    </style>

    <div class="footer-padding"></div>  <!-- space to avoid overlap -->
    <div class="sticky-footer">
        Created with care by Nouf
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Main router
# -----------------------------------------------------------------------------
def main():
    """Handle navigation between landing, predictor, and tips views."""
    inject_custom_styles()

    if "active_view" not in st.session_state:
        st.session_state.active_view = "Home"

    st.sidebar.title("Navigate")
    selection = st.sidebar.radio(
        "Choose a page",
        options=["Home", MODULE_TITLE, TIPS_TITLE],
        index=["Home", MODULE_TITLE, TIPS_TITLE].index(st.session_state.active_view),
    )
    st.session_state.active_view = selection

    if selection == "Home":
        render_landing_page()
    elif selection == MODULE_TITLE:
        render_prediction_module()
    else:
        render_tips_page()


if __name__ == "__main__":
    main()
