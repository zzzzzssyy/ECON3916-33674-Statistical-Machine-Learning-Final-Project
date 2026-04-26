import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Netflix Watch Time Predictor", layout="wide")

st.title("Netflix Watch Time Predictor")
st.markdown("Predict how many hours a user is likely to watch based on their profile.")

st.warning(
    "**Model Limitation:** This model achieves R² ≈ 0.00 on held-out data, meaning the available "
    "features (age, country, subscription type, genre) carry almost no predictive signal for watch time. "
    "Predictions are shown for demonstration purposes only. The naive baseline (mean watch time) "
    "is displayed alongside each prediction for honest comparison."
)

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

try:
    model, feature_cols = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("model.pkl not found. Run the notebook to generate it first.")
    model_loaded = False

TRAIN_MEAN = 245.0
TRAIN_RMSE = 285.0

COUNTRIES = [
    "Australia", "Brazil", "Canada", "France", "Germany",
    "India", "Japan", "Mexico", "Spain", "UK", "USA"
]
SUBSCRIPTION_TYPES = ["Basic", "Premium", "Standard"]
GENRES = ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]

# ── Sidebar inputs ───────────────────────────────────────────────────────────
st.sidebar.header("User Profile")
age = st.sidebar.slider("Age", min_value=18, max_value=80, value=30)
country = st.sidebar.selectbox("Country", COUNTRIES)
subscription = st.sidebar.selectbox("Subscription Type", SUBSCRIPTION_TYPES)
genre = st.sidebar.selectbox("Favorite Genre", GENRES)

# ── Build input dataframe ────────────────────────────────────────────────────
raw = pd.DataFrame({
    "Age": [age],
    "Country": [country],
    "Subscription_Type": [subscription],
    "Favorite_Genre": [genre],
})
X_input = pd.get_dummies(raw, drop_first=True)

# Align to training columns
for col in feature_cols:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_cols]

# ── Predict ──────────────────────────────────────────────────────────────────
if model_loaded:
    prediction = model.predict(X_input)[0]
    lower = max(0, prediction - 1.96 * TRAIN_RMSE)
    upper = prediction + 1.96 * TRAIN_RMSE

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Watch Time", f"{prediction:.1f} hrs")
    col2.metric("Naive Baseline (mean)", f"{TRAIN_MEAN:.1f} hrs")
    col3.metric("95% Prediction Interval", f"[{lower:.0f}, {upper:.0f}] hrs")

    # ── Visualization ─────────────────────────────────────────────────────────
    st.subheader("Prediction vs. Baseline")
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(
        ["Naive Baseline", "Model Prediction"],
        [TRAIN_MEAN, prediction],
        color=["#aec6cf", "#4a90d9"],
        height=0.4
    )
    ax.errorbar(
        x=prediction, y=1,
        xerr=[[prediction - lower], [upper - prediction]],
        fmt="none", color="black", capsize=5, linewidth=1.5
    )
    ax.set_xlabel("Watch Time (hours)")
    ax.set_title("Predicted Watch Time with 95% Interval")
    for bar, val in zip(bars, [TRAIN_MEAN, prediction]):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} hrs", va="center", fontsize=10)
    ax.set_xlim(0, max(TRAIN_MEAN, upper) * 1.15)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "The 95% prediction interval is computed as prediction ± 1.96 × RMSE (test set). "
        "Wide intervals reflect the model's low predictive accuracy."
    )

    st.subheader("Your Input Summary")
    st.dataframe(pd.DataFrame({
        "Age": [age],
        "Country": [country],
        "Subscription Type": [subscription],
        "Favorite Genre": [genre],
    }), use_container_width=True)
