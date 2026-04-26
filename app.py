import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Netflix Watch Time Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f0f0f; color: #e5e5e5; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #e50914;
    }
    [data-testid="stSidebar"] * { color: #e5e5e5 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { color: #e50914 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #aaa !important; }

    /* Section headers */
    h1 { color: #e50914 !important; font-family: 'Arial Black', sans-serif; }
    h2, h3 { color: #ffffff !important; }

    /* Warning box */
    .stAlert { border-left: 4px solid #e50914 !important; background: #1a1a1a !important; }

    /* Divider */
    hr { border-color: #333 !important; }

    /* DataFrame */
    [data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 8px; }

    /* Buttons & selects */
    .stSelectbox > div, .stSlider > div { background: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
    <span style="font-size:3rem;">🎬</span>
    <div>
        <h1 style="margin:0; font-size:2.4rem; letter-spacing:2px;">NETFLIX WATCH TIME PREDICTOR</h1>
        <p style="color:#aaa; margin:4px 0 0 0; font-size:1rem;">
            ECON 3916 — Statistical Machine Learning Final Project
        </p>
    </div>
</div>
<hr>
""", unsafe_allow_html=True)

# ── Model limitation banner ───────────────────────────────────────────────────
st.warning(
    "⚠️  **Model Transparency Notice** — This Random Forest achieves R² ≈ 0.00 on held-out data. "
    "The available features (age, country, subscription, genre) carry almost no signal for predicting "
    "watch time in this synthetic dataset. Predictions are shown for **demonstration purposes only**. "
    "The naive baseline (mean watch time) is displayed alongside each prediction for honest comparison."
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

try:
    model, feature_cols = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("❌  `model.pkl` not found. Run the notebook to generate it first.")
    model_loaded = False
    st.stop()

TRAIN_MEAN = 245.0
TRAIN_RMSE = 285.0

COUNTRIES = sorted([
    "Australia", "Brazil", "Canada", "France", "Germany",
    "India", "Japan", "Mexico", "Spain", "UK", "USA"
])
SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium"]
GENRES = sorted(["Action", "Comedy", "Documentary", "Drama",
                 "Horror", "Romance", "Sci-Fi", "Thriller"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 User Profile")
    st.markdown("Adjust the sliders and dropdowns to build a user profile.")
    st.markdown("---")

    age = st.slider("🎂 Age", min_value=18, max_value=80, value=30)
    country = st.selectbox("🌍 Country", COUNTRIES, index=COUNTRIES.index("USA"))
    subscription = st.selectbox("💳 Subscription Type", SUBSCRIPTION_TYPES, index=1)
    genre = st.selectbox("🎭 Favorite Genre", GENRES, index=GENRES.index("Drama"))

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem; color:#888; line-height:1.6;">
        <b>Dataset:</b> Netflix Users Database<br>
        <b>N =</b> 25,000 observations<br>
        <b>Model:</b> Random Forest (50 trees)<br>
        <b>CV R² =</b> ≈ 0.00<br>
        <b>RMSE =</b> ≈ 285 hrs
    </div>
    """, unsafe_allow_html=True)

# ── Build input & predict ─────────────────────────────────────────────────────
raw = pd.DataFrame({
    "Age": [age], "Country": [country],
    "Subscription_Type": [subscription], "Favorite_Genre": [genre],
})
X_input = pd.get_dummies(raw, drop_first=True)
for col in feature_cols:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_cols]

prediction = model.predict(X_input)[0]
lower = max(0, prediction - 1.96 * TRAIN_RMSE)
upper = prediction + 1.96 * TRAIN_RMSE
diff = prediction - TRAIN_MEAN
delta_str = f"{diff:+.1f} hrs vs baseline"

# ── Metrics row ───────────────────────────────────────────────────────────────
st.markdown("### 📊 Prediction Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Predicted Watch Time", f"{prediction:.1f} hrs", delta_str)
c2.metric("📈 Naive Baseline (mean)", f"{TRAIN_MEAN:.1f} hrs")
c3.metric("📉 95% Interval Lower", f"{lower:.0f} hrs")
c4.metric("📉 95% Interval Upper", f"{upper:.0f} hrs")

st.markdown("---")

# ── Two-panel chart ───────────────────────────────────────────────────────────
st.markdown("### 📉 Visual Breakdown")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor("#0f0f0f")

# Left: prediction vs baseline bar chart
ax = axes[0]
ax.set_facecolor("#1a1a1a")
categories = ["Naive Baseline", "Model Prediction"]
values = [TRAIN_MEAN, prediction]
colors = ["#555", "#e50914"]
bars = ax.barh(categories, values, color=colors, height=0.45, zorder=2)
ax.errorbar(x=prediction, y=1,
            xerr=[[prediction - lower], [upper - prediction]],
            fmt="none", color="white", capsize=6, linewidth=2, zorder=3)
for bar, val in zip(bars, values):
    ax.text(val + 3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} hrs", va="center", color="white", fontsize=10)
ax.set_xlim(0, max(TRAIN_MEAN, upper) * 1.2)
ax.set_xlabel("Watch Time (hours)", color="#aaa")
ax.set_title("Prediction vs. Baseline", color="white", fontsize=12, pad=10)
ax.tick_params(colors="#aaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.grid(axis="x", color="#333", linestyle="--", alpha=0.6, zorder=1)

# Right: feature profile radar / bar
ax2 = axes[1]
ax2.set_facecolor("#1a1a1a")
top_n = min(10, len(feature_cols))
importances = pd.Series(model.feature_importances_, index=feature_cols).nlargest(top_n).sort_values()
bar_colors = ["#e50914" if (col in X_input.columns and X_input[col].values[0] != 0) or col == "Age"
              else "#555" for col in importances.index]
importances.plot(kind="barh", ax=ax2, color=bar_colors, zorder=2)
ax2.set_xlabel("Feature Importance (Gini)", color="#aaa")
ax2.set_title("Top Feature Importances", color="white", fontsize=12, pad=10)
ax2.tick_params(colors="#aaa")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.grid(axis="x", color="#333", linestyle="--", alpha=0.6, zorder=1)
red_patch = mpatches.Patch(color="#e50914", label="Active in your profile")
gray_patch = mpatches.Patch(color="#555", label="Not active")
ax2.legend(handles=[red_patch, gray_patch], fontsize=8,
           facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
ax2.text(0.98, 0.02, "Predictive importance only.\nDoes not imply causal effect.",
         transform=ax2.transAxes, fontsize=8, ha="right", va="bottom",
         style="italic", color="#e50914",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a0000", edgecolor="#e50914", alpha=0.8))

plt.tight_layout(pad=2.0)
st.pyplot(fig)

# ── User profile summary ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧾 Input Summary")
summary_df = pd.DataFrame({
    "Feature": ["Age", "Country", "Subscription Type", "Favorite Genre"],
    "Value": [age, country, subscription, genre],
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.caption(
    "95% prediction interval = prediction ± 1.96 × RMSE (test set). "
    "Wide intervals reflect the model's inherently low predictive accuracy on this dataset."
)
