"""
Credit Card Fraud Detection — Streamlit App
=============================================
Loads artifacts produced by the training pipeline (model.pkl, scaler.pkl,
config.json) and serves predictions on single transactions or batch CSVs.

Feature engineering here MUST mirror the training script exactly:
  - Time kept (cyclic + raw)
  - Amount-derived features
  - V-column aggregates
  - amount_to_mean / is_high_amount computed using TRAIN stats from config.json
    (never recomputed on new/incoming data — that would be leakage in reverse,
    i.e. inconsistent scaling vs. what the model was trained on)
"""

import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (cached — only runs once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)
    return model, scaler, config


try:
    model, scaler, config = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing artifact file: {e.filename}. "
        "Make sure model.pkl, scaler.pkl, and config.json are in the same "
        "folder as app.py."
    )
    st.stop()

FEATURE_ORDER = config["feature_order"]
NUMERIC_FEATURES = config["numeric_features"]
BEST_THRESHOLD = config["best_threshold"]
TRAIN_MEAN = config["train_mean"]
TRAIN_P99 = config["train_p99"]
V_COLS = [c for c in FEATURE_ORDER if c.startswith("V") and c not in
          ("v_sum_sq", "v_max_abs")]


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — mirrors training script exactly
# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    df_raw must contain: Time, Amount, V1..V28 (standard creditcard.csv schema).
    Returns a DataFrame with every engineered column, in FEATURE_ORDER.
    Uses TRAIN-set statistics (train_mean, train_p99) from config.json —
    never recomputed from the incoming data.
    """
    df = df_raw.copy()

    # --- Amount features ---
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_log_sq"] = df["log_amount"] ** 2
    df["is_round_amount"] = (df["Amount"] % 1 == 0).astype(int)
    df["is_small_amount"] = (df["Amount"] < 1).astype(int)

    # --- Time features ---
    # --- Time features ---
    df["Hour"] = (df["Time"] // 3600) % 24
    df["hour_of_day"] = df["Hour"]

    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df["is_night"] = df["Hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    # --- V-column aggregates ---
    v_cols_present = [c for c in df.columns if c.startswith("V")]
    df["v_sum_sq"] = (df[v_cols_present] ** 2).sum(axis=1)
    df["v_max_abs"] = df[v_cols_present].abs().max(axis=1)

    # --- Leakage-safe stats features — TRAIN stats only ---
    df["amount_to_mean"] = df["Amount"] / (TRAIN_MEAN + 1e-9)
    df["is_high_amount"] = (df["Amount"] > TRAIN_P99).astype(int)

    # Reindex to exact training column order. Raises if anything is missing.
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after engineering: {missing}")

    return df[FEATURE_ORDER]


def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Returns df_raw with prob_fraud and prediction columns appended."""
    X = engineer_features(df_raw)
    X_scaled = X.copy()
    X_scaled[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])

    probs = model.predict_proba(X_scaled.values)[:, 1]
    preds = (probs >= BEST_THRESHOLD).astype(int)

    out = df_raw.copy()
    out["prob_fraud"] = probs
    out["prediction"] = np.where(preds == 1, "FRAUD", "LEGIT")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.caption(
    f"Model: **{config['model_type']}**  |  "
    f"Decision threshold: **{BEST_THRESHOLD:.4f}**  |  "
    f"Tuned for recall ≥ {config.get('target_recall', 0.85)}"
)

tab_single, tab_batch, tab_about = st.tabs(
    ["🔎 Single Transaction", "📁 Batch CSV", "ℹ️ Model Info"]
)

# ── TAB 1: Single transaction ────────────────────────────────────────────────
with tab_single:
    st.subheader("Score one transaction")
    st.write(
        "Enter the raw transaction fields below. V1–V28 are the anonymized "
        "PCA components from the original dataset — paste them in if you have "
        "them, or leave at 0 for a rough test."
    )

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input(
            "Time (seconds since first transaction in dataset)",
            min_value=0.0, value=50000.0, step=1.0
        )
    with col2:
        amount_val = st.number_input(
            "Amount ($)", min_value=0.0, value=100.0, step=1.0
        )

    with st.expander("V1 – V28 (PCA features)", expanded=False):
        v_values = {}
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(
                    f"V{i}", value=0.0, format="%.4f", key=f"v_{i}"
                )

    if st.button("Predict", type="primary"):
        row = {"Time": time_val, "Amount": amount_val, **v_values}
        single_df = pd.DataFrame([row])

        result = predict(single_df)
        prob = result["prob_fraud"].iloc[0]
        pred = result["prediction"].iloc[0]

        if pred == "FRAUD":
            st.error(f"🚨 **FRAUD** — probability {prob:.4f}")
        else:
            st.success(f"✅ **LEGIT** — fraud probability {prob:.4f}")

        st.progress(min(float(prob), 1.0))

# ── TAB 2: Batch CSV ─────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📂 Batch Fraud Detection")


    st.markdown("""
    **CSV Requirements**

    Upload a CSV containing:

    - Time
    - Amount
    - V1 → V28

    The file structure must match the original credit card dataset
    (excluding the `Class` column).
    """)

    uploaded = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded is not None:

        st.info(
            f"📄 {uploaded.name} • "
            f"{uploaded.size / (1024 * 1024):.2f} MB"
        )

        # File Upload + Validation
        try:
            with st.spinner("📤 Uploading and validating file..."):
                batch_df = pd.read_csv(uploaded)

        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        required_cols = (
            {"Time", "Amount"}
            | {f"V{i}" for i in range(1, 29)}
        )

        missing_cols = required_cols - set(batch_df.columns)

        if missing_cols:
            st.error(
                "Missing required columns:\n\n"
                + ", ".join(sorted(missing_cols))
            )
            st.stop()

        st.success(
            f"✅ Loaded {len(batch_df):,} transactions successfully"
        )

        # Model Prediction
        try:
            with st.spinner(
                f"🔍 Analyzing {len(batch_df):,} transactions..."
            ):
                result = predict(batch_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Metrics
        n_fraud = (
            result["prediction"] == "FRAUD"
        ).sum()

        fraud_rate = (
            n_fraud / len(result) * 100
        )

        st.divider()

        st.markdown("## 📊 Analysis Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Transactions",
            f"{len(result):,}"
        )

        col2.metric(
            "Frauds Detected",
            f"{n_fraud:,}"
        )

        col3.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

        # Risk Assessment
        st.markdown("### Risk Assessment")

        if fraud_rate > 5:
            st.error(
                f"🔴 HIGH RISK • {n_fraud:,} suspicious transactions "
                f"detected ({fraud_rate:.2f}% of total transactions)"
            )

        elif fraud_rate > 1:
            st.warning(
                f"🟡 MEDIUM RISK • {n_fraud:,} suspicious transactions "
                f"detected ({fraud_rate:.2f}% of total transactions)"
            )

        else:
            st.success(
                f"🟢 LOW RISK • {n_fraud:,} suspicious transactions "
                f"detected ({fraud_rate:.2f}% of total transactions)"
            )

        st.divider()

        # Hidden Detailed Report
        with st.expander(
            "📑 View Detailed Report",
            expanded=False
        ):

            st.markdown(
                "### 🚨 Highest Risk Transactions"
            )

            st.dataframe(
                result.sort_values(
                    "prob_fraud",
                    ascending=False
                ).head(100),
                use_container_width=True,
                height=450
            )

            csv_out = (
                result.to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "⬇️ Download Full Report",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )



# ── TAB 3: Model info ────────────────────────────────────────────────────────
with tab_about:
    st.subheader("Model configuration")
    info = {
        "Model type": config["model_type"],
        "Decision threshold": BEST_THRESHOLD,
        "Target recall (tuning goal)": config.get("target_recall"),
        "scale_pos_weight used": config.get("scale_pos_weight"),
        "Optuna trials run": config.get("optuna_n_trials"),
        "Train-set Amount mean (for amount_to_mean)": TRAIN_MEAN,
        "Train-set Amount 99th pct (for is_high_amount)": TRAIN_P99,
        "Number of features": len(FEATURE_ORDER),
    }
    st.json(info)

    with st.expander("Best hyperparameters found by Optuna"):
        st.json(config.get("best_params", {}))

    with st.expander("Full feature order (model input columns)"):
        st.write(FEATURE_ORDER)

    
    