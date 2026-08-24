import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


# ============================================================
# RISKGRAPH AI
# Payment Risk Command Center
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "riskgraph_fraud_model_v2.joblib"
DEFAULT_DATA_PATH = "sample_transactions.csv"

MODEL_FEATURES = [
    "amount",
    "log_amount",
    "account_age_days",
    "device_age_days",
    "transactions_last_10min",
    "failed_attempts",
    "location_change",
    "amount_deviation",
    "behavior_risk_count",
    "hour",
    "day_of_week",
    "is_weekend",
    "high_value_transaction",
    "high_velocity",
    "high_failure_activity",
    "new_device",
    "new_account",
]

ANOMALY_FEATURES = [
    "amount_deviation",
    "transactions_last_10min",
    "failed_attempts",
    "device_age_days",
    "location_change",
    "account_age_days",
    "behavior_risk_count",
]

VERIFY_COST = 25.0
REVIEW_COST = 75.0

VERIFY_FRAUD_REDUCTION = 0.80
REVIEW_FRAUD_REDUCTION = 0.95

APPROVE_THRESHOLD = 60
REVIEW_THRESHOLD = 75


# ============================================================
# NOTEBOOK METRICS
# These are the evaluation values already reported by the
# Razorpay.ipynb project.
# ============================================================

NOTEBOOK_METRICS = {
    "Future holdout ROC-AUC": 0.9985,
    "Future holdout Precision": 0.8511,
    "Future holdout Recall": 0.9836,
    "Future holdout F1": 0.9125,
    "Anomaly ROC-AUC": 0.9912,
    "Graph Risk ROC-AUC": 0.3673,
}


# ============================================================
# PROFESSIONAL ORANGE / WHITE UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #ffffff;
        color: #111111;
    }

    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }

    [data-testid="stHeader"] {
        background: #ffffff;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #111111 !important;
        font-family: Arial, Helvetica, sans-serif;
    }

    p, label, span, div {
        font-family: Arial, Helvetica, sans-serif;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #252525;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .sidebar-brand {
        padding: 10px 0 25px 0;
    }

    .sidebar-brand-title {
        font-size: 25px;
        font-weight: 800;
        color: #ffffff;
    }

    .sidebar-brand-title span {
        color: #ff7900;
    }

    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #aaaaaa !important;
        margin-top: 5px;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: #111111;
        border-radius: 18px;
        padding: 35px 40px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -90px;
        top: -90px;
        background: #ff7900;
        border-radius: 50%;
        opacity: 0.16;
    }

    .hero-title {
        color: #ffffff;
        font-size: 44px;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 2;
    }

    .hero-title span {
        color: #ff7900;
    }

    .hero-subtitle {
        color: #d0d0d0;
        font-size: 17px;
        margin-top: 8px;
        position: relative;
        z-index: 2;
    }

    .hero-line {
        color: #ff7900;
        font-weight: 700;
        margin-top: 20px;
        position: relative;
        z-index: 2;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 15px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
    }

    .metric-label {
        color: #777777;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }

    .metric-value {
        color: #111111;
        font-size: 30px;
        font-weight: 800;
        margin-top: 10px;
    }

    .metric-accent {
        color: #ff7900;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        color: #111111;
        font-size: 25px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #777777;
        font-size: 13px;
        margin-bottom: 20px;
    }

    /* ---------- STATUS ---------- */

    .status-online {
        display: inline-block;
        padding: 8px 14px;
        background: #fff4e8;
        border: 1px solid #ffd4ad;
        color: #e56600;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 800;
    }

    /* ---------- RISK CARDS ---------- */

    .risk-review {
        background: #fff0f0;
        border-left: 5px solid #e53935;
        padding: 15px;
        border-radius: 10px;
    }

    .risk-verify {
        background: #fff5e8;
        border-left: 5px solid #ff7900;
        padding: 15px;
        border-radius: 10px;
    }

    .risk-approve {
        background: #eefaf2;
        border-left: 5px solid #20a464;
        padding: 15px;
        border-radius: 10px;
    }

    /* ---------- ORANGE BUTTON ---------- */

    .stButton > button {
        background: #ff7900 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #e86600 !important;
        color: white !important;
    }

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
    }

    /* ---------- INFO BOX ---------- */

    .orange-info {
        background: #fff5eb;
        border: 1px solid #ffd5b5;
        border-radius: 12px;
        padding: 15px 18px;
        color: #333333;
        margin-top: 15px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        border-top: 1px solid #eeeeee;
        margin-top: 50px;
        padding-top: 20px;
        color: #888888;
        font-size: 12px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "₹0.00"


def percentage(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def safe_numeric(df, column, default=0):
    if column not in df.columns:
        df[column] = default

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(default)

    return df


def decision_class(action):
    if action == "REVIEW":
        return "risk-review"
    elif action == "VERIFY":
        return "risk-verify"
    return "risk-approve"


def decision_text(action):
    if action == "REVIEW":
        return "🔴 REVIEW"
    elif action == "VERIFY":
        return "🟠 VERIFY"
    return "🟢 APPROVE"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_bundle = joblib.load(MODEL_PATH)

    if isinstance(model_bundle, dict):

        if "model" not in model_bundle:
            raise ValueError(
                "The model file does not contain a 'model' object."
            )

        model = model_bundle["model"]

        saved_features = model_bundle.get(
            "features",
            MODEL_FEATURES
        )

    else:

        model = model_bundle
        saved_features = MODEL_FEATURES

    return model, list(saved_features)


# ============================================================
# LOAD DEFAULT DATA
# ============================================================

@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA_PATH)


# ============================================================
# VALIDATE RAW DATA
# ============================================================

def validate_input_data(df):

    required_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "timestamp",
        "amount",
        "location",
        "account_age_days",
        "device_age_days",
        "transactions_last_10min",
        "failed_attempts",
        "location_change",
        "amount_deviation",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Your CSV is missing required columns: "
            + ", ".join(missing)
        )

    if len(df) == 0:
        raise ValueError(
            "The uploaded CSV contains no transactions."
        )

    return True


# ============================================================
# FEATURE ENGINEERING
# IMPORTANT:
# This reproduces the features used by the trained model.
# ============================================================

def engineer_features(raw_df):

    df = raw_df.copy()

    validate_input_data(df)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamp values could not be parsed."
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    numeric_columns = [
        "amount",
        "account_age_days",
        "device_age_days",
        "transactions_last_10min",
        "failed_attempts",
        "location_change",
        "amount_deviation",
    ]

    for column in numeric_columns:
        df = safe_numeric(
            df,
            column,
            default=0
        )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Amount transformation
    # --------------------------------------------------------

    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    # --------------------------------------------------------
    # HIGH VALUE TRANSACTION
    #
    # THIS WAS THE MISSING FEATURE CAUSING YOUR ERROR.
    # --------------------------------------------------------

    df["high_value_transaction"] = (
        df["amount"] > 10000
    ).astype(int)

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    df["high_velocity"] = (
        df["transactions_last_10min"] >= 4
    ).astype(int)

    # --------------------------------------------------------
    # Failed attempts
    # --------------------------------------------------------

    df["high_failure_activity"] = (
        df["failed_attempts"] >= 3
    ).astype(int)

    # --------------------------------------------------------
    # Device age
    # --------------------------------------------------------

    df["new_device"] = (
        df["device_age_days"] < 14
    ).astype(int)

    # --------------------------------------------------------
    # Account age
    # --------------------------------------------------------

    df["new_account"] = (
        df["account_age_days"] < 60
    ).astype(int)

    # --------------------------------------------------------
    # Behaviour risk count
    # --------------------------------------------------------

    df["behavior_risk_count"] = (
        df["high_velocity"]
        + df["high_failure_activity"]
        + df["new_device"]
        + df["new_account"]
        + df["location_change"]
        + (
            df["amount_deviation"] > 3
        ).astype(int)
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    missing_model_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_model_features:
        raise ValueError(
            "Feature engineering failed. Missing model features: "
            + ", ".join(missing_model_features)
        )

    return df


# ============================================================
# ANOMALY ENGINE
# ============================================================

def calculate_anomaly_scores(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(work) < 10:

        work["anomaly_score"] = 0.0

        return work

    split = max(
        int(len(work) * 0.80),
        1
    )

    historical = work.iloc[:split].copy()

    if "is_fraud" in historical.columns:

        legitimate = historical[
            historical["is_fraud"] == 0
        ].copy()

    else:

        legitimate = historical.copy()

    if len(legitimate) < 20:
        legitimate = historical

    X_train = legitimate[
        ANOMALY_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    X_all = work[
        ANOMALY_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    anomaly_model.fit(X_train)

    train_raw = (
        -anomaly_model.decision_function(
            X_train
        )
    )

    all_raw = (
        -anomaly_model.decision_function(
            X_all
        )
    )

    sorted_training_scores = np.sort(
        train_raw
    )

    if len(sorted_training_scores) == 0:

        work["anomaly_score"] = 0.0

        return work

    percentile_scores = (
        np.searchsorted(
            sorted_training_scores,
            all_raw,
            side="right"
        )
        / len(sorted_training_scores)
    )

    work["anomaly_score"] = np.clip(
        percentile_scores * 100,
        0,
        100
    )

    return work


# ============================================================
# GRAPH / ENTITY INTELLIGENCE
# ============================================================

def calculate_graph_features(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    split = max(
        int(len(work) * 0.80),
        1
    )

    historical = work.iloc[:split].copy()

    device_customers = (
        historical
        .groupby("device_id")["customer_id"]
        .nunique()
    )

    ip_customers = (
        historical
        .groupby("ip_id")["customer_id"]
        .nunique()
    )

    device_transactions = (
        historical
        .groupby("device_id")
        .size()
    )

    ip_transactions = (
        historical
        .groupby("ip_id")
        .size()
    )

    work["device_customer_count"] = (
        work["device_id"]
        .map(device_customers)
        .fillna(0)
    )

    work["ip_customer_count"] = (
        work["ip_id"]
        .map(ip_customers)
        .fillna(0)
    )

    work["device_transaction_count"] = (
        work["device_id"]
        .map(device_transactions)
        .fillna(0)
    )

    work["ip_transaction_count"] = (
        work["ip_id"]
        .map(ip_transactions)
        .fillna(0)
    )

    work["shared_device"] = (
        work["device_customer_count"] > 1
    ).astype(int)

    work["shared_ip"] = (
        work["ip_customer_count"] > 1
    ).astype(int)

    work["entity_risk_count"] = (
        work["shared_device"]
        + work["shared_ip"]
    )

    work["graph_risk_score"] = (

        np.minimum(
            work["device_customer_count"] * 8,
            30
        )

        +

        np.minimum(
            work["ip_customer_count"] * 8,
            30
        )

        +

        np.minimum(
            work["device_transaction_count"] * 1.5,
            20
        )

        +

        np.minimum(
            work["ip_transaction_count"] * 1.5,
            20
        )
    )

    work["graph_risk_score"] = np.clip(
        work["graph_risk_score"],
        0,
        100
    )

    return work


# ============================================================
# RISK EXPLANATION
# ============================================================

def risk_explanation(row):

    reasons = []

    if row["fraud_probability"] >= 0.50:

        reasons.append(
            f"High fraud probability "
            f"({row['fraud_probability']:.1%})"
        )

    if row["anomaly_score"] >= 75:

        reasons.append(
            f"Highly unusual behavioural pattern "
            f"({row['anomaly_score']:.1f}/100)"
        )

    if row["financial_exposure_score"] >= 75:

        reasons.append(
            f"High financial exposure "
            f"({row['financial_exposure_score']:.1f}/100)"
        )

    if row["shared_device"]:

        reasons.append(
            f"Device shared across "
            f"{int(row['device_customer_count'])} customers"
        )

    if row["shared_ip"]:

        reasons.append(
            f"IP shared across "
            f"{int(row['ip_customer_count'])} customers"
        )

    if row["transactions_last_10min"] >= 4:

        reasons.append(
            f"High transaction velocity "
            f"({int(row['transactions_last_10min'])} "
            f"transactions / 10 min)"
        )

    if row["failed_attempts"] >= 3:

        reasons.append(
            f"High failed-attempt activity "
            f"({int(row['failed_attempts'])})"
        )

    if row["new_device"]:

        reasons.append(
            "New device detected"
        )

    if row["new_account"]:

        reasons.append(
            "New account detected"
        )

    if row["location_change"]:

        reasons.append(
            "Location change detected"
        )

    if row["amount_deviation"] > 3:

        reasons.append(
            f"Transaction amount is "
            f"{row['amount_deviation']:.1f}× "
            f"the normal amount"
        )

    if not reasons:

        reasons.append(
            "No major risk indicators crossed the configured thresholds."
        )

    return reasons


# ============================================================
# COMPLETE RISK PIPELINE
# ============================================================

def run_pipeline(raw_df):

    # --------------------------------------------------------
    # 1. Feature engineering
    # --------------------------------------------------------

    df = engineer_features(raw_df)

    # --------------------------------------------------------
    # 2. Load model
    # --------------------------------------------------------

    model, saved_features = load_model()

    # --------------------------------------------------------
    # 3. Guarantee saved model features exist
    # --------------------------------------------------------

    for feature in saved_features:

        if feature not in df.columns:

            # Known model features should already exist.
            # If the model bundle contains an additional
            # harmless metadata feature, create a numeric zero.
            if feature in MODEL_FEATURES:

                raise ValueError(
                    f"Required model feature '{feature}' "
                    f"was not created during feature engineering."
                )

            df[feature] = 0

    # --------------------------------------------------------
    # 4. Correct model feature order
    # --------------------------------------------------------

    X = (
        df[saved_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # 5. Fraud probability
    # --------------------------------------------------------

    if not hasattr(model, "predict_proba"):

        raise ValueError(
            "The loaded model does not support predict_proba()."
        )

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] == 2:

        df["fraud_probability"] = probabilities[:, 1]

    else:

        df["fraud_probability"] = probabilities[:, -1]

    df["fraud_probability"] = np.clip(
        df["fraud_probability"],
        0,
        1
    )

    # --------------------------------------------------------
    # 6. Anomaly detection
    # --------------------------------------------------------

    df = calculate_anomaly_scores(df)

    # --------------------------------------------------------
    # 7. Entity graph
    # --------------------------------------------------------

    df = calculate_graph_features(df)

    # --------------------------------------------------------
    # 8. Fraud signal
    # --------------------------------------------------------

    df["fraud_signal"] = (
        df["fraud_probability"] * 100
    )

    df["anomaly_signal"] = (
        df["anomaly_score"]
    )

    # --------------------------------------------------------
    # 9. Raw risk
    # --------------------------------------------------------

    df["raw_risk_score"] = (
        0.70 * df["fraud_signal"]
        +
        0.30 * df["anomaly_signal"]
    )

    df["raw_risk_score"] = np.clip(
        df["raw_risk_score"],
        0,
        100
    )

    # --------------------------------------------------------
    # 10. Expected fraud loss
    # --------------------------------------------------------

    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        * df["amount"]
    )

    # --------------------------------------------------------
    # 11. Cost-aware actions
    # --------------------------------------------------------

    df["approve_cost"] = (
        df["expected_fraud_loss"]
    )

    df["verify_cost"] = (
        VERIFY_COST
        +
        df["expected_fraud_loss"]
        *
        (1 - VERIFY_FRAUD_REDUCTION)
    )

    df["review_cost"] = (
        REVIEW_COST
        +
        df["expected_fraud_loss"]
        *
        (1 - REVIEW_FRAUD_REDUCTION)
    )

    cost_matrix = df[
        [
            "approve_cost",
            "verify_cost",
            "review_cost"
        ]
    ]

    df["recommended_action"] = (
        cost_matrix
        .idxmin(axis=1)
        .map(
            {
                "approve_cost": "APPROVE",
                "verify_cost": "VERIFY",
                "review_cost": "REVIEW",
            }
        )
    )

    # --------------------------------------------------------
    # 12. Financial exposure
    # --------------------------------------------------------

    loss_cap = (
        df["expected_fraud_loss"]
        .quantile(0.95)
    )

    if not np.isfinite(loss_cap) or loss_cap <= 0:
        loss_cap = 1

    df["financial_exposure_score"] = (
        df["expected_fraud_loss"]
        / loss_cap
        * 100
    )

    df["financial_exposure_score"] = np.clip(
        df["financial_exposure_score"],
        0,
        100
    )

    # --------------------------------------------------------
    # 13. Final RiskGraph score
    # --------------------------------------------------------

    df["risk_score"] = (

        0.50 * df["fraud_signal"]

        +

        0.30 * df["anomaly_signal"]

        +

        0.20 * df["financial_exposure_score"]
    )

    df["risk_score"] = np.clip(
        df["risk_score"],
        0,
        100
    )

    # --------------------------------------------------------
    # 14. Final decision policy
    # --------------------------------------------------------

    def final_decision(score):

        if score < APPROVE_THRESHOLD:
            return "APPROVE"

        elif score < REVIEW_THRESHOLD:
            return "VERIFY"

        else:
            return "REVIEW"

    df["final_action"] = (
        df["risk_score"]
        .apply(final_decision)
    )

    # --------------------------------------------------------
    # 15. Risk bands
    # --------------------------------------------------------

    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[
            -0.01,
            30,
            60,
            75,
            100
        ],
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL"
        ]
    ).astype(str)

    # --------------------------------------------------------
    # 16. Signal count
    # --------------------------------------------------------

    df["signal_count"] = (

        (df["fraud_probability"] >= 0.50)
        .astype(int)

        +

        (df["anomaly_score"] >= 75)
        .astype(int)

        +

        (df["financial_exposure_score"] >= 75)
        .astype(int)

        +

        (df["shared_device"] == 1)
        .astype(int)

        +

        (df["shared_ip"] == 1)
        .astype(int)

        +

        (df["behavior_risk_count"] >= 3)
        .astype(int)
    )

    return df


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">
            🛡️ RiskGraph <span>AI</span>
        </div>

        <div class="sidebar-brand-subtitle">
            AI Payment Risk Manager
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### DATA")

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help="Upload a CSV using the same transaction schema as sample_transactions.csv.",
)

# ------------------------------------------------------------
# Data source
# ------------------------------------------------------------

if uploaded_file is not None:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

        data_source = (
            f"Uploaded: {uploaded_file.name}"
        )

    except Exception as exc:

        st.sidebar.error(
            f"Could not read CSV: {exc}"
        )

        st.stop()

else:

    try:

        raw_data = load_default_data()

        data_source = (
            "Default sample_transactions.csv"
        )

    except Exception as exc:

        st.sidebar.error(
            "sample_transactions.csv could not be loaded."
        )

        st.sidebar.caption(
            str(exc)
        )

        st.stop()


# ============================================================
# PIPELINE
# ============================================================

try:

    data = run_pipeline(
        raw_data
    )

except Exception as exc:

    st.error(
        "RiskGraph AI could not process this dataset."
    )

    st.error(
        f"{type(exc).__name__}: {exc}"
    )

    st.markdown(
        """
        <div class="orange-info">
        <b>CSV format required</b><br><br>
        Your uploaded file must contain the transaction-level
        fields used by the RiskGraph model. Use
        <b>sample_transactions.csv</b> as the reference format.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# SIDEBAR STATUS
# ============================================================

st.sidebar.success(
    f"Loaded {len(data):,} transactions"
)

st.sidebar.caption(
    data_source
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Processed Transactions",
    f"{len(data):,}"
)

st.sidebar.metric(
    "Interventions",
    f"{(data['final_action'] != 'APPROVE').sum():,}"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        background:#1b1b1b;
        border:1px solid #333;
        border-radius:10px;
        padding:12px;
        font-size:12px;
    ">
    <span style="color:#ff7900;font-weight:700;">
    RISK ENGINE ONLINE
    </span><br><br>
    Fraud probability<br>
    Behavioral anomaly<br>
    Entity intelligence<br>
    Financial exposure
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Command Center",
        "Investigations",
        "Entity Network",
        "Model Intelligence",
        "Business Impact",
        "Data Explorer",
    ],
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                RiskGraph <span>AI</span>
            </div>

            <div class="hero-subtitle">
                Payment Risk Command Center
            </div>

            <div class="hero-line">
                DETECT &nbsp; • &nbsp; EXPLAIN &nbsp; • &nbsp; DECIDE
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    total_transactions = len(data)

    total_value = data[
        "amount"
    ].sum()

    approved = (
        data["final_action"] == "APPROVE"
    ).sum()

    verify = (
        data["final_action"] == "VERIFY"
    ).sum()

    review = (
        data["final_action"] == "REVIEW"
    ).sum()

    critical = (
        data["risk_score"] >= 75
    ).sum()

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    cols = st.columns(5)

    with cols[0]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Transactions
                </div>
                <div class="metric-value">
                    {total_transactions:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Transaction Value
                </div>
                <div class="metric-value">
                    ₹{total_value:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Approved
                </div>
                <div class="metric-value">
                    {approved:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[3]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Verify
                </div>
                <div class="metric-value metric-accent">
                    {verify:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[4]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Review
                </div>
                <div class="metric-value">
                    {review:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    st.markdown(
        """
        <br>
        <span class="status-online">
            ● RISK ENGINE ONLINE
        </span>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Risk Distribution
        </div>

        <div class="section-subtitle">
            Distribution of transactions across the 0–100 RiskGraph score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_bins = pd.cut(
        data["risk_score"],
        bins=[
            -0.01,
            30,
            60,
            75,
            100
        ],
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL"
        ],
    )

    risk_distribution = (
        risk_bins
        .value_counts()
        .reindex(
            [
                "LOW",
                "MODERATE",
                "HIGH",
                "CRITICAL"
            ],
            fill_value=0,
        )
    )

    st.bar_chart(
        risk_distribution,
        height=320,
    )

    st.caption(
        "LOW < 30  •  MODERATE 30–59.99  •  "
        "HIGH 60–74.99  •  CRITICAL ≥ 75"
    )

    # --------------------------------------------------------
    # DECISION MIX
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Decision Mix
        </div>
        """,
        unsafe_allow_html=True,
    )

    decision_distribution = (
        data["final_action"]
        .value_counts()
        .reindex(
            [
                "APPROVE",
                "VERIFY",
                "REVIEW"
            ],
            fill_value=0,
        )
    )

    st.bar_chart(
        decision_distribution,
        height=300,
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.markdown(
            f"""
            <div class="risk-approve">
                <b>🟢 APPROVE</b><br>
                Low intervention risk<br>
                <b>{approved:,}</b> transactions
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:

        st.markdown(
            f"""
            <div class="risk-verify">
                <b>🟠 VERIFY</b><br>
                Additional verification<br>
                <b>{verify:,}</b> transactions
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d3:

        st.markdown(
            f"""
            <div class="risk-review">
                <b>🔴 REVIEW</b><br>
                Analyst investigation<br>
                <b>{review:,}</b> transactions
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # HIGH RISK QUEUE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Priority Risk Queue
        </div>

        <div class="section-subtitle">
            Transactions requiring the highest level of attention.
        </div>
        """,
        unsafe_allow_html=True,
    )

    priority_columns = [
        "transaction_id",
        "amount",
        "fraud_probability",
        "anomaly_score",
        "graph_risk_score",
        "financial_exposure_score",
        "risk_score",
        "final_action",
    ]

    priority = (
        data
        .sort_values(
            "risk_score",
            ascending=False
        )
        [priority_columns]
        .head(15)
        .copy()
    )

    priority["amount"] = (
        priority["amount"]
        .map(lambda x: f"₹{x:,.2f}")
    )

    priority["fraud_probability"] = (
        priority["fraud_probability"]
        .map(lambda x: f"{x:.2%}")
    )

    priority["anomaly_score"] = (
        priority["anomaly_score"]
        .round(1)
    )

    priority["graph_risk_score"] = (
        priority["graph_risk_score"]
        .round(1)
    )

    priority["financial_exposure_score"] = (
        priority["financial_exposure_score"]
        .round(1)
    )

    priority["risk_score"] = (
        priority["risk_score"]
        .round(1)
    )

    st.dataframe(
        priority,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="orange-info">
            <b>How RiskGraph decides:</b><br><br>
            50% fraud probability + 30% behavioural anomaly
            + 20% financial exposure = final 0–100 risk score.<br><br>
            <b>0–59.99 → APPROVE</b>
            &nbsp;&nbsp;
            <b>60–74.99 → VERIFY</b>
            &nbsp;&nbsp;
            <b>75–100 → REVIEW</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# INVESTIGATIONS
# ============================================================

elif page == "Investigations":

    st.title("Transaction Investigation")

    st.caption(
        "Explain the risk before deciding what to do."
    )

    search = st.text_input(
        "Search transaction, customer, merchant, device or IP",
        placeholder="TX_0013175 / CUST_0001 / DEV_0001",
    )

    actions = st.multiselect(
        "Decision",
        [
            "APPROVE",
            "VERIFY",
            "REVIEW"
        ],
        default=[
            "VERIFY",
            "REVIEW"
        ],
    )

    min_risk = st.slider(
        "Minimum risk score",
        0,
        100,
        0,
    )

    queue = data[
        data["final_action"].isin(actions)
        &
        (data["risk_score"] >= min_risk)
    ].copy()

    if search.strip():

        q = search.strip().lower()

        mask = (

            queue["transaction_id"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)

            |

            queue["customer_id"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)

            |

            queue["merchant_id"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)

            |

            queue["device_id"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)

            |

            queue["ip_id"]
            .astype(str)
            .str.lower()
            .str.contains(q, na=False)
        )

        queue = queue[mask]

    queue = queue.sort_values(
        [
            "risk_score",
            "amount"
        ],
        ascending=False,
    )

    st.metric(
        "Transactions in queue",
        f"{len(queue):,}"
    )

    if len(queue) == 0:

        st.info(
            "No transactions match the current filters."
        )

    else:

        table_columns = [
            "transaction_id",
            "customer_id",
            "amount",
            "fraud_probability",
            "anomaly_score",
            "graph_risk_score",
            "risk_score",
            "risk_band",
            "final_action",
        ]

        display_queue = queue[
            table_columns
        ].copy()

        display_queue["amount"] = (
            display_queue["amount"]
            .map(lambda x: f"₹{x:,.2f}")
        )

        display_queue["fraud_probability"] = (
            display_queue["fraud_probability"]
            .map(lambda x: f"{x:.2%}")
        )

        display_queue["risk_score"] = (
            display_queue["risk_score"]
            .round(1)
        )

        st.dataframe(
            display_queue,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        selected_id = st.selectbox(
            "Select transaction to investigate",
            queue[
                "transaction_id"
            ].tolist(),
        )

        row = queue[
            queue["transaction_id"] == selected_id
        ].iloc[0]

        st.markdown(
            f"### Investigation — `{selected_id}`"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Risk Score",
            f"{row['risk_score']:.1f}/100"
        )

        c2.metric(
            "Fraud Probability",
            f"{row['fraud_probability']:.2%}"
        )

        c3.metric(
            "Anomaly Score",
            f"{row['anomaly_score']:.1f}/100"
        )

        c4.metric(
            "Transaction Amount",
            money(row["amount"])
        )

        action = row["final_action"]

        st.markdown(
            f"""
            <div class="{decision_class(action)}">
                <b>{decision_text(action)}</b><br>
                RiskGraph final decision
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Why was this transaction flagged?"
        )

        reasons = risk_explanation(row)

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

        st.markdown(
            "### Entity context"
        )

        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "Device Customers",
            int(row["device_customer_count"])
        )

        e2.metric(
            "IP Customers",
            int(row["ip_customer_count"])
        )

        e3.metric(
            "Device Transactions",
            int(row["device_transaction_count"])
        )

        e4.metric(
            "IP Transactions",
            int(row["ip_transaction_count"])
        )

        st.markdown(
            "### Transaction details"
        )

        detail = pd.DataFrame(
            {
                "Field": [
                    "Customer",
                    "Merchant",
                    "Device",
                    "IP",
                    "Timestamp",
                    "Location",
                    "Account age",
                    "Device age",
                    "Transactions / 10 min",
                    "Failed attempts",
                    "Location change",
                    "Amount deviation",
                    "Behavior risk count",
                    "Expected fraud loss",
                    "Financial exposure",
                    "Graph risk",
                ],

                "Value": [
                    row["customer_id"],
                    row["merchant_id"],
                    row["device_id"],
                    row["ip_id"],
                    str(row["timestamp"]),
                    row["location"],
                    f"{int(row['account_age_days'])} days",
                    f"{int(row['device_age_days'])} days",
                    int(row["transactions_last_10min"]),
                    int(row["failed_attempts"]),
                    "Yes" if row["location_change"] else "No",
                    f"{row['amount_deviation']:.2f}×",
                    int(row["behavior_risk_count"]),
                    money(row["expected_fraud_loss"]),
                    f"{row['financial_exposure_score']:.1f}/100",
                    f"{row['graph_risk_score']:.1f}/100",
                ],
            }
        )

        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ENTITY NETWORK
# ============================================================

elif page == "Entity Network":

    st.title("Entity Network")

    st.caption(
        "Investigate relationships between customers, devices and IP addresses."
    )

    entity_type = st.selectbox(
        "Entity type",
        [
            "Device",
            "IP",
            "Customer"
        ],
    )

    if entity_type == "Device":

        counts = (
            data
            .groupby("device_id")
            .agg(
                customers=(
                    "customer_id",
                    "nunique"
                ),
                transactions=(
                    "transaction_id",
                    "count"
                ),
                avg_risk=(
                    "risk_score",
                    "mean"
                ),
            )
            .sort_values(
                [
                    "customers",
                    "transactions"
                ],
                ascending=False
            )
            .head(30)
            .reset_index()
        )

        entity_col = "device_id"

    elif entity_type == "IP":

        counts = (
            data
            .groupby("ip_id")
            .agg(
                customers=(
                    "customer_id",
                    "nunique"
                ),
                transactions=(
                    "transaction_id",
                    "count"
                ),
                avg_risk=(
                    "risk_score",
                    "mean"
                ),
            )
            .sort_values(
                [
                    "customers",
                    "transactions"
                ],
                ascending=False
            )
            .head(30)
            .reset_index()
        )

        entity_col = "ip_id"

    else:

        counts = (
            data
            .groupby("customer_id")
            .agg(
                devices=(
                    "device_id",
                    "nunique"
                ),
                ips=(
                    "ip_id",
                    "nunique"
                ),
                transactions=(
                    "transaction_id",
                    "count"
                ),
                avg_risk=(
                    "risk_score",
                    "mean"
                ),
            )
            .sort_values(
                [
                    "transactions",
                    "avg_risk"
                ],
                ascending=False
            )
            .head(30)
            .reset_index()
        )

        entity_col = "customer_id"

    st.dataframe(
        counts,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Inspect entity"
    )

    if len(counts) > 0:

        selected_entity = st.selectbox(
            f"Select {entity_type.lower()}",
            counts[
                entity_col
            ].tolist(),
        )

        if entity_type == "Device":

            related = data[
                data["device_id"]
                == selected_entity
            ]

        elif entity_type == "IP":

            related = data[
                data["ip_id"]
                == selected_entity
            ]

        else:

            related = data[
                data["customer_id"]
                == selected_entity
            ]

        st.markdown(
            "### Related transactions"
        )

        related_display = (
            related[
                [
                    "transaction_id",
                    "customer_id",
                    "device_id",
                    "ip_id",
                    "amount",
                    "risk_score",
                    "final_action",
                ]
            ]
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(50)
            .copy()
        )

        related_display["amount"] = (
            related_display["amount"]
            .map(lambda x: f"₹{x:,.2f}")
        )

        related_display["risk_score"] = (
            related_display["risk_score"]
            .round(1)
        )

        st.dataframe(
            related_display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.title("Model Intelligence")

    st.caption(
        "Model architecture and evaluation results from Razorpay.ipynb."
    )

    st.markdown(
        """
        <div class="orange-info">
            The metrics below are the evaluation results recorded
            during the notebook modelling process. The dashboard
            does not claim to recompute those historical holdout
            metrics from the uploaded CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Model Performance"
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Future Holdout ROC-AUC",
        f"{NOTEBOOK_METRICS['Future holdout ROC-AUC']:.4f}"
    )

    m2.metric(
        "Future Holdout Precision",
        f"{NOTEBOOK_METRICS['Future holdout Precision']:.4f}"
    )

    m3.metric(
        "Future Holdout Recall",
        f"{NOTEBOOK_METRICS['Future holdout Recall']:.4f}"
    )

    m4, m5, m6 = st.columns(3)

    m4.metric(
        "Future Holdout F1",
        f"{NOTEBOOK_METRICS['Future holdout F1']:.4f}"
    )

    m5.metric(
        "Anomaly ROC-AUC",
        f"{NOTEBOOK_METRICS['Anomaly ROC-AUC']:.4f}"
    )

    m6.metric(
        "Graph Risk ROC-AUC",
        f"{NOTEBOOK_METRICS['Graph Risk ROC-AUC']:.4f}"
    )

    st.markdown(
        "### RiskGraph Architecture"
    )

    architecture = pd.DataFrame(
        {
            "Component": [
                "Supervised fraud model",
                "Behavioural anomaly engine",
                "Entity intelligence",
                "Risk fusion",
                "Financial exposure",
                "Final decision policy",
            ],

            "Purpose": [
                "Predict fraud probability",
                "Detect unusual transaction behaviour",
                "Identify device and IP relationships",
                "Combine fraud and anomaly signals",
                "Estimate expected fraud loss",
                "Convert risk score into an action",
            ],

            "Implementation": [
                "Random Forest V2",
                "Isolation Forest",
                "Device / IP graph signals",
                "70% fraud + 30% anomaly",
                "Fraud probability × amount",
                "50% fraud + 30% anomaly + 20% exposure",
            ],
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Model Features"
    )

    feature_table = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Description": [
                "Transaction amount",
                "Log-transformed amount",
                "Customer account age",
                "Device age",
                "Transactions in last 10 minutes",
                "Failed payment attempts",
                "Location change",
                "Amount deviation from normal",
                "Behaviour risk count",
                "Transaction hour",
                "Day of week",
                "Weekend indicator",
                "High-value transaction flag",
                "High transaction velocity",
                "High failure activity",
                "New device flag",
                "New account flag",
            ],
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Current Dataset"
    )

    if "is_fraud" in data.columns:

        fraud_rate = (
            data["is_fraud"]
            .mean()
        )

        a, b, c = st.columns(3)

        a.metric(
            "Fraud Cases",
            int(
                data["is_fraud"].sum()
            )
        )

        b.metric(
            "Fraud Rate",
            f"{fraud_rate:.2%}"
        )

        c.metric(
            "Average Risk",
            f"{data['risk_score'].mean():.1f}/100"
        )

    else:

        st.info(
            "This uploaded dataset does not contain an is_fraud label, "
            "so actual fraud-rate metrics cannot be calculated."
        )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.title("Business Impact")

    st.caption(
        "Cost-aware decisioning using the modelling assumptions defined in the project."
    )

    expected_loss = (
        data["expected_fraud_loss"]
        .sum()
    )

    approve_loss = (
        data.loc[
            data["final_action"] == "APPROVE",
            "expected_fraud_loss",
        ]
        .sum()
    )

    verify_loss = (
        data.loc[
            data["final_action"] == "VERIFY",
            "expected_fraud_loss",
        ]
        *
        (1 - VERIFY_FRAUD_REDUCTION)
    ).sum()

    review_loss = (
        data.loc[
            data["final_action"] == "REVIEW",
            "expected_fraud_loss",
        ]
        *
        (1 - REVIEW_FRAUD_REDUCTION)
    ).sum()

    residual_loss = (
        approve_loss
        +
        verify_loss
        +
        review_loss
    )

    intervention_cost = (

        (
            data["final_action"] == "VERIFY"
        ).sum()
        * VERIFY_COST

        +

        (
            data["final_action"] == "REVIEW"
        ).sum()
        * REVIEW_COST
    )

    total_decision_cost = (
        residual_loss
        +
        intervention_cost
    )

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Baseline Expected Loss",
        money(expected_loss)
    )

    b2.metric(
        "Residual Expected Loss",
        money(residual_loss)
    )

    b3.metric(
        "Intervention Cost",
        money(intervention_cost)
    )

    b4.metric(
        "Total Decision Cost",
        money(total_decision_cost)
    )

    st.markdown(
        "### Decision Economics"
    )

    summary = (
        data
        .groupby("final_action")
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            transaction_value=(
                "amount",
                "sum"
            ),
            expected_loss=(
                "expected_fraud_loss",
                "sum"
            ),
        )
        .reset_index()
    )

    summary["transaction_value"] = (
        summary["transaction_value"]
        .map(money)
    )

    summary["expected_loss"] = (
        summary["expected_loss"]
        .map(money)
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Business Interpretation"
    )

    st.markdown(
        f"""
        <div class="orange-info">

        <b>Expected fraud exposure:</b>
        {money(expected_loss)}

        <br><br>

        <b>Residual exposure after decisions:</b>
        {money(residual_loss)}

        <br><br>

        <b>Estimated intervention cost:</b>
        {money(intervention_cost)}

        <br><br>

        <b>Total modelled decision cost:</b>
        {money(total_decision_cost)}

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "₹25 VERIFY cost, ₹75 REVIEW cost and the fraud-reduction "
        "assumptions are prototype modelling assumptions. "
        "They are not Razorpay operational costs."
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("Data Explorer")

    st.caption(
        "Inspect transactions and every generated RiskGraph signal."
    )

    default_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "amount",
        "timestamp",
        "fraud_probability",
        "anomaly_score",
        "graph_risk_score",
        "financial_exposure_score",
        "risk_score",
        "risk_band",
        "final_action",
    ]

    default_columns = [
        c
        for c in default_columns
        if c in data.columns
    ]

    selected_columns = st.multiselect(
        "Select columns",
        data.columns.tolist(),
        default=default_columns,
    )

    if selected_columns:

        display_data = data[
            selected_columns
        ].copy()

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = (
            display_data
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download RiskGraph Results",
            data=csv_data,
            file_name="riskgraph_results.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "Select at least one column."
        )

    st.markdown(
        "### Dataset Overview"
    )

    overview = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Total transaction value",
                "Average transaction",
                "Average risk score",
                "APPROVE count",
                "VERIFY count",
                "REVIEW count",
            ],

            "Value": [
                f"{len(data):,}",
                f"{len(data.columns):,}",
                money(
                    data["amount"].sum()
                ),
                money(
                    data["amount"].mean()
                ),
                f"{data['risk_score'].mean():.2f}",
                f"{(
                    data['final_action']
                    == 'APPROVE'
                ).sum():,}",
                f"{(
                    data['final_action']
                    == 'VERIFY'
                ).sum():,}",
                f"{(
                    data['final_action']
                    == 'REVIEW'
                ).sum():,}",
            ],
        }
    )

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        RiskGraph AI · AI Payment Risk Manager ·
        Fraud Detection · Behavioural Anomaly Detection ·
        Entity Intelligence · Cost-Aware Decisioning
        <br><br>
        Synthetic-data prototype built from the Razorpay modelling pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)
