import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


# ============================================================
# RISKGRAPH AI
# AI PAYMENT RISK MANAGER
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILES
# ============================================================

MODEL_PATH = "riskgraph_fraud_model_v2.joblib"
DEFAULT_CSV = "sample_transactions.csv"


# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# RISK POLICY
# ============================================================

VERIFY_THRESHOLD = 60
REVIEW_THRESHOLD = 75

VERIFY_COST = 25
REVIEW_COST = 75

VERIFY_FRAUD_REDUCTION = 0.80
REVIEW_FRAUD_REDUCTION = 0.95


# ============================================================
# PROFESSIONAL LIGHT THEME
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background: #F8FAFC !important;
    color: #0F172A !important;
}

[data-testid="stAppViewContainer"] {
    background: #F8FAFC !important;
}

[data-testid="stHeader"] {
    background: #F8FAFC !important;
}

.block-container {
    max-width: 1450px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}


/* ============================================================
   TEXT
   ============================================================ */

h1,
h2,
h3,
h4 {
    color: #0F172A !important;
    font-family: Inter, Arial, sans-serif !important;
    font-weight: 800 !important;
}

p {
    color: #475569 !important;
}

[data-testid="stCaptionContainer"] {
    color: #64748B !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] p {
    color: #CBD5E1 !important;
}

section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}


/* ============================================================
   SIDEBAR FILE UPLOADER
   ============================================================ */

section[data-testid="stSidebar"]
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 5px !important;
}

section[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFC !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stSidebar"]
[data-testid="stFileUploaderDropzone"] small {
    color: #475569 !important;
}


/* ============================================================
   SIDEBAR METRIC CARDS
   ============================================================ */

section[data-testid="stSidebar"]
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-top: 8px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15) !important;
}

section[data-testid="stSidebar"]
[data-testid="stMetricLabel"],
section[data-testid="stSidebar"]
[data-testid="stMetricLabel"] *,
section[data-testid="stSidebar"]
[data-testid="stMetricLabel"] p {
    color: #64748B !important;
    background: transparent !important;
}

section[data-testid="stSidebar"]
[data-testid="stMetricValue"],
section[data-testid="stSidebar"]
[data-testid="stMetricValue"] *,
section[data-testid="stSidebar"]
[data-testid="stMetricValue"] div {
    color: #0F172A !important;
    background: transparent !important;
    font-weight: 800 !important;
}


/* ============================================================
   MAIN METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: #64748B !important;
    background: transparent !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div {
    color: #0F172A !important;
    background: transparent !important;
    font-weight: 800 !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    background: #F97316 !important;
    color: #FFFFFF !important;
    border: 1px solid #F97316 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-height: 42px !important;
}

.stButton > button:hover {
    background: #EA580C !important;
    border-color: #EA580C !important;
    color: #FFFFFF !important;
}

.stDownloadButton > button {
    background: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid #0F172A !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stDownloadButton > button:hover {
    background: #1E293B !important;
    color: #FFFFFF !important;
}


/* ============================================================
   INPUTS
   ============================================================ */

div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] * {
    color: #0F172A !important;
}

input {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

input::placeholder {
    color: #94A3B8 !important;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #FFFFFF !important;
}


/* ============================================================
   ALERTS
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 10px !important;
}


/* ============================================================
   DIVIDER
   ============================================================ */

hr {
    border: 0 !important;
    border-top: 1px solid #E2E8F0 !important;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    background: linear-gradient(
        135deg,
        #FFFFFF 0%,
        #FFF7ED 100%
    );

    border: 1px solid #FED7AA;
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 24px;

    box-shadow:
        0 8px 25px
        rgba(15, 23, 42, 0.05);
}

.hero-title {
    color: #0F172A !important;
    font-size: 2.4rem;
    font-weight: 850;
    letter-spacing: -1px;
}

.hero-title span {
    color: #F97316 !important;
}

.hero-subtitle {
    color: #475569 !important;
    font-size: 1.05rem;
}

.hero-line {
    color: #EA580C !important;
    font-weight: 700;
}


/* ============================================================
   STATUS
   ============================================================ */

.status-card {
    padding: 12px 16px;
    border-radius: 10px;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #15803D !important;
    font-weight: 700;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #64748B !important;
    font-size: 0.82rem;
    padding-top: 25px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
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


def risk_band(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MODERATE"

    return "LOW"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):

        model = bundle.get("model")

        if model is None:
            raise ValueError(
                "The joblib file does not contain a 'model'."
            )

        features = bundle.get(
            "features",
            MODEL_FEATURES
        )

        return model, list(features)

    model = bundle

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = MODEL_FEATURES

    return model, features


# ============================================================
# LOAD DEFAULT DATA
# ============================================================

@st.cache_data
def load_default_data():

    return pd.read_csv(
        DEFAULT_CSV
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(df):

    required = [
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
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if len(df) == 0:
        raise ValueError(
            "The uploaded CSV contains no transactions."
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(raw_df):

    df = raw_df.copy()

    validate_input(df)

    # --------------------------------------------------------
    # TIMESTAMP
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
    # NUMERIC COLUMNS
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
            column
        )

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # AMOUNT FEATURES
    # --------------------------------------------------------

    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    # IMPORTANT:
    # This fixes the previous
    # high_value_transaction error.

    df["high_value_transaction"] = (
        df["amount"] >= 10000
    ).astype(int)

    # --------------------------------------------------------
    # VELOCITY
    # --------------------------------------------------------

    df["high_velocity"] = (
        df["transactions_last_10min"] >= 4
    ).astype(int)

    # --------------------------------------------------------
    # FAILED ATTEMPTS
    # --------------------------------------------------------

    df["high_failure_activity"] = (
        df["failed_attempts"] >= 3
    ).astype(int)

    # --------------------------------------------------------
    # DEVICE / ACCOUNT AGE
    # --------------------------------------------------------

    df["new_device"] = (
        df["device_age_days"] < 14
    ).astype(int)

    df["new_account"] = (
        df["account_age_days"] < 60
    ).astype(int)

    # --------------------------------------------------------
    # BEHAVIOUR RISK COUNT
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
    # MAKE SURE ALL MODEL FEATURES EXIST
    # --------------------------------------------------------

    for feature in MODEL_FEATURES:

        if feature not in df.columns:
            df[feature] = 0

    return df


# ============================================================
# ANOMALY DETECTION
# ============================================================

def calculate_anomaly_score(df):

    work = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    anomaly_features = [
        "amount_deviation",
        "transactions_last_10min",
        "failed_attempts",
        "device_age_days",
        "location_change",
        "account_age_days",
        "behavior_risk_count",
    ]

    if len(work) < 10:

        work["anomaly_score"] = 0.0

        return work

    train_size = max(
        10,
        int(len(work) * 0.80)
    )

    training = work.iloc[
        :train_size
    ]

    X_train = (
        training[anomaly_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    X_all = (
        work[anomaly_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    isolation = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    isolation.fit(X_train)

    training_scores = (
        -isolation.decision_function(
            X_train
        )
    )

    all_scores = (
        -isolation.decision_function(
            X_all
        )
    )

    sorted_scores = np.sort(
        training_scores
    )

    percentile = (
        np.searchsorted(
            sorted_scores,
            all_scores,
            side="right"
        )
        /
        len(sorted_scores)
    )

    work["anomaly_score"] = np.clip(
        percentile * 100,
        0,
        100
    )

    return work


# ============================================================
# ENTITY / GRAPH RISK
# ============================================================

def calculate_entity_risk(df):

    work = df.copy()

    device_customers = (
        work
        .groupby("device_id")["customer_id"]
        .nunique()
    )

    ip_customers = (
        work
        .groupby("ip_id")["customer_id"]
        .nunique()
    )

    device_transactions = (
        work
        .groupby("device_id")
        .size()
    )

    ip_transactions = (
        work
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
        +
        work["shared_ip"]
    )

    work["graph_risk_score"] = np.clip(

        np.minimum(
            work["device_customer_count"] * 12,
            35
        )

        +

        np.minimum(
            work["ip_customer_count"] * 12,
            35
        )

        +

        np.minimum(
            work["device_transaction_count"],
            15
        )

        +

        np.minimum(
            work["ip_transaction_count"],
            15
        ),

        0,
        100
    )

    return work


# ============================================================
# RISK EXPLANATION
# ============================================================

def explain_transaction(row):

    reasons = []

    if row["fraud_probability"] >= 0.75:

        reasons.append(
            "Very high fraud probability"
        )

    elif row["fraud_probability"] >= 0.50:

        reasons.append(
            "Elevated fraud probability"
        )

    if row["anomaly_score"] >= 75:

        reasons.append(
            "Highly unusual behavioural pattern"
        )

    elif row["anomaly_score"] >= 50:

        reasons.append(
            "Behaviour differs from normal activity"
        )

    if row["financial_exposure_score"] >= 75:

        reasons.append(
            "High financial exposure"
        )

    if row["transactions_last_10min"] >= 4:

        reasons.append(
            "High transaction velocity"
        )

    if row["failed_attempts"] >= 3:

        reasons.append(
            "Multiple failed attempts"
        )

    if row["location_change"] == 1:

        reasons.append(
            "Location change detected"
        )

    if row["new_device"] == 1:

        reasons.append(
            "New device detected"
        )

    if row["new_account"] == 1:

        reasons.append(
            "New account detected"
        )

    if row["amount_deviation"] > 3:

        reasons.append(
            "Transaction amount is significantly above normal behaviour"
        )

    if row["shared_device"] == 1:

        reasons.append(
            "Device is associated with multiple customers"
        )

    if row["shared_ip"] == 1:

        reasons.append(
            "IP address is associated with multiple customers"
        )

    if not reasons:

        reasons.append(
            "No major risk indicators detected"
        )

    return reasons


# ============================================================
# MAIN RISK PIPELINE
# ============================================================

def run_pipeline(raw_df):

    df = engineer_features(
        raw_df
    )

    model, model_features = (
        load_model()
    )

    # --------------------------------------------------------
    # MODEL FEATURE VALIDATION
    # --------------------------------------------------------

    missing = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Model features missing: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    X = (
        df[model_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # FRAUD PROBABILITY
    # --------------------------------------------------------

    if not hasattr(
        model,
        "predict_proba"
    ):
        raise ValueError(
            "The loaded model does not support predict_proba()."
        )

    probabilities = (
        model.predict_proba(X)
    )

    if probabilities.shape[1] >= 2:

        df["fraud_probability"] = (
            probabilities[:, 1]
        )

    else:

        df["fraud_probability"] = (
            probabilities[:, 0]
        )

    df["fraud_probability"] = np.clip(
        df["fraud_probability"],
        0,
        1
    )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    df = calculate_anomaly_score(
        df
    )

    # --------------------------------------------------------
    # ENTITY
    # --------------------------------------------------------

    df = calculate_entity_risk(
        df
    )

    # --------------------------------------------------------
    # FINANCIAL EXPOSURE
    # --------------------------------------------------------

    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        *
        df["amount"]
    )

    reference = (
        df["expected_fraud_loss"]
        .quantile(0.95)
    )

    if (
        not np.isfinite(reference)
        or reference <= 0
    ):
        reference = 1

    df["financial_exposure_score"] = np.clip(

        (
            df["expected_fraud_loss"]
            /
            reference
        )
        *
        100,

        0,
        100
    )

    # --------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------

    df["fraud_signal"] = (
        df["fraud_probability"]
        *
        100
    )

    df["risk_score"] = np.clip(

        (
            0.50
            *
            df["fraud_signal"]
        )

        +

        (
            0.30
            *
            df["anomaly_score"]
        )

        +

        (
            0.20
            *
            df["financial_exposure_score"]
        ),

        0,
        100
    )

    # --------------------------------------------------------
    # RISK BAND
    # --------------------------------------------------------

    df["risk_band"] = (
        df["risk_score"]
        .apply(risk_band)
    )

    # --------------------------------------------------------
    # DECISION POLICY
    # --------------------------------------------------------

    def decision(score):

        if score >= REVIEW_THRESHOLD:
            return "REVIEW"

        if score >= VERIFY_THRESHOLD:
            return "VERIFY"

        return "APPROVE"

    df["final_action"] = (
        df["risk_score"]
        .apply(decision)
    )

    # --------------------------------------------------------
    # COST-AWARE METRICS
    # --------------------------------------------------------

    df["approve_cost"] = (
        df["expected_fraud_loss"]
    )

    df["verify_cost"] = (

        VERIFY_COST

        +

        (
            df["expected_fraud_loss"]
            *
            (
                1
                -
                VERIFY_FRAUD_REDUCTION
            )
        )
    )

    df["review_cost"] = (

        REVIEW_COST

        +

        (
            df["expected_fraud_loss"]
            *
            (
                1
                -
                REVIEW_FRAUD_REDUCTION
            )
        )
    )

    # --------------------------------------------------------
    # SIGNAL COUNT
    # --------------------------------------------------------

    df["signal_count"] = (

        (
            df["fraud_probability"]
            >= 0.50
        ).astype(int)

        +

        (
            df["anomaly_score"]
            >= 75
        ).astype(int)

        +

        (
            df["financial_exposure_score"]
            >= 75
        ).astype(int)

        +

        df["shared_device"]

        +

        df["shared_ip"]

        +

        (
            df["behavior_risk_count"]
            >= 3
        ).astype(int)
    )

    return df


# ============================================================
# SIDEBAR BRAND
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        padding: 8px 0 12px 0;
    ">
        <div style="
            font-size: 1.35rem;
            font-weight: 800;
            color: #FFFFFF;
        ">
            🛡️ RiskGraph <span style="color:#F97316;">AI</span>
        </div>

        <div style="
            margin-top: 5px;
            color: #94A3B8;
            font-size: 0.82rem;
        ">
            AI Payment Risk Manager
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()


# ============================================================
# DATA UPLOAD
# ============================================================

st.sidebar.markdown(
    "**DATA**"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help="Upload a CSV using sample_transactions.csv as the reference format."
)


# ============================================================
# READ DATA
# ============================================================

if uploaded_file is not None:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

        source_name = uploaded_file.name

    except Exception as error:

        st.sidebar.error(
            f"Could not read CSV: {error}"
        )

        st.stop()

else:

    try:

        raw_data = load_default_data()

        source_name = DEFAULT_CSV

    except Exception as error:

        st.sidebar.error(
            "sample_transactions.csv could not be loaded."
        )

        st.sidebar.code(
            str(error)
        )

        st.stop()


# ============================================================
# RUN PIPELINE
# ============================================================

try:

    data = run_pipeline(
        raw_data
    )

except Exception as error:

    st.error(
        "RiskGraph AI could not process this dataset."
    )

    st.error(
        f"{type(error).__name__}: {error}"
    )

    st.info(
        "Please use sample_transactions.csv "
        "as the reference schema."
    )

    st.stop()


# ============================================================
# SIDEBAR STATUS
# ============================================================

st.sidebar.success(
    f"Loaded {len(data):,} transactions"
)

st.sidebar.caption(
    f"Source: {source_name}"
)

st.sidebar.metric(
    "Processed",
    f"{len(data):,}"
)

interventions = (
    data["final_action"]
    != "APPROVE"
).sum()

st.sidebar.metric(
    "Interventions",
    f"{interventions:,}"
)

st.sidebar.markdown(
    """
    <div class="status-card">
        🟢 Risk Engine Online
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown(
    "**RISK OPERATIONS**"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Command Center",
        "Investigate",
        "Model Intelligence",
        "Business Impact",
        "Entity Network",
        "Data Explorer",
    ]
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
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_transactions = len(data)

    total_value = data["amount"].sum()

    approved = (
        data["final_action"]
        == "APPROVE"
    ).sum()

    verify = (
        data["final_action"]
        == "VERIFY"
    ).sum()

    review = (
        data["final_action"]
        == "REVIEW"
    ).sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Transaction Value",
        money(total_value)
    )

    c3.metric(
        "Approved",
        f"{approved:,}"
    )

    c4.metric(
        "Verify",
        f"{verify:,}"
    )

    c5.metric(
        "Review",
        f"{review:,}"
    )

    st.divider()

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Risk Score Distribution"
        )

        distribution = (
            data["risk_band"]
            .value_counts()
            .reindex(
                [
                    "LOW",
                    "MODERATE",
                    "HIGH",
                    "CRITICAL"
                ],
                fill_value=0
            )
        )

        st.bar_chart(
            distribution,
            height=330
        )

        st.caption(
            "LOW < 30  •  MODERATE 30–59.99  •  "
            "HIGH 60–74.99  •  CRITICAL ≥ 75"
        )

    with right:

        st.subheader(
            "Decision Mix"
        )

        decision_mix = (
            data["final_action"]
            .value_counts()
            .reindex(
                [
                    "APPROVE",
                    "VERIFY",
                    "REVIEW"
                ],
                fill_value=0
            )
        )

        st.bar_chart(
            decision_mix,
            height=330
        )

        st.write(
            f"🟢 **APPROVE** — {approved:,}"
        )

        st.write(
            f"🟠 **VERIFY** — {verify:,}"
        )

        st.write(
            f"🔴 **REVIEW** — {review:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # PRIORITY QUEUE
    # --------------------------------------------------------

    st.subheader(
        "Priority Risk Queue"
    )

    st.caption(
        "Highest-risk transactions requiring analyst attention."
    )

    queue = (
        data
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(15)
        .copy()
    )

    queue_display = queue[
        [
            "transaction_id",
            "amount",
            "fraud_probability",
            "anomaly_score",
            "graph_risk_score",
            "financial_exposure_score",
            "risk_score",
            "final_action",
        ]
    ].copy()

    queue_display["amount"] = (
        queue_display["amount"]
        .map(money)
    )

    queue_display["fraud_probability"] = (
        queue_display["fraud_probability"]
        .map(percentage)
    )

    queue_display["anomaly_score"] = (
        queue_display["anomaly_score"]
        .round(1)
    )

    queue_display["graph_risk_score"] = (
        queue_display["graph_risk_score"]
        .round(1)
    )

    queue_display["financial_exposure_score"] = (
        queue_display["financial_exposure_score"]
        .round(1)
    )

    queue_display["risk_score"] = (
        queue_display["risk_score"]
        .round(1)
    )

    st.dataframe(
        queue_display,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "RiskGraph combines fraud probability, "
        "behavioural anomaly, entity intelligence "
        "and financial exposure into a final 0–100 risk score."
    )


# ============================================================
# INVESTIGATE
# ============================================================

elif page == "Investigate":

    st.title(
        "Transaction Investigation"
    )

    st.caption(
        "Explain the risk before deciding what to do."
    )

    st.divider()

    search = st.text_input(
        "Search transaction, customer, merchant, device or IP",
        placeholder="Example: TX_0000141"
    )

    action_filter = st.multiselect(
        "Decision filter",
        [
            "APPROVE",
            "VERIFY",
            "REVIEW"
        ],
        default=[
            "VERIFY",
            "REVIEW"
        ]
    )

    minimum_risk = st.slider(
        "Minimum Risk Score",
        0,
        100,
        0
    )

    filtered = data[
        (
            data["final_action"]
            .isin(action_filter)
        )
        &
        (
            data["risk_score"]
            >= minimum_risk
        )
    ].copy()

    if search.strip():

        query = search.strip().lower()

        mask = (
            filtered["transaction_id"]
            .astype(str)
            .str.lower()
            .str.contains(query, na=False)
        )

        for column in [
            "customer_id",
            "merchant_id",
            "device_id",
            "ip_id"
        ]:

            mask = (
                mask
                |
                filtered[column]
                .astype(str)
                .str.lower()
                .str.contains(query, na=False)
            )

        filtered = filtered[mask]

    st.metric(
        "Matching Transactions",
        f"{len(filtered):,}"
    )

    if len(filtered) == 0:

        st.warning(
            "No transactions match the current filters."
        )

    else:

        display = filtered[
            [
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
        ].copy()

        display["amount"] = (
            display["amount"]
            .map(money)
        )

        display["fraud_probability"] = (
            display["fraud_probability"]
            .map(percentage)
        )

        display["anomaly_score"] = (
            display["anomaly_score"]
            .round(1)
        )

        display["graph_risk_score"] = (
            display["graph_risk_score"]
            .round(1)
        )

        display["risk_score"] = (
            display["risk_score"]
            .round(1)
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        selected = st.selectbox(
            "Select transaction to investigate",
            filtered[
                "transaction_id"
            ].tolist()
        )

        row = filtered[
            filtered["transaction_id"]
            == selected
        ].iloc[0]

        st.subheader(
            f"Investigation — {selected}"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Risk Score",
            f"{row['risk_score']:.1f}/100"
        )

        b.metric(
            "Fraud Probability",
            percentage(
                row["fraud_probability"]
            )
        )

        c.metric(
            "Anomaly Score",
            f"{row['anomaly_score']:.1f}/100"
        )

        d.metric(
            "Amount",
            money(
                row["amount"]
            )
        )

        st.divider()

        if row["final_action"] == "REVIEW":

            st.error(
                "🔴 REVIEW — Analyst investigation recommended."
            )

        elif row["final_action"] == "VERIFY":

            st.warning(
                "🟠 VERIFY — Additional verification recommended."
            )

        else:

            st.success(
                "🟢 APPROVE — Low intervention risk."
            )

        st.subheader(
            "Why was this transaction flagged?"
        )

        reasons = explain_transaction(
            row
        )

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

        st.subheader(
            "Entity Context"
        )

        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "Customers / Device",
            int(
                row["device_customer_count"]
            )
        )

        e2.metric(
            "Customers / IP",
            int(
                row["ip_customer_count"]
            )
        )

        e3.metric(
            "Device Transactions",
            int(
                row["device_transaction_count"]
            )
        )

        e4.metric(
            "IP Transactions",
            int(
                row["ip_transaction_count"]
            )
        )

        st.subheader(
            "Transaction Details"
        )

        details = pd.DataFrame(
            {
                "Field": [
                    "Customer",
                    "Merchant",
                    "Device",
                    "IP",
                    "Timestamp",
                    "Location",
                    "Account Age",
                    "Device Age",
                    "Transactions / 10 min",
                    "Failed Attempts",
                    "Location Change",
                    "Amount Deviation",
                    "Behaviour Risk Count",
                    "Expected Fraud Loss",
                    "Financial Exposure",
                    "Graph Risk Score",
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
                ]
            }
        )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.title(
        "Model Intelligence"
    )

    st.caption(
        "Understand how RiskGraph converts transaction signals into risk."
    )

    st.divider()

    st.subheader(
        "RiskGraph Decision Architecture"
    )

    architecture = pd.DataFrame(
        {
            "Layer": [
                "Fraud Detection",
                "Behavioural Anomaly",
                "Entity Intelligence",
                "Financial Exposure",
                "Risk Fusion",
                "Decision Policy",
            ],

            "Purpose": [
                "Estimate fraud probability",
                "Identify unusual transaction behaviour",
                "Identify customer, device and IP relationships",
                "Estimate potential monetary exposure",
                "Combine multiple risk signals",
                "Convert risk into APPROVE / VERIFY / REVIEW",
            ],
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Final Risk Score"
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Fraud Signal",
        "50%"
    )

    r2.metric(
        "Anomaly Signal",
        "30%"
    )

    r3.metric(
        "Financial Exposure",
        "20%"
    )

    st.info(
        "Final Risk Score = 50% Fraud Signal + "
        "30% Behavioural Anomaly + 20% Financial Exposure."
    )

    st.subheader(
        "Decision Policy"
    )

    policy = pd.DataFrame(
        {
            "Risk Score": [
                "0 – 59.99",
                "60 – 74.99",
                "75 – 100",
            ],

            "Action": [
                "APPROVE",
                "VERIFY",
                "REVIEW",
            ],

            "Meaning": [
                "Low intervention risk",
                "Additional verification",
                "Analyst investigation",
            ],
        }
    )

    st.dataframe(
        policy,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Model Features"
    )

    feature_descriptions = [
        "Transaction amount",
        "Log-transformed transaction amount",
        "Customer account age",
        "Device age",
        "Transaction velocity",
        "Failed payment attempts",
        "Location change",
        "Amount deviation",
        "Behaviour risk count",
        "Transaction hour",
        "Day of week",
        "Weekend indicator",
        "High-value transaction",
        "High transaction velocity",
        "High failure activity",
        "New device",
        "New account",
    ]

    feature_table = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Purpose": feature_descriptions,
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.title(
        "Business Impact"
    )

    st.caption(
        "Understand the financial trade-off behind intervention decisions."
    )

    st.divider()

    baseline_loss = (
        data["expected_fraud_loss"]
        .sum()
    )

    approved_loss = (
        data.loc[
            data["final_action"]
            == "APPROVE",
            "expected_fraud_loss"
        ]
        .sum()
    )

    verify_loss = (
        data.loc[
            data["final_action"]
            == "VERIFY",
            "expected_fraud_loss"
        ]
        .sum()
        *
        (
            1
            -
            VERIFY_FRAUD_REDUCTION
        )
    )

    review_loss = (
        data.loc[
            data["final_action"]
            == "REVIEW",
            "expected_fraud_loss"
        ]
        .sum()
        *
        (
            1
            -
            REVIEW_FRAUD_REDUCTION
        )
    )

    residual_loss = (
        approved_loss
        +
        verify_loss
        +
        review_loss
    )

    verification_cost = (
        (
            data["final_action"]
            == "VERIFY"
        )
        .sum()
        *
        VERIFY_COST
    )

    review_cost = (
        (
            data["final_action"]
            == "REVIEW"
        )
        .sum()
        *
        REVIEW_COST
    )

    intervention_cost = (
        verification_cost
        +
        review_cost
    )

    total_cost = (
        residual_loss
        +
        intervention_cost
    )

    savings = max(
        0,
        baseline_loss - total_cost
    )

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Baseline Expected Loss",
        money(baseline_loss)
    )

    b2.metric(
        "Residual Loss",
        money(residual_loss)
    )

    b3.metric(
        "Intervention Cost",
        money(intervention_cost)
    )

    b4.metric(
        "Estimated Savings",
        money(savings)
    )

    st.divider()

    st.subheader(
        "Decision Economics"
    )

    economics = (
        data
        .groupby("final_action")
        .agg(
            Transactions=(
                "transaction_id",
                "count"
            ),

            Transaction_Value=(
                "amount",
                "sum"
            ),

            Expected_Fraud_Loss=(
                "expected_fraud_loss",
                "sum"
            )
        )
        .reset_index()
    )

    economics["Transaction_Value"] = (
        economics["Transaction_Value"]
        .map(money)
    )

    economics["Expected_Fraud_Loss"] = (
        economics["Expected_Fraud_Loss"]
        .map(money)
    )

    st.dataframe(
        economics,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "VERIFY and REVIEW costs are prototype assumptions "
        "used to demonstrate cost-aware decisioning."
    )


# ============================================================
# ENTITY NETWORK
# ============================================================

elif page == "Entity Network":

    st.title(
        "Entity Network"
    )

    st.caption(
        "Explore relationships between customers, devices and IP addresses."
    )

    st.divider()

    entity_type = st.selectbox(
        "Entity Type",
        [
            "Device",
            "IP Address",
            "Customer"
        ]
    )

    if entity_type == "Device":

        summary = (
            data
            .groupby("device_id")
            .agg(
                Customers=(
                    "customer_id",
                    "nunique"
                ),

                Transactions=(
                    "transaction_id",
                    "count"
                ),

                Average_Risk=(
                    "risk_score",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                [
                    "Customers",
                    "Transactions"
                ],
                ascending=False
            )
            .head(30)
        )

        entity_column = "device_id"

    elif entity_type == "IP Address":

        summary = (
            data
            .groupby("ip_id")
            .agg(
                Customers=(
                    "customer_id",
                    "nunique"
                ),

                Transactions=(
                    "transaction_id",
                    "count"
                ),

                Average_Risk=(
                    "risk_score",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                [
                    "Customers",
                    "Transactions"
                ],
                ascending=False
            )
            .head(30)
        )

        entity_column = "ip_id"

    else:

        summary = (
            data
            .groupby("customer_id")
            .agg(
                Devices=(
                    "device_id",
                    "nunique"
                ),

                IPs=(
                    "ip_id",
                    "nunique"
                ),

                Transactions=(
                    "transaction_id",
                    "count"
                ),

                Average_Risk=(
                    "risk_score",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                "Transactions",
                ascending=False
            )
            .head(30)
        )

        entity_column = "customer_id"

    summary["Average_Risk"] = (
        summary["Average_Risk"]
        .round(1)
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        f"Investigate {entity_type}"
    )

    selected_entity = st.selectbox(
        "Select entity",
        summary[
            entity_column
        ].tolist()
    )

    if entity_type == "Device":

        related = data[
            data["device_id"]
            == selected_entity
        ]

    elif entity_type == "IP Address":

        related = data[
            data["ip_id"]
            == selected_entity
        ]

    else:

        related = data[
            data["customer_id"]
            == selected_entity
        ]

    related_display = related[
        [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "device_id",
            "ip_id",
            "amount",
            "risk_score",
            "final_action",
        ]
    ].copy()

    related_display["amount"] = (
        related_display["amount"]
        .map(money)
    )

    related_display["risk_score"] = (
        related_display["risk_score"]
        .round(1)
    )

    st.dataframe(
        related_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title(
        "Data Explorer"
    )

    st.caption(
        "Inspect, filter and export RiskGraph results."
    )

    st.divider()

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
        col
        for col in default_columns
        if col in data.columns
    ]

    selected_columns = st.multiselect(
        "Columns to display",
        data.columns.tolist(),
        default=default_columns
    )

    if selected_columns:

        display = data[
            selected_columns
        ].copy()

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        csv_data = (
            display
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download RiskGraph Results",
            data=csv_data,
            file_name="riskgraph_results.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Select at least one column."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>RiskGraph AI</strong>
        • AI Payment Risk Manager
        • Fraud Detection
        • Behavioural Anomaly Detection
        • Entity Intelligence
        • Cost-Aware Decisioning
        <br><br>
        Synthetic-data prototype
    </div>
    """,
    unsafe_allow_html=True
)
