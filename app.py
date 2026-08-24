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

VERIFY_THRESHOLD = 60
REVIEW_THRESHOLD = 75


# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APPLICATION
       ======================================================== */

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
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1,
    h2,
    h3,
    h4 {
        color: #111111 !important;
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: 800 !important;
    }

    p {
        color: #444444;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #111111 !important;
        border-right: 1px solid #292929;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p {
        color: #dddddd !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }


    /* ========================================================
       SIDEBAR METRIC CARDS
       ======================================================== */

    section[data-testid="stSidebar"]
    [data-testid="stMetric"] {

        background: #ffffff !important;

        border: 1px solid #dddddd !important;

        border-radius: 12px !important;

        padding: 16px !important;

        margin-top: 8px !important;

        margin-bottom: 8px !important;

        box-shadow:
            0 4px 14px
            rgba(0, 0, 0, 0.20) !important;
    }

    /* Sidebar metric label */

    section[data-testid="stSidebar"]
    [data-testid="stMetricLabel"] {

        color: #555555 !important;

        background: transparent !important;

        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricLabel"] * {

        color: #555555 !important;

        background: transparent !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricLabel"] p {

        color: #555555 !important;

        background: transparent !important;
    }

    /* Sidebar metric value */

    section[data-testid="stSidebar"]
    [data-testid="stMetricValue"] {

        color: #111111 !important;

        background: transparent !important;

        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricValue"] * {

        color: #111111 !important;

        background: transparent !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricValue"] div {

        color: #111111 !important;

        background: transparent !important;
    }

    /* Sidebar metric delta */

    section[data-testid="stSidebar"]
    [data-testid="stMetricDelta"] {

        color: #555555 !important;

        background: transparent !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stMetricDelta"] * {

        color: #555555 !important;

        background: transparent !important;
    }


    /* ========================================================
       MAIN METRIC CARDS
       ======================================================== */

    [data-testid="stMetric"] {

        background: #ffffff !important;

        border: 1px solid #e5e5e5 !important;

        border-radius: 12px !important;

        padding: 18px !important;

        box-shadow:
            0 4px 16px
            rgba(0, 0, 0, 0.05) !important;
    }

    [data-testid="stMetricLabel"] {

        color: #555555 !important;

        background: transparent !important;

        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] * {

        color: #555555 !important;

        background: transparent !important;
    }

    [data-testid="stMetricLabel"] p {

        color: #555555 !important;

        background: transparent !important;
    }

    [data-testid="stMetricValue"] {

        color: #111111 !important;

        background: transparent !important;

        font-weight: 800 !important;
    }

    [data-testid="stMetricValue"] * {

        color: #111111 !important;

        background: transparent !important;
    }

    [data-testid="stMetricValue"] div {

        color: #111111 !important;

        background: transparent !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        background: #ff7900 !important;

        color: #ffffff !important;

        border: 1px solid #ff7900 !important;

        border-radius: 8px !important;

        font-weight: 700 !important;

        min-height: 42px !important;
    }

    .stButton > button:hover {

        background: #e96800 !important;

        border-color: #e96800 !important;

        color: #ffffff !important;
    }


    /* ========================================================
       DOWNLOAD BUTTON
       ======================================================== */

    .stDownloadButton > button {

        background: #ff7900 !important;

        color: #ffffff !important;

        border: 1px solid #ff7900 !important;

        border-radius: 8px !important;

        font-weight: 700 !important;
    }

    .stDownloadButton > button:hover {

        background: #e96800 !important;

        color: #ffffff !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {

        background: #ffffff !important;

        border-radius: 10px !important;
    }


    /* ========================================================
       SELECTBOX / INPUT
       ======================================================== */

    div[data-baseweb="select"] > div {

        border-radius: 8px !important;

        background: #ffffff !important;
    }

    input {

        border-radius: 8px !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {

        border: 1px solid #e5e5e5 !important;

        border-radius: 10px !important;

        overflow: hidden !important;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {

        border-color: #eeeeee !important;
    }


    /* ========================================================
       INFO / WARNING / SUCCESS / ERROR
       ======================================================== */

    [data-testid="stAlert"] {

        border-radius: 10px !important;
    }


    /* ========================================================
       ORANGE ACCENT
       ======================================================== */

    .orange-text {

        color: #ff7900 !important;
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


def pct(value):

    try:
        return f"{float(value) * 100:.2f}%"

    except Exception:
        return "0.00%"


def safe_numeric(
    df,
    column,
    default=0
):

    if column not in df.columns:

        df[column] = default

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(default)

    return df


def risk_label(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 60:
        return "HIGH"

    elif score >= 30:
        return "MODERATE"

    return "LOW"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(
        MODEL_PATH
    )

    if isinstance(bundle, dict):

        if "model" not in bundle:

            raise ValueError(
                "The joblib file does not contain a trained model."
            )

        model = bundle["model"]

        features = bundle.get(
            "features",
            MODEL_FEATURES
        )

        return model, list(features)

    model = bundle

    if hasattr(
        model,
        "feature_names_in_"
    ):

        features = list(
            model.feature_names_in_
        )

    else:

        features = MODEL_FEATURES

    return model, features


# ============================================================
# LOAD DEFAULT CSV
# ============================================================

@st.cache_data
def load_default_data():

    return pd.read_csv(
        DEFAULT_DATA_PATH
    )


# ============================================================
# VALIDATE CSV
# ============================================================

def validate_input(df):

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

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing)
        )

    if df.empty:

        raise ValueError(
            "The uploaded CSV contains no transactions."
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(
    raw_data
):

    df = raw_data.copy()

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
            "One or more timestamp values could not be parsed."
        )

    # --------------------------------------------------------
    # NUMERIC FEATURES
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
            0
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
        df["amount"].clip(
            lower=0
        )
    )

    # --------------------------------------------------------
    # IMPORTANT MODEL FEATURE
    # --------------------------------------------------------

    df["high_value_transaction"] = (
        df["amount"] > 10000
    ).astype(int)

    # --------------------------------------------------------
    # BEHAVIOUR FEATURES
    # --------------------------------------------------------

    df["high_velocity"] = (
        df["transactions_last_10min"] >= 4
    ).astype(int)

    df["high_failure_activity"] = (
        df["failed_attempts"] >= 3
    ).astype(int)

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

        +

        df["high_failure_activity"]

        +

        df["new_device"]

        +

        df["new_account"]

        +

        df["location_change"]

        +

        (
            df["amount_deviation"] > 3
        ).astype(int)
    )

    # --------------------------------------------------------
    # FINAL FEATURE VALIDATION
    # --------------------------------------------------------

    missing_features = [

        feature

        for feature in MODEL_FEATURES

        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Model features missing after feature engineering: "
            + ", ".join(missing_features)
        )

    return df


# ============================================================
# ANOMALY DETECTION
# ============================================================

def calculate_anomaly_scores(
    df
):

    work = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    if len(work) < 10:

        work["anomaly_score"] = 0.0

        return work

    split = max(
        1,
        int(len(work) * 0.80)
    )

    historical = (
        work.iloc[:split]
        .copy()
    )

    if (
        "is_fraud" in historical.columns
        and historical["is_fraud"].nunique() > 1
    ):

        legitimate = historical[
            historical["is_fraud"] == 0
        ].copy()

    else:

        legitimate = historical.copy()

    if len(legitimate) < 20:

        legitimate = historical.copy()

    X_train = (
        legitimate[
            ANOMALY_FEATURES
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    X_all = (
        work[
            ANOMALY_FEATURES
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    iso = IsolationForest(
        n_estimators=250,
        random_state=42,
        contamination="auto",
        n_jobs=-1
    )

    iso.fit(
        X_train
    )

    train_scores = (
        -iso.decision_function(
            X_train
        )
    )

    all_scores = (
        -iso.decision_function(
            X_all
        )
    )

    train_sorted = np.sort(
        train_scores
    )

    if len(train_sorted) == 0:

        work["anomaly_score"] = 0.0

        return work

    percentile = (
        np.searchsorted(
            train_sorted,
            all_scores,
            side="right"
        )
        /
        len(train_sorted)
    )

    work["anomaly_score"] = np.clip(
        percentile * 100,
        0,
        100
    )

    return work


# ============================================================
# ENTITY / GRAPH FEATURES
# ============================================================

def calculate_graph_features(
    df
):

    work = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    split = max(
        1,
        int(len(work) * 0.80)
    )

    historical = (
        work.iloc[:split]
        .copy()
    )

    device_customer_count = (
        historical
        .groupby("device_id")["customer_id"]
        .nunique()
    )

    ip_customer_count = (
        historical
        .groupby("ip_id")["customer_id"]
        .nunique()
    )

    device_transaction_count = (
        historical
        .groupby("device_id")
        .size()
    )

    ip_transaction_count = (
        historical
        .groupby("ip_id")
        .size()
    )

    work["device_customer_count"] = (
        work["device_id"]
        .map(
            device_customer_count
        )
        .fillna(0)
    )

    work["ip_customer_count"] = (
        work["ip_id"]
        .map(
            ip_customer_count
        )
        .fillna(0)
    )

    work["device_transaction_count"] = (
        work["device_id"]
        .map(
            device_transaction_count
        )
        .fillna(0)
    )

    work["ip_transaction_count"] = (
        work["ip_id"]
        .map(
            ip_transaction_count
        )
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

def get_reasons(
    row
):

    reasons = []

    if row["fraud_probability"] >= 0.50:

        reasons.append(
            f"High fraud probability: "
            f"{row['fraud_probability']:.1%}"
        )

    if row["anomaly_score"] >= 75:

        reasons.append(
            f"Highly unusual behaviour: "
            f"{row['anomaly_score']:.1f}/100"
        )

    if row["financial_exposure_score"] >= 75:

        reasons.append(
            f"High financial exposure: "
            f"{row['financial_exposure_score']:.1f}/100"
        )

    if row["transactions_last_10min"] >= 4:

        reasons.append(
            f"High transaction velocity: "
            f"{int(row['transactions_last_10min'])} "
            f"transactions in 10 minutes"
        )

    if row["failed_attempts"] >= 3:

        reasons.append(
            f"Multiple failed attempts: "
            f"{int(row['failed_attempts'])}"
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
            f"Amount deviation: "
            f"{row['amount_deviation']:.1f}×"
        )

    if row["shared_device"]:

        reasons.append(
            f"Shared device linked to "
            f"{int(row['device_customer_count'])} customers"
        )

    if row["shared_ip"]:

        reasons.append(
            f"Shared IP linked to "
            f"{int(row['ip_customer_count'])} customers"
        )

    if not reasons:

        reasons.append(
            "No major risk indicators detected."
        )

    return reasons


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    raw_data
):

    df = engineer_features(
        raw_data
    )

    model, model_features = (
        load_model()
    )

    # --------------------------------------------------------
    # MODEL FEATURE CHECK
    # --------------------------------------------------------

    for feature in model_features:

        if feature not in df.columns:

            raise ValueError(
                f"Model features missing: {feature}"
            )

    X = (
        df[
            model_features
        ]
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
            "Loaded fraud model does not support predict_proba()."
        )

    probabilities = (
        model.predict_proba(X)
    )

    if probabilities.shape[1] == 2:

        df["fraud_probability"] = (
            probabilities[:, 1]
        )

    else:

        df["fraud_probability"] = (
            probabilities[:, -1]
        )

    df["fraud_probability"] = np.clip(
        df["fraud_probability"],
        0,
        1
    )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    df = calculate_anomaly_scores(
        df
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    df = calculate_graph_features(
        df
    )

    # --------------------------------------------------------
    # FRAUD SIGNAL
    # --------------------------------------------------------

    df["fraud_signal"] = (
        df["fraud_probability"]
        * 100
    )

    df["anomaly_signal"] = (
        df["anomaly_score"]
    )

    # --------------------------------------------------------
    # FINANCIAL EXPOSURE
    # --------------------------------------------------------

    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        *
        df["amount"]
    )

    exposure_reference = (
        df["expected_fraud_loss"]
        .quantile(0.95)
    )

    if (
        not np.isfinite(
            exposure_reference
        )
        or exposure_reference <= 0
    ):

        exposure_reference = 1

    df["financial_exposure_score"] = np.clip(

        (
            df["expected_fraud_loss"]
            /
            exposure_reference
            *
            100
        ),

        0,
        100
    )

    # --------------------------------------------------------
    # FINAL RISK SCORE
    # --------------------------------------------------------

    df["risk_score"] = np.clip(

        (
            0.50 * df["fraud_signal"]

            +

            0.30 * df["anomaly_signal"]

            +

            0.20 * df["financial_exposure_score"]
        ),

        0,
        100
    )

    # --------------------------------------------------------
    # COST-AWARE ACTION
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
            (1 - VERIFY_FRAUD_REDUCTION)
        )
    )

    df["review_cost"] = (

        REVIEW_COST

        +

        (
            df["expected_fraud_loss"]
            *
            (1 - REVIEW_FRAUD_REDUCTION)
        )
    )

    cost_columns = [
        "approve_cost",
        "verify_cost",
        "review_cost"
    ]

    df["recommended_action"] = (
        df[
            cost_columns
        ]
        .idxmin(axis=1)
        .map(
            {
                "approve_cost": "APPROVE",
                "verify_cost": "VERIFY",
                "review_cost": "REVIEW"
            }
        )
    )

    # --------------------------------------------------------
    # FINAL POLICY
    # --------------------------------------------------------

    def choose_action(
        score
    ):

        if score < VERIFY_THRESHOLD:

            return "APPROVE"

        if score < REVIEW_THRESHOLD:

            return "VERIFY"

        return "REVIEW"

    df["final_action"] = (
        df["risk_score"]
        .apply(
            choose_action
        )
    )

    # --------------------------------------------------------
    # RISK BAND
    # --------------------------------------------------------

    df["risk_band"] = (
        df["risk_score"]
        .apply(
            risk_label
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

        (
            df["shared_device"]
            == 1
        ).astype(int)

        +

        (
            df["shared_ip"]
            == 1
        ).astype(int)

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

st.sidebar.title(
    "🛡️ RiskGraph AI"
)

st.sidebar.caption(
    "AI Payment Risk Manager"
)

st.sidebar.divider()


# ============================================================
# DATA UPLOAD
# ============================================================

st.sidebar.subheader(
    "DATA"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help=(
        "Upload a transaction CSV using "
        "sample_transactions.csv as the schema reference."
    )
)


if uploaded_file is not None:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

        source_name = (
            uploaded_file.name
        )

    except Exception as error:

        st.sidebar.error(
            f"CSV could not be read: {error}"
        )

        st.stop()

else:

    try:

        raw_data = load_default_data()

        source_name = (
            DEFAULT_DATA_PATH
        )

    except Exception as error:

        st.sidebar.error(
            "sample_transactions.csv "
            "could not be loaded."
        )

        st.sidebar.code(
            str(error)
        )

        st.stop()


# ============================================================
# RUN RISK ENGINE
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
        "Use sample_transactions.csv "
        "as the reference CSV format."
    )

    st.stop()


# ============================================================
# SIDEBAR STATUS
# ============================================================

st.sidebar.success(
    f"Loaded {len(data):,} transactions"
)

st.sidebar.caption(
    f"Uploaded: {source_name}"
)

st.sidebar.metric(
    "Processed",
    f"{len(data):,}"
)

st.sidebar.metric(
    "Interventions",
    f"{int((data['final_action'] != 'APPROVE').sum()):,}"
)

st.sidebar.divider()

st.sidebar.success(
    "🟠 Risk Engine Online"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.subheader(
    "RISK OPERATIONS"
)

page = st.sidebar.radio(
    "Navigate",
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

    st.title(
        "RiskGraph AI"
    )

    st.subheader(
        "Payment Risk Command Center"
    )

    st.caption(
        "Detect • Explain • Decide"
    )

    st.divider()

    total_transactions = (
        len(data)
    )

    total_value = (
        data["amount"].sum()
    )

    approved = (
        data["final_action"]
        ==
        "APPROVE"
    ).sum()

    verify = (
        data["final_action"]
        ==
        "VERIFY"
    ).sum()

    review = (
        data["final_action"]
        ==
        "REVIEW"
    ).sum()

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Transaction Value",
        f"₹{total_value:,.0f}"
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

    chart_left, chart_right = (
        st.columns(2)
    )

    with chart_left:

        st.subheader(
            "Risk Score Distribution"
        )

        risk_distribution = (
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
            risk_distribution,
            height=320
        )

        st.caption(
            "LOW < 30 • MODERATE 30–59.99 • "
            "HIGH 60–74.99 • CRITICAL ≥ 75"
        )

    with chart_right:

        st.subheader(
            "Decision Mix"
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
                fill_value=0
            )
        )

        st.bar_chart(
            decision_distribution,
            height=320
        )

        st.write(
            f"🟢 APPROVE — {approved:,}"
        )

        st.write(
            f"🟠 VERIFY — {verify:,}"
        )

        st.write(
            f"🔴 REVIEW — {review:,}"
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
        queue_display[
            "fraud_probability"
        ]
        .map(pct)
    )

    queue_display["anomaly_score"] = (
        queue_display[
            "anomaly_score"
        ]
        .round(1)
    )

    queue_display["graph_risk_score"] = (
        queue_display[
            "graph_risk_score"
        ]
        .round(1)
    )

    queue_display[
        "financial_exposure_score"
    ] = (
        queue_display[
            "financial_exposure_score"
        ]
        .round(1)
    )

    queue_display["risk_score"] = (
        queue_display[
            "risk_score"
        ]
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
        "Search transaction / customer / merchant / device / IP",
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
        min_value=0,
        max_value=100,
        value=0
    )

    filtered = data[
        data["final_action"].isin(
            action_filter
        )
        &
        (
            data["risk_score"]
            >= minimum_risk
        )
    ].copy()

    if search.strip():

        q = search.strip().lower()

        mask = (

            filtered[
                "transaction_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                q,
                na=False
            )

            |

            filtered[
                "customer_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                q,
                na=False
            )

            |

            filtered[
                "merchant_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                q,
                na=False
            )

            |

            filtered[
                "device_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                q,
                na=False
            )

            |

            filtered[
                "ip_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                q,
                na=False
            )
        )

        filtered = filtered[
            mask
        ]

    st.metric(
        "Matching Transactions",
        f"{len(filtered):,}"
    )

    if len(filtered) == 0:

        st.warning(
            "No transactions match your filters."
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
            display[
                "fraud_probability"
            ]
            .map(pct)
        )

        display["anomaly_score"] = (
            display[
                "anomaly_score"
            ]
            .round(1)
        )

        display["graph_risk_score"] = (
            display[
                "graph_risk_score"
            ]
            .round(1)
        )

        display["risk_score"] = (
            display[
                "risk_score"
            ]
            .round(1)
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        selected_transaction = (
            st.selectbox(
                "Select transaction to investigate",
                filtered[
                    "transaction_id"
                ].tolist()
            )
        )

        row = filtered[
            filtered[
                "transaction_id"
            ]
            ==
            selected_transaction
        ].iloc[0]

        st.subheader(
            f"Investigation: {selected_transaction}"
        )

        a, b, c, d = (
            st.columns(4)
        )

        a.metric(
            "Risk Score",
            f"{row['risk_score']:.1f}/100"
        )

        b.metric(
            "Fraud Probability",
            pct(
                row[
                    "fraud_probability"
                ]
            )
        )

        c.metric(
            "Anomaly Score",
            f"{row['anomaly_score']:.1f}/100"
        )

        d.metric(
            "Transaction Amount",
            money(
                row["amount"]
            )
        )

        action = row[
            "final_action"
        ]

        if action == "REVIEW":

            st.error(
                "🔴 REVIEW — Analyst investigation recommended."
            )

        elif action == "VERIFY":

            st.warning(
                "🟠 VERIFY — Additional verification recommended."
            )

        else:

            st.success(
                "🟢 APPROVE — Low intervention risk."
            )

        st.subheader(
            "Risk Explanation"
        )

        reasons = get_reasons(
            row
        )

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

        st.subheader(
            "Entity Context"
        )

        e1, e2, e3, e4 = (
            st.columns(4)
        )

        e1.metric(
            "Customers / Device",
            int(
                row[
                    "device_customer_count"
                ]
            )
        )

        e2.metric(
            "Customers / IP",
            int(
                row[
                    "ip_customer_count"
                ]
            )
        )

        e3.metric(
            "Device Transactions",
            int(
                row[
                    "device_transaction_count"
                ]
            )
        )

        e4.metric(
            "IP Transactions",
            int(
                row[
                    "ip_transaction_count"
                ]
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

                    row[
                        "customer_id"
                    ],

                    row[
                        "merchant_id"
                    ],

                    row[
                        "device_id"
                    ],

                    row[
                        "ip_id"
                    ],

                    str(
                        row[
                            "timestamp"
                        ]
                    ),

                    row[
                        "location"
                    ],

                    f"{int(row['account_age_days'])} days",

                    f"{int(row['device_age_days'])} days",

                    int(
                        row[
                            "transactions_last_10min"
                        ]
                    ),

                    int(
                        row[
                            "failed_attempts"
                        ]
                    ),

                    (
                        "Yes"
                        if row[
                            "location_change"
                        ]
                        else
                        "No"
                    ),

                    f"{row['amount_deviation']:.2f}×",

                    int(
                        row[
                            "behavior_risk_count"
                        ]
                    ),

                    money(
                        row[
                            "expected_fraud_loss"
                        ]
                    ),

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
        "RiskGraph AI model and feature pipeline."
    )

    st.divider()

    st.subheader(
        "RiskGraph Architecture"
    )

    architecture = pd.DataFrame(
        {
            "Layer": [

                "Fraud Detection",

                "Behavioural Detection",

                "Entity Intelligence",

                "Financial Exposure",

                "Risk Fusion",

                "Decision Policy",
            ],

            "Purpose": [

                "Predict fraud probability",

                "Detect unusual behaviour",

                "Detect device and IP relationships",

                "Estimate monetary exposure",

                "Combine multiple risk signals",

                "Convert risk into action",
            ]
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Model Features"
    )

    feature_descriptions = [

        "Transaction amount",

        "Log transformed transaction amount",

        "Customer account age",

        "Device age",

        "Transactions during the last 10 minutes",

        "Failed attempts",

        "Location change indicator",

        "Amount deviation",

        "Behaviour risk count",

        "Transaction hour",

        "Day of week",

        "Weekend indicator",

        "High-value transaction indicator",

        "High transaction velocity",

        "High failure activity",

        "New device indicator",

        "New account indicator",
    ]

    feature_table = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,

            "Purpose": feature_descriptions
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Current Dataset Statistics"
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    m1.metric(
        "Transactions",
        f"{len(data):,}"
    )

    m2.metric(
        "Average Risk",
        f"{data['risk_score'].mean():.1f}"
    )

    m3.metric(
        "Average Fraud Probability",
        pct(
            data[
                "fraud_probability"
            ].mean()
        )
    )

    if "is_fraud" in data.columns:

        m4.metric(
            "Actual Fraud Cases",
            f"{int(data['is_fraud'].sum()):,}"
        )

    else:

        m4.metric(
            "Actual Fraud Cases",
            "N/A"
        )

    st.info(
        "Razorpay.ipynb contains the model-development and evaluation work. "
        "The deployed Streamlit application loads the trained "
        "riskgraph_fraud_model_v2.joblib model."
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.title(
        "Business Impact"
    )

    st.caption(
        "Cost-aware fraud decisioning."
    )

    st.divider()

    baseline_loss = (
        data[
            "expected_fraud_loss"
        ].sum()
    )

    approve_loss = (
        data.loc[
            data["final_action"]
            ==
            "APPROVE",
            "expected_fraud_loss"
        ].sum()
    )

    verify_loss = (
        data.loc[
            data["final_action"]
            ==
            "VERIFY",
            "expected_fraud_loss"
        ].sum()
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
            ==
            "REVIEW",
            "expected_fraud_loss"
        ].sum()
        *
        (
            1
            -
            REVIEW_FRAUD_REDUCTION
        )
    )

    residual_loss = (
        approve_loss
        +
        verify_loss
        +
        review_loss
    )

    verification_cost = (
        (
            data["final_action"]
            ==
            "VERIFY"
        ).sum()
        *
        VERIFY_COST
    )

    review_cost = (
        (
            data["final_action"]
            ==
            "REVIEW"
        ).sum()
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

    b1, b2, b3, b4 = (
        st.columns(4)
    )

    b1.metric(
        "Baseline Expected Loss",
        money(
            baseline_loss
        )
    )

    b2.metric(
        "Residual Loss",
        money(
            residual_loss
        )
    )

    b3.metric(
        "Intervention Cost",
        money(
            intervention_cost
        )
    )

    b4.metric(
        "Total Decision Cost",
        money(
            total_cost
        )
    )

    st.divider()

    st.subheader(
        "Decision Economics"
    )

    economics = (
        data
        .groupby(
            "final_action"
        )
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
            )
        )
        .reset_index()
    )

    economics[
        "transaction_value"
    ] = (
        economics[
            "transaction_value"
        ]
        .map(money)
    )

    economics[
        "expected_loss"
    ] = (
        economics[
            "expected_loss"
        ]
        .map(money)
    )

    st.dataframe(
        economics,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "VERIFY and REVIEW costs are prototype assumptions "
        "for cost-aware decisioning and are not claimed to "
        "represent actual Razorpay operating costs."
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

        entity_summary = (
            data
            .groupby(
                "device_id"
            )
            .agg(
                customers=(
                    "customer_id",
                    "nunique"
                ),

                transactions=(
                    "transaction_id",
                    "count"
                ),

                average_risk=(
                    "risk_score",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                [
                    "customers",
                    "transactions"
                ],
                ascending=False
            )
            .head(30)
        )

        entity_column = (
            "device_id"
        )

    elif entity_type == "IP Address":

        entity_summary = (
            data
            .groupby(
                "ip_id"
            )
            .agg(
                customers=(
                    "customer_id",
                    "nunique"
                ),

                transactions=(
                    "transaction_id",
                    "count"
                ),

                average_risk=(
                    "risk_score",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                [
                    "customers",
                    "transactions"
                ],
                ascending=False
            )
            .head(30)
        )

        entity_column = (
            "ip_id"
        )

    else:

        entity_summary = (
            data
            .groupby(
                "customer_id"
            )
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

                average_risk=(
                    "risk_score",
                    "mean"
                )
            )
            .reset_index()
            .sort_values(
                "transactions",
                ascending=False
            )
            .head(30)
        )

        entity_column = (
            "customer_id"
        )

    st.dataframe(
        entity_summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        f"Investigate {entity_type}"
    )

    selected_entity = st.selectbox(
        "Select entity",
        entity_summary[
            entity_column
        ].tolist()
    )

    if entity_type == "Device":

        related = data[
            data["device_id"]
            ==
            selected_entity
        ]

    elif entity_type == "IP Address":

        related = data[
            data["ip_id"]
            ==
            selected_entity
        ]

    else:

        related = data[
            data["customer_id"]
            ==
            selected_entity
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
            "final_action"
        ]
    ].copy()

    related_display[
        "amount"
    ] = (
        related_display[
            "amount"
        ]
        .map(money)
    )

    related_display[
        "risk_score"
    ] = (
        related_display[
            "risk_score"
        ]
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
        "Inspect and export RiskGraph results."
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

        "recommended_action",

        "final_action",
    ]

    default_columns = [

        column

        for column in default_columns

        if column in data.columns
    ]

    selected_columns = st.multiselect(
        "Columns to display",
        data.columns.tolist(),
        default=default_columns
    )

    if selected_columns:

        display_data = data[
            selected_columns
        ].copy()

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        csv_output = (
            display_data
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            "Download RiskGraph Results",
            data=csv_output,
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

st.caption(
    "RiskGraph AI • AI Payment Risk Manager • "
    "Fraud Detection • Behavioural Anomaly Detection • "
    "Entity Intelligence • Cost-Aware Decisioning"
)

st.caption(
    "Synthetic-data prototype • Razorpay RiskGraph project"
)
