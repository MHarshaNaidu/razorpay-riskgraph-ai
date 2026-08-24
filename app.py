import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT FILES
# ============================================================

MODEL_PATH = "riskgraph_fraud_model_v2.joblib"
DATA_PATH = "sample_transactions.csv"


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


# ============================================================
# PROFESSIONAL LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC;
    }

    [data-testid="stHeader"] {
        background-color: #F8FAFC;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    p {
        color: #475569 !important;
    }

    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    section[data-testid="stSidebar"] p {
        color: #CBD5E1 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155;
    }

    /* METRICS */

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #64748B !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* BUTTONS */

    .stButton > button {
        background-color: #F97316;
        color: white;
        border: 1px solid #F97316;
        border-radius: 8px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #EA580C;
        border-color: #EA580C;
        color: white;
    }

    /* FILE UPLOADER */

    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border-radius: 10px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF;
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
    }

    /* SELECTBOX */

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        color: #0F172A;
        border-color: #CBD5E1;
    }

    div[data-baseweb="select"] * {
        color: #0F172A !important;
    }

    /* DATAFRAME */

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        background: white;
    }

    /* TEXT INPUT */

    input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
    }

    /* DOWNLOAD */

    .stDownloadButton > button {
        background-color: #0F172A;
        color: white;
        border: 1px solid #0F172A;
        border-radius: 8px;
        font-weight: 700;
    }

    .stDownloadButton > button:hover {
        background-color: #1E293B;
        color: white;
    }

    /* DIVIDER */

    hr {
        border-color: #E2E8F0 !important;
    }

    /* CUSTOM NATIVE-STYLE CARDS */

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
    }

    .orange-label {
        color: #F97316;
        font-weight: 800;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .risk-review {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        padding: 14px;
        border-radius: 10px;
        color: #991B1B;
        font-weight: 700;
    }

    .risk-verify {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        padding: 14px;
        border-radius: 10px;
        color: #92400E;
        font-weight: 700;
    }

    .risk-approve {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        padding: 14px;
        border-radius: 10px;
        color: #166534;
        font-weight: 700;
    }

    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.8rem;
        padding-top: 20px;
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


def pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


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

    model_data = joblib.load(MODEL_PATH)

    if isinstance(model_data, dict):

        model = model_data.get("model")

        if model is None:
            raise ValueError(
                "Model file does not contain a model."
            )

        features = model_data.get(
            "features",
            MODEL_FEATURES
        )

        return model, list(features)

    model = model_data

    if hasattr(model, "feature_names_in_"):

        features = list(
            model.feature_names_in_
        )

    else:

        features = MODEL_FEATURES

    return model, features


# ============================================================
# LOAD SAMPLE DATA
# ============================================================

@st.cache_data
def load_sample_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(df):

    df = df.copy()

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
            "Missing required columns: "
            + ", ".join(missing)
        )

    # Numeric columns

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

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    # Timestamp

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values detected."
        )

    # Time features

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Amount

    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    # IMPORTANT MODEL FEATURE

    df["high_value_transaction"] = (
        df["amount"] >= 10000
    ).astype(int)

    # Velocity

    df["high_velocity"] = (
        df["transactions_last_10min"] >= 4
    ).astype(int)

    # Failed activity

    df["high_failure_activity"] = (
        df["failed_attempts"] >= 3
    ).astype(int)

    # Device

    df["new_device"] = (
        df["device_age_days"] < 14
    ).astype(int)

    # Account

    df["new_account"] = (
        df["account_age_days"] < 60
    ).astype(int)

    # Behaviour risk

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

    # Ensure every expected feature exists

    for feature in MODEL_FEATURES:

        if feature not in df.columns:
            df[feature] = 0

    return df


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(df):

    df = df.copy()

    anomaly_features = [
        "amount_deviation",
        "transactions_last_10min",
        "failed_attempts",
        "device_age_days",
        "location_change",
        "account_age_days",
        "behavior_risk_count",
    ]

    if len(df) < 10:

        df["anomaly_score"] = 0.0

        return df

    X = (
        df[anomaly_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    detector = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42
    )

    detector.fit(X)

    raw = -detector.decision_function(X)

    low = raw.min()
    high = raw.max()

    if high - low == 0:

        score = np.zeros(
            len(raw)
        )

    else:

        score = (
            (raw - low)
            /
            (high - low)
        ) * 100

    df["anomaly_score"] = np.clip(
        score,
        0,
        100
    )

    return df


# ============================================================
# ENTITY RISK
# ============================================================

def calculate_entity_risk(df):

    df = df.copy()

    device_customers = (
        df.groupby("device_id")["customer_id"]
        .nunique()
    )

    ip_customers = (
        df.groupby("ip_id")["customer_id"]
        .nunique()
    )

    device_transactions = (
        df.groupby("device_id")
        .size()
    )

    ip_transactions = (
        df.groupby("ip_id")
        .size()
    )

    df["device_customer_count"] = (
        df["device_id"]
        .map(device_customers)
        .fillna(0)
    )

    df["ip_customer_count"] = (
        df["ip_id"]
        .map(ip_customers)
        .fillna(0)
    )

    df["device_transaction_count"] = (
        df["device_id"]
        .map(device_transactions)
        .fillna(0)
    )

    df["ip_transaction_count"] = (
        df["ip_id"]
        .map(ip_transactions)
        .fillna(0)
    )

    df["shared_device"] = (
        df["device_customer_count"] > 1
    ).astype(int)

    df["shared_ip"] = (
        df["ip_customer_count"] > 1
    ).astype(int)

    df["graph_risk_score"] = np.clip(

        np.minimum(
            df["device_customer_count"] * 12,
            35
        )

        +

        np.minimum(
            df["ip_customer_count"] * 12,
            35
        )

        +

        np.minimum(
            df["device_transaction_count"],
            15
        )

        +

        np.minimum(
            df["ip_transaction_count"],
            15
        ),

        0,
        100
    )

    return df


# ============================================================
# EXPLANATION
# ============================================================

def get_reasons(row):

    reasons = []

    if row["fraud_probability"] >= 0.75:

        reasons.append(
            "Very high fraud probability."
        )

    elif row["fraud_probability"] >= 0.50:

        reasons.append(
            "Elevated fraud probability."
        )

    if row["anomaly_score"] >= 75:

        reasons.append(
            "Highly unusual behavioural pattern."
        )

    elif row["anomaly_score"] >= 50:

        reasons.append(
            "Behaviour differs from normal activity."
        )

    if row["financial_exposure_score"] >= 75:

        reasons.append(
            "High financial exposure."
        )

    if row["transactions_last_10min"] >= 4:

        reasons.append(
            "High transaction velocity."
        )

    if row["failed_attempts"] >= 3:

        reasons.append(
            "Multiple failed payment attempts."
        )

    if row["location_change"] == 1:

        reasons.append(
            "Location change detected."
        )

    if row["new_device"] == 1:

        reasons.append(
            "New device detected."
        )

    if row["new_account"] == 1:

        reasons.append(
            "New account detected."
        )

    if row["amount_deviation"] > 3:

        reasons.append(
            "Transaction amount is significantly above normal."
        )

    if row["shared_device"] == 1:

        reasons.append(
            "Device is associated with multiple customers."
        )

    if row["shared_ip"] == 1:

        reasons.append(
            "IP address is associated with multiple customers."
        )

    if not reasons:

        reasons.append(
            "No major risk indicators detected."
        )

    return reasons


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def process_transactions(raw_df):

    df = prepare_features(
        raw_df
    )

    model, model_features = load_model()

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Model features missing: "
            + ", ".join(missing_features)
        )

    X = (
        df[model_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # Fraud probability

    if not hasattr(
        model,
        "predict_proba"
    ):

        raise ValueError(
            "The trained model does not support probability prediction."
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

    # Anomaly

    df = calculate_anomaly_score(
        df
    )

    # Entity

    df = calculate_entity_risk(
        df
    )

    # Financial exposure

    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        *
        df["amount"]
    )

    reference = (
        df["expected_fraud_loss"]
        .quantile(0.95)
    )

    if reference <= 0:
        reference = 1

    df["financial_exposure_score"] = np.clip(

        (
            df["expected_fraud_loss"]
            /
            reference
        )
        * 100,

        0,
        100
    )

    # Risk score

    df["risk_score"] = np.clip(

        (
            0.50
            *
            df["fraud_probability"]
            *
            100
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

    # Risk band

    df["risk_band"] = (
        df["risk_score"]
        .apply(risk_band)
    )

    # Decision

    def make_decision(score):

        if score >= REVIEW_THRESHOLD:
            return "REVIEW"

        if score >= VERIFY_THRESHOLD:
            return "VERIFY"

        return "APPROVE"

    df["final_action"] = (
        df["risk_score"]
        .apply(make_decision)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

st.sidebar.title("🛡️ RiskGraph AI")

st.sidebar.caption(
    "AI Payment Risk Manager"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Data"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

        source_name = uploaded_file.name

    except Exception as error:

        st.sidebar.error(
            f"CSV error: {error}"
        )

        st.stop()

else:

    try:

        raw_data = load_sample_data()

        source_name = DATA_PATH

    except Exception as error:

        st.sidebar.error(
            "Could not load sample_transactions.csv"
        )

        st.sidebar.code(
            str(error)
        )

        st.stop()


# ============================================================
# PROCESS
# ============================================================

try:

    data = process_transactions(
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
        "Use sample_transactions.csv as the reference CSV format."
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
    "Processed Transactions",
    len(data)
)

interventions = (
    data["final_action"]
    != "APPROVE"
).sum()

st.sidebar.metric(
    "Interventions",
    interventions
)

st.sidebar.divider()

st.sidebar.subheader(
    "Risk Operations"
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
        "🛡️ RiskGraph AI"
    )

    st.subheader(
        "Payment Risk Command Center"
    )

    st.caption(
        "Detect • Explain • Decide"
    )

    st.divider()

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
        approved
    )

    c4.metric(
        "Verify",
        verify
    )

    c5.metric(
        "Review",
        review
    )

    st.divider()

    left, right = st.columns(2)

    with left:

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

    with right:

        st.subheader(
            "Decision Mix"
        )

        decisions = (
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
            decisions,
            height=320
        )

        st.success(
            f"APPROVE — {approved:,} transactions"
        )

        st.warning(
            f"VERIFY — {verify:,} transactions"
        )

        st.error(
            f"REVIEW — {review:,} transactions"
        )

    st.divider()

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
        .map(pct)
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
        "RiskGraph combines fraud probability, behavioural anomaly, "
        "entity intelligence and financial exposure into a final "
        "0–100 risk score."
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
        "Decision",
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

    if search:

        query = search.lower()

        mask = (
            filtered["transaction_id"]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
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
                .str.contains(
                    query,
                    na=False
                )
            )

        filtered = filtered[mask]

    st.metric(
        "Matching Transactions",
        len(filtered)
    )

    if len(filtered) == 0:

        st.warning(
            "No transactions match the selected filters."
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
            .map(pct)
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
            "Select transaction",
            filtered[
                "transaction_id"
            ].tolist()
        )

        row = filtered[
            filtered["transaction_id"]
            == selected
        ].iloc[0]

        st.subheader(
            f"Investigation: {selected}"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Risk Score",
            f"{row['risk_score']:.1f}/100"
        )

        b.metric(
            "Fraud Probability",
            pct(row["fraud_probability"])
        )

        c.metric(
            "Anomaly Score",
            f"{row['anomaly_score']:.1f}/100"
        )

        d.metric(
            "Transaction Amount",
            money(row["amount"])
        )

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
                    f"{row['account_age_days']:.0f} days",
                    f"{row['device_age_days']:.0f} days",
                    int(row["transactions_last_10min"]),
                    int(row["failed_attempts"]),
                    "Yes" if row["location_change"] else "No",
                    f"{row['amount_deviation']:.2f}x",
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
        "How RiskGraph converts transaction signals into risk."
    )

    st.divider()

    st.subheader(
        "Risk Decision Pipeline"
    )

    pipeline = pd.DataFrame(
        {
            "Stage": [
                "1. Transaction Input",
                "2. Feature Engineering",
                "3. Fraud Probability",
                "4. Behavioural Anomaly",
                "5. Entity Intelligence",
                "6. Financial Exposure",
                "7. Risk Fusion",
                "8. Decision",
            ],

            "Purpose": [
                "Receive payment transaction data",
                "Create model-ready behavioural features",
                "Estimate probability of fraud",
                "Identify unusual transaction behaviour",
                "Analyse device and IP relationships",
                "Estimate potential monetary exposure",
                "Combine risk signals into 0–100 score",
                "Approve, Verify or Review",
            ]
        }
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Risk Score Composition"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Fraud Signal",
        "50%"
    )

    b.metric(
        "Anomaly Signal",
        "30%"
    )

    c.metric(
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

            "Decision": [
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

    feature_table = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Role": [
                "Transaction amount",
                "Log transaction amount",
                "Customer account age",
                "Device age",
                "Transaction velocity",
                "Failed attempts",
                "Location change",
                "Amount deviation",
                "Behaviour risk indicators",
                "Transaction hour",
                "Day of week",
                "Weekend indicator",
                "High-value transaction",
                "High transaction velocity",
                "High failure activity",
                "New device",
                "New account",
            ]
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
        "Cost-aware payment risk decisioning."
    )

    st.divider()

    baseline_loss = (
        data["expected_fraud_loss"]
        .sum()
    )

    approved_loss = (
        data.loc[
            data["final_action"] == "APPROVE",
            "expected_fraud_loss"
        ]
        .sum()
    )

    verify_loss = (
        data.loc[
            data["final_action"] == "VERIFY",
            "expected_fraud_loss"
        ]
        .sum()
        * 0.20
    )

    review_loss = (
        data.loc[
            data["final_action"] == "REVIEW",
            "expected_fraud_loss"
        ]
        .sum()
        * 0.05
    )

    residual_loss = (
        approved_loss
        +
        verify_loss
        +
        review_loss
    )

    intervention_cost = (
        (
            data["final_action"] == "VERIFY"
        ).sum()
        * 25
    ) + (
        (
            data["final_action"] == "REVIEW"
        ).sum()
        * 75
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
        "Intervention costs are prototype assumptions used "
        "to demonstrate cost-aware risk decisioning."
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
        "Entity type",
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
        "Inspect and export processed RiskGraph results."
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
        column
        for column in default_columns
        if column in data.columns
    ]

    columns = st.multiselect(
        "Columns",
        data.columns.tolist(),
        default=default_columns
    )

    if columns:

        display = data[
            columns
        ].copy()

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        csv = (
            display
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download RiskGraph Results",
            data=csv,
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
