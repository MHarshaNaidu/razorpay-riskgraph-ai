import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


# ============================================================
# RISKGRAPH AI
# PAYMENT RISK INTELLIGENCE PLATFORM
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "riskgraph_fraud_model_v2.joblib"
DEFAULT_DATA_PATH = "sample_transactions.csv"

APPROVE_THRESHOLD = 60
VERIFY_THRESHOLD = 75

ANOMALY_FEATURES = [
    "amount_deviation",
    "transactions_last_10min",
    "failed_attempts",
    "device_age_days",
    "location_change",
    "account_age_days",
    "behavior_risk_count",
]


# ============================================================
# COLORS
# ============================================================

ORANGE = "#FF7900"
ORANGE_LIGHT = "#FF9638"
BLACK = "#111111"
DARK = "#222222"
GRAY = "#666666"
LIGHT_GRAY = "#F4F4F4"
MID_GRAY = "#E6E6E6"
WHITE = "#FFFFFF"

GREEN = "#168A52"
RED = "#D92D20"


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

* {{
    font-family: Inter, Arial, sans-serif;
}}

html {{
    scroll-behavior: smooth;
}}

body {{
    background: {WHITE};
}}

.stApp {{
    background: {WHITE};
    color: {BLACK};
}}

.main {{
    background: {WHITE};
}}

.block-container {{
    max-width: 1240px;
    padding-top: 1rem;
    padding-bottom: 5rem;
}}


/* ============================================================
   TOP NAVIGATION
   ============================================================ */

.nav-wrapper {{
    width: 100%;
    border-bottom: 1px solid {MID_GRAY};
    background: {WHITE};
    padding: 8px 0 12px 0;
    margin-bottom: 30px;
}}

.brand {{
    font-size: 22px;
    font-weight: 800;
    color: {BLACK};
    letter-spacing: -0.8px;
}}

.brand-orange {{
    color: {ORANGE};
}}

.brand-sub {{
    color: {GRAY};
    font-size: 11px;
    margin-top: 2px;
}}


/* ============================================================
   HERO
   ============================================================ */

.hero {{
    padding: 50px 0 55px 0;
}}

.hero-kicker {{
    color: {ORANGE};
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 15px;
}}

.hero-title {{
    color: {BLACK};
    font-size: 58px;
    line-height: 1.02;
    font-weight: 800;
    letter-spacing: -3px;
    max-width: 900px;
    margin-bottom: 20px;
}}

.hero-title span {{
    color: {ORANGE};
}}

.hero-description {{
    color: {GRAY};
    font-size: 18px;
    line-height: 1.65;
    max-width: 700px;
}}

.hero-line {{
    width: 65px;
    height: 5px;
    background: {ORANGE};
    margin-top: 28px;
}}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section {{
    padding: 55px 0;
    border-top: 1px solid {MID_GRAY};
}}

.section-label {{
    color: {ORANGE};
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 10px;
}}

.section-title {{
    color: {BLACK};
    font-size: 34px;
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -1.4px;
}}

.section-text {{
    color: {GRAY};
    font-size: 15px;
    line-height: 1.7;
    max-width: 700px;
}}


/* ============================================================
   METRICS
   ============================================================ */

.metric-box {{
    padding: 25px 0;
    border-top: 3px solid {BLACK};
}}

.metric-number {{
    color: {BLACK};
    font-size: 35px;
    font-weight: 800;
    letter-spacing: -1.5px;
}}

.metric-label {{
    color: {GRAY};
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-top: 5px;
}}

.metric-orange {{
    border-top-color: {ORANGE};
}}


/* ============================================================
   ORANGE FEATURE
   ============================================================ */

.orange-panel {{
    background: {ORANGE};
    color: {WHITE};
    padding: 42px;
    min-height: 250px;
}}

.orange-panel h2 {{
    color: {WHITE};
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
}}

.orange-panel p {{
    color: {WHITE};
    opacity: 0.92;
    line-height: 1.7;
}}


/* ============================================================
   LIGHT PANEL
   ============================================================ */

.light-panel {{
    background: {LIGHT_GRAY};
    padding: 35px;
}}

.light-panel-title {{
    color: {BLACK};
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 10px;
}}

.light-panel-text {{
    color: {GRAY};
    line-height: 1.6;
}}


/* ============================================================
   RISK BARS
   ============================================================ */

.risk-row {{
    margin-bottom: 20px;
}}

.risk-header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
}}

.risk-name {{
    font-size: 13px;
    font-weight: 700;
    color: {BLACK};
}}

.risk-value {{
    font-size: 13px;
    font-weight: 700;
    color: {ORANGE};
}}

.risk-track {{
    width: 100%;
    height: 9px;
    background: {MID_GRAY};
}}

.risk-fill {{
    height: 100%;
    background: {ORANGE};
}}


/* ============================================================
   DECISION BADGES
   ============================================================ */

.badge {{
    display: inline-block;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.5px;
}}

.badge-approve {{
    background: #E9F7EF;
    color: {GREEN};
}}

.badge-verify {{
    background: #FFF2E5;
    color: {ORANGE};
}}

.badge-review {{
    background: #FDECEC;
    color: {RED};
}}


/* ============================================================
   FEATURE GRID
   ============================================================ */

.feature-box {{
    padding: 28px;
    border: 1px solid {MID_GRAY};
    min-height: 180px;
    background: {WHITE};
}}

.feature-number {{
    color: {ORANGE};
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}}

.feature-title {{
    color: {BLACK};
    font-size: 20px;
    font-weight: 800;
    margin: 15px 0 10px 0;
}}

.feature-description {{
    color: {GRAY};
    font-size: 14px;
    line-height: 1.6;
}}


/* ============================================================
   UPLOAD
   ============================================================ */

.upload-box {{
    background: {LIGHT_GRAY};
    padding: 28px;
    border-left: 5px solid {ORANGE};
}}

.upload-title {{
    font-size: 18px;
    color: {BLACK};
    font-weight: 800;
}}

.upload-sub {{
    color: {GRAY};
    font-size: 13px;
    margin-top: 5px;
}}


/* ============================================================
   STREAMLIT CONTROLS
   ============================================================ */

.stButton > button {{
    background: {ORANGE} !important;
    color: {WHITE} !important;
    border: 1px solid {ORANGE} !important;
    border-radius: 0px !important;
    font-weight: 800 !important;
    padding: 9px 22px !important;
}}

.stButton > button:hover {{
    background: {BLACK} !important;
    border-color: {BLACK} !important;
    color: {WHITE} !important;
}}

.stDownloadButton > button {{
    background: {BLACK} !important;
    color: {WHITE} !important;
    border-radius: 0px !important;
    border: 1px solid {BLACK} !important;
    font-weight: 700 !important;
}}

.stDownloadButton > button:hover {{
    background: {ORANGE} !important;
    border-color: {ORANGE} !important;
}}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {{
    background: {WHITE};
}}

[data-testid="stFileUploaderDropzone"] {{
    background: {WHITE};
    border: 2px dashed {MID_GRAY};
    border-radius: 0px;
}}

[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {ORANGE};
}}


/* ============================================================
   INPUTS
   ============================================================ */

input,
textarea {{
    background: {WHITE} !important;
    color: {BLACK} !important;
    border: 1px solid {MID_GRAY} !important;
    border-radius: 0px !important;
}}

div[data-baseweb="select"] > div {{
    background: {WHITE} !important;
    color: {BLACK} !important;
    border: 1px solid {MID_GRAY} !important;
    border-radius: 0px !important;
}}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {{
    border: 1px solid {MID_GRAY};
}}


/* ============================================================
   ALERTS
   ============================================================ */

[data-testid="stAlert"] {{
    border-radius: 0px;
}}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {{
    background: {BLACK};
    color: {WHITE};
    padding: 45px;
    margin-top: 60px;
}}

.footer-title {{
    color: {WHITE};
    font-size: 20px;
    font-weight: 800;
}}

.footer-text {{
    color: #BBBBBB;
    font-size: 12px;
    line-height: 1.7;
}}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 800px) {{

    .hero-title {{
        font-size: 42px;
        letter-spacing: -2px;
    }}

    .section-title {{
        font-size: 28px;
    }}

    .hero {{
        padding: 35px 0;
    }}
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    value = float(value)

    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"

    if value >= 100_000:
        return f"₹{value / 100_000:.2f} L"

    if value >= 1_000:
        return f"₹{value / 1_000:.1f} K"

    return f"₹{value:,.0f}"


def money_full(value):
    return f"₹{float(value):,.2f}"


def pct(value):
    return f"{float(value) * 100:.2f}%"


def decision_badge(action):

    if action == "APPROVE":
        return '<span class="badge badge-approve">APPROVE</span>'

    if action == "VERIFY":
        return '<span class="badge badge-verify">VERIFY</span>'

    return '<span class="badge badge-review">REVIEW</span>'


def risk_reason(row):

    reasons = []

    if row["fraud_probability"] >= 0.50:
        reasons.append(
            f"High fraud probability: "
            f"{row['fraud_probability']:.1%}"
        )

    if row["anomaly_score"] >= 75:
        reasons.append(
            f"Highly unusual behavioral pattern "
            f"({row['anomaly_score']:.1f}/100)"
        )

    if row["graph_risk_score"] >= 60:
        reasons.append(
            f"Elevated entity/network risk "
            f"({row['graph_risk_score']:.1f}/100)"
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

    if row["location_change"]:
        reasons.append(
            "Location change detected"
        )

    if row["new_device"]:
        reasons.append(
            "New device detected"
        )

    if row["new_account"]:
        reasons.append(
            "New account detected"
        )

    if row["amount_deviation"] > 3:
        reasons.append(
            "Transaction amount is significantly "
            "above normal behavior"
        )

    if not reasons:
        reasons.append(
            "No major risk indicators detected"
        )

    return reasons


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
                "Model bundle does not contain model."
            )

        if "features" not in bundle:
            raise ValueError(
                "Model bundle does not contain features."
            )

        return (
            bundle["model"],
            bundle["features"]
        )

    raise ValueError(
        "Unsupported model format."
    )


# ============================================================
# LOAD DEFAULT DATA
# ============================================================

@st.cache_data
def load_default_data():

    return pd.read_csv(
        DEFAULT_DATA_PATH
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

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
        x for x in required
        if x not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp values found."
        )

    numeric = [
        "amount",
        "account_age_days",
        "device_age_days",
        "transactions_last_10min",
        "failed_attempts",
        "location_change",
        "amount_deviation",
    ]

    for col in numeric:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["log_amount"] = np.log1p(
        df["amount"]
    )

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

    return df


# ============================================================
# ANOMALY ENGINE
# ============================================================

def calculate_anomaly(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    split = max(
        int(len(work) * 0.80),
        50
    )

    historical = work.iloc[
        :split
    ].copy()

    if "is_fraud" in historical.columns:

        legitimate = historical[
            historical["is_fraud"] == 0
        ]

        if len(legitimate) < 50:
            legitimate = historical

    else:

        legitimate = historical

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        legitimate[
            ANOMALY_FEATURES
        ]
    )

    train_score = -model.decision_function(
        legitimate[
            ANOMALY_FEATURES
        ]
    )

    all_score = -model.decision_function(
        work[
            ANOMALY_FEATURES
        ]
    )

    sorted_score = np.sort(
        train_score
    )

    percentile = (
        np.searchsorted(
            sorted_score,
            all_score,
            side="right",
        )
        / len(sorted_score)
    )

    work["anomaly_score"] = np.clip(
        percentile * 100,
        0,
        100,
    )

    return work


# ============================================================
# ENTITY ENGINE
# ============================================================

def calculate_entities(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    split = max(
        int(len(work) * 0.80),
        50
    )

    historical = work.iloc[
        :split
    ]

    device_customers = (
        historical
        .groupby("device_id")[
            "customer_id"
        ]
        .nunique()
    )

    ip_customers = (
        historical
        .groupby("ip_id")[
            "customer_id"
        ]
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
# MAIN RISK PIPELINE
# ============================================================

def run_pipeline(raw):

    df = engineer_features(
        raw
    )

    model, features = load_model()

    missing = [
        x for x in features
        if x not in df.columns
    ]

    if missing:

        raise ValueError(
            "Model features missing: "
            + ", ".join(missing)
        )

    # Fraud probability

    df["fraud_probability"] = (
        model
        .predict_proba(
            df[features]
        )[:, 1]
    )

    # Anomaly

    df = calculate_anomaly(
        df
    )

    # Entity intelligence

    df = calculate_entities(
        df
    )

    # Financial exposure

    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        * df["amount"]
    )

    cap = max(
        df["expected_fraud_loss"]
        .quantile(0.95),
        1
    )

    df["financial_exposure_score"] = (
        df["expected_fraud_loss"]
        / cap
        * 100
    )

    df["financial_exposure_score"] = np.clip(
        df["financial_exposure_score"],
        0,
        100
    )

    # Final risk score

    df["risk_score"] = (

        0.50
        * df["fraud_probability"]
        * 100

        +

        0.30
        * df["anomaly_score"]

        +

        0.20
        * df["financial_exposure_score"]
    )

    df["risk_score"] = np.clip(
        df["risk_score"],
        0,
        100
    )

    # Final action

    df["final_action"] = np.select(
        [
            df["risk_score"] < APPROVE_THRESHOLD,
            df["risk_score"] < VERIFY_THRESHOLD,
        ],
        [
            "APPROVE",
            "VERIFY",
        ],
        default="REVIEW",
    )

    # Risk band

    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[
            -0.01,
            30,
            60,
            75,
            100.01,
        ],
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ],
    ).astype(str)

    return df


# ============================================================
# DATA UPLOAD
# ============================================================

if "data" not in st.session_state:

    try:

        st.session_state.data = (
            load_default_data()
        )

    except Exception as e:

        st.error(
            "Could not load sample_transactions.csv"
        )

        st.exception(e)

        st.stop()


uploaded = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded is not None:

    try:

        uploaded_df = pd.read_csv(
            uploaded
        )

        st.session_state.data = uploaded_df

        st.success(
            f"Loaded {len(uploaded_df):,} transactions from {uploaded.name}"
        )

    except Exception as e:

        st.error(
            "Could not read the uploaded CSV."
        )

        st.exception(e)

        st.stop()


raw_data = st.session_state.data


# ============================================================
# RUN PIPELINE
# ============================================================

try:

    with st.spinner(
        "RiskGraph AI is analyzing transactions..."
    ):

        data = run_pipeline(
            raw_data
        )

except Exception as e:

    st.error(
        "RiskGraph AI could not process this dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

nav_left, nav_center, nav_right = st.columns(
    [2.5, 6, 1.5]
)

with nav_left:

    st.markdown(
        """
        <div class="brand">
            Risk<span class="brand-orange">Graph</span> AI
        </div>
        <div class="brand-sub">
            AI PAYMENT RISK INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_center:

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Investigate",
            "Entity Intelligence",
            "Model Intelligence",
            "Business Impact",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

with nav_right:

    st.markdown(
        f"""
        <div style="
            text-align:right;
            padding-top:10px;
            font-size:12px;
            font-weight:700;
            color:{GREEN};
        ">
            ● ENGINE ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="nav-wrapper"></div>',
    unsafe_allow_html=True,
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                Payment Risk Command Center
            </div>

            <div class="hero-title">
                Detect risk.<br>
                <span>Explain decisions.</span><br>
                Protect payments.
            </div>

            <div class="hero-description">
                RiskGraph AI combines fraud probability,
                behavioral anomalies, entity intelligence
                and financial exposure into an actionable
                payment risk decision.
            </div>

            <div class="hero-line"></div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload area

    st.markdown(
        """
        <div class="upload-box">

            <div class="upload-title">
                Analyze a transaction dataset
            </div>

            <div class="upload-sub">
                Upload a CSV to run the complete
                RiskGraph AI decision pipeline.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload is already rendered above at top;
    # this section explains it.

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">LIVE DATA</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Risk at a glance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-text">'
        f"Currently analyzing {len(data):,} transactions."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    total = len(data)

    total_value = data["amount"].sum()

    approve = (
        data["final_action"] == "APPROVE"
    ).sum()

    verify = (
        data["final_action"] == "VERIFY"
    ).sum()

    review = (
        data["final_action"] == "REVIEW"
    ).sum()

    m1, m2, m3, m4, m5 = st.columns(5)

    metrics = [
        (m1, total, "Transactions", False),
        (m2, money(total_value), "Transaction Value", True),
        (m3, approve, "Approved", False),
        (m4, verify, "Verify", True),
        (m5, review, "Review", False),
    ]

    for col, value, label, orange in metrics:

        with col:

            st.markdown(
                f"""
                <div class="metric-box {'metric-orange' if orange else ''}">
                    <div class="metric-number">
                        {value}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Risk section

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.2, 1]
    )

    with left:

        st.markdown(
            '<div class="section-label">'
            'RISK INTELLIGENCE'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">'
            'Understand the risk.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-text">'
            'Every transaction receives a 0–100 risk score '
            'built from multiple independent signals.'
            '</div>',
            unsafe_allow_html=True,
        )

        risk_counts = (
            data["risk_band"]
            .value_counts()
            .reindex(
                [
                    "LOW",
                    "MODERATE",
                    "HIGH",
                    "CRITICAL",
                ]
            )
            .fillna(0)
        )

        for label in [
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ]:

            count = int(
                risk_counts[label]
            )

            percentage = (
                count / total * 100
                if total
                else 0
            )

            st.markdown(
                f"""
                <div class="risk-row">

                    <div class="risk-header">

                        <div class="risk-name">
                            {label}
                        </div>

                        <div class="risk-value">
                            {count:,} · {percentage:.1f}%
                        </div>

                    </div>

                    <div class="risk-track">

                        <div class="risk-fill"
                             style="width:{percentage}%;">
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:

        st.markdown(
            """
            <div class="orange-panel">

                <h2>
                    One score.
                    Multiple signals.
                </h2>

                <p>
                    RiskGraph turns complex payment signals
                    into a single decision-ready risk score.
                </p>

                <p>
                    <b>Fraud Model</b><br>
                    Probability of fraudulent behavior.
                </p>

                <p>
                    <b>Anomaly Engine</b><br>
                    Detects unusual transaction behavior.
                </p>

                <p>
                    <b>Entity Intelligence</b><br>
                    Connects customers, devices and IPs.
                </p>

                <p>
                    <b>Financial Exposure</b><br>
                    Measures potential monetary impact.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Decision section

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">DECISION ENGINE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        'From prediction to action.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-text">'
        'The system converts risk into an operational '
        'decision instead of stopping at a model prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.markdown(
            """
            <div class="feature-box">

                <div class="feature-number">
                    01
                </div>

                <div class="feature-title">
                    APPROVE
                </div>

                <div class="feature-description">
                    Low intervention risk.
                    Payment proceeds normally.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:

        st.markdown(
            """
            <div class="feature-box">

                <div class="feature-number">
                    02
                </div>

                <div class="feature-title">
                    VERIFY
                </div>

                <div class="feature-description">
                    Medium risk.
                    Request additional verification.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with d3:

        st.markdown(
            """
            <div class="feature-box">

                <div class="feature-number">
                    03
                </div>

                <div class="feature-title">
                    REVIEW
                </div>

                <div class="feature-description">
                    High risk.
                    Send the transaction to investigation.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Priority queue

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">'
        'PRIORITY QUEUE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        'Transactions that need attention.'
        '</div>',
        unsafe_allow_html=True,
    )

    queue = (
        data
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(12)
        .copy()
    )

    display = queue[
        [
            "transaction_id",
            "amount",
            "fraud_probability",
            "anomaly_score",
            "graph_risk_score",
            "risk_score",
            "final_action",
        ]
    ].copy()

    display["amount"] = display[
        "amount"
    ].map(money_full)

    display["fraud_probability"] = display[
        "fraud_probability"
    ].map(pct)

    display["anomaly_score"] = display[
        "anomaly_score"
    ].round(1)

    display["graph_risk_score"] = display[
        "graph_risk_score"
    ].round(1)

    display["risk_score"] = display[
        "risk_score"
    ].round(1)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# INVESTIGATE
# ============================================================

elif page == "Investigate":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                Risk Investigation
            </div>

            <div class="hero-title">
                Explain <span>why</span>
                a payment is risky.
            </div>

            <div class="hero-description">
                Move from a risk score to the underlying
                evidence behind the decision.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Search transaction, customer, merchant, device or IP",
        placeholder="TX_0013175",
    )

    actions = st.multiselect(
        "Decision",
        [
            "APPROVE",
            "VERIFY",
            "REVIEW",
        ],
        default=[
            "VERIFY",
            "REVIEW",
        ],
    )

    filtered = data[
        data["final_action"].isin(actions)
    ].copy()

    if search.strip():

        q = search.lower().strip()

        mask = (
            filtered[
                "transaction_id"
            ].astype(str).str.lower().str.contains(
                q, na=False
            )
            |
            filtered[
                "customer_id"
            ].astype(str).str.lower().str.contains(
                q, na=False
            )
            |
            filtered[
                "merchant_id"
            ].astype(str).str.lower().str.contains(
                q, na=False
            )
            |
            filtered[
                "device_id"
            ].astype(str).str.lower().str.contains(
                q, na=False
            )
            |
            filtered[
                "ip_id"
            ].astype(str).str.lower().str.contains(
                q, na=False
            )
        )

        filtered = filtered[
            mask
        ]

    if filtered.empty:

        st.warning(
            "No matching transactions found."
        )

        st.stop()

    selected = st.selectbox(
        "Select transaction",
        filtered[
            "transaction_id"
        ].tolist(),
    )

    row = filtered[
        filtered["transaction_id"]
        == selected
    ].iloc[0]

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            f"""
            <div class="metric-box metric-orange">
                <div class="metric-number">
                    {row['risk_score']:.1f}
                </div>
                <div class="metric-label">
                    Final Risk Score
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-number">
                    {row['fraud_probability']:.1%}
                </div>
                <div class="metric-label">
                    Fraud Probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-number">
                    {row['anomaly_score']:.1f}
                </div>
                <div class="metric-label">
                    Anomaly Score
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-number">
                    {money(row['amount'])}
                </div>
                <div class="metric-label">
                    Transaction Value
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        decision_badge(
            row["final_action"]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.markdown(
            """
            <div class="light-panel">

                <div class="light-panel-title">
                    Risk explanation
                </div>

                <div class="light-panel-text">
                    RiskGraph combines independent signals
                    to determine the transaction's final
                    intervention level.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        for reason in risk_reason(row):

            st.markdown(
                f"**•** {reason}"
            )

    with right:

        signals = {
            "Fraud probability":
                row["fraud_probability"] * 100,

            "Behavior anomaly":
                row["anomaly_score"],

            "Entity graph":
                row["graph_risk_score"],

            "Financial exposure":
                row["financial_exposure_score"],
        }

        for name, value in signals.items():

            value = float(
                np.clip(value, 0, 100)
            )

            st.markdown(
                f"""
                <div class="risk-row">

                    <div class="risk-header">

                        <div class="risk-name">
                            {name}
                        </div>

                        <div class="risk-value">
                            {value:.1f}
                        </div>

                    </div>

                    <div class="risk-track">

                        <div class="risk-fill"
                             style="width:{value}%;">
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader(
        "Transaction evidence"
    )

    evidence = pd.DataFrame(
        {
            "Field": [
                "Transaction ID",
                "Customer",
                "Merchant",
                "Device",
                "IP",
                "Timestamp",
                "Location",
                "Amount",
                "Account age",
                "Device age",
                "Transactions / 10 min",
                "Failed attempts",
                "Location change",
                "Amount deviation",
                "Device customers",
                "IP customers",
            ],

            "Value": [
                row["transaction_id"],
                row["customer_id"],
                row["merchant_id"],
                row["device_id"],
                row["ip_id"],
                str(row["timestamp"]),
                row["location"],
                money_full(row["amount"]),
                f"{int(row['account_age_days'])} days",
                f"{int(row['device_age_days'])} days",
                int(row["transactions_last_10min"]),
                int(row["failed_attempts"]),
                "Yes" if row["location_change"] else "No",
                f"{row['amount_deviation']:.2f}×",
                int(row["device_customer_count"]),
                int(row["ip_customer_count"]),
            ],
        }
    )

    st.dataframe(
        evidence,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ENTITY INTELLIGENCE
# ============================================================

elif page == "Entity Intelligence":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                Entity Intelligence
            </div>

            <div class="hero-title">
                See the <span>network</span>
                behind the payment.
            </div>

            <div class="hero-description">
                Identify relationships between customers,
                devices and IP addresses that individual
                transaction models can miss.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    entity_type = st.selectbox(
        "Explore",
        [
            "Devices",
            "IP Addresses",
            "Customers",
        ],
    )

    if entity_type == "Devices":

        summary = (
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
                average_risk=(
                    "risk_score",
                    "mean"
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "customers",
                    "transactions",
                ],
                ascending=False,
            )
        )

        entity_column = "device_id"

    elif entity_type == "IP Addresses":

        summary = (
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
                average_risk=(
                    "risk_score",
                    "mean"
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "customers",
                    "transactions",
                ],
                ascending=False,
            )
        )

        entity_column = "ip_id"

    else:

        summary = (
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
                average_risk=(
                    "risk_score",
                    "mean"
                ),
            )
            .reset_index()
            .sort_values(
                "transactions",
                ascending=False,
            )
        )

        entity_column = "customer_id"

    summary["average_risk"] = (
        summary["average_risk"]
        .round(1)
    )

    st.dataframe(
        summary.head(30),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    if not summary.empty:

        selected = st.selectbox(
            "Inspect entity",
            summary[
                entity_column
            ].head(30).tolist(),
        )

        related = data[
            data[
                entity_column
            ] == selected
        ].copy()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Transactions",
            len(related)
        )

        c2.metric(
            "Average Risk",
            f"{related['risk_score'].mean():.1f}"
        )

        c3.metric(
            "High Risk",
            int(
                (
                    related["risk_score"]
                    >= 75
                ).sum()
            )
        )

        st.subheader(
            "Related transactions"
        )

        related_display = related[
            [
                "transaction_id",
                "customer_id",
                "device_id",
                "ip_id",
                "amount",
                "risk_score",
                "final_action",
            ]
        ].copy()

        related_display["amount"] = (
            related_display["amount"]
            .map(money_full)
        )

        related_display["risk_score"] = (
            related_display["risk_score"]
            .round(1)
        )

        st.dataframe(
            related_display
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(50),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                Model Intelligence
            </div>

            <div class="hero-title">
                Built for <span>high-recall</span>
                fraud detection.
            </div>

            <div class="hero-description">
                RiskGraph AI combines supervised fraud
                detection with behavioral anomaly detection
                and entity-level intelligence.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    values = [
        ("99.85%", "Future Holdout ROC-AUC"),
        ("85.11%", "Precision"),
        ("98.36%", "Recall"),
        ("91.25%", "F1 Score"),
    ]

    for col, (value, label) in zip(
        [c1, c2, c3, c4],
        values,
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-box metric-orange">

                    <div class="metric-number">
                        {value}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">'
        'RISKGRAPH ARCHITECTURE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        'Six layers. One decision.'
        '</div>',
        unsafe_allow_html=True,
    )

    features = [
        (
            "01",
            "Fraud Model",
            "Produces transaction-level fraud probability."
        ),
        (
            "02",
            "Anomaly Engine",
            "Detects behavioral patterns that differ from normal activity."
        ),
        (
            "03",
            "Entity Intelligence",
            "Connects customers, devices and IP addresses."
        ),
        (
            "04",
            "Financial Exposure",
            "Estimates the monetary impact of fraud."
        ),
        (
            "05",
            "Risk Fusion",
            "Combines signals into a 0–100 risk score."
        ),
        (
            "06",
            "Decision Policy",
            "Converts risk into APPROVE, VERIFY or REVIEW."
        ),
    ]

    cols = st.columns(3)

    for i, feature in enumerate(features):

        with cols[i % 3]:

            st.markdown(
                f"""
                <div class="feature-box">

                    <div class="feature-number">
                        {feature[0]}
                    </div>

                    <div class="feature-title">
                        {feature[1]}
                    </div>

                    <div class="feature-description">
                        {feature[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader(
        "Evaluation evidence"
    )

    evaluation = pd.DataFrame(
        {
            "Metric": [
                "Future Holdout ROC-AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "Anomaly ROC-AUC",
                "Final Intervention Precision",
                "Final Intervention Recall",
                "Final Intervention F1",
            ],

            "Result": [
                "0.9985",
                "0.8511",
                "0.9836",
                "0.9125",
                "0.9912",
                "0.7785",
                "0.9795",
                "0.8675",
            ],
        }
    )

    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Decision thresholds"
    )

    policy = pd.DataFrame(
        {
            "Risk Score": [
                "< 60",
                "60 – 74.99",
                "≥ 75",
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
        hide_index=True,
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-kicker">
                Business Impact
            </div>

            <div class="hero-title">
                Fraud detection is only useful
                when it <span>changes outcomes.</span>
            </div>

            <div class="hero-description">
                RiskGraph translates model predictions
                into intervention decisions and financial
                exposure estimates.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline = (
        data["expected_fraud_loss"]
        .sum()
    )

    residual = 0.0

    for _, row in data.iterrows():

        if row["final_action"] == "REVIEW":

            residual += (
                row["expected_fraud_loss"]
                * 0.05
            )

        elif row["final_action"] == "VERIFY":

            residual += (
                row["expected_fraud_loss"]
                * 0.20
            )

        else:

            residual += (
                row["expected_fraud_loss"]
            )

    avoided = (
        baseline
        - residual
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Baseline Expected Loss",
        money(baseline)
    )

    c2.metric(
        "Residual Expected Loss",
        money(residual)
    )

    c3.metric(
        "Estimated Loss Avoided",
        money(avoided)
    )

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="orange-panel">

            <h2>
                Cost-aware risk management
            </h2>

            <p>
                The objective is not simply to block as many
                transactions as possible.
            </p>

            <p>
                RiskGraph balances fraud detection,
                customer friction and financial exposure
                through a three-level intervention policy.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader(
        "Decision distribution"
    )

    decision_summary = (
        data["final_action"]
        .value_counts()
        .reindex(
            [
                "APPROVE",
                "VERIFY",
                "REVIEW",
            ]
        )
        .fillna(0)
        .astype(int)
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Decision":
                    decision_summary.index,

                "Transactions":
                    decision_summary.values,

                "Percentage":
                    [
                        f"{x / len(data) * 100:.2f}%"
                        for x
                        in decision_summary.values
                    ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Financial impact figures are prototype estimates "
        "based on the project's modelling assumptions and "
        "should not be interpreted as actual Razorpay costs."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <div class="footer-title">
            RiskGraph AI
        </div>

        <div class="footer-text">
            AI Payment Risk Manager<br><br>

            Detect · Explain · Decide<br><br>

            Fraud detection · Anomaly intelligence ·
            Entity intelligence · Financial exposure
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)
