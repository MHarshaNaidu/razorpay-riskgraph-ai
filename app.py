import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


# ============================================================
# RISKGRAPH AI
# AI-powered payment fraud risk management system
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

MODEL_FILE = "riskgraph_fraud_model_v2.joblib"
DATA_FILE = "sample_transactions.csv"

APPROVE_THRESHOLD = 60
REVIEW_THRESHOLD = 75

VERIFY_COST = 25.0
REVIEW_COST = 75.0

VERIFY_FRAUD_REDUCTION = 0.80
REVIEW_FRAUD_REDUCTION = 0.95

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
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 10px 0 25px 0;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #6b7280;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .risk-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        background: rgba(128,128,128,.04);
    }

    .risk-title {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 5px;
    }

    .risk-value {
        font-size: 28px;
        font-weight: 750;
    }

    .status-review {
        background: #fee2e2;
        color: #991b1b;
        padding: 10px 15px;
        border-radius: 10px;
        font-weight: 700;
    }

    .status-verify {
        background: #fef3c7;
        color: #92400e;
        padding: 10px 15px;
        border-radius: 10px;
        font-weight: 700;
    }

    .status-approve {
        background: #dcfce7;
        color: #166534;
        padding: 10px 15px;
        border-radius: 10px;
        font-weight: 700;
    }

    .info-box {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 12px;
        padding: 18px;
        margin-top: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    return f"₹{float(value):,.2f}"


def percentage(value):
    return f"{float(value) * 100:.2f}%"


def risk_color(action):
    if action == "REVIEW":
        return "🔴"
    if action == "VERIFY":
        return "🟠"
    return "🟢"


def action_message(action):

    if action == "REVIEW":
        return (
            '<div class="status-review">'
            '🔴 REVIEW — High-risk transaction requiring investigation'
            '</div>'
        )

    if action == "VERIFY":
        return (
            '<div class="status-verify">'
            '🟠 VERIFY — Additional verification recommended'
            '</div>'
        )

    return (
        '<div class="status-approve">'
        '🟢 APPROVE — Transaction is below intervention threshold'
        '</div>'
    )


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(MODEL_FILE)

    if not isinstance(bundle, dict):
        raise ValueError(
            "The model file is not a valid RiskGraph model bundle."
        )

    if "model" not in bundle:
        raise ValueError(
            "The model bundle does not contain 'model'."
        )

    if "features" not in bundle:
        raise ValueError(
            "The model bundle does not contain 'features'."
        )

    return bundle["model"], bundle["features"]


@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_FILE)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamp values could not be parsed."
        )

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

    # Time features
    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Amount features
    df["log_amount"] = np.log1p(
        df["amount"]
    )

    df["high_value_transaction"] = (
        df["amount"] > 10000
    ).astype(int)

    # Velocity
    df["high_velocity"] = (
        df["transactions_last_10min"] >= 4
    ).astype(int)

    # Failed attempts
    df["high_failure_activity"] = (
        df["failed_attempts"] >= 3
    ).astype(int)

    # New device/account
    df["new_device"] = (
        df["device_age_days"] < 14
    ).astype(int)

    df["new_account"] = (
        df["account_age_days"] < 60
    ).astype(int)

    # Behavioral risk
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
# ANOMALY DETECTION
# ============================================================

def calculate_anomaly_score(df):

    work = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    split = max(
        int(len(work) * 0.80),
        1
    )

    historical = work.iloc[:split].copy()

    if "is_fraud" in historical.columns:

        training = historical[
            historical["is_fraud"] == 0
        ].copy()

    else:

        training = historical.copy()

    if len(training) < 50:
        training = historical.copy()

    detector = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    detector.fit(
        training[ANOMALY_FEATURES]
    )

    training_scores = -detector.decision_function(
        training[ANOMALY_FEATURES]
    )

    all_scores = -detector.decision_function(
        work[ANOMALY_FEATURES]
    )

    reference = np.sort(
        training_scores
    )

    percentile = (
        np.searchsorted(
            reference,
            all_scores,
            side="right"
        )
        / len(reference)
    )

    work["anomaly_score"] = np.clip(
        percentile * 100,
        0,
        100
    )

    return work


# ============================================================
# ENTITY / GRAPH INTELLIGENCE
# ============================================================

def calculate_graph_features(df):

    work = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
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
# COMPLETE RISK PIPELINE
# ============================================================

@st.cache_data
def run_risk_pipeline(raw_df):

    df = engineer_features(
        raw_df
    )

    model, saved_features = load_model()

    missing_features = [
        feature
        for feature in saved_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Model features missing: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # 1. Supervised fraud model
    # --------------------------------------------------------

    df["fraud_probability"] = (
        model.predict_proba(
            df[saved_features]
        )[:, 1]
    )

    # --------------------------------------------------------
    # 2. Behavioral anomaly
    # --------------------------------------------------------

    df = calculate_anomaly_score(df)

    # --------------------------------------------------------
    # 3. Entity / graph intelligence
    # --------------------------------------------------------

    df = calculate_graph_features(df)

    # --------------------------------------------------------
    # 4. Financial exposure
    # --------------------------------------------------------

    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        * df["amount"]
    )

    loss_cap = (
        df["expected_fraud_loss"]
        .quantile(0.95)
    )

    df["financial_exposure_score"] = (
        df["expected_fraud_loss"]
        / max(loss_cap, 1)
        * 100
    )

    df["financial_exposure_score"] = np.clip(
        df["financial_exposure_score"],
        0,
        100
    )

    # --------------------------------------------------------
    # 5. Final RiskGraph score
    #
    # 50% fraud probability
    # 30% anomaly
    # 20% financial exposure
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Cost-aware recommendation
    # --------------------------------------------------------

    df["approve_cost"] = (
        df["expected_fraud_loss"]
    )

    df["verify_cost"] = (
        VERIFY_COST
        +
        df["expected_fraud_loss"]
        * (1 - VERIFY_FRAUD_REDUCTION)
    )

    df["review_cost"] = (
        REVIEW_COST
        +
        df["expected_fraud_loss"]
        * (1 - REVIEW_FRAUD_REDUCTION)
    )

    cost_columns = [
        "approve_cost",
        "verify_cost",
        "review_cost",
    ]

    action_map = {
        "approve_cost": "APPROVE",
        "verify_cost": "VERIFY",
        "review_cost": "REVIEW",
    }

    df["recommended_action"] = (
        df[cost_columns]
        .idxmin(axis=1)
        .map(action_map)
    )

    # --------------------------------------------------------
    # 7. Final policy
    # --------------------------------------------------------

    df["final_action"] = np.select(
        [
            df["risk_score"] < APPROVE_THRESHOLD,
            df["risk_score"] < REVIEW_THRESHOLD,
        ],
        [
            "APPROVE",
            "VERIFY",
        ],
        default="REVIEW"
    )

    # --------------------------------------------------------
    # 8. Risk band
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

    return df


# ============================================================
# LOAD DATA
# ============================================================

st.sidebar.title("🛡️ RiskGraph AI")

st.sidebar.caption(
    "AI Payment Fraud Risk Manager"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

try:

    if uploaded_file is not None:

        raw_data = pd.read_csv(
            uploaded_file
        )

        data_source = "Uploaded CSV"

    else:

        raw_data = load_default_data()

        data_source = (
            "sample_transactions.csv"
        )

    data = run_risk_pipeline(
        raw_data
    )

except Exception as error:

    st.error(
        "RiskGraph AI could not start."
    )

    st.exception(error)

    st.stop()


st.sidebar.success(
    "Risk engine ready"
)

st.sidebar.caption(
    f"Data source: {data_source}"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Command Center",
        "Investigation",
        "Entity Intelligence",
        "Model Evaluation",
        "Business Impact",
        "Transactions",
    ]
)


st.sidebar.divider()

st.sidebar.metric(
    "Transactions",
    f"{len(data):,}"
)

st.sidebar.metric(
    "Average Risk",
    f"{data['risk_score'].mean():.1f}"
)

st.sidebar.caption(
    "Risk policy: <60 Approve • "
    "60–74.99 Verify • ≥75 Review"
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                RiskGraph AI
            </div>

            <div class="hero-subtitle">
                Intelligent payment fraud detection,
                risk scoring and decision management.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_transactions = len(data)

    total_value = data[
        "amount"
    ].sum()

    fraud_cases = (
        int(data["is_fraud"].sum())
        if "is_fraud" in data.columns
        else 0
    )

    approve_count = (
        data["final_action"]
        == "APPROVE"
    ).sum()

    verify_count = (
        data["final_action"]
        == "VERIFY"
    ).sum()

    review_count = (
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
        "Fraud Cases",
        f"{fraud_cases:,}"
    )

    c4.metric(
        "Verify",
        f"{verify_count:,}"
    )

    c5.metric(
        "Review",
        f"{review_count:,}"
    )

    st.divider()

    left, right = st.columns(
        [1.5, 1]
    )

    with left:

        st.markdown(
            '<div class="section-title">'
            'Risk Score Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        bins = pd.cut(
            data["risk_score"],
            bins=np.arange(
                0,
                105,
                5
            ),
            include_lowest=True
        )

        distribution = (
            bins
            .value_counts()
            .sort_index()
        )

        distribution.index = [
            f"{int(x.left)}–{int(x.right)}"
            for x in distribution.index
        ]

        st.bar_chart(
            distribution,
            height=350
        )

    with right:

        st.markdown(
            '<div class="section-title">'
            'Decision Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        decision_data = pd.Series(
            {
                "APPROVE": approve_count,
                "VERIFY": verify_count,
                "REVIEW": review_count
            }
        )

        st.bar_chart(
            decision_data,
            height=250
        )

        st.info(
            "Risk score < 60 → APPROVE\n\n"
            "Risk score 60–74.99 → VERIFY\n\n"
            "Risk score ≥ 75 → REVIEW"
        )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Highest Risk Transactions'
        '</div>',
        unsafe_allow_html=True
    )

    top = (
        data
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(15)
        .copy()
    )

    display = top[
        [
            "transaction_id",
            "amount",
            "fraud_probability",
            "anomaly_score",
            "graph_risk_score",
            "financial_exposure_score",
            "risk_score",
            "final_action"
        ]
    ].copy()

    display["amount"] = display[
        "amount"
    ].map(money)

    display[
        "fraud_probability"
    ] = display[
        "fraud_probability"
    ].map(percentage)

    for column in [
        "anomaly_score",
        "graph_risk_score",
        "financial_exposure_score",
        "risk_score"
    ]:

        display[column] = display[
            column
        ].round(1)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INVESTIGATION
# ============================================================

elif page == "Investigation":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Investigation
            </div>

            <div class="hero-subtitle">
                Investigate a transaction and understand
                exactly why it received its risk decision.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    search = st.text_input(
        "Search transaction / customer / merchant / device / IP",
        placeholder="Example: TX_0009808"
    )

    candidates = data.copy()

    if search.strip():

        q = search.strip().lower()

        mask = (
            candidates[
                "transaction_id"
            ].astype(str)
            .str.lower()
            .str.contains(q)

            |

            candidates[
                "customer_id"
            ].astype(str)
            .str.lower()
            .str.contains(q)

            |

            candidates[
                "merchant_id"
            ].astype(str)
            .str.lower()
            .str.contains(q)

            |

            candidates[
                "device_id"
            ].astype(str)
            .str.lower()
            .str.contains(q)

            |

            candidates[
                "ip_id"
            ].astype(str)
            .str.lower()
            .str.contains(q)
        )

        candidates = candidates[
            mask
        ]

    if candidates.empty:

        st.warning(
            "No matching transaction found."
        )

        st.stop()

    candidates = candidates.sort_values(
        "risk_score",
        ascending=False
    )

    selected_transaction = st.selectbox(
        "Select transaction",
        candidates[
            "transaction_id"
        ].tolist()
    )

    row = candidates[
        candidates["transaction_id"]
        == selected_transaction
    ].iloc[0]

    st.divider()

    st.subheader(
        f"Transaction: {selected_transaction}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Risk Score",
        f"{row['risk_score']:.1f}/100"
    )

    c2.metric(
        "Fraud Probability",
        percentage(
            row["fraud_probability"]
        )
    )

    c3.metric(
        "Anomaly Score",
        f"{row['anomaly_score']:.1f}/100"
    )

    c4.metric(
        "Amount",
        money(row["amount"])
    )

    st.markdown(
        action_message(
            row["final_action"]
        ),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Risk Signal Breakdown'
        '</div>',
        unsafe_allow_html=True
    )

    signals = pd.DataFrame(
        {
            "Risk Signal": [
                "Fraud Probability",
                "Behavioral Anomaly",
                "Graph Risk",
                "Financial Exposure",
                "Final Risk Score"
            ],

            "Score": [
                row["fraud_probability"] * 100,
                row["anomaly_score"],
                row["graph_risk_score"],
                row["financial_exposure_score"],
                row["risk_score"]
            ]
        }
    )

    signals["Score"] = signals[
        "Score"
    ].round(2)

    st.dataframe(
        signals,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">'
        'Why was this transaction flagged?'
        '</div>',
        unsafe_allow_html=True
    )

    reasons = []

    if row["fraud_probability"] >= 0.50:

        reasons.append(
            f"High fraud probability "
            f"({row['fraud_probability']:.1%})"
        )

    if row["anomaly_score"] >= 75:

        reasons.append(
            f"Highly unusual behavioral pattern "
            f"({row['anomaly_score']:.1f}/100)"
        )

    if row["financial_exposure_score"] >= 75:

        reasons.append(
            f"High financial exposure "
            f"({row['financial_exposure_score']:.1f}/100)"
        )

    if row["shared_device"]:

        reasons.append(
            "Device is associated with multiple customers."
        )

    if row["shared_ip"]:

        reasons.append(
            "IP address is associated with multiple customers."
        )

    if row["transactions_last_10min"] >= 4:

        reasons.append(
            "High transaction velocity detected."
        )

    if row["failed_attempts"] >= 3:

        reasons.append(
            "Multiple failed attempts detected."
        )

    if row["new_device"]:

        reasons.append(
            "New device detected."
        )

    if row["new_account"]:

        reasons.append(
            "New account detected."
        )

    if row["location_change"]:

        reasons.append(
            "Location change detected."
        )

    if row["amount_deviation"] > 3:

        reasons.append(
            "Transaction amount is significantly above normal behavior."
        )

    if not reasons:

        reasons.append(
            "No major configured risk signal was triggered."
        )

    for reason in reasons:

        st.write(
            "• " + reason
        )

    st.markdown(
        '<div class="section-title">'
        'Entity Context'
        '</div>',
        unsafe_allow_html=True
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Customers on Device",
        int(row["device_customer_count"])
    )

    e2.metric(
        "Customers on IP",
        int(row["ip_customer_count"])
    )

    e3.metric(
        "Transactions on Device",
        int(row["device_transaction_count"])
    )

    e4.metric(
        "Transactions on IP",
        int(row["ip_transaction_count"])
    )

    st.markdown(
        '<div class="section-title">'
        'Transaction Details'
        '</div>',
        unsafe_allow_html=True
    )

    details = pd.DataFrame(
        {
            "Field": [
                "Customer ID",
                "Merchant ID",
                "Device ID",
                "IP ID",
                "Timestamp",
                "Location",
                "Account Age",
                "Device Age",
                "Transactions / 10 min",
                "Failed Attempts",
                "Amount Deviation",
                "Behavior Risk Count",
                "Expected Fraud Loss",
                "Recommended Action",
                "Final Action"
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
                f"{row['amount_deviation']:.2f}x",
                int(row["behavior_risk_count"]),
                money(row["expected_fraud_loss"]),
                row["recommended_action"],
                row["final_action"]
            ]
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ENTITY INTELLIGENCE
# ============================================================

elif page == "Entity Intelligence":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Entity Intelligence
            </div>

            <div class="hero-subtitle">
                Detect relationships between customers,
                devices and IP addresses.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    entity_type = st.selectbox(
        "Entity type",
        [
            "Device",
            "IP Address",
            "Customer"
        ]
    )

    if entity_type == "Device":

        table = (
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
                )
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

        key = "device_id"

    elif entity_type == "IP Address":

        table = (
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
                )
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

        key = "ip_id"

    else:

        table = (
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
                )
            )
            .sort_values(
                [
                    "transactions",
                    "average_risk"
                ],
                ascending=False
            )
            .head(30)
            .reset_index()
        )

        key = "customer_id"

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    selected = st.selectbox(
        "Select entity",
        table[key].tolist()
    )

    if key == "device_id":

        related = data[
            data["device_id"] == selected
        ]

    elif key == "ip_id":

        related = data[
            data["ip_id"] == selected
        ]

    else:

        related = data[
            data["customer_id"] == selected
        ]

    st.markdown(
        '<div class="section-title">'
        'Related Transactions'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        related[
            [
                "transaction_id",
                "customer_id",
                "device_id",
                "ip_id",
                "amount",
                "risk_score",
                "final_action"
            ]
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(50),

        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Model Evaluation
            </div>

            <div class="hero-subtitle">
                Performance of the RiskGraph AI fraud detection pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "The following metrics are the evaluation results "
        "reported by the completed RiskGraph AI modelling pipeline."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ROC-AUC",
        "99.85%"
    )

    c2.metric(
        "Precision",
        "85.11%"
    )

    c3.metric(
        "Recall",
        "98.36%"
    )

    c4.metric(
        "F1 Score",
        "0.9125"
    )

    st.markdown(
        '<div class="section-title">'
        'Anomaly Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.metric(
        "Anomaly ROC-AUC",
        "99.12%"
    )

    st.markdown(
        '<div class="section-title">'
        'Final Decision Policy'
        '</div>',
        unsafe_allow_html=True
    )

    policy = pd.DataFrame(
        {
            "Risk Score": [
                "< 60",
                "60 – 74.99",
                "≥ 75"
            ],

            "Decision": [
                "APPROVE",
                "VERIFY",
                "REVIEW"
            ],

            "Meaning": [
                "Low intervention risk",
                "Additional verification",
                "Manual investigation"
            ]
        }
    )

    st.dataframe(
        policy,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">'
        'Final Policy Evaluation'
        '</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4, p5 = st.columns(5)

    p1.metric(
        "Intervention Precision",
        "77.85%"
    )

    p2.metric(
        "Intervention Recall",
        "97.95%"
    )

    p3.metric(
        "Intervention F1",
        "0.8675"
    )

    p4.metric(
        "False Positives",
        "68"
    )

    p5.metric(
        "False Negatives",
        "5"
    )

    st.markdown(
        '<div class="section-title">'
        'Business Evaluation'
        '</div>',
        unsafe_allow_html=True
    )

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "False-positive value",
        "₹4,83,752.82"
    )

    b2.metric(
        "Missed-fraud value",
        "₹4,292.70"
    )

    b3.metric(
        "Estimated FP Cost",
        "₹4,837.53"
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Business Impact
            </div>

            <div class="hero-subtitle">
                Understand how risk decisions affect fraud exposure
                and intervention costs.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    baseline_loss = data[
        "expected_fraud_loss"
    ].sum()

    residual_loss = 0.0

    for _, row in data.iterrows():

        if row["final_action"] == "VERIFY":

            residual_loss += (
                row["expected_fraud_loss"]
                * (1 - VERIFY_FRAUD_REDUCTION)
            )

        elif row["final_action"] == "REVIEW":

            residual_loss += (
                row["expected_fraud_loss"]
                * (1 - REVIEW_FRAUD_REDUCTION)
            )

        else:

            residual_loss += (
                row["expected_fraud_loss"]
            )

    loss_avoided = (
        baseline_loss
        - residual_loss
    )

    intervention_cost = (

        (
            data["final_action"]
            == "VERIFY"
        ).sum()
        * VERIFY_COST

        +

        (
            data["final_action"]
            == "REVIEW"
        ).sum()
        * REVIEW_COST
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Baseline Expected Loss",
        money(baseline_loss)
    )

    c2.metric(
        "Residual Loss",
        money(residual_loss)
    )

    c3.metric(
        "Estimated Loss Avoided",
        money(loss_avoided)
    )

    c4.metric(
        "Intervention Cost",
        money(intervention_cost)
    )

    st.divider()

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

            expected_fraud_loss=(
                "expected_fraud_loss",
                "sum"
            )
        )
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "VERIFY and REVIEW costs are prototype modelling "
        "assumptions from the project, not Razorpay's actual "
        "operating costs."
    )


# ============================================================
# TRANSACTIONS
# ============================================================

elif page == "Transactions":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Transaction Explorer
            </div>

            <div class="hero-subtitle">
                Search, filter and export RiskGraph AI decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_actions = st.multiselect(
        "Decision",
        [
            "APPROVE",
            "VERIFY",
            "REVIEW"
        ],
        default=[
            "APPROVE",
            "VERIFY",
            "REVIEW"
        ]
    )

    minimum_risk = st.slider(
        "Minimum risk score",
        min_value=0,
        max_value=100,
        value=0
    )

    filtered = data[
        data["final_action"].isin(
            selected_actions
        )
        &
        (
            data["risk_score"]
            >= minimum_risk
        )
    ].copy()

    st.write(
        f"Showing {len(filtered):,} transactions"
    )

    display = filtered[
        [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "amount",
            "fraud_probability",
            "anomaly_score",
            "graph_risk_score",
            "financial_exposure_score",
            "risk_score",
            "risk_band",
            "final_action"
        ]
    ].copy()

    display["amount"] = display[
        "amount"
    ].map(money)

    display[
        "fraud_probability"
    ] = display[
        "fraud_probability"
    ].map(percentage)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download RiskGraph Results",
        data=csv_data,
        file_name="riskgraph_results.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RiskGraph AI • AI-powered payment fraud risk manager • "
    "Prototype / academic project"
)
