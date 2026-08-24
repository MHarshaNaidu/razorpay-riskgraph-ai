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
# CLEAN STREAMLIT STYLING
# No custom HTML components
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 12px;
        padding: 12px;
        background: rgba(128,128,128,0.04);
    }

    .stButton button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    return f"₹{float(value):,.2f}"


def percentage(value):
    return f"{float(value) * 100:.2f}%"


def risk_band(score):

    score = float(score)

    if score >= 75:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MODERATE"

    return "LOW"


def risk_reasons(row):

    reasons = []

    if row["fraud_probability"] >= 0.50:
        reasons.append(
            f"Fraud model probability is "
            f"{row['fraud_probability']:.1%}"
        )

    if row["anomaly_score"] >= 75:
        reasons.append(
            f"Behavioral anomaly score is "
            f"{row['anomaly_score']:.1f}/100"
        )

    if row["financial_exposure_score"] >= 75:
        reasons.append(
            f"Financial exposure score is "
            f"{row['financial_exposure_score']:.1f}/100"
        )

    if row["shared_device"]:
        reasons.append(
            f"Device is associated with "
            f"{int(row['device_customer_count'])} customers"
        )

    if row["shared_ip"]:
        reasons.append(
            f"IP is associated with "
            f"{int(row['ip_customer_count'])} customers"
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
            f"Transaction amount is "
            f"{row['amount_deviation']:.1f}× "
            f"the normal amount"
        )

    if not reasons:
        reasons.append(
            "No major risk indicators crossed "
            "the configured thresholds"
        )

    return reasons


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(bundle, dict):
        raise ValueError(
            "Invalid model bundle."
        )

    if "model" not in bundle:
        raise ValueError(
            "Model bundle does not contain 'model'."
        )

    if "features" not in bundle:
        raise ValueError(
            "Model bundle does not contain 'features'."
        )

    return (
        bundle["model"],
        bundle["features"]
    )


# ============================================================
# DEFAULT DATA
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

    df["hour"] = (
        df["timestamp"].dt.hour
    )

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

    # Device/account age

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
# ANOMALY ENGINE
# ============================================================

def calculate_anomaly_scores(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    split_index = int(
        len(work) * 0.80
    )

    historical = work.iloc[
        :split_index
    ].copy()

    if (
        "is_fraud" in historical.columns
    ):

        legitimate = historical[
            historical["is_fraud"] == 0
        ].copy()

    else:

        legitimate = historical.copy()

    if len(legitimate) < 100:

        legitimate = historical.copy()

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    X_train = legitimate[
        ANOMALY_FEATURES
    ]

    X_all = work[
        ANOMALY_FEATURES
    ]

    anomaly_model.fit(
        X_train
    )

    train_scores = (
        -anomaly_model
        .decision_function(X_train)
    )

    all_scores = (
        -anomaly_model
        .decision_function(X_all)
    )

    sorted_scores = np.sort(
        train_scores
    )

    percentiles = (
        np.searchsorted(
            sorted_scores,
            all_scores,
            side="right"
        )
        / len(sorted_scores)
    )

    work["anomaly_score"] = np.clip(
        percentiles * 100,
        0,
        100
    )

    return work


# ============================================================
# ENTITY / GRAPH ENGINE
# ============================================================

def calculate_graph_features(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    split_index = int(
        len(work) * 0.80
    )

    historical = work.iloc[
        :split_index
    ].copy()

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
# COMPLETE RISK PIPELINE
#
# IMPORTANT:
# No @st.cache_data here.
#
# This guarantees a newly uploaded CSV is processed again.
# ============================================================

def run_pipeline(raw_df):

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
    # 1. FRAUD MODEL
    # --------------------------------------------------------

    X = df[
        saved_features
    ]

    df["fraud_probability"] = (
        model
        .predict_proba(X)[:, 1]
    )

    # --------------------------------------------------------
    # 2. ANOMALY ENGINE
    # --------------------------------------------------------

    df = calculate_anomaly_scores(
        df
    )

    # --------------------------------------------------------
    # 3. ENTITY GRAPH
    # --------------------------------------------------------

    df = calculate_graph_features(
        df
    )

    # --------------------------------------------------------
    # 4. FRAUD SIGNAL
    # --------------------------------------------------------

    df["fraud_signal"] = (
        df["fraud_probability"]
        * 100
    )

    # --------------------------------------------------------
    # 5. EXPECTED FRAUD LOSS
    # --------------------------------------------------------

    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        * df["amount"]
    )

    # --------------------------------------------------------
    # 6. FINANCIAL EXPOSURE
    # --------------------------------------------------------

    loss_cap = (
        df["expected_fraud_loss"]
        .quantile(0.95)
    )

    df["financial_exposure_score"] = (
        df["expected_fraud_loss"]
        / max(loss_cap, 1)
        * 100
    )

    df["financial_exposure_score"] = (
        np.clip(
            df["financial_exposure_score"],
            0,
            100
        )
    )

    # --------------------------------------------------------
    # 7. FINAL RISK SCORE
    #
    # 50% fraud
    # 30% anomaly
    # 20% financial exposure
    # --------------------------------------------------------

    df["risk_score"] = (

        0.50
        * df["fraud_signal"]

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
    # 8. COST-AWARE RECOMMENDATION
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

    df["recommended_action"] = (
        df[cost_columns]
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
    # 9. FINAL DECISION POLICY
    # --------------------------------------------------------

    df["final_action"] = np.select(
        [
            df["risk_score"]
            < APPROVE_THRESHOLD,

            df["risk_score"]
            < REVIEW_THRESHOLD,
        ],
        [
            "APPROVE",
            "VERIFY",
        ],
        default="REVIEW",
    )

    # --------------------------------------------------------
    # 10. RISK BAND
    # --------------------------------------------------------

    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[
            -0.01,
            30,
            60,
            75,
            100,
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
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ RiskGraph AI"
)

st.sidebar.caption(
    "AI Payment Risk Manager"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Data"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help=(
        "Upload a CSV containing the same transaction "
        "fields used by the RiskGraph model."
    ),
)

# ------------------------------------------------------------
# DATA SOURCE
# ------------------------------------------------------------

if uploaded_file is not None:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

        data_source = (
            f"Uploaded: {uploaded_file.name}"
        )

        st.sidebar.success(
            f"Loaded {len(raw_data):,} rows"
        )

    except Exception as error:

        st.sidebar.error(
            f"CSV could not be read: {error}"
        )

        st.stop()

else:

    try:

        raw_data = load_default_data()

        data_source = (
            "sample_transactions.csv"
        )

        st.sidebar.info(
            f"Demo dataset: {len(raw_data):,} rows"
        )

    except Exception as error:

        st.sidebar.error(
            "sample_transactions.csv could not be loaded."
        )

        st.exception(error)

        st.stop()


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

if "transaction_id" not in raw_data.columns:

    st.error(
        "This CSV does not contain transaction_id."
    )

    st.stop()


# ------------------------------------------------------------
# RUN PIPELINE
# ------------------------------------------------------------

with st.spinner(
    "RiskGraph AI is processing transactions..."
):

    try:

        data = run_pipeline(
            raw_data
        )

    except Exception as error:

        st.error(
            "Risk pipeline failed."
        )

        st.exception(error)

        st.stop()


# ------------------------------------------------------------
# SIDEBAR STATUS
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.success(
    "Risk Engine Online"
)

st.sidebar.caption(
    data_source
)

st.sidebar.metric(
    "Processed Transactions",
    f"{len(data):,}"
)

intervention_count = (
    data["final_action"]
    != "APPROVE"
).sum()

st.sidebar.metric(
    "Risk Interventions",
    f"{intervention_count:,}"
)

st.sidebar.divider()

st.sidebar.caption(
    "Decision Policy"
)

st.sidebar.caption(
    "0 – 59.99  → APPROVE"
)

st.sidebar.caption(
    "60 – 74.99 → VERIFY"
)

st.sidebar.caption(
    "75 – 100   → REVIEW"
)


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Command Center",
        "Investigate",
        "Entity Intelligence",
        "Model Intelligence",
        "Business Impact",
        "Transactions",
    ],
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
        "Detect · Explain · Decide"
    )

    st.divider()

    total_transactions = len(
        data
    )

    total_value = data[
        "amount"
    ].sum()

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
        "Approved",
        f"{approve_count:,}"
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

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Risk Score Distribution"
        )

        bins = [
            0,
            30,
            60,
            75,
            100.01,
        ]

        labels = [
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ]

        distribution = pd.cut(
            data["risk_score"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        risk_counts = (
            distribution
            .value_counts()
            .reindex(labels)
            .fillna(0)
            .astype(int)
        )

        st.bar_chart(
            risk_counts,
            height=350,
        )

        st.caption(
            "LOW < 30 · MODERATE 30–59.99 · "
            "HIGH 60–74.99 · CRITICAL ≥ 75"
        )

    # --------------------------------------------------------
    # DECISION MIX
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Decision Mix"
        )

        decisions = pd.Series(
            {
                "APPROVE": approve_count,
                "VERIFY": verify_count,
                "REVIEW": review_count,
            }
        )

        st.bar_chart(
            decisions,
            height=300,
        )

        st.markdown(
            "🟢 **APPROVE** — low intervention risk"
        )

        st.markdown(
            "🟠 **VERIFY** — additional verification"
        )

        st.markdown(
            "🔴 **REVIEW** — analyst investigation"
        )

    st.divider()

    st.subheader(
        "Priority Risk Queue"
    )

    priority = (
        data
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(15)
        .copy()
    )

    priority_display = priority[
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

    priority_display[
        "amount"
    ] = priority_display[
        "amount"
    ].map(money)

    priority_display[
        "fraud_probability"
    ] = priority_display[
        "fraud_probability"
    ].map(percentage)

    priority_display[
        "anomaly_score"
    ] = priority_display[
        "anomaly_score"
    ].round(1)

    priority_display[
        "graph_risk_score"
    ] = priority_display[
        "graph_risk_score"
    ].round(1)

    priority_display[
        "financial_exposure_score"
    ] = priority_display[
        "financial_exposure_score"
    ].round(1)

    priority_display[
        "risk_score"
    ] = priority_display[
        "risk_score"
    ].round(1)

    st.dataframe(
        priority_display,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "RiskGraph combines fraud probability, behavioral "
        "anomaly, entity intelligence and financial exposure "
        "into a final 0–100 risk score."
    )


# ============================================================
# INVESTIGATION
# ============================================================

elif page == "Investigate":

    st.title(
        "🔎 Transaction Investigation"
    )

    st.subheader(
        "Investigate why a payment received its risk decision."
    )

    st.divider()

    search = st.text_input(
        "Search transaction, customer, merchant, device or IP",
        placeholder="Example: TX_0013175",
    )

    decision_filter = st.multiselect(
        "Decision filter",
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

    minimum_risk = st.slider(
        "Minimum risk score",
        0,
        100,
        0,
    )

    queue = data[
        data["final_action"]
        .isin(decision_filter)
        &
        (
            data["risk_score"]
            >= minimum_risk
        )
    ].copy()

    if search.strip():

        query = (
            search
            .strip()
            .lower()
        )

        search_mask = (
            queue[
                "transaction_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
            |

            queue[
                "customer_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
            |

            queue[
                "merchant_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
            |

            queue[
                "device_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
            |

            queue[
                "ip_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                query,
                na=False
            )
        )

        queue = queue[
            search_mask
        ]

    queue = queue.sort_values(
        "risk_score",
        ascending=False,
    )

    st.metric(
        "Matching Transactions",
        f"{len(queue):,}"
    )

    if queue.empty:

        st.warning(
            "No transactions match the current filters."
        )

        st.stop()

    st.dataframe(
        queue[
            [
                "transaction_id",
                "customer_id",
                "amount",
                "fraud_probability",
                "anomaly_score",
                "graph_risk_score",
                "risk_score",
                "final_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    selected_transaction = st.selectbox(
        "Open transaction investigation",
        queue[
            "transaction_id"
        ].tolist(),
    )

    row = queue[
        queue[
            "transaction_id"
        ]
        == selected_transaction
    ].iloc[0]

    st.subheader(
        f"Transaction: {selected_transaction}"
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

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
        money(row["amount"])
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    st.subheader(
        "Final Decision"
    )

    if row["final_action"] == "REVIEW":

        st.error(
            "🔴 REVIEW — High-risk transaction"
        )

    elif row["final_action"] == "VERIFY":

        st.warning(
            "🟠 VERIFY — Additional verification required"
        )

    else:

        st.success(
            "🟢 APPROVE — Below intervention threshold"
        )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    st.subheader(
        "Risk Signal Breakdown"
    )

    signal_table = pd.DataFrame(
        {
            "Signal": [
                "Fraud probability",
                "Behavior anomaly",
                "Graph risk",
                "Financial exposure",
                "Final risk score",
            ],
            "Score": [
                row["fraud_probability"] * 100,
                row["anomaly_score"],
                row["graph_risk_score"],
                row["financial_exposure_score"],
                row["risk_score"],
            ],
        }
    )

    signal_table[
        "Score"
    ] = signal_table[
        "Score"
    ].round(2)

    st.dataframe(
        signal_table,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "Why did RiskGraph make this decision?"
    )

    reasons = risk_reasons(
        row
    )

    for reason in reasons:

        st.markdown(
            f"• {reason}"
        )

    # --------------------------------------------------------
    # ENTITY CONTEXT
    # --------------------------------------------------------

    st.subheader(
        "Entity Context"
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Device Customers",
        int(
            row["device_customer_count"]
        )
    )

    e2.metric(
        "IP Customers",
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

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    st.subheader(
        "Transaction Evidence"
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
                "Account age",
                "Device age",
                "Transactions / 10 min",
                "Failed attempts",
                "Location change",
                "Amount deviation",
                "Behavior risk count",
                "Expected fraud loss",
                "Recommended action",
                "Final action",
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
                int(
                    row["transactions_last_10min"]
                ),
                int(
                    row["failed_attempts"]
                ),
                "Yes"
                if row["location_change"]
                else "No",
                f"{row['amount_deviation']:.2f}×",
                int(
                    row["behavior_risk_count"]
                ),
                money(
                    row["expected_fraud_loss"]
                ),
                row["recommended_action"],
                row["final_action"],
            ],
        }
    )

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ENTITY INTELLIGENCE
# ============================================================

elif page == "Entity Intelligence":

    st.title(
        "🕸️ Entity Intelligence"
    )

    st.subheader(
        "Investigate relationships between customers, "
        "devices and IP addresses."
    )

    st.divider()

    entity_type = st.selectbox(
        "Select entity type",
        [
            "Device",
            "IP Address",
            "Customer",
        ],
    )

    if entity_type == "Device":

        entity_summary = (
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
            .sort_values(
                [
                    "customers",
                    "transactions",
                ],
                ascending=False,
            )
            .reset_index()
        )

        entity_column = "device_id"

    elif entity_type == "IP Address":

        entity_summary = (
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
            .sort_values(
                [
                    "customers",
                    "transactions",
                ],
                ascending=False,
            )
            .reset_index()
        )

        entity_column = "ip_id"

    else:

        entity_summary = (
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
            .sort_values(
                [
                    "transactions",
                    "average_risk",
                ],
                ascending=False,
            )
            .reset_index()
        )

        entity_column = "customer_id"

    st.subheader(
        "Highest-activity entities"
    )

    display_entities = (
        entity_summary
        .head(30)
        .copy()
    )

    display_entities[
        "average_risk"
    ] = display_entities[
        "average_risk"
    ].round(2)

    st.dataframe(
        display_entities,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Inspect Entity"
    )

    selected_entity = st.selectbox(
        "Select entity",
        display_entities[
            entity_column
        ].tolist(),
    )

    if entity_column == "device_id":

        related = data[
            data["device_id"]
            == selected_entity
        ]

    elif entity_column == "ip_id":

        related = data[
            data["ip_id"]
            == selected_entity
        ]

    else:

        related = data[
            data["customer_id"]
            == selected_entity
        ]

    x1, x2, x3 = st.columns(3)

    x1.metric(
        "Transactions",
        f"{len(related):,}"
    )

    x2.metric(
        "Average Risk",
        f"{related['risk_score'].mean():.1f}"
    )

    x3.metric(
        "High-risk Transactions",
        f"{(related['risk_score'] >= 75).sum():,}"
    )

    st.subheader(
        "Related Transactions"
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

    related_display[
        "amount"
    ] = related_display[
        "amount"
    ].map(money)

    related_display[
        "risk_score"
    ] = related_display[
        "risk_score"
    ].round(1)

    st.dataframe(
        related_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.title(
        "🧠 Model Intelligence"
    )

    st.subheader(
        "Evidence from the completed RiskGraph AI "
        "fraud detection pipeline."
    )

    st.divider()

    st.info(
        "These are the evaluation results produced by "
        "the completed notebook pipeline."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Future Holdout ROC-AUC",
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
        "91.25%"
    )

    st.divider()

    st.subheader(
        "Anomaly Engine"
    )

    st.metric(
        "Anomaly ROC-AUC",
        "99.12%"
    )

    st.divider()

    st.subheader(
        "RiskGraph Architecture"
    )

    architecture = pd.DataFrame(
        {
            "Layer": [
                "Fraud Model",
                "Anomaly Engine",
                "Entity Graph",
                "Financial Exposure",
                "Risk Fusion",
                "Decision Policy",
            ],

            "Output": [
                "Fraud probability",
                "Anomaly score",
                "Graph risk score",
                "Expected fraud loss",
                "Risk score 0–100",
                "APPROVE / VERIFY / REVIEW",
            ],
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Final Decision Policy"
    )

    policy = pd.DataFrame(
        {
            "Risk Score": [
                "< 60",
                "60 – 74.99",
                "≥ 75",
            ],

            "Action": [
                "APPROVE",
                "VERIFY",
                "REVIEW",
            ],

            "Meaning": [
                "Low intervention risk",
                "Additional verification",
                "Manual investigation",
            ],
        }
    )

    st.dataframe(
        policy,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Final Policy Evaluation"
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
        "86.75%"
    )

    p4.metric(
        "False Positives",
        "68"
    )

    p5.metric(
        "False Negatives",
        "5"
    )

    st.divider()

    st.subheader(
        "Final Business Metrics"
    )

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "False-positive Value",
        "₹4,83,752.82"
    )

    b2.metric(
        "Missed-fraud Value",
        "₹4,292.70"
    )

    b3.metric(
        "Estimated FP Cost",
        "₹4,837.53"
    )

    st.caption(
        "Prototype business-cost assumptions are modelling "
        "assumptions and are not Razorpay operational costs."
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.title(
        "💰 Business Impact"
    )

    st.subheader(
        "Translate fraud predictions into financial exposure "
        "and intervention decisions."
    )

    st.divider()

    baseline_loss = (
        data["expected_fraud_loss"]
        .sum()
    )

    residual_loss = 0.0

    for _, row in data.iterrows():

        if row["final_action"] == "VERIFY":

            residual_loss += (
                row["expected_fraud_loss"]
                * (
                    1
                    - VERIFY_FRAUD_REDUCTION
                )
            )

        elif row["final_action"] == "REVIEW":

            residual_loss += (
                row["expected_fraud_loss"]
                * (
                    1
                    - REVIEW_FRAUD_REDUCTION
                )
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
        money(
            baseline_loss
        )
    )

    c2.metric(
        "Residual Expected Loss",
        money(
            residual_loss
        )
    )

    c3.metric(
        "Estimated Loss Avoided",
        money(
            loss_avoided
        )
    )

    c4.metric(
        "Intervention Cost",
        money(
            intervention_cost
        )
    )

    st.divider()

    st.subheader(
        "Decision-level financial exposure"
    )

    business_table = (
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
            ),
        )
        .reset_index()
    )

    business_table[
        "transaction_value"
    ] = business_table[
        "transaction_value"
    ].map(money)

    business_table[
        "expected_fraud_loss"
    ] = business_table[
        "expected_fraud_loss"
    ].map(money)

    st.dataframe(
        business_table,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Financial intervention costs and fraud-reduction "
        "assumptions are prototype assumptions for project "
        "evaluation, not production Razorpay figures."
    )


# ============================================================
# TRANSACTIONS / DATA EXPLORER
# ============================================================

elif page == "Transactions":

    st.title(
        "📊 Transaction Explorer"
    )

    st.subheader(
        "Search, filter and export RiskGraph decisions."
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        selected_actions = st.multiselect(
            "Decision",
            [
                "APPROVE",
                "VERIFY",
                "REVIEW",
            ],
            default=[
                "APPROVE",
                "VERIFY",
                "REVIEW",
            ],
        )

    with c2:

        minimum_risk = st.slider(
            "Minimum Risk Score",
            0,
            100,
            0,
        )

    search_text = st.text_input(
        "Search transaction / customer / merchant / device / IP",
        placeholder="Search...",
    )

    filtered = data[
        data["final_action"]
        .isin(selected_actions)
        &
        (
            data["risk_score"]
            >= minimum_risk
        )
    ].copy()

    if search_text.strip():

        q = (
            search_text
            .strip()
            .lower()
        )

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
            "final_action",
        ]
    ].copy()

    display[
        "amount"
    ] = display[
        "amount"
    ].map(money)

    display[
        "fraud_probability"
    ] = display[
        "fraud_probability"
    ].map(percentage)

    display[
        "anomaly_score"
    ] = display[
        "anomaly_score"
    ].round(1)

    display[
        "graph_risk_score"
    ] = display[
        "graph_risk_score"
    ].round(1)

    display[
        "financial_exposure_score"
    ] = display[
        "financial_exposure_score"
    ].round(1)

    display[
        "risk_score"
    ] = display[
        "risk_score"
    ].round(1)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    csv_data = filtered.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ Download RiskGraph Results",
        data=csv_data,
        file_name="riskgraph_results.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RiskGraph AI · AI Payment Risk Manager · "
    "Synthetic-data prototype · Defense-only fraud risk detection"
)
