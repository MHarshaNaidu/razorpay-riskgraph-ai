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
DATA_PATH = "sample_transactions.csv"

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
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 10px 0 25px 0;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #6b7280;
        font-size: 16px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .decision-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.20);
        background: rgba(128,128,128,.04);
    }

    .decision-title {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 5px;
    }

    .decision-value {
        font-size: 30px;
        font-weight: 800;
    }

    .risk-box {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.20);
        background: rgba(128,128,128,.04);
    }

    .small {
        font-size: 13px;
        color: #6b7280;
    }

    .critical {
        border-left: 5px solid #dc2626;
    }

    .high {
        border-left: 5px solid #ea580c;
    }

    .moderate {
        border-left: 5px solid #d97706;
    }

    .low {
        border-left: 5px solid #16a34a;
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


def pct(value):
    return f"{float(value) * 100:.2f}%"


def action_icon(action):
    if action == "REVIEW":
        return "🔴"
    if action == "VERIFY":
        return "🟠"
    return "🟢"


def risk_band(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MODERATE"

    return "LOW"


def explanation(row):

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
            f"{int(row['transactions_last_10min'])} transactions "
            f"in the last 10 minutes"
        )

    if row["failed_attempts"] >= 3:
        reasons.append(
            f"Multiple failed attempts: "
            f"{int(row['failed_attempts'])}"
        )

    if row["new_device"]:
        reasons.append("New device detected")

    if row["new_account"]:
        reasons.append("New account detected")

    if row["location_change"]:
        reasons.append("Location change detected")

    if row["amount_deviation"] > 3:
        reasons.append(
            f"Amount is {row['amount_deviation']:.1f}× "
            f"the normal transaction amount"
        )

    if not reasons:
        reasons.append(
            "No major risk indicator crossed the configured thresholds"
        )

    return reasons


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(bundle, dict):
        raise ValueError(
            "Invalid RiskGraph model bundle."
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


@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
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
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp detected."
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

    for col in numeric_columns:

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

    df["high_value_transaction"] = (
        df["amount"] > 10000
    ).astype(int)

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

def calculate_anomaly_scores(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    split = int(
        len(work) * 0.80
    )

    historical = work.iloc[
        :split
    ].copy()

    if "is_fraud" in historical.columns:

        legitimate = historical[
            historical["is_fraud"] == 0
        ].copy()

    else:

        legitimate = historical.copy()

    if len(legitimate) < 100:
        legitimate = historical.copy()

    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    detector.fit(
        legitimate[
            ANOMALY_FEATURES
        ]
    )

    training_scores = -detector.decision_function(
        legitimate[
            ANOMALY_FEATURES
        ]
    )

    all_scores = -detector.decision_function(
        work[
            ANOMALY_FEATURES
        ]
    )

    reference = np.sort(
        training_scores
    )

    percentiles = (
        np.searchsorted(
            reference,
            all_scores,
            side="right"
        )
        / len(reference)
    )

    work["anomaly_score"] = np.clip(
        percentiles * 100,
        0,
        100
    )

    return work


# ============================================================
# ENTITY GRAPH ENGINE
# ============================================================

def calculate_graph_features(df):

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    split = int(
        len(work) * 0.80
    )

    historical = work.iloc[
        :split
    ].copy()

    device_customers = (
        historical
        .groupby("device_id")
        ["customer_id"]
        .nunique()
    )

    ip_customers = (
        historical
        .groupby("ip_id")
        ["customer_id"]
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
# ============================================================

@st.cache_data
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

    # 1. Supervised fraud model
    df["fraud_probability"] = (
        model.predict_proba(
            df[saved_features]
        )[:, 1]
    )

    # 2. Behavioral anomaly
    df = calculate_anomaly_scores(
        df
    )

    # 3. Entity graph
    df = calculate_graph_features(
        df
    )

    # 4. Expected fraud loss
    df["expected_fraud_loss"] = (
        df["fraud_probability"]
        * df["amount"]
    )

    # 5. Financial exposure
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

    # 6. Final RiskGraph score
    #
    # 50% fraud
    # 30% anomaly
    # 20% financial exposure

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

    # 7. Cost-aware recommendation

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

    costs = df[
        [
            "approve_cost",
            "verify_cost",
            "review_cost"
        ]
    ]

    df["recommended_action"] = (
        costs
        .idxmin(axis=1)
        .map(
            {
                "approve_cost": "APPROVE",
                "verify_cost": "VERIFY",
                "review_cost": "REVIEW",
            }
        )
    )

    # 8. Final decision policy

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

        default="REVIEW"
    )

    # 9. Risk band

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
# LOAD DATA
# ============================================================

try:

    raw_data = load_data()

except Exception as error:

    st.error(
        "Could not load sample_transactions.csv"
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "# 🛡️ RiskGraph AI"
)

st.sidebar.caption(
    "AI Payment Risk Manager"
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload transactions",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.sidebar.error(
            "Unable to read CSV."
        )

        st.stop()


try:

    data = run_pipeline(
        raw_data
    )

except Exception as error:

    st.error(
        "RiskGraph AI pipeline failed."
    )

    st.exception(error)

    st.stop()


st.sidebar.success(
    "Risk engine online"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Command Center",
        "Investigate",
        "Entity Intelligence",
        "Model & Results",
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
    "Interventions",
    f"{(data['final_action'] != 'APPROVE').sum():,}"
)

st.sidebar.caption(
    "Risk policy"
)

st.sidebar.caption(
    "0–59.99  APPROVE"
)

st.sidebar.caption(
    "60–74.99  VERIFY"
)

st.sidebar.caption(
    "75–100  REVIEW"
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
                Payment Risk Command Center
                · Detect · Explain · Decide
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    total = len(data)

    total_value = data[
        "amount"
    ].sum()

    approve = (
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

    critical = (
        data["risk_score"]
        >= 75
    ).sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Transactions",
        f"{total:,}"
    )

    c2.metric(
        "Value monitored",
        money(total_value)
    )

    c3.metric(
        "Auto-approved",
        f"{approve:,}"
    )

    c4.metric(
        "Verification",
        f"{verify:,}"
    )

    c5.metric(
        "High-risk review",
        f"{review:,}"
    )

    st.divider()

    left, right = st.columns(
        [1.55, 1]
    )

    with left:

        st.markdown(
            '<div class="section-title">'
            'Risk Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        histogram = pd.cut(
            data["risk_score"],
            bins=np.arange(
                0,
                105,
                5
            ),
            include_lowest=True
        )

        counts = (
            histogram
            .value_counts()
            .sort_index()
        )

        counts.index = [
            f"{int(x.left)}–{int(x.right)}"
            for x in counts.index
        ]

        st.bar_chart(
            counts,
            height=350
        )

        st.caption(
            "Risk score thresholds: "
            "60 = VERIFY · 75 = REVIEW"
        )

    with right:

        st.markdown(
            '<div class="section-title">'
            'Decision Mix'
            '</div>',
            unsafe_allow_html=True
        )

        decision_mix = pd.Series(
            {
                "APPROVE": approve,
                "VERIFY": verify,
                "REVIEW": review,
            }
        )

        st.bar_chart(
            decision_mix,
            height=260
        )

        st.markdown(
            """
            **Decision policy**

            🟢 APPROVE — below 60

            🟠 VERIFY — 60 to 74.99

            🔴 REVIEW — 75 or higher
            """
        )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Priority Risk Queue'
        '</div>',
        unsafe_allow_html=True
    )

    priority = (
        data
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(12)
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

    priority_display["amount"] = (
        priority_display["amount"]
        .map(money)
    )

    priority_display[
        "fraud_probability"
    ] = (
        priority_display[
            "fraud_probability"
        ].map(pct)
    )

    st.dataframe(
        priority_display,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "RiskGraph combines supervised fraud prediction, "
        "behavioral anomaly detection, entity relationships "
        "and financial exposure into a final decision."
    )


# ============================================================
# INVESTIGATION
# ============================================================

elif page == "Investigate":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                Transaction Investigation
            </div>

            <div class="hero-subtitle">
                Understand exactly why RiskGraph AI
                assigned a transaction its decision.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    search = st.text_input(
        "Search transaction, customer, merchant, device or IP",
        placeholder="Example: TX_0009808"
    )

    candidates = data.copy()

    if search.strip():

        q = search.lower().strip()

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

    selected = st.selectbox(
        "Select transaction",
        candidates[
            "transaction_id"
        ].tolist()
    )

    row = candidates[
        candidates["transaction_id"]
        == selected
    ].iloc[0]

    st.divider()

    st.subheader(
        f"Investigation: {selected}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Risk Score",
        f"{row['risk_score']:.1f}/100"
    )

    c2.metric(
        "Fraud Probability",
        pct(
            row["fraud_probability"]
        )
    )

    c3.metric(
        "Anomaly",
        f"{row['anomaly_score']:.1f}/100"
    )

    c4.metric(
        "Transaction",
        money(row["amount"])
    )

    action = row["final_action"]

    if action == "REVIEW":

        st.error(
            "🔴 REVIEW — High-risk transaction"
        )

    elif action == "VERIFY":

        st.warning(
            "🟠 VERIFY — Additional verification required"
        )

    else:

        st.success(
            "🟢 APPROVE — Below intervention threshold"
        )

    st.markdown(
        '<div class="section-title">'
        'Risk Signal Breakdown'
        '</div>',
        unsafe_allow_html=True
    )

    signals = pd.DataFrame(
        {
            "Signal": [
                "Fraud probability",
                "Behavior anomaly",
                "Entity graph",
                "Financial exposure",
                "Final risk",
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

    signals["Score"] = (
        signals["Score"]
        .round(2)
    )

    st.dataframe(
        signals,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">'
        'Why this decision?'
        '</div>',
        unsafe_allow_html=True
    )

    reasons = explanation(
        row
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
        "Device customers",
        int(
            row["device_customer_count"]
        )
    )

    e2.metric(
        "IP customers",
        int(
            row["ip_customer_count"]
        )
    )

    e3.metric(
        "Device transactions",
        int(
            row["device_transaction_count"]
        )
    )

    e4.metric(
        "IP transactions",
        int(
            row["ip_transaction_count"]
        )
    )

    st.markdown(
        '<div class="section-title">'
        'Transaction Evidence'
        '</div>',
        unsafe_allow_html=True
    )

    evidence = pd.DataFrame(
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
        evidence,
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
                Investigate device, IP and customer relationships
                that provide additional fraud context.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    entity_type = st.selectbox(
        "Entity",
        [
            "Device",
            "IP Address",
            "Customer",
        ]
    )

    if entity_type == "Device":

        entity_table = (
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

        entity_column = "device_id"

    elif entity_type == "IP Address":

        entity_table = (
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

        entity_column = "ip_id"

    else:

        entity_table = (
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

        entity_column = "customer_id"

    st.dataframe(
        entity_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">'
        'Inspect Entity'
        '</div>',
        unsafe_allow_html=True
    )

    selected_entity = st.selectbox(
        "Select",
        entity_table[
            entity_column
        ].tolist()
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

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Transactions",
        f"{len(related):,}"
    )

    r2.metric(
        "Average Risk",
        f"{related['risk_score'].mean():.1f}"
    )

    r3.metric(
        "High-risk cases",
        f"{(related['risk_score'] >= 75).sum():,}"
    )

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
                "final_action",
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
# MODEL & RESULTS
# ============================================================

elif page == "Model & Results":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                Model Intelligence
            </div>

            <div class="hero-subtitle">
                Validation evidence from the completed
                RiskGraph AI modelling pipeline.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "These metrics are the measured results from the "
        "completed held-out evaluation."
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
        "91.25%"
    )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Anomaly Engine'
        '</div>',
        unsafe_allow_html=True
    )

    st.metric(
        "Anomaly ROC-AUC",
        "99.12%"
    )

    st.markdown(
        '<div class="section-title">'
        'Risk Decision Architecture'
        '</div>',
        unsafe_allow_html=True
    )

    architecture = pd.DataFrame(
        {
            "Layer": [
                "Supervised model",
                "Behavior anomaly",
                "Entity intelligence",
                "Financial exposure",
                "Risk fusion",
                "Decision policy",
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
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">'
        'Final Decision Policy'
        '</div>',
        unsafe_allow_html=True
    )

    policy = pd.DataFrame(
        {
            "Risk score": [
                "< 60",
                "60 – 74.99",
                "≥ 75",
            ],

            "Decision": [
                "APPROVE",
                "VERIFY",
                "REVIEW",
            ],

            "Purpose": [
                "Low intervention risk",
                "Additional verification",
                "Manual investigation",
            ],
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

    st.markdown(
        "### Business evaluation from final policy"
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
        "Estimated FP cost",
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

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                Business Impact
            </div>

            <div class="hero-subtitle">
                Translate model decisions into financial risk
                and intervention impact.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    baseline_loss = (
        data["expected_fraud_loss"]
        .sum()
    )

    residual_loss = 0

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
        "Business intervention costs and fraud-reduction "
        "assumptions are prototype assumptions used for "
        "project evaluation."
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
                Search, filter and export RiskGraph decisions.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    actions = st.multiselect(
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
        ]
    )

    min_risk = st.slider(
        "Minimum risk score",
        0,
        100,
        0
    )

    filtered = data[
        data["final_action"]
        .isin(actions)
        &
        (
            data["risk_score"]
            >= min_risk
        )
    ].copy()

    st.metric(
        "Matching transactions",
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

    display["amount"] = (
        display["amount"]
        .map(money)
    )

    display[
        "fraud_probability"
    ] = (
        display[
            "fraud_probability"
        ].map(pct)
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download RiskGraph Results",
        data=csv,
        file_name="riskgraph_results.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RiskGraph AI · AI Payment Risk Manager · "
    "Defense-only fraud risk detection and decisioning"
)
