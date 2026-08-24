import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# ============================================================
# RISKGRAPH AI — STREAMLIT APPLICATION
# Built around Razorpay.ipynb
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
    }
    .risk-high {
        padding: 14px;
        border-radius: 10px;
        border-left: 5px solid #d62728;
        background: rgba(214,39,40,.08);
    }
    .risk-medium {
        padding: 14px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        background: rgba(255,152,0,.08);
    }
    .risk-low {
        padding: 14px;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        background: rgba(44,160,44,.08);
    }
    .small-muted {color: #777; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Constants from Razorpay.ipynb
# -----------------------------
MODEL_PATH = "riskgraph_fraud_model_v2.joblib"
DEFAULT_DATA_PATH = "sample_transactions.csv"

FEATURES = [
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
FALSE_POSITIVE_RATE = 0.01
VERIFY_FRAUD_REDUCTION = 0.80
REVIEW_FRAUD_REDUCTION = 0.95

APPROVE_THRESHOLD = 60
REVIEW_THRESHOLD = 75

# Notebook-reported evaluation numbers
NOTEBOOK_METRICS = {
    "Future holdout ROC-AUC": 0.9985,
    "Future holdout Precision": 0.8511,
    "Future holdout Recall": 0.9836,
    "Future holdout F1": 0.9125,
    "Anomaly ROC-AUC": 0.9912,
    "Graph Risk ROC-AUC": 0.3673,
}

# -----------------------------
# Loading
# -----------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA_PATH)


# -----------------------------
# Feature engineering
# Matches Razorpay.ipynb V2
# -----------------------------
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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamp values could not be parsed.")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["account_age_days"] = pd.to_numeric(
        df["account_age_days"], errors="coerce"
    ).fillna(0)
    df["device_age_days"] = pd.to_numeric(
        df["device_age_days"], errors="coerce"
    ).fillna(0)
    df["transactions_last_10min"] = pd.to_numeric(
        df["transactions_last_10min"], errors="coerce"
    ).fillna(0)
    df["failed_attempts"] = pd.to_numeric(
        df["failed_attempts"], errors="coerce"
    ).fillna(0)
    df["location_change"] = pd.to_numeric(
        df["location_change"], errors="coerce"
    ).fillna(0)
    df["amount_deviation"] = pd.to_numeric(
        df["amount_deviation"], errors="coerce"
    ).fillna(0)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["log_amount"] = np.log1p(df["amount"])

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
        + (df["amount_deviation"] > 3).astype(int)
    )

    return df


# -----------------------------
# Anomaly engine
# Same Isolation Forest logic as notebook
# -----------------------------
@st.cache_data
def calculate_anomaly_scores(df):
    work = df.copy().sort_values("timestamp").reset_index(drop=True)

    split = int(len(work) * 0.80)
    historical = work.iloc[:split].copy()

    legitimate = historical[
        historical["is_fraud"] == 0
    ] if "is_fraud" in historical.columns else historical

    if len(legitimate) < 100:
        legitimate = historical

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    X_train = legitimate[ANOMALY_FEATURES]
    X_all = work[ANOMALY_FEATURES]

    anomaly_model.fit(X_train)

    train_raw = -anomaly_model.decision_function(X_train)
    all_raw = -anomaly_model.decision_function(X_all)

    sorted_training_scores = np.sort(train_raw)

    percentiles = (
        np.searchsorted(
            sorted_training_scores,
            all_raw,
            side="right",
        )
        / len(sorted_training_scores)
    )

    scores = np.clip(percentiles * 100, 0, 100)

    work["anomaly_score"] = scores

    return work


# -----------------------------
# Graph intelligence
# Mirrors notebook graph-risk formulation
# -----------------------------
def calculate_graph_features(df):
    work = df.copy().sort_values("timestamp").reset_index(drop=True)

    split = int(len(work) * 0.80)
    historical = work.iloc[:split].copy()

    # Historical entity relationships
    device_customers = historical.groupby("device_id")["customer_id"].nunique()
    ip_customers = historical.groupby("ip_id")["customer_id"].nunique()

    device_transactions = historical.groupby("device_id").size()
    ip_transactions = historical.groupby("ip_id").size()

    work["device_customer_count"] = (
        work["device_id"].map(device_customers).fillna(0)
    )

    work["ip_customer_count"] = (
        work["ip_id"].map(ip_customers).fillna(0)
    )

    work["device_transaction_count"] = (
        work["device_id"].map(device_transactions).fillna(0)
    )

    work["ip_transaction_count"] = (
        work["ip_id"].map(ip_transactions).fillna(0)
    )

    work["shared_device"] = (
        work["device_customer_count"] > 1
    ).astype(int)

    work["shared_ip"] = (
        work["ip_customer_count"] > 1
    ).astype(int)

    work["entity_risk_count"] = (
        work["shared_device"] + work["shared_ip"]
    )

    work["graph_risk_score"] = (
        np.minimum(work["device_customer_count"] * 8, 30)
        + np.minimum(work["ip_customer_count"] * 8, 30)
        + np.minimum(work["device_transaction_count"] * 1.5, 20)
        + np.minimum(work["ip_transaction_count"] * 1.5, 20)
    )

    work["graph_risk_score"] = np.clip(
        work["graph_risk_score"], 0, 100
    )

    return work


# -----------------------------
# Full risk pipeline
# -----------------------------
@st.cache_data
def run_pipeline(raw_df):
    df = engineer_features(raw_df)

    model, saved_features = load_model()

    # Preserve exact saved model feature ordering.
    X = df[saved_features]

    df["fraud_probability"] = model.predict_proba(X)[:, 1]

    # Anomaly score
    df = calculate_anomaly_scores(df)

    # Graph intelligence
    df = calculate_graph_features(df)

    # Risk fusion
    df["fraud_signal"] = df["fraud_probability"] * 100
    df["anomaly_signal"] = df["anomaly_score"]

    df["raw_risk_score"] = (
        0.70 * df["fraud_signal"]
        + 0.30 * df["anomaly_signal"]
    )

    df["raw_risk_score"] = np.clip(
        df["raw_risk_score"], 0, 100
    )

    # Financial exposure
    df["expected_fraud_loss"] = (
        df["fraud_probability"] * df["amount"]
    )

    # Cost-aware recommendation
    df["approve_cost"] = df["expected_fraud_loss"]

    df["verify_cost"] = (
        VERIFY_COST
        + df["expected_fraud_loss"]
        * (1 - VERIFY_FRAUD_REDUCTION)
    )

    df["review_cost"] = (
        REVIEW_COST
        + df["expected_fraud_loss"]
        * (1 - REVIEW_FRAUD_REDUCTION)
    )

    cost_matrix = df[
        ["approve_cost", "verify_cost", "review_cost"]
    ]

    df["recommended_action"] = (
        cost_matrix.idxmin(axis=1)
        .map(
            {
                "approve_cost": "APPROVE",
                "verify_cost": "VERIFY",
                "review_cost": "REVIEW",
            }
        )
    )

    # User-facing score from notebook:
    # 50% fraud signal
    # 30% anomaly signal
    # 20% financial exposure
    loss_cap = df["expected_fraud_loss"].quantile(0.95)

    df["financial_exposure_score"] = (
        df["expected_fraud_loss"]
        / max(loss_cap, 1)
        * 100
    )

    df["financial_exposure_score"] = np.clip(
        df["financial_exposure_score"], 0, 100
    )

    df["risk_score"] = (
        0.50 * df["fraud_signal"]
        + 0.30 * df["anomaly_signal"]
        + 0.20 * df["financial_exposure_score"]
    )

    df["risk_score"] = np.clip(
        df["risk_score"], 0, 100
    )

    # Final decision policy from notebook
    def final_decision(score):
        if score < APPROVE_THRESHOLD:
            return "APPROVE"
        elif score < REVIEW_THRESHOLD:
            return "VERIFY"
        return "REVIEW"

    df["final_action"] = df["risk_score"].apply(final_decision)

    # Risk bands for UI
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 30, 60, 75, 100],
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        ],
    ).astype(str)

    # Explanation flags
    df["signal_count"] = (
        (df["fraud_probability"] >= 0.50).astype(int)
        + (df["anomaly_score"] >= 75).astype(int)
        + (df["financial_exposure_score"] >= 75).astype(int)
        + (df["shared_device"] == 1).astype(int)
        + (df["shared_ip"] == 1).astype(int)
        + (df["behavior_risk_count"] >= 3).astype(int)
    )

    return df


# -----------------------------
# Helpers
# -----------------------------
def money(value):
    return f"₹{value:,.2f}"


def pct(value):
    return f"{value * 100:.2f}%"


def show_risk_badge(action):
    if action == "REVIEW":
        st.error("🔴 REVIEW")
    elif action == "VERIFY":
        st.warning("🟠 VERIFY")
    else:
        st.success("🟢 APPROVE")


def risk_explanation(row):
    reasons = []

    if row["fraud_probability"] >= 0.50:
        reasons.append(
            f"supervised fraud probability is {row['fraud_probability']:.1%}"
        )

    if row["anomaly_score"] >= 75:
        reasons.append(
            f"behavioral anomaly score is {row['anomaly_score']:.1f}/100"
        )

    if row["financial_exposure_score"] >= 75:
        reasons.append(
            f"financial exposure is {row['financial_exposure_score']:.1f}/100"
        )

    if row["shared_device"]:
        reasons.append(
            f"device is shared across {int(row['device_customer_count'])} customers"
        )

    if row["shared_ip"]:
        reasons.append(
            f"IP is shared across {int(row['ip_customer_count'])} customers"
        )

    if row["transactions_last_10min"] >= 4:
        reasons.append(
            f"high velocity ({int(row['transactions_last_10min'])} transactions/10 min)"
        )

    if row["failed_attempts"] >= 3:
        reasons.append(
            f"high failed-attempt activity ({int(row['failed_attempts'])})"
        )

    if row["new_device"]:
        reasons.append("new device")

    if row["new_account"]:
        reasons.append("new account")

    if row["location_change"]:
        reasons.append("location change")

    if row["amount_deviation"] > 3:
        reasons.append(
            f"transaction amount is {row['amount_deviation']:.1f}× the normal amount"
        )

    if not reasons:
        reasons.append("no major risk signal crossed the configured thresholds")

    return reasons


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🛡️ RiskGraph AI")
st.sidebar.caption("Fraud intelligence & decision engine")

uploaded = st.sidebar.file_uploader(
    "Upload transactions CSV",
    type=["csv"],
)

if uploaded is not None:
    try:
        raw_data = pd.read_csv(uploaded)
        data_source = "Uploaded CSV"
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    try:
        raw_data = load_default_data()
        data_source = "sample_transactions.csv"
    except FileNotFoundError:
        st.error(
            "sample_transactions.csv was not found. "
            "Put it beside app.py or upload a CSV."
        )
        st.stop()

st.sidebar.caption(f"Data source: {data_source}")

if "transaction_id" not in raw_data.columns:
    st.error("The CSV does not contain transaction_id.")
    st.stop()

try:
    data = run_pipeline(raw_data)
except Exception as exc:
    st.error(f"Risk pipeline failed: {exc}")
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    [
        "Command Center",
        "Investigations",
        "Entity Network",
        "Model Intelligence",
        "Business Impact",
        "Data Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.metric(
    "Transactions",
    f"{len(data):,}",
)

st.sidebar.metric(
    "Interventions",
    f"{(data['final_action'] != 'APPROVE').sum():,}",
)

st.sidebar.caption(
    "Prototype cost assumptions are modelling assumptions, "
    "not Razorpay operational costs."
)


# ============================================================
# COMMAND CENTER
# ============================================================
if page == "Command Center":

    st.title("RiskGraph AI")
    st.subheader("Real-time fraud risk command center")
    st.caption(
        "Combines supervised fraud probability, behavioral anomaly "
        "detection, entity relationships and financial exposure."
    )

    total_tx = len(data)
    total_amount = data["amount"].sum()
    review_count = (data["final_action"] == "REVIEW").sum()
    verify_count = (data["final_action"] == "VERIFY").sum()
    high_risk = (data["risk_score"] >= 75).sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Transactions", f"{total_tx:,}")
    c2.metric("Transaction Value", money(total_amount))
    c3.metric("Critical Risk", f"{high_risk:,}")
    c4.metric("Verify", f"{verify_count:,}")
    c5.metric("Review", f"{review_count:,}")

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("### Risk distribution")

        hist = px.histogram(
            data,
            x="risk_score",
            nbins=30,
            title="User-facing risk score",
            labels={"risk_score": "Risk Score"},
        )

        hist.add_vline(
            x=60,
            line_dash="dash",
            annotation_text="VERIFY",
        )

        hist.add_vline(
            x=75,
            line_dash="dash",
            annotation_text="REVIEW",
        )

        st.plotly_chart(
            hist,
            use_container_width=True,
        )

    with right:
        st.markdown("### Final decisions")

        decision_counts = (
            data["final_action"]
            .value_counts()
            .rename_axis("Action")
            .reset_index(name="Transactions")
        )

        pie = px.pie(
            decision_counts,
            names="Action",
            values="Transactions",
            hole=0.55,
        )

        st.plotly_chart(
            pie,
            use_container_width=True,
        )

    st.markdown("### Highest-risk transactions")

    cols = [
        "transaction_id",
        "amount",
        "fraud_probability",
        "anomaly_score",
        "graph_risk_score",
        "financial_exposure_score",
        "risk_score",
        "final_action",
    ]

    top = (
        data.sort_values("risk_score", ascending=False)
        [cols]
        .head(15)
        .copy()
    )

    top["amount"] = top["amount"].map(lambda x: f"₹{x:,.2f}")
    top["fraud_probability"] = top["fraud_probability"].map(
        lambda x: f"{x:.2%}"
    )
    top["anomaly_score"] = top["anomaly_score"].round(1)
    top["graph_risk_score"] = top["graph_risk_score"].round(1)
    top["financial_exposure_score"] = (
        top["financial_exposure_score"].round(1)
    )
    top["risk_score"] = top["risk_score"].round(1)

    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Decision policy: risk score < 60 → APPROVE; "
        "60–74.99 → VERIFY; ≥ 75 → REVIEW."
    )


# ============================================================
# INVESTIGATIONS
# ============================================================
elif page == "Investigations":

    st.title("Investigation Queue")
    st.caption(
        "Prioritize transactions that require analyst attention."
    )

    search = st.text_input(
        "Search transaction / customer / merchant / device / IP",
        placeholder="TX_0013175, CUST_..., DEV_..., IP_...",
    )

    action_filter = st.multiselect(
        "Decision",
        ["APPROVE", "VERIFY", "REVIEW"],
        default=["VERIFY", "REVIEW"],
    )

    risk_min = st.slider(
        "Minimum risk score",
        0,
        100,
        0,
    )

    queue = data[
        data["final_action"].isin(action_filter)
        & (data["risk_score"] >= risk_min)
    ].copy()

    if search.strip():
        q = search.strip().lower()
        mask = (
            queue["transaction_id"].astype(str).str.lower().str.contains(q)
            | queue["customer_id"].astype(str).str.lower().str.contains(q)
            | queue["merchant_id"].astype(str).str.lower().str.contains(q)
            | queue["device_id"].astype(str).str.lower().str.contains(q)
            | queue["ip_id"].astype(str).str.lower().str.contains(q)
        )
        queue = queue[mask]

    queue = queue.sort_values(
        ["risk_score", "amount"],
        ascending=False,
    )

    st.metric(
        "Transactions in queue",
        f"{len(queue):,}",
    )

    display_cols = [
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

    st.dataframe(
        queue[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    if len(queue) > 0:
        selected_id = st.selectbox(
            "Open investigation",
            queue["transaction_id"].tolist(),
        )

        row = queue[
            queue["transaction_id"] == selected_id
        ].iloc[0]

        st.markdown(f"## Investigation — `{selected_id}`")

        a, b, c, d = st.columns(4)

        a.metric(
            "Risk Score",
            f"{row['risk_score']:.1f}/100",
        )
        b.metric(
            "Fraud Probability",
            f"{row['fraud_probability']:.2%}",
        )
        c.metric(
            "Anomaly",
            f"{row['anomaly_score']:.1f}/100",
        )
        d.metric(
            "Amount",
            money(row["amount"]),
        )

        st.markdown("### Decision")
        show_risk_badge(row["final_action"])

        st.markdown("### Why was this transaction flagged?")

        reasons = risk_explanation(row)

        for reason in reasons:
            st.write("• " + reason)

        st.markdown("### Entity context")

        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "Device customers",
            int(row["device_customer_count"]),
        )
        e2.metric(
            "IP customers",
            int(row["ip_customer_count"]),
        )
        e3.metric(
            "Device transactions",
            int(row["device_transaction_count"]),
        )
        e4.metric(
            "IP transactions",
            int(row["ip_transaction_count"]),
        )

        st.markdown("### Transaction details")

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
                    "Financial exposure score",
                    "Graph risk score",
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

    st.title("Entity Network Intelligence")
    st.caption(
        "Investigate relationships between customers, devices and IPs."
    )

    entity_type = st.selectbox(
        "Entity type",
        ["Device", "IP", "Customer"],
    )

    if entity_type == "Device":
        counts = (
            data.groupby("device_id")
            .agg(
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "count"),
                avg_risk=("risk_score", "mean"),
            )
            .sort_values(
                ["customers", "transactions"],
                ascending=False,
            )
            .head(30)
            .reset_index()
        )
        entity_col = "device_id"

    elif entity_type == "IP":
        counts = (
            data.groupby("ip_id")
            .agg(
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "count"),
                avg_risk=("risk_score", "mean"),
            )
            .sort_values(
                ["customers", "transactions"],
                ascending=False,
            )
            .head(30)
            .reset_index()
        )
        entity_col = "ip_id"

    else:
        counts = (
            data.groupby("customer_id")
            .agg(
                devices=("device_id", "nunique"),
                ips=("ip_id", "nunique"),
                transactions=("transaction_id", "count"),
                avg_risk=("risk_score", "mean"),
            )
            .sort_values(
                ["transactions", "avg_risk"],
                ascending=False,
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

    if entity_type in ["Device", "IP"]:
        chart = px.scatter(
            counts,
            x="transactions",
            y="customers",
            size="avg_risk",
            hover_name=entity_col,
            title=f"{entity_type} relationship density",
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    selected = st.selectbox(
        f"Inspect {entity_type.lower()}",
        counts[entity_col].tolist(),
    )

    if entity_type == "Device":
        related = data[data["device_id"] == selected]
    elif entity_type == "IP":
        related = data[data["ip_id"] == selected]
    else:
        related = data[data["customer_id"] == selected]

    st.markdown("### Related transactions")

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
        .sort_values("risk_score", ascending=False)
        .head(50),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================
elif page == "Model Intelligence":

    st.title("Model Intelligence")
    st.caption(
        "Evaluation results reported by Razorpay.ipynb."
    )

    st.info(
        "These metrics are the notebook's recorded evaluation results. "
        "The dashboard does not recompute the historical holdout metrics."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Future Holdout ROC-AUC",
        f"{NOTEBOOK_METRICS['Future holdout ROC-AUC']:.4f}",
    )

    m2.metric(
        "Future Holdout Recall",
        f"{NOTEBOOK_METRICS['Future holdout Recall']:.4f}",
    )

    m3.metric(
        "Future Holdout F1",
        f"{NOTEBOOK_METRICS['Future holdout F1']:.4f}",
    )

    m4, m5, m6 = st.columns(3)

    m4.metric(
        "Future Holdout Precision",
        f"{NOTEBOOK_METRICS['Future holdout Precision']:.4f}",
    )

    m5.metric(
        "Anomaly ROC-AUC",
        f"{NOTEBOOK_METRICS['Anomaly ROC-AUC']:.4f}",
    )

    m6.metric(
        "Graph Risk ROC-AUC",
        f"{NOTEBOOK_METRICS['Graph Risk ROC-AUC']:.4f}",
    )

    st.divider()

    st.markdown("### Model architecture")

    architecture = pd.DataFrame(
        {
            "Layer": [
                "Supervised model",
                "Behavioral anomaly engine",
                "Entity graph",
                "Risk fusion",
                "Financial exposure",
                "Final policy",
            ],
            "Implementation": [
                "Random Forest V2",
                "Isolation Forest",
                "Device / IP relationships",
                "70% fraud + 30% anomaly",
                "Expected fraud loss",
                "50% fraud + 30% anomaly + 20% exposure",
            ],
        }
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Feature importance")

    try:
        model, saved_features = load_model()

        importance = pd.DataFrame(
            {
                "feature": saved_features,
                "importance": model.feature_importances_,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        fig = px.bar(
            importance.head(15).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top model features",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except Exception as exc:
        st.warning(
            f"Could not display model feature importance: {exc}"
        )

    st.markdown("### Current data distribution")

    if "is_fraud" in data.columns:
        fraud_rate = data["is_fraud"].mean()
        st.metric(
            "Dataset fraud rate",
            f"{fraud_rate:.2%}",
        )


# ============================================================
# BUSINESS IMPACT
# ============================================================
elif page == "Business Impact":

    st.title("Business Impact")
    st.caption(
        "Cost-aware decisioning using the modelling assumptions "
        "defined in Razorpay.ipynb."
    )

    total_exposure = data["expected_fraud_loss"].sum()

    approve_loss = data.loc[
        data["final_action"] == "APPROVE",
        "expected_fraud_loss",
    ].sum()

    verify_residual = (
        data.loc[
            data["final_action"] == "VERIFY",
            "expected_fraud_loss",
        ]
        * (1 - VERIFY_FRAUD_REDUCTION)
    ).sum()

    review_residual = (
        data.loc[
            data["final_action"] == "REVIEW",
            "expected_fraud_loss",
        ]
        * (1 - REVIEW_FRAUD_REDUCTION)
    ).sum()

    residual_loss = (
        approve_loss
        + verify_residual
        + review_residual
    )

    intervention_cost = (
        (data["final_action"] == "VERIFY").sum() * VERIFY_COST
        + (data["final_action"] == "REVIEW").sum() * REVIEW_COST
    )

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Expected Fraud Loss",
        money(total_exposure),
    )

    b2.metric(
        "Residual Loss",
        money(residual_loss),
    )

    b3.metric(
        "Intervention Cost",
        money(intervention_cost),
    )

    b4.metric(
        "Total Decision Cost",
        money(residual_loss + intervention_cost),
    )

    st.divider()

    decision_summary = (
        data.groupby("final_action")
        .agg(
            transactions=("transaction_id", "count"),
            transaction_value=("amount", "sum"),
            expected_loss=("expected_fraud_loss", "sum"),
        )
        .reset_index()
    )

    st.markdown("### Decision economics")

    st.dataframe(
        decision_summary,
        use_container_width=True,
        hide_index=True,
    )

    chart = px.bar(
        decision_summary,
        x="final_action",
        y="expected_loss",
        title="Expected fraud loss by final action",
        labels={
            "final_action": "Decision",
            "expected_loss": "Expected Fraud Loss (₹)",
        },
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    st.warning(
        "VERIFY_COST = ₹25, REVIEW_COST = ₹75 and the fraud-reduction "
        "assumptions are prototype modelling assumptions from the notebook, "
        "not Razorpay's actual operating costs."
    )


# ============================================================
# DATA EXPLORER
# ============================================================
elif page == "Data Explorer":

    st.title("Data Explorer")

    st.caption(
        "Inspect transactions and all generated risk signals."
    )

    columns = st.multiselect(
        "Columns",
        data.columns.tolist(),
        default=[
            "transaction_id",
            "customer_id",
            "amount",
            "timestamp",
            "fraud_probability",
            "anomaly_score",
            "graph_risk_score",
            "financial_exposure_score",
            "risk_score",
            "final_action",
        ],
    )

    if columns:
        st.dataframe(
            data[columns],
            use_container_width=True,
            hide_index=True,
        )

        csv = data[columns].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download filtered results",
            data=csv,
            file_name="riskgraph_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Select at least one column.")

    st.divider()

    st.markdown("### Dataset overview")

    overview = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Total amount",
                "Average transaction",
                "Fraud cases",
                "Fraud rate",
                "Average risk score",
                "Review count",
                "Verify count",
            ],
            "Value": [
                f"{len(data):,}",
                f"{len(data.columns):,}",
                money(data["amount"].sum()),
                money(data["amount"].mean()),
                f"{int(data['is_fraud'].sum())}"
                if "is_fraud" in data.columns
                else "N/A",
                f"{data['is_fraud'].mean():.2%}"
                if "is_fraud" in data.columns
                else "N/A",
                f"{data['risk_score'].mean():.2f}",
                f"{(data['final_action'] == 'REVIEW').sum():,}",
                f"{(data['final_action'] == 'VERIFY').sum():,}",
            ],
        }
    )

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "RiskGraph AI • Prototype fraud intelligence system • "
    "Built from the Razorpay.ipynb modelling pipeline"
)
