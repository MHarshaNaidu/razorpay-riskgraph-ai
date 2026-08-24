import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# RISKGRAPH AI
# AI Risk Manager — Razorpay Buildathon Track 02
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "sample_transactions.csv"
    )

    return df


df = load_data()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🛡️ RiskGraph AI")

st.caption(
    "Cost-Aware Payment Risk Intelligence"
)

st.markdown(
    """
    **AI Risk Manager**

    Detect suspicious payment behavior, prioritize intervention,
    and reduce merchant loss while controlling customer friction.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Risk Operations")

page = st.sidebar.radio(
    "Navigate",
    [
        "Command Center",
        "Transaction Investigation",
        "Model Performance",
        "Business Impact",
        "Entity Context"
    ]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):

    return f"₹{value:,.2f}"


def pct(value):

    return f"{value * 100:.2f}%"


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.header("Risk Command Center")

    total_transactions = len(df)

    fraud_count = int(
        df["is_fraud"].sum()
    )

    review_count = int(
        (df["final_action"] == "REVIEW").sum()
    )

    verify_count = int(
        (df["final_action"] == "VERIFY").sum()
    )

    approve_count = int(
        (df["final_action"] == "APPROVE").sum()
    )

    false_positive_cost = 4837.53

    intervention_recall = 0.9795


    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Cases",
        f"{fraud_count:,}"
    )

    col3.metric(
        "Intervention Recall",
        pct(intervention_recall)
    )

    col4.metric(
        "Estimated FP Cost",
        money(false_positive_cost)
    )


    st.divider()


    # --------------------------------------------------------
    # DECISION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Risk Decision Distribution")

    decision_data = pd.DataFrame(
        {
            "Decision": [
                "APPROVE",
                "VERIFY",
                "REVIEW"
            ],
            "Transactions": [
                approve_count,
                verify_count,
                review_count
            ]
        }
    )

    st.bar_chart(
        decision_data.set_index(
            "Decision"
        )
    )


    st.divider()


    # --------------------------------------------------------
    # HIGH RISK QUEUE
    # --------------------------------------------------------

    st.subheader("High-Risk Transaction Queue")

    high_risk = (
        df[
            df["final_action"] == "REVIEW"
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )


    display_columns = [
        "transaction_id",
        "amount",
        "risk_score",
        "fraud_probability",
        "anomaly_score",
        "final_action"
    ]


    st.dataframe(
        high_risk[display_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TRANSACTION INVESTIGATION
# ============================================================

elif page == "Transaction Investigation":

    st.header("🔎 Transaction Investigation")

    transaction_ids = (
        df["transaction_id"]
        .astype(str)
        .tolist()
    )


    selected_id = st.selectbox(
        "Select transaction",
        transaction_ids
    )


    row = df[
        df["transaction_id"].astype(str)
        == selected_id
    ].iloc[0]


    st.divider()


    # --------------------------------------------------------
    # MAIN RISK CARD
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Risk Score",
        f"{row['risk_score']:.1f} / 100"
    )


    col2.metric(
        "Fraud Signal",
        f"{row['fraud_probability'] * 100:.1f} / 100"
    )


    col3.metric(
        "Anomaly Signal",
        f"{row['anomaly_score']:.1f} / 100"
    )


    action = row["final_action"]


    if action == "REVIEW":

        st.error(
            f"🔴 RECOMMENDED ACTION: {action}"
        )

    elif action == "VERIFY":

        st.warning(
            f"🟡 RECOMMENDED ACTION: {action}"
        )

    else:

        st.success(
            f"🟢 RECOMMENDED ACTION: {action}"
        )


    st.divider()


    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    st.subheader("Transaction Details")


    c1, c2, c3, c4 = st.columns(4)


    c1.write("**Transaction**")
    c1.write(row["transaction_id"])


    c2.write("**Amount**")
    c2.write(
        money(row["amount"])
    )


    c3.write("**Customer**")
    c3.write(row["customer_id"])


    c4.write("**Merchant**")
    c4.write(row["merchant_id"])


    st.divider()


    # --------------------------------------------------------
    # MODEL EVIDENCE
    # --------------------------------------------------------

    st.subheader("Model Evidence")


    evidence = []


    if row["fraud_probability"] >= 0.70:

        evidence.append(
            "High fraud-model signal"
        )


    if row["anomaly_score"] >= 80:

        evidence.append(
            "Highly unusual behavioral pattern"
        )


    if "amount_deviation" in row.index:

        if row["amount_deviation"] >= 4:

            evidence.append(
                "Transaction amount significantly "
                "exceeds normal behavior"
            )


    if "transactions_last_10min" in row.index:

        if row["transactions_last_10min"] >= 4:

            evidence.append(
                "Elevated transaction velocity"
            )


    if "failed_attempts" in row.index:

        if row["failed_attempts"] >= 3:

            evidence.append(
                "Multiple recent failed attempts"
            )


    if "device_age_days" in row.index:

        if row["device_age_days"] < 14:

            evidence.append(
                "Recently observed device"
            )


    if "location_change" in row.index:

        if row["location_change"] == 1:

            evidence.append(
                "Unusual transaction location"
            )


    if len(evidence) == 0:

        evidence.append(
            "No major risk indicators detected"
        )


    for item in evidence:

        st.write(
            f"• {item}"
        )


    st.divider()


    # --------------------------------------------------------
    # ENTITY CONTEXT
    # --------------------------------------------------------

    st.subheader("Entity Context")


    ec1, ec2, ec3 = st.columns(3)


    ec1.metric(
        "Graph Risk Context",
        f"{row.get('graph_risk_score', 0):.1f}"
    )


    ec2.metric(
        "Device",
        str(row.get("device_id", "N/A"))
    )


    ec3.metric(
        "IP",
        str(row.get("ip_id", "N/A"))
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📊 Model Performance")

    st.info(
        "Evaluation uses a temporal holdout: "
        "historical transactions were used for training "
        "and later transactions were used for testing."
    )


    st.subheader(
        "Supervised Fraud Model"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Precision",
        "85.11%"
    )

    c2.metric(
        "Recall",
        "98.36%"
    )

    c3.metric(
        "F1 Score",
        "91.25%"
    )

    c4.metric(
        "ROC-AUC",
        "99.85%"
    )


    st.divider()


    st.subheader(
        "Behavioral Anomaly Engine"
    )


    st.metric(
        "ROC-AUC",
        "99.12%"
    )


    st.divider()


    st.subheader(
        "Graph Signal Evaluation"
    )


    st.metric(
        "Graph ROC-AUC",
        "0.3673"
    )


    st.warning(
        """
        The graph signal was deliberately excluded from
        predictive risk scoring because it did not provide
        reliable fraud discrimination on the temporal
        holdout. It is retained as investigation context.
        """
    )


    st.divider()


    st.subheader(
        "Final Risk Policy"
    )


    policy = pd.DataFrame(
        {
            "Risk Score": [
                "0–59",
                "60–74",
                "75–100"
            ],
            "Decision": [
                "APPROVE",
                "VERIFY",
                "REVIEW"
            ]
        }
    )


    st.table(policy)


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.header("💰 Business Impact")

    st.caption(
        "Prototype economics based on explicit modelling assumptions."
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "False Positives",
        "68"
    )

    c2.metric(
        "False Negatives",
        "5"
    )

    c3.metric(
        "FP Estimated Cost",
        "₹4,837.53"
    )

    c4.metric(
        "Missed Fraud Value",
        "₹4,292.70"
    )


    st.divider()


    st.subheader(
        "Decision Policy"
    )


    st.markdown(
        """
        **APPROVE — Risk < 60**

        Low-risk transactions proceed normally.

        **VERIFY — Risk 60–74**

        Additional verification is requested when risk is
        meaningful but not sufficient for manual review.

        **REVIEW — Risk ≥ 75**

        High-risk transactions are escalated for investigation.
        """
    )


    st.divider()


    st.subheader(
        "Cost Assumptions")


    st.write(
        "The prototype uses configurable assumptions "
        "because proprietary merchant cost data is unavailable."
    )


    assumptions = pd.DataFrame(
        {
            "Parameter": [
                "Verification cost",
                "Review cost",
                "False-positive cost rate",
                "Verification fraud reduction",
                "Review fraud reduction"
            ],
            "Prototype value": [
                "₹25",
                "₹75",
                "1%",
                "80%",
                "95%"
            ]
        }
    )


    st.table(
        assumptions
    )


# ============================================================
# ENTITY CONTEXT
# ============================================================

elif page == "Entity Context":

    st.header("🕸️ Entity Investigation")

    st.info(
        """
        RiskGraph's entity layer is an investigation and
        relationship-context system. It is not used as a
        predictive fraud score.
        """
    )


    entity_cols = [
        "transaction_id",
        "customer_id",
        "device_id",
        "ip_id",
        "graph_risk_score"
    ]


    available = [
        col for col in entity_cols
        if col in df.columns
    ]


    st.dataframe(
        df[available]
        .sort_values(
            "graph_risk_score",
            ascending=False
        )
        .head(50),
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "Why the graph is contextual"
    )


    st.write(
        """
        The graph-derived signal was evaluated independently
        on the future holdout. Its ROC-AUC was 0.3673, so it
        was intentionally excluded from the predictive risk
        score rather than forcing a weak signal into the model.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RiskGraph AI • AI Risk Manager • Razorpay Buildathon Track 02"
)

st.caption(
    "Synthetic-data prototype • Temporal holdout evaluation"
)
