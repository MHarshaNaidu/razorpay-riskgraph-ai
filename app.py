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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f6f8fb;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */

.brand {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
}

.brand span {
    color: #635bff;
}

.subtitle {
    color: #667085;
    font-size: 14px;
    margin-top: -8px;
}

/* Status */

.status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 20px;
    background: #ecfdf3;
    color: #027a48;
    font-size: 12px;
    font-weight: 700;
}

.dot {
    width: 7px;
    height: 7px;
    background: #12b76a;
    border-radius: 50%;
}

/* Metric cards */

.metric-card {
    background: white;
    border: 1px solid #eaecf0;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
    box-shadow: 0 1px 2px rgba(16,24,40,.03);
}

.metric-label {
    color: #667085;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .04em;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
    color: #101828;
}

.metric-note {
    color: #98a2b3;
    font-size: 11px;
    margin-top: 4px;
}

/* Section */

.section-title {
    font-size: 18px;
    font-weight: 750;
    color: #101828;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #667085;
    font-size: 12px;
    margin-bottom: 15px;
}

/* Risk cards */

.risk-card {
    background: white;
    border: 1px solid #eaecf0;
    border-radius: 14px;
    padding: 18px;
    height: 100%;
}

.risk-number {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -2px;
}

.risk-label {
    font-size: 13px;
    font-weight: 600;
}

.risk-desc {
    color: #667085;
    font-size: 11px;
    margin-top: 6px;
}

/* Investigation */

.investigation {
    background: white;
    border: 1px solid #eaecf0;
    border-radius: 16px;
    padding: 24px;
}

.score-ring {
    text-align: center;
    padding: 12px;
}

.score-value {
    font-size: 54px;
    font-weight: 800;
    letter-spacing: -3px;
}

.score-label {
    color: #667085;
    font-size: 12px;
    font-weight: 600;
}

/* Action */

.action-review {
    background: #fef3f2;
    color: #b42318;
    border: 1px solid #fecdca;
    border-radius: 10px;
    padding: 12px 16px;
    font-weight: 800;
    text-align: center;
}

.action-verify {
    background: #fffaeb;
    color: #b54708;
    border: 1px solid #fedf89;
    border-radius: 10px;
    padding: 12px 16px;
    font-weight: 800;
    text-align: center;
}

.action-approve {
    background: #ecfdf3;
    color: #027a48;
    border: 1px solid #abefc6;
    border-radius: 10px;
    padding: 12px 16px;
    font-weight: 800;
    text-align: center;
}

/* Evidence */

.evidence {
    background: #f9fafb;
    border-radius: 10px;
    padding: 12px 15px;
    margin-bottom: 8px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "sample_transactions.csv"
    )


df = load_data()


# ============================================================
# HELPERS
# ============================================================

def money(value):
    return f"₹{value:,.2f}"


def action_html(action):

    if action == "REVIEW":
        return '<div class="action-review">● REVIEW</div>'

    if action == "VERIFY":
        return '<div class="action-verify">● VERIFY</div>'

    return '<div class="action-approve">● APPROVE</div>'


def risk_level(score):

    if score >= 75:
        return "HIGH"

    if score >= 60:
        return "MEDIUM"

    return "LOW"


def evidence_for(row):

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
                "Transaction amount significantly exceeds "
                "normal customer behavior"
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

    if not evidence:
        evidence.append(
            "No major risk indicators detected"
        )

    return evidence


# ============================================================
# HEADER
# ============================================================

h1, h2 = st.columns([6, 2])

with h1:

    st.markdown(
        '<div class="brand">RiskGraph <span>AI</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Cost-aware payment risk intelligence'
        '</div>',
        unsafe_allow_html=True
    )

with h2:

    st.markdown(
        '<div style="text-align:right;margin-top:8px;">'
        '<span class="status">'
        '<span class="dot"></span>'
        'RISK ENGINE ONLINE'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "",
    [
        "Command Center",
        "Investigate",
        "Performance",
        "Business Impact",
        "Entity Context"
    ],
    horizontal=True
)


st.divider()


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        '<div class="section-title">'
        'Risk Command Center'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Real-time prioritization of suspicious payment activity'
        '</div>',
        unsafe_allow_html=True
    )


    total = len(df)

    fraud_cases = int(
        df["is_fraud"].sum()
    )

    approve = int(
        (df["final_action"] == "APPROVE").sum()
    )

    verify = int(
        (df["final_action"] == "VERIFY").sum()
    )

    review = int(
        (df["final_action"] == "REVIEW").sum()
    )


    cards = st.columns(4)


    with cards[0]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Transactions</div>
                <div class="metric-value">{total:,}</div>
                <div class="metric-note">Future holdout evaluated</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with cards[1]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Fraud Cases</div>
                <div class="metric-value">{fraud_cases:,}</div>
                <div class="metric-note">Actual fraud labels</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with cards[2]:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Detection Recall</div>
                <div class="metric-value">97.95%</div>
                <div class="metric-note">Final intervention policy</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with cards[3]:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">FP Cost</div>
                <div class="metric-value">₹4,837</div>
                <div class="metric-note">Prototype estimate</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # --------------------------------------------------------
    # RISK SUMMARY
    # --------------------------------------------------------

    left, right = st.columns([1.4, 1])


    with left:

        st.markdown(
            '<div class="section-title">Risk distribution</div>',
            unsafe_allow_html=True
        )

        chart = pd.DataFrame(
            {
                "Decision": [
                    "APPROVE",
                    "VERIFY",
                    "REVIEW"
                ],
                "Transactions": [
                    approve,
                    verify,
                    review
                ]
            }
        )

        st.bar_chart(
            chart.set_index("Decision"),
            height=280
        )


    with right:

        st.markdown(
            '<div class="section-title">Decision mix</div>',
            unsafe_allow_html=True
        )

        st.write("")

        for label, count in [
            ("APPROVE", approve),
            ("VERIFY", verify),
            ("REVIEW", review)
        ]:

            percentage = (
                count / total * 100
            )

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    padding:12px 0;
                    border-bottom:1px solid #eaecf0;
                ">
                    <span style="font-weight:600">{label}</span>
                    <span style="color:#667085">
                    {count:,} · {percentage:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


    st.write("")


    # --------------------------------------------------------
    # HIGH RISK QUEUE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">High-risk queue</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Transactions requiring immediate analyst attention'
        '</div>',
        unsafe_allow_html=True
    )


    high = (
        df[
            df["final_action"] == "REVIEW"
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(8)
        .copy()
    )


    display = high[
        [
            "transaction_id",
            "amount",
            "risk_score",
            "fraud_probability",
            "anomaly_score",
            "final_action"
        ]
    ].copy()


    display["amount"] = (
        display["amount"]
        .map(money)
    )

    display["risk_score"] = (
        display["risk_score"]
        .round(1)
    )

    display["fraud_probability"] = (
        display["fraud_probability"]
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )

    display["anomaly_score"] = (
        display["anomaly_score"]
        .round(1)
    )


    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INVESTIGATE
# ============================================================

elif page == "Investigate":

    st.markdown(
        '<div class="section-title">Transaction Investigation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Inspect model evidence and entity context for a payment'
        '</div>',
        unsafe_allow_html=True
    )


    ids = df[
        "transaction_id"
    ].astype(str).tolist()


    selected = st.selectbox(
        "Transaction",
        ids
    )


    row = df[
        df["transaction_id"].astype(str)
        == selected
    ].iloc[0]


    st.write("")


    # --------------------------------------------------------
    # TOP RISK CARD
    # --------------------------------------------------------

    a, b = st.columns([2, 1])


    with a:

        st.markdown(
            f"""
            <div class="investigation">

            <div style="
                color:#667085;
                font-size:12px;
                font-weight:600;
            ">
            TRANSACTION
            </div>

            <div style="
                font-size:24px;
                font-weight:800;
                margin-top:5px;
            ">
            {row["transaction_id"]}
            </div>

            <div style="
                color:#667085;
                margin-top:4px;
            ">
            {row["customer_id"]} · {row["merchant_id"]}
            </div>

            <hr style="border:none;border-top:1px solid #eaecf0;">

            <div style="font-size:28px;font-weight:800;">
            {money(row["amount"])}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with b:

        action = row["final_action"]

        st.markdown(
            f"""
            <div class="investigation">
                <div class="score-ring">
                    <div class="score-value">
                        {row["risk_score"]:.0f}
                    </div>
                    <div class="score-label">
                        RISK SCORE / 100
                    </div>
                    <br>
                    {action_html(action)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    s1, s2, s3 = st.columns(3)


    with s1:

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="metric-label">
                    Fraud Signal
                </div>
                <div class="risk-number">
                    {row["fraud_probability"] * 100:.1f}
                </div>
                <div class="risk-label">
                    / 100
                </div>
                <div class="risk-desc">
                    Supervised fraud model
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with s2:

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="metric-label">
                    Anomaly Signal
                </div>
                <div class="risk-number">
                    {row["anomaly_score"]:.1f}
                </div>
                <div class="risk-label">
                    / 100
                </div>
                <div class="risk-desc">
                    Behavioral anomaly engine
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with s3:

        graph = row.get(
            "graph_risk_score",
            0
        )

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="metric-label">
                    Graph Context
                </div>
                <div class="risk-number">
                    {graph:.1f}
                </div>
                <div class="risk-label">
                    / 100
                </div>
                <div class="risk-desc">
                    Investigation context only
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    left, right = st.columns([1.2, 1])


    with left:

        st.markdown(
            '<div class="section-title">Why was this flagged?</div>',
            unsafe_allow_html=True
        )

        st.write("")

        for item in evidence_for(row):

            st.markdown(
                f"""
                <div class="evidence">
                    ● &nbsp; {item}
                </div>
                """,
                unsafe_allow_html=True
            )


    with right:

        st.markdown(
            '<div class="section-title">Entity context</div>',
            unsafe_allow_html=True
        )

        st.write("")

        entity_data = {
            "Customer": row.get(
                "customer_id", "N/A"
            ),
            "Merchant": row.get(
                "merchant_id", "N/A"
            ),
            "Device": row.get(
                "device_id", "N/A"
            ),
            "IP": row.get(
                "ip_id", "N/A"
            )
        }


        for key, value in entity_data.items():

            st.markdown(
                f"""
                <div class="evidence">
                    <b>{key}</b><br>
                    {value}
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Temporal holdout evaluation — later transactions were '
        'held out from model development'
        '</div>',
        unsafe_allow_html=True
    )


    st.write("")


    st.subheader("Fraud Detection Model")


    p1, p2, p3, p4 = st.columns(4)


    p1.metric(
        "Precision",
        "85.11%"
    )

    p2.metric(
        "Recall",
        "98.36%"
    )

    p3.metric(
        "F1",
        "91.25%"
    )

    p4.metric(
        "ROC-AUC",
        "99.85%"
    )


    st.write("")


    st.subheader("Anomaly Engine")


    st.metric(
        "Anomaly ROC-AUC",
        "99.12%"
    )


    st.write("")


    st.subheader("Graph Evaluation")


    st.metric(
        "Graph ROC-AUC",
        "0.3673"
    )


    st.warning(
        """
        Graph-derived features were deliberately excluded from
        predictive risk scoring because their standalone holdout
        ROC-AUC was 0.3673. The graph is retained as an
        investigation/context layer.
        """
    )


    st.write("")


    st.subheader("Final Decision Policy")


    policy = pd.DataFrame(
        {
            "Risk score": [
                "0–59",
                "60–74",
                "75–100"
            ],
            "Decision": [
                "APPROVE",
                "VERIFY",
                "REVIEW"
            ],
            "Purpose": [
                "Low customer friction",
                "Additional verification",
                "Analyst investigation"
            ]
        }
    )


    st.dataframe(
        policy,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.markdown(
        '<div class="section-title">Business Impact</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Merchant-loss and customer-friction analysis'
        '</div>',
        unsafe_allow_html=True
    )


    b1, b2, b3, b4 = st.columns(4)


    b1.metric(
        "False Positives",
        "68"
    )

    b2.metric(
        "False Negatives",
        "5"
    )

    b3.metric(
        "FP Cost",
        "₹4,837.53"
    )

    b4.metric(
        "Missed Fraud Value",
        "₹4,292.70"
    )


    st.write("")


    st.subheader(
        "Risk policy"
    )


    policy = pd.DataFrame(
        {
            "Range": [
                "0–59",
                "60–74",
                "75–100"
            ],
            "Action": [
                "APPROVE",
                "VERIFY",
                "REVIEW"
            ],
            "Intent": [
                "Minimize friction",
                "Challenge uncertain risk",
                "Escalate high-risk payments"
            ]
        }
    )


    st.dataframe(
        policy,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        """
        Prototype economics use explicit configurable assumptions.
        They are not claimed to represent proprietary Razorpay
        operating costs.
        """
    )


# ============================================================
# ENTITY CONTEXT
# ============================================================

elif page == "Entity Context":

    st.markdown(
        '<div class="section-title">Entity Investigation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Customer, device, IP and merchant relationship context'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        """
        The entity layer is intentionally investigative rather
        than predictive. Its measured ROC-AUC was 0.3673, so it
        is not included in the final risk score.
        """
    )


    columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "graph_risk_score",
        "risk_score",
        "final_action"
    ]


    available = [
        c for c in columns
        if c in df.columns
    ]


    entity = (
        df[available]
        .sort_values(
            "graph_risk_score",
            ascending=False
        )
        .head(50)
    )


    st.dataframe(
        entity,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#98a2b3;
        font-size:11px;
    ">
    RiskGraph AI · AI Risk Manager · Razorpay Buildathon Track 02
    <br>
    Synthetic-data prototype · Temporal holdout evaluation
    </div>
    """,
    unsafe_allow_html=True
)
