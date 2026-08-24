import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# RISKGRAPH AI
# AI Risk Manager — Razorpay Buildathon Track 02
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DATA
# ============================================================

DATA_FILE = Path("sample_transactions.csv")


@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    # Normalize timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    # Make sure numerical fields are numeric
    numeric_columns = [
        "amount",
        "fraud_probability",
        "anomaly_score",
        "graph_risk_score",
        "risk_score",
        "amount_deviation",
        "transactions_last_10min",
        "failed_attempts",
        "device_age_days",
        "account_age_days",
        "location_change",
        "is_fraud"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


df = load_data()


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

FINAL_PRECISION = 0.7785
FINAL_RECALL = 0.9795
FINAL_F1 = 0.8675

MODEL_PRECISION = 0.8511
MODEL_RECALL = 0.9836
MODEL_F1 = 0.9125
MODEL_AUC = 0.9985

ANOMALY_AUC = 0.9912
GRAPH_AUC = 0.3673

FALSE_POSITIVES = 68
FALSE_NEGATIVES = 5

FP_VALUE = 483752.82
MISSED_FRAUD_VALUE = 4292.70
FP_COST = 4837.53

BASELINE_LOSS = 2060929.95
RESIDUAL_LOSS = 111460.20
LOSS_AVOIDED = 1949469.75


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):

    if pd.isna(value):
        return "₹0"

    return f"₹{float(value):,.0f}"


def money_decimal(value):

    if pd.isna(value):
        return "₹0.00"

    return f"₹{float(value):,.2f}"


def percent(value):

    return f"{float(value) * 100:.2f}%"


def score(value):

    if pd.isna(value):
        return "0"

    return f"{float(value):.0f}"


def safe(row, column, default="—"):

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def risk_band(value):

    value = float(value)

    if value >= 75:
        return "HIGH"

    if value >= 60:
        return "MEDIUM"

    return "LOW"


def action_class(action):

    if action == "REVIEW":
        return "review"

    if action == "VERIFY":
        return "verify"

    return "approve"


def build_evidence(row):

    evidence = []

    fraud_prob = float(
        safe(row, "fraud_probability", 0)
    )

    anomaly = float(
        safe(row, "anomaly_score", 0)
    )

    velocity = float(
        safe(row, "transactions_last_10min", 0)
    )

    failures = float(
        safe(row, "failed_attempts", 0)
    )

    amount_deviation = float(
        safe(row, "amount_deviation", 0)
    )

    device_age = float(
        safe(row, "device_age_days", 9999)
    )

    location_change = float(
        safe(row, "location_change", 0)
    )

    if fraud_prob >= 0.70:

        evidence.append(
            (
                "HIGH",
                "Fraud model",
                "Strong supervised fraud signal"
            )
        )

    if anomaly >= 80:

        evidence.append(
            (
                "HIGH",
                "Anomaly engine",
                "Highly unusual behavioral pattern"
            )
        )

    if velocity >= 4:

        evidence.append(
            (
                "MEDIUM",
                "Velocity",
                f"{velocity:.0f} transactions detected in 10 minutes"
            )
        )

    if failures >= 3:

        evidence.append(
            (
                "MEDIUM",
                "Authentication",
                f"{failures:.0f} recent failed attempts"
            )
        )

    if amount_deviation >= 4:

        evidence.append(
            (
                "HIGH",
                "Amount deviation",
                f"Transaction is {amount_deviation:.1f}× normal amount"
            )
        )

    if device_age < 14:

        evidence.append(
            (
                "MEDIUM",
                "Device",
                f"Device observed only {device_age:.0f} days ago"
            )
        )

    if location_change == 1:

        evidence.append(
            (
                "MEDIUM",
                "Location",
                "Transaction location differs from normal behavior"
            )
        )

    if not evidence:

        evidence.append(
            (
                "LOW",
                "Baseline",
                "No major risk indicators detected"
            )
        )

    return evidence


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   BASE
========================================================= */

:root {
    --bg: #f5f7fb;
    --surface: #ffffff;
    --border: #e7eaf0;
    --text: #101828;
    --muted: #667085;
    --subtle: #98a2b3;

    --purple: #635bff;
    --purple-soft: #efedff;

    --green: #12b76a;
    --green-soft: #ecfdf3;

    --yellow: #f79009;
    --yellow-soft: #fffaeb;

    --red: #f04438;
    --red-soft: #fef3f2;

    --blue: #1570ef;
    --blue-soft: #eff8ff;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: #d0d5dd;
}

.sidebar-brand {
    padding: 8px 8px 28px 8px;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: linear-gradient(
        135deg,
        #635bff,
        #8b5cf6
    );
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 18px;
}

.logo-title {
    color: white;
    font-size: 18px;
    font-weight: 800;
}

.logo-title span {
    color: #8b83ff;
}

.sidebar-caption {
    font-size: 11px;
    color: #667085 !important;
    margin-top: 5px;
}

.nav-title {
    color: #667085 !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 20px 8px 8px 8px;
}

.sidebar-status {
    margin-top: 25px;
    padding: 13px;
    border: 1px solid #27303d;
    border-radius: 12px;
    background: #151a21;
}

.sidebar-status-title {
    color: #98a2b3 !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.sidebar-status-value {
    color: #d1fadf !important;
    font-size: 12px;
    font-weight: 700;
    margin-top: 5px;
}

/* =========================================================
   TOP BAR
========================================================= */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
}

.top-title {
    font-size: 30px;
    font-weight: 850;
    letter-spacing: -1.3px;
    color: #101828;
}

.top-title span {
    color: var(--purple);
}

.top-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-top: 3px;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 999px;
    background: var(--green-soft);
    border: 1px solid #abefc6;
    color: #027a48;
    font-size: 11px;
    font-weight: 800;
}

.live-dot {
    width: 7px;
    height: 7px;
    background: var(--green);
    border-radius: 50%;
}

/* =========================================================
   PAGE LABEL
========================================================= */

.page-heading {
    margin-bottom: 18px;
}

.page-heading-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.3px;
}

.page-heading-subtitle {
    font-size: 12px;
    color: var(--muted);
    margin-top: 3px;
}

/* =========================================================
   KPI CARDS
========================================================= */

.kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px 19px;
    min-height: 126px;
    box-shadow: 0 1px 2px rgba(16,24,40,.025);
}

.kpi-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.kpi-value {
    color: var(--text);
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -1px;
    margin-top: 12px;
}

.kpi-note {
    color: var(--subtle);
    font-size: 10px;
    margin-top: 5px;
}

/* =========================================================
   PANELS
========================================================= */

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 1px 2px rgba(16,24,40,.025);
}

.panel-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 800;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 11px;
    margin-top: 4px;
    margin-bottom: 18px;
}

/* =========================================================
   DECISION BARS
========================================================= */

.decision-row {
    margin: 13px 0 18px 0;
}

.decision-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
}

.decision-name {
    font-size: 11px;
    font-weight: 800;
}

.decision-value {
    color: var(--muted);
    font-size: 11px;
}

.bar-track {
    height: 8px;
    background: #eef0f4;
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.approve-fill {
    background: #12b76a;
}

.verify-fill {
    background: #f79009;
}

.review-fill {
    background: #f04438;
}

/* =========================================================
   RISK DISTRIBUTION
========================================================= */

.histogram {
    display: flex;
    align-items: flex-end;
    gap: 7px;
    height: 180px;
    padding: 8px 2px 0 2px;
}

.hist-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    height: 100%;
}

.hist-bar {
    width: 100%;
    min-height: 5px;
    border-radius: 5px 5px 2px 2px;
    background: linear-gradient(
        180deg,
        #756df5,
        #635bff
    );
}

.hist-label {
    color: var(--subtle);
    font-size: 8px;
    text-align: center;
    margin-top: 7px;
}

/* =========================================================
   QUEUE TABLE
========================================================= */

.queue {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
}

.queue th {
    color: #667085;
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .07em;
    text-align: left;
    padding: 11px 12px;
    border-bottom: 1px solid var(--border);
}

.queue td {
    padding: 13px 12px;
    border-bottom: 1px solid #f0f2f5;
    color: #344054;
    font-size: 11px;
}

.queue tr:last-child td {
    border-bottom: none;
}

.tx-id {
    color: #101828;
    font-weight: 800;
}

.risk-high {
    color: #d92d20;
    font-weight: 800;
}

.risk-medium {
    color: #b54708;
    font-weight: 800;
}

.risk-low {
    color: #027a48;
    font-weight: 800;
}

.action-pill {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 9px;
    font-weight: 800;
}

.action-pill-review {
    background: var(--red-soft);
    color: #b42318;
}

.action-pill-verify {
    background: var(--yellow-soft);
    color: #b54708;
}

.action-pill-approve {
    background: var(--green-soft);
    color: #027a48;
}

/* =========================================================
   INVESTIGATION HERO
========================================================= */

.investigation-hero {
    background: #111827;
    border-radius: 18px;
    padding: 25px;
    color: white;
}

.investigation-label {
    color: #98a2b3;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
}

.investigation-id {
    color: white;
    font-size: 24px;
    font-weight: 850;
    margin-top: 6px;
}

.investigation-amount {
    color: white;
    font-size: 36px;
    font-weight: 850;
    letter-spacing: -1px;
    margin-top: 25px;
}

.investigation-meta {
    color: #98a2b3;
    font-size: 11px;
    margin-top: 5px;
}

.dark-score {
    text-align: center;
    padding: 10px;
}

.dark-score-number {
    color: white;
    font-size: 62px;
    line-height: 1;
    font-weight: 900;
}

.dark-score-label {
    color: #98a2b3;
    font-size: 9px;
    letter-spacing: .1em;
    font-weight: 800;
    margin-top: 8px;
}

/* =========================================================
   SIGNAL CARDS
========================================================= */

.signal {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
}

.signal-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .07em;
}

.signal-score {
    font-size: 32px;
    font-weight: 850;
    letter-spacing: -1px;
    margin-top: 10px;
}

.signal-desc {
    color: var(--muted);
    font-size: 10px;
    margin-top: 5px;
}

.signal-progress {
    margin-top: 14px;
    height: 5px;
    background: #eef0f4;
    border-radius: 999px;
    overflow: hidden;
}

.signal-progress-inner {
    height: 100%;
    background: #635bff;
    border-radius: 999px;
}

/* =========================================================
   EVIDENCE
========================================================= */

.evidence-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 9px;
    background: white;
}

.evidence-top {
    display: flex;
    justify-content: space-between;
}

.evidence-title {
    color: #101828;
    font-size: 11px;
    font-weight: 800;
}

.evidence-text {
    color: var(--muted);
    font-size: 10px;
    margin-top: 5px;
}

.evidence-high {
    color: #b42318;
    background: #fef3f2;
}

.evidence-medium {
    color: #b54708;
    background: #fffaeb;
}

.evidence-low {
    color: #027a48;
    background: #ecfdf3;
}

.severity {
    font-size: 8px;
    font-weight: 900;
    padding: 4px 7px;
    border-radius: 999px;
}

/* =========================================================
   ENTITY GRAPH
========================================================= */

.entity-node {
    border: 1px solid var(--border);
    background: #fafbfc;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}

.entity-icon {
    font-size: 19px;
}

.entity-type {
    color: var(--subtle);
    font-size: 8px;
    text-transform: uppercase;
    font-weight: 800;
    margin-top: 7px;
}

.entity-value {
    color: #101828;
    font-size: 10px;
    font-weight: 800;
    margin-top: 4px;
}

/* =========================================================
   METRIC LARGE
========================================================= */

.big-metric {
    font-size: 38px;
    font-weight: 900;
    letter-spacing: -1.5px;
}

.metric-caption {
    color: var(--muted);
    font-size: 10px;
    margin-top: 4px;
}

/* =========================================================
   FOOTER
========================================================= */

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 9px;
    padding: 30px 0 10px 0;
}

/* =========================================================
   STREAMLIT WIDGET CLEANUP
========================================================= */

div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stSelectbox"] label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #475467 !important;
}

div[data-testid="stMetric"] {
    background: transparent;
}

button[kind="secondary"] {
    border-radius: 9px;
}

/* Hide deploy menu branding elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">

                <div class="logo-mark">
                    ◈
                </div>

                <div class="logo-title">
                    RiskGraph <span>AI</span>
                </div>

            </div>

            <div class="sidebar-caption">
                Payment risk intelligence
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nav-title">Risk Operations</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Investigate",
            "Model Intelligence",
            "Business Impact",
            "Entity Network"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="sidebar-status">

            <div class="sidebar-status-title">
                Risk Engine
            </div>

            <div class="sidebar-status-value">
                ● Operational
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">

        <div>

            <div class="top-title">
                RiskGraph <span>AI</span>
            </div>

            <div class="top-subtitle">
                Cost-aware payment risk intelligence
            </div>

        </div>

        <div>
            <span class="live-pill">
                <span class="live-dot"></span>
                RISK ENGINE ONLINE
            </span>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        """
        <div class="page-heading">

            <div class="page-heading-title">
                Command Center
            </div>

            <div class="page-heading-subtitle">
                Detect suspicious payments, prioritize intervention,
                and control merchant loss.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total = len(df)

    fraud_count = int(
        df["is_fraud"].sum()
    ) if "is_fraud" in df.columns else 244

    review_count = int(
        (df["final_action"] == "REVIEW").sum()
    ) if "final_action" in df.columns else 266

    verify_count = int(
        (df["final_action"] == "VERIFY").sum()
    ) if "final_action" in df.columns else 41

    approve_count = int(
        (df["final_action"] == "APPROVE").sum()
    ) if "final_action" in df.columns else 3693

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.markdown(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Transactions
                </div>

                <div class="kpi-value">
                    {total:,}
                </div>

                <div class="kpi-note">
                    Temporal future holdout
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:

        st.markdown(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Fraud cases
                </div>

                <div class="kpi-value">
                    {fraud_count:,}
                </div>

                <div class="kpi-note">
                    Actual fraud labels
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:

        st.markdown(
            """
            <div class="kpi">

                <div class="kpi-label">
                    Detection recall
                </div>

                <div class="kpi-value">
                    97.95%
                </div>

                <div class="kpi-note">
                    Final intervention policy
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:

        st.markdown(
            """
            <div class="kpi">

                <div class="kpi-label">
                    Estimated FP cost
                </div>

                <div class="kpi-value">
                    ₹4,838
                </div>

                <div class="kpi-note">
                    Prototype economics
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # RISK OVERVIEW
    # --------------------------------------------------------

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    Risk landscape
                </div>

                <div class="panel-subtitle">
                    Distribution of final risk scores across
                    the evaluated payment population
                </div>
            """,
            unsafe_allow_html=True
        )

        if "risk_score" in df.columns:

            risk_values = (
                df["risk_score"]
                .fillna(0)
                .clip(0, 100)
            )

            bins = [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100.1
            ]

            counts, _ = np.histogram(
                risk_values,
                bins=bins
            )

            max_count = max(
                counts.max(),
                1
            )

            html = '<div class="histogram">'

            for i, count in enumerate(counts):

                height = max(
                    5,
                    int(
                        (count / max_count) * 155
                    )
                )

                label = (
                    f"{int(bins[i])}"
                )

                html += f"""
                <div class="hist-column">

                    <div
                        class="hist-bar"
                        style="height:{height}px;"
                    ></div>

                    <div class="hist-label">
                        {label}
                    </div>

                </div>
                """

            html += "</div>"

            st.markdown(
                html,
                unsafe_allow_html=True
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    Decision engine
                </div>

                <div class="panel-subtitle">
                    Final intervention policy
                </div>
            """,
            unsafe_allow_html=True
        )

        total_safe = max(total, 1)

        decisions = [
            (
                "APPROVE",
                approve_count,
                "approve-fill"
            ),
            (
                "VERIFY",
                verify_count,
                "verify-fill"
            ),
            (
                "REVIEW",
                review_count,
                "review-fill"
            )
        ]

        for name, count, css in decisions:

            p = count / total_safe * 100

            st.markdown(
                f"""
                <div class="decision-row">

                    <div class="decision-header">

                        <span class="decision-name">
                            {name}
                        </span>

                        <span class="decision-value">
                            {count:,} · {p:.2f}%
                        </span>

                    </div>

                    <div class="bar-track">

                        <div
                            class="bar-fill {css}"
                            style="width:{min(p,100):.2f}%"
                        ></div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="
                margin-top:18px;
                padding:11px;
                background:#f9fafb;
                border-radius:10px;
                color:#667085;
                font-size:10px;
            ">
                <b style="color:#344054;">
                Policy logic
                </b><br>
                0–59 → APPROVE &nbsp; · &nbsp;
                60–74 → VERIFY &nbsp; · &nbsp;
                75–100 → REVIEW
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # HIGH RISK QUEUE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="panel">

            <div class="panel-title">
                High-risk queue
            </div>

            <div class="panel-subtitle">
                Priority transactions requiring analyst attention
            </div>
        """,
        unsafe_allow_html=True
    )

    if "risk_score" in df.columns:

        queue = (
            df[
                df.get(
                    "final_action",
                    pd.Series(
                        ["APPROVE"] * len(df)
                    )
                ) == "REVIEW"
            ]
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(10)
        )

    else:

        queue = df.head(10)

    rows_html = ""

    for _, row in queue.iterrows():

        tx = safe(
            row,
            "transaction_id"
        )

        amount = money_decimal(
            safe(row, "amount", 0)
        )

        risk = float(
            safe(row, "risk_score", 0)
        )

        fraud = float(
            safe(row, "fraud_probability", 0)
        ) * 100

        anomaly = float(
            safe(row, "anomaly_score", 0)
        )

        action = safe(
            row,
            "final_action",
            "REVIEW"
        )

        band = risk_band(risk)

        if band == "HIGH":
            risk_css = "risk-high"
        elif band == "MEDIUM":
            risk_css = "risk-medium"
        else:
            risk_css = "risk-low"

        action_css = (
            "action-pill-review"
            if action == "REVIEW"
            else
            "action-pill-verify"
            if action == "VERIFY"
            else
            "action-pill-approve"
        )

        rows_html += f"""
        <tr>

            <td>
                <span class="tx-id">
                    {tx}
                </span>
            </td>

            <td>
                {amount}
            </td>

            <td>
                <span class="{risk_css}">
                    {risk:.0f}
                </span>
            </td>

            <td>
                {fraud:.1f}%
            </td>

            <td>
                {anomaly:.1f}
            </td>

            <td>
                <span class="action-pill {action_css}">
                    {action}
                </span>
            </td>

        </tr>
        """

    st.markdown(
        f"""
        <table class="queue">

            <thead>

                <tr>
                    <th>Transaction</th>
                    <th>Amount</th>
                    <th>Risk</th>
                    <th>Fraud signal</th>
                    <th>Anomaly</th>
                    <th>Action</th>
                </tr>

            </thead>

            <tbody>

                {rows_html}

            </tbody>

        </table>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            RiskGraph AI · AI Risk Manager · Razorpay Buildathon Track 02
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INVESTIGATION
# ============================================================

elif page == "Investigate":

    st.markdown(
        """
        <div class="page-heading">

            <div class="page-heading-title">
                Transaction Investigation
            </div>

            <div class="page-heading-subtitle">
                Explain the risk before deciding what to do.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SELECT TRANSACTION
    # --------------------------------------------------------

    if "transaction_id" in df.columns:

        transaction_ids = (
            df["transaction_id"]
            .astype(str)
            .tolist()
        )

        selected_id = st.selectbox(
            "Select transaction to investigate",
            transaction_ids
        )

        selected_row = df[
            df["transaction_id"].astype(str)
            == selected_id
        ]

        if selected_row.empty:
            st.error(
                "Transaction not found."
            )
            st.stop()

        row = selected_row.iloc[0]

    else:

        row = df.iloc[0]

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    action = safe(
        row,
        "final_action",
        "REVIEW"
    )

    risk = float(
        safe(row, "risk_score", 0)
    )

    amount = float(
        safe(row, "amount", 0)
    )

    action_color = (
        "#f04438"
        if action == "REVIEW"
        else "#f79009"
        if action == "VERIFY"
        else "#12b76a"
    )

    c1, c2 = st.columns(
        [1.65, 1]
    )

    with c1:

        st.markdown(
            f"""
            <div class="investigation-hero">

                <div class="investigation-label">
                    Payment under investigation
                </div>

                <div class="investigation-id">
                    {safe(row, "transaction_id")}
                </div>

                <div class="investigation-amount">
                    {money_decimal(amount)}
                </div>

                <div class="investigation-meta">
                    Customer {safe(row, "customer_id")}
                    &nbsp; · &nbsp;
                    Merchant {safe(row, "merchant_id")}
                </div>

                <div class="investigation-meta">
                    Device {safe(row, "device_id")}
                    &nbsp; · &nbsp;
                    IP {safe(row, "ip_id")}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="investigation-hero">

                <div class="dark-score">

                    <div
                        class="dark-score-number"
                        style="color:{action_color};"
                    >
                        {risk:.0f}
                    </div>

                    <div class="dark-score-label">
                        RISK SCORE / 100
                    </div>

                    <div style="
                        margin-top:18px;
                        padding:10px;
                        border-radius:9px;
                        background:rgba(255,255,255,.06);
                        color:{action_color};
                        font-weight:900;
                        font-size:12px;
                    ">
                        ● {action}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # SIGNAL CARDS
    # --------------------------------------------------------

    fraud = float(
        safe(row, "fraud_probability", 0)
    ) * 100

    anomaly = float(
        safe(row, "anomaly_score", 0)
    )

    graph = float(
        safe(row, "graph_risk_score", 0)
    )

    s1, s2, s3 = st.columns(3)

    signal_data = [
        (
            s1,
            "Fraud model",
            fraud,
            "Supervised fraud probability",
            "#635bff"
        ),
        (
            s2,
            "Anomaly engine",
            anomaly,
            "Behavioral deviation signal",
            "#f04438"
        ),
        (
            s3,
            "Graph context",
            graph,
            "Investigation signal only",
            "#1570ef"
        )
    ]

    for col, label, value, desc, color in signal_data:

        with col:

            st.markdown(
                f"""
                <div class="signal">

                    <div class="signal-label">
                        {label}
                    </div>

                    <div class="signal-score">
                        {value:.1f}
                        <span style="
                            font-size:13px;
                            color:#98a2b3;
                            font-weight:600;
                        ">
                            /100
                        </span>
                    </div>

                    <div class="signal-desc">
                        {desc}
                    </div>

                    <div class="signal-progress">

                        <div
                            class="signal-progress-inner"
                            style="
                                width:{min(value,100):.1f}%;
                                background:{color};
                            "
                        ></div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # --------------------------------------------------------
    # EVIDENCE + ENTITY
    # --------------------------------------------------------

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    Why RiskGraph flagged this payment
                </div>

                <div class="panel-subtitle">
                    Evidence generated from independent risk signals
                </div>
            """,
            unsafe_allow_html=True
        )

        for severity, title, text in build_evidence(row):

            css = (
                "evidence-high"
                if severity == "HIGH"
                else
                "evidence-medium"
                if severity == "MEDIUM"
                else
                "evidence-low"
            )

            st.markdown(
                f"""
                <div class="evidence-card">

                    <div class="evidence-top">

                        <div class="evidence-title">
                            {title}
                        </div>

                        <div class="severity {css}">
                            {severity}
                        </div>

                    </div>

                    <div class="evidence-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    Entity context
                </div>

                <div class="panel-subtitle">
                    Payment relationships available to an analyst
                </div>
            """,
            unsafe_allow_html=True
        )

        e1, e2 = st.columns(2)

        entities = [
            (
                e1,
                "Customer",
                safe(row, "customer_id"),
                "C"
            ),
            (
                e2,
                "Merchant",
                safe(row, "merchant_id"),
                "M"
            ),
            (
                e1,
                "Device",
                safe(row, "device_id"),
                "D"
            ),
            (
                e2,
                "IP",
                safe(row, "ip_id"),
                "IP"
            )
        ]

        for col, kind, value, icon in entities:

            with col:

                st.markdown(
                    f"""
                    <div class="entity-node"
                         style="margin-bottom:9px;">

                        <div class="entity-icon">
                            {icon}
                        </div>

                        <div class="entity-type">
                            {kind}
                        </div>

                        <div class="entity-value">
                            {value}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
            """
            <div style="
                margin-top:5px;
                padding:10px;
                background:#eff8ff;
                color:#175cd3;
                border-radius:9px;
                font-size:9px;
            ">
                Graph features are intentionally used as
                investigation context rather than predictive
                fraud evidence.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="panel">

            <div class="panel-title">
                Behavioral profile
            </div>

            <div class="panel-subtitle">
                Raw signals supporting the investigation
            </div>
        """,
        unsafe_allow_html=True
    )

    details = [
        (
            "Amount deviation",
            f"{float(safe(row,'amount_deviation',0)):.2f}×"
        ),
        (
            "Transactions / 10 min",
            f"{float(safe(row,'transactions_last_10min',0)):.0f}"
        ),
        (
            "Failed attempts",
            f"{float(safe(row,'failed_attempts',0)):.0f}"
        ),
        (
            "Device age",
            f"{float(safe(row,'device_age_days',0)):.0f} days"
        ),
        (
            "Account age",
            f"{float(safe(row,'account_age_days',0)):.0f} days"
        ),
        (
            "Location change",
            "YES"
            if float(safe(row,'location_change',0)) == 1
            else "NO"
        )
    ]

    cols = st.columns(6)

    for col, (label, value) in zip(
        cols,
        details
    ):

        with col:

            st.markdown(
                f"""
                <div style="
                    padding:12px;
                    background:#f9fafb;
                    border-radius:10px;
                ">

                    <div style="
                        color:#667085;
                        font-size:8px;
                        text-transform:uppercase;
                        font-weight:800;
                    ">
                        {label}
                    </div>

                    <div style="
                        color:#101828;
                        font-size:16px;
                        font-weight:850;
                        margin-top:7px;
                    ">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.markdown(
        """
        <div class="page-heading">

            <div class="page-heading-title">
                Model Intelligence
            </div>

            <div class="page-heading-subtitle">
                Honest temporal-holdout evaluation of each
                component in the RiskGraph architecture.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MODEL CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (
            c1,
            "Precision",
            "85.11%",
            "Fraud model"
        ),
        (
            c2,
            "Recall",
            "98.36%",
            "Fraud model"
        ),
        (
            c3,
            "F1 score",
            "91.25%",
            "Fraud model"
        ),
        (
            c4,
            "ROC-AUC",
            "99.85%",
            "Future holdout"
        )
    ]

    for col, label, value, note in metrics:

        with col:

            st.markdown(
                f"""
                <div class="kpi">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-note">
                        {note}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="panel">

            <div class="panel-title">
                RiskGraph architecture
            </div>

            <div class="panel-subtitle">
                Three independent signals feed a cost-aware
                intervention policy.
            </div>

        """,
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    architecture = [
        (
            a1,
            "01",
            "Fraud model",
            "Supervised classification",
            "99.85% ROC-AUC",
            "#635bff"
        ),
        (
            a2,
            "02",
            "Anomaly engine",
            "Unsupervised behavioral detection",
            "99.12% ROC-AUC",
            "#f04438"
        ),
        (
            a3,
            "03",
            "Entity graph",
            "Investigation and relationship context",
            "0.3673 ROC-AUC",
            "#1570ef"
        )
    ]

    for col, number, title, desc, metric, color in architecture:

        with col:

            st.markdown(
                f"""
                <div style="
                    border:1px solid #e7eaf0;
                    border-radius:13px;
                    padding:17px;
                    min-height:150px;
                ">

                    <div style="
                        color:{color};
                        font-size:10px;
                        font-weight:900;
                    ">
                        {number}
                    </div>

                    <div style="
                        font-size:15px;
                        font-weight:850;
                        margin-top:10px;
                    ">
                        {title}
                    </div>

                    <div style="
                        color:#667085;
                        font-size:10px;
                        margin-top:6px;
                    ">
                        {desc}
                    </div>

                    <div style="
                        margin-top:18px;
                        font-size:12px;
                        font-weight:850;
                    ">
                        {metric}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # GRAPH HONESTY
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="panel">

            <div class="panel-title">
                Model governance decision
            </div>

            <div class="panel-subtitle">
                Why the graph is not part of the predictive score
            </div>

            <div style="
                padding:16px;
                background:#fffaeb;
                border:1px solid #fedf89;
                border-radius:11px;
            ">

                <div style="
                    color:#b54708;
                    font-weight:850;
                    font-size:13px;
                ">
                    Graph ROC-AUC: {GRAPH_AUC:.4f}
                </div>

                <div style="
                    color:#7a2e0b;
                    font-size:10px;
                    line-height:1.6;
                    margin-top:7px;
                ">
                    The graph signal was evaluated independently
                    and did not provide reliable fraud
                    discrimination on the holdout. RiskGraph
                    therefore keeps it as analyst context rather
                    than forcing it into the predictive model.
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # FINAL POLICY
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="panel">

            <div class="panel-title">
                Final intervention policy
            </div>

            <div class="panel-subtitle">
                Risk score translated into operational action
            </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    policy_cards = [
        (
            p1,
            "0–59",
            "APPROVE",
            "Low risk",
            "#12b76a",
            "#ecfdf3"
        ),
        (
            p2,
            "60–74",
            "VERIFY",
            "Uncertain risk",
            "#f79009",
            "#fffaeb"
        ),
        (
            p3,
            "75–100",
            "REVIEW",
            "High risk",
            "#f04438",
            "#fef3f2"
        )
    ]

    for col, range_text, action, desc, color, bg in policy_cards:

        with col:

            st.markdown(
                f"""
                <div style="
                    padding:20px;
                    background:{bg};
                    border-radius:12px;
                    border:1px solid {color}33;
                ">

                    <div style="
                        color:{color};
                        font-size:22px;
                        font-weight:900;
                    ">
                        {range_text}
                    </div>

                    <div style="
                        color:#101828;
                        font-size:14px;
                        font-weight:850;
                        margin-top:6px;
                    ">
                        {action}
                    </div>

                    <div style="
                        color:#667085;
                        font-size:10px;
                        margin-top:5px;
                    ">
                        {desc}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    st.markdown(
        """
        <div class="page-heading">

            <div class="page-heading-title">
                Business Impact
            </div>

            <div class="page-heading-subtitle">
                Risk decisions are evaluated not only by accuracy,
                but also by merchant loss and customer friction.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # IMPACT KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    impact = [
        (
            c1,
            "False positives",
            f"{FALSE_POSITIVES}",
            "Transactions incorrectly intervened"
        ),
        (
            c2,
            "False negatives",
            f"{FALSE_NEGATIVES}",
            "Fraud transactions missed"
        ),
        (
            c3,
            "FP cost",
            money_decimal(FP_COST),
            "Estimated customer-friction cost"
        ),
        (
            c4,
            "Loss avoided",
            money_decimal(LOSS_AVOIDED),
            "Prototype estimated benefit"
        )
    ]

    for col, label, value, note in impact:

        with col:

            st.markdown(
                f"""
                <div class="kpi">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-note">
                        {note}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # --------------------------------------------------------
    # LOSS VIEW
    # --------------------------------------------------------

    left, right = st.columns(
        [1.15, 1]
    )

    with left:

        st.markdown(
            f"""
            <div class="panel">

                <div class="panel-title">
                    Merchant loss exposure
                </div>

                <div class="panel-subtitle">
                    Prototype economics from the final policy
                </div>

                <div style="
                    padding:18px;
                    border-bottom:1px solid #eaecf0;
                ">

                    <div style="
                        color:#667085;
                        font-size:10px;
                    ">
                        Baseline expected fraud loss
                    </div>

                    <div class="big-metric">
                        ₹2.06M
                    </div>

                </div>

                <div style="
                    padding:18px;
                    border-bottom:1px solid #eaecf0;
                ">

                    <div style="
                        color:#667085;
                        font-size:10px;
                    ">
                        Residual expected loss
                    </div>

                    <div class="big-metric">
                        ₹111K
                    </div>

                </div>

                <div style="
                    padding:18px;
                ">

                    <div style="
                        color:#027a48;
                        font-size:10px;
                        font-weight:800;
                    ">
                        ESTIMATED LOSS AVOIDED
                    </div>

                    <div class="big-metric"
                         style="color:#027a48;">
                        ₹1.95M
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    Cost-aware intervention
                </div>

                <div class="panel-subtitle">
                    Why the system does not simply block everything
                </div>
            """,
            unsafe_allow_html=True
        )

        items = [
            (
                "APPROVE",
                "Minimize friction",
                "#12b76a"
            ),
            (
                "VERIFY",
                "Challenge uncertain risk",
                "#f79009"
            ),
            (
                "REVIEW",
                "Escalate high-risk cases",
                "#f04438"
            )
        ]

        for name, desc, color in items:

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    gap:12px;
                    align-items:center;
                    padding:13px 0;
                    border-bottom:1px solid #f0f2f5;
                ">

                    <div style="
                        width:9px;
                        height:9px;
                        background:{color};
                        border-radius:50%;
                    "></div>

                    <div>

                        <div style="
                            font-size:11px;
                            font-weight:850;
                        ">
                            {name}
                        </div>

                        <div style="
                            color:#667085;
                            font-size:10px;
                            margin-top:3px;
                        ">
                            {desc}
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="
                margin-top:15px;
                padding:12px;
                background:#f9fafb;
                border-radius:10px;
                color:#667085;
                font-size:9px;
                line-height:1.6;
            ">
                Business metrics use explicit prototype assumptions
                because proprietary merchant operating-cost data
                is unavailable.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # ERROR COST
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="panel">

            <div class="panel-title">
                Error cost profile
            </div>

            <div class="panel-subtitle">
                The final policy produced {FALSE_POSITIVES} false positives
                and {FALSE_NEGATIVES} false negatives.
            </div>

            <div style="
                display:flex;
                gap:20px;
                flex-wrap:wrap;
            ">

                <div style="
                    flex:1;
                    min-width:220px;
                    padding:17px;
                    background:#fef3f2;
                    border-radius:11px;
                ">

                    <div style="
                        color:#b42318;
                        font-size:10px;
                        font-weight:800;
                    ">
                        FALSE POSITIVE VALUE
                    </div>

                    <div style="
                        color:#101828;
                        font-size:23px;
                        font-weight:850;
                        margin-top:7px;
                    ">
                        ₹483,753
                    </div>

                </div>

                <div style="
                    flex:1;
                    min-width:220px;
                    padding:17px;
                    background:#fffaeb;
                    border-radius:11px;
                ">

                    <div style="
                        color:#b54708;
                        font-size:10px;
                        font-weight:800;
                    ">
                        MISSED FRAUD VALUE
                    </div>

                    <div style="
                        color:#101828;
                        font-size:23px;
                        font-weight:850;
                        margin-top:7px;
                    ">
                        ₹4,293
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ENTITY NETWORK
# ============================================================

elif page == "Entity Network":

    st.markdown(
        """
        <div class="page-heading">

            <div class="page-heading-title">
                Entity Network
            </div>

            <div class="page-heading-subtitle">
                Investigate relationships between customers,
                devices, IPs and merchants.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Graph signals are intentionally treated as investigation "
        "context because the independently measured graph ROC-AUC "
        "was 0.3673."
    )

    # --------------------------------------------------------
    # ENTITY STATS
    # --------------------------------------------------------

    device_count = (
        df["device_id"].nunique()
        if "device_id" in df.columns
        else 0
    )

    ip_count = (
        df["ip_id"].nunique()
        if "ip_id" in df.columns
        else 0
    )

    customer_count = (
        df["customer_id"].nunique()
        if "customer_id" in df.columns
        else 0
    )

    merchant_count = (
        df["merchant_id"].nunique()
        if "merchant_id" in df.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    entity_stats = [
        (
            c1,
            "Customers",
            customer_count
        ),
        (
            c2,
            "Merchants",
            merchant_count
        ),
        (
            c3,
            "Devices",
            device_count
        ),
        (
            c4,
            "IPs",
            ip_count
        )
    ]

    for col, label, value in entity_stats:

        with col:

            st.markdown(
                f"""
                <div class="kpi">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value:,}
                    </div>

                    <div class="kpi-note">
                        Unique entities observed
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # --------------------------------------------------------
    # SELECT ENTITY
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 1.4]
    )

    with left:

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    Relationship lookup
                </div>

                <div class="panel-subtitle">
                    Select an entity to inspect associated payments
                </div>
            """,
            unsafe_allow_html=True
        )

        entity_type = st.selectbox(
            "Entity type",
            [
                "Customer",
                "Device",
                "IP",
                "Merchant"
            ]
        )

        mapping = {
            "Customer": "customer_id",
            "Device": "device_id",
            "IP": "ip_id",
            "Merchant": "merchant_id"
        }

        selected_column = mapping[
            entity_type
        ]

        values = (
            df[selected_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_entity = st.selectbox(
            entity_type,
            values[:5000]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with right:

        related = df[
            df[selected_column].astype(str)
            == str(selected_entity)
        ].copy()

        st.markdown(
            f"""
            <div class="panel">

                <div class="panel-title">
                    {selected_entity}
                </div>

                <div class="panel-subtitle">
                    {len(related):,} associated transaction(s)
                </div>
            """,
            unsafe_allow_html=True
        )

        if not related.empty:

            high_related = related[
                related["risk_score"] >= 75
            ] if "risk_score" in related.columns else related

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    gap:12px;
                ">

                    <div style="
                        flex:1;
                        padding:14px;
                        background:#f9fafb;
                        border-radius:10px;
                    ">

                        <div style="
                            color:#667085;
                            font-size:9px;
                            font-weight:800;
                        ">
                            TRANSACTIONS
                        </div>

                        <div style="
                            font-size:22px;
                            font-weight:850;
                            margin-top:5px;
                        ">
                            {len(related):,}
                        </div>

                    </div>

                    <div style="
                        flex:1;
                        padding:14px;
                        background:#fef3f2;
                        border-radius:10px;
                    ">

                        <div style="
                            color:#b42318;
                            font-size:9px;
                            font-weight:800;
                        ">
                            HIGH RISK
                        </div>

                        <div style="
                            font-size:22px;
                            font-weight:850;
                            margin-top:5px;
                        ">
                            {len(high_related):,}
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # RELATED TRANSACTIONS
    # --------------------------------------------------------

    related = df[
        df[selected_column].astype(str)
        == str(selected_entity)
    ].copy()

    if not related.empty:

        related = related.sort_values(
            "risk_score",
            ascending=False
        ).head(20)

        st.markdown(
            """
            <div class="panel">

                <div class="panel-title">
                    Related transaction activity
                </div>

                <div class="panel-subtitle">
                    Highest-risk relationships first
                </div>
            """,
            unsafe_allow_html=True
        )

        rows_html = ""

        for _, row in related.iterrows():

            action = safe(
                row,
                "final_action",
                "APPROVE"
            )

            action_css = (
                "action-pill-review"
                if action == "REVIEW"
                else
                "action-pill-verify"
                if action == "VERIFY"
                else
                "action-pill-approve"
            )

            rows_html += f"""
            <tr>

                <td>
                    <span class="tx-id">
                        {safe(row,'transaction_id')}
                    </span>
                </td>

                <td>
                    {money_decimal(
                        safe(row,'amount',0)
                    )}
                </td>

                <td>
                    {float(
                        safe(row,'risk_score',0)
                    ):.1f}
                </td>

                <td>
                    <span class="action-pill {action_css}">
                        {action}
                    </span>
                </td>

            </tr>
            """

        st.markdown(
            f"""
            <table class="queue">

                <thead>

                    <tr>
                        <th>Transaction</th>
                        <th>Amount</th>
                        <th>Risk</th>
                        <th>Action</th>
                    </tr>

                </thead>

                <tbody>
                    {rows_html}
                </tbody>

            </table>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        RiskGraph AI · AI Risk Manager · Razorpay Buildathon Track 02
        <br>
        Synthetic-data prototype · Temporal holdout evaluation
    </div>
    """,
    unsafe_allow_html=True
)
