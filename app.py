import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from textwrap import dedent


# ============================================================
# RISKGRAPH AI
# AI RISK MANAGER — RAZORPAY BUILDAThon TRACK 02
# ============================================================

st.set_page_config(
    page_title="RiskGraph AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("sample_transactions.csv")


# Your measured evaluation results
FINAL_PRECISION = 0.7785
FINAL_RECALL = 0.9795
FINAL_F1 = 0.8675

FRAUD_PRECISION = 0.8511
FRAUD_RECALL = 0.9836
FRAUD_F1 = 0.9125
FRAUD_AUC = 0.9985

ANOMALY_AUC = 0.9912
GRAPH_AUC = 0.3673

FALSE_POSITIVES = 68
FALSE_NEGATIVES = 5

FALSE_POSITIVE_VALUE = 483752.82
MISSED_FRAUD_VALUE = 4292.70
ESTIMATED_FP_COST = 4837.53

BASELINE_LOSS = 2060929.95
RESIDUAL_LOSS = 111460.20
LOSS_AVOIDED = 1949469.75


# ============================================================
# SAFE HTML RENDERER
# ============================================================

def html(content):
    """
    Safely render multiline HTML without accidental Markdown
    code-block rendering caused by Python indentation.
    """
    st.markdown(
        dedent(content).strip(),
        unsafe_allow_html=True
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_FILE)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

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
        "behavior_risk_count",
        "high_velocity",
        "new_account",
        "new_device",
        "is_weekend",
        "is_fraud",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


df = load_data()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_value(row, column, default=0):

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def money(value):

    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return "₹0"


def money2(value):

    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "₹0.00"


def risk_color(score):

    try:
        score = float(score)
    except Exception:
        score = 0

    if score >= 75:
        return "#F04438"

    if score >= 60:
        return "#F79009"

    return "#12B76A"


def action_color(action):

    if action == "REVIEW":
        return "#F04438"

    if action == "VERIFY":
        return "#F79009"

    return "#12B76A"


def action_background(action):

    if action == "REVIEW":
        return "#FEF3F2"

    if action == "VERIFY":
        return "#FFFAEB"

    return "#ECFDF3"


def get_action(row):

    if "final_action" in row.index:
        action = str(row["final_action"])
        if action in ["APPROVE", "VERIFY", "REVIEW"]:
            return action

    risk = float(get_value(row, "risk_score", 0))

    if risk >= 75:
        return "REVIEW"

    if risk >= 60:
        return "VERIFY"

    return "APPROVE"


def evidence_for(row):

    evidence = []

    fraud_probability = float(
        get_value(row, "fraud_probability", 0)
    )

    anomaly = float(
        get_value(row, "anomaly_score", 0)
    )

    velocity = float(
        get_value(row, "transactions_last_10min", 0)
    )

    failed = float(
        get_value(row, "failed_attempts", 0)
    )

    deviation = float(
        get_value(row, "amount_deviation", 0)
    )

    device_age = float(
        get_value(row, "device_age_days", 9999)
    )

    location_change = float(
        get_value(row, "location_change", 0)
    )

    behavior = float(
        get_value(row, "behavior_risk_count", 0)
    )

    if fraud_probability >= 0.70:
        evidence.append(
            (
                "HIGH",
                "Fraud model",
                f"Fraud probability is {fraud_probability * 100:.1f}%."
            )
        )

    if anomaly >= 80:
        evidence.append(
            (
                "HIGH",
                "Anomaly engine",
                f"Behavioral anomaly score is {anomaly:.1f}/100."
            )
        )

    if deviation >= 4:
        evidence.append(
            (
                "HIGH",
                "Amount deviation",
                f"Transaction amount is {deviation:.1f}× the behavioral baseline."
            )
        )

    if velocity >= 4:
        evidence.append(
            (
                "MEDIUM",
                "Transaction velocity",
                f"{velocity:.0f} transactions occurred within the 10-minute window."
            )
        )

    if failed >= 3:
        evidence.append(
            (
                "MEDIUM",
                "Failed attempts",
                f"{failed:.0f} failed attempts were observed."
            )
        )

    if device_age < 14:
        evidence.append(
            (
                "MEDIUM",
                "New device",
                f"Device age is only {device_age:.0f} days."
            )
        )

    if location_change == 1:
        evidence.append(
            (
                "MEDIUM",
                "Location change",
                "Transaction location differs from the established behavior."
            )
        )

    if behavior >= 3:
        evidence.append(
            (
                "MEDIUM",
                "Behavior risk",
                f"{behavior:.0f} behavioral risk indicators are active."
            )
        )

    if not evidence:
        evidence.append(
            (
                "LOW",
                "Baseline",
                "No major risk indicators were detected."
            )
        )

    return evidence


# ============================================================
# GLOBAL CSS
# ============================================================

html(
    """
    <style>

    /* ======================================================
       ROOT
    ====================================================== */

    :root {
        --background: #F5F7FA;
        --surface: #FFFFFF;
        --surface-soft: #F8FAFC;
        --border: #E4E7EC;

        --text: #101828;
        --muted: #667085;
        --subtle: #98A2B3;

        --purple: #635BFF;
        --purple-soft: #F0EEFF;

        --green: #12B76A;
        --green-soft: #ECFDF3;

        --orange: #F79009;
        --orange-soft: #FFFAEB;

        --red: #F04438;
        --red-soft: #FEF3F2;

        --blue: #1570EF;
        --blue-soft: #EFF8FF;
    }

    /* ======================================================
       MAIN APP
    ====================================================== */

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 30px;
        padding-bottom: 60px;
    }

    /* ======================================================
       REMOVE STREAMLIT CHROME
    ====================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background: #0B1017;
        border-right: 1px solid #202833;
    }

    section[data-testid="stSidebar"] * {
        color: #D0D5DD;
    }

    .brand {
        padding: 10px 8px 28px 8px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: linear-gradient(
            135deg,
            #635BFF,
            #8B5CF6
        );
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 20px;
        box-shadow: 0 8px 25px rgba(99,91,255,.28);
    }

    .brand-name {
        color: #FFFFFF;
        font-size: 18px;
        font-weight: 850;
        letter-spacing: -.4px;
    }

    .brand-name span {
        color: #8B83FF;
    }

    .brand-caption {
        color: #667085;
        font-size: 10px;
        margin-top: 6px;
        margin-left: 2px;
    }

    .side-section {
        color: #667085;
        font-size: 9px;
        font-weight: 850;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 24px 7px 8px;
    }

    .engine-card {
        margin-top: 30px;
        border: 1px solid #242C37;
        background: #121820;
        border-radius: 12px;
        padding: 13px;
    }

    .engine-label {
        color: #667085;
        font-size: 9px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .engine-online {
        color: #6CE9A6;
        font-size: 11px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* ======================================================
       HEADER
    ====================================================== */

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 30px;
    }

    .header-title {
        color: #101828;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: -1.3px;
    }

    .header-title span {
        color: var(--purple);
    }

    .header-subtitle {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }

    .online-pill {
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 8px 13px;
        border: 1px solid #ABEFC6;
        background: var(--green-soft);
        color: #027A48;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 850;
    }

    .online-dot {
        width: 7px;
        height: 7px;
        background: var(--green);
        border-radius: 50%;
    }

    /* ======================================================
       SECTION TITLES
    ====================================================== */

    .section-title {
        color: #101828;
        font-size: 19px;
        font-weight: 850;
        letter-spacing: -.3px;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 11px;
        margin-top: 3px;
        margin-bottom: 17px;
    }

    /* ======================================================
       KPI
    ====================================================== */

    .kpi {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px;
        min-height: 125px;
        box-shadow: 0 1px 2px rgba(16,24,40,.03);
    }

    .kpi-label {
        color: var(--muted);
        font-size: 9px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .09em;
    }

    .kpi-value {
        color: #101828;
        font-size: 28px;
        font-weight: 900;
        letter-spacing: -1px;
        margin-top: 11px;
    }

    .kpi-note {
        color: var(--subtle);
        font-size: 9px;
        margin-top: 5px;
    }

    /* ======================================================
       PANEL
    ====================================================== */

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 1px 2px rgba(16,24,40,.025);
    }

    .panel-title {
        color: #101828;
        font-size: 14px;
        font-weight: 850;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 10px;
        line-height: 1.5;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    /* ======================================================
       DECISION MIX
    ====================================================== */

    .decision {
        margin-bottom: 17px;
    }

    .decision-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 7px;
    }

    .decision-name {
        font-size: 10px;
        font-weight: 850;
    }

    .decision-number {
        color: var(--muted);
        font-size: 10px;
    }

    .track {
        height: 7px;
        border-radius: 99px;
        background: #EEF0F3;
        overflow: hidden;
    }

    .fill-green {
        height: 100%;
        background: #12B76A;
        border-radius: 99px;
    }

    .fill-orange {
        height: 100%;
        background: #F79009;
        border-radius: 99px;
    }

    .fill-red {
        height: 100%;
        background: #F04438;
        border-radius: 99px;
    }

    /* ======================================================
       RISK HISTOGRAM
    ====================================================== */

    .histogram {
        display: flex;
        align-items: flex-end;
        gap: 6px;
        height: 185px;
        padding-top: 8px;
    }

    .hist-column {
        flex: 1;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }

    .hist-bar {
        width: 100%;
        border-radius: 5px 5px 2px 2px;
        background: linear-gradient(
            180deg,
            #8B83FF,
            #635BFF
        );
        min-height: 5px;
    }

    .hist-label {
        text-align: center;
        color: #98A2B3;
        font-size: 8px;
        margin-top: 6px;
    }

    /* ======================================================
       TABLE
    ====================================================== */

    .risk-table {
        width: 100%;
        border-collapse: collapse;
    }

    .risk-table th {
        color: #667085;
        font-size: 8px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .08em;
        padding: 10px 11px;
        text-align: left;
        border-bottom: 1px solid #EAECF0;
    }

    .risk-table td {
        color: #344054;
        font-size: 10px;
        padding: 12px 11px;
        border-bottom: 1px solid #F2F4F7;
    }

    .risk-table tr:last-child td {
        border-bottom: none;
    }

    .transaction-id {
        color: #101828;
        font-weight: 850;
    }

    .risk-high {
        color: #D92D20;
        font-weight: 850;
    }

    .risk-medium {
        color: #B54708;
        font-weight: 850;
    }

    .risk-low {
        color: #027A48;
        font-weight: 850;
    }

    .pill {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 8px;
        font-weight: 900;
    }

    .pill-review {
        color: #B42318;
        background: #FEF3F2;
    }

    .pill-verify {
        color: #B54708;
        background: #FFFAEB;
    }

    .pill-approve {
        color: #027A48;
        background: #ECFDF3;
    }

    /* ======================================================
       INVESTIGATION HERO
    ====================================================== */

    .hero-dark {
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(99,91,255,.25),
                transparent 35%
            ),
            #101722;
        border-radius: 17px;
        padding: 25px;
        color: white;
        min-height: 205px;
    }

    .hero-label {
        color: #98A2B3;
        font-size: 9px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .1em;
    }

    .hero-id {
        color: white;
        font-size: 23px;
        font-weight: 900;
        margin-top: 6px;
    }

    .hero-amount {
        color: white;
        font-size: 34px;
        font-weight: 900;
        letter-spacing: -1px;
        margin-top: 25px;
    }

    .hero-meta {
        color: #98A2B3;
        font-size: 10px;
        margin-top: 5px;
    }

    .hero-score {
        text-align: center;
        padding-top: 10px;
    }

    .hero-score-number {
        font-size: 60px;
        font-weight: 950;
        line-height: 1;
    }

    .hero-score-label {
        color: #98A2B3;
        font-size: 8px;
        font-weight: 850;
        letter-spacing: .1em;
        margin-top: 8px;
    }

    /* ======================================================
       SIGNAL CARDS
    ====================================================== */

    .signal {
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px;
        min-height: 145px;
    }

    .signal-label {
        color: #667085;
        font-size: 9px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .signal-value {
        color: #101828;
        font-size: 31px;
        font-weight: 900;
        letter-spacing: -1px;
        margin-top: 9px;
    }

    .signal-description {
        color: #667085;
        font-size: 9px;
        margin-top: 3px;
    }

    .signal-track {
        height: 5px;
        background: #EEF0F3;
        border-radius: 99px;
        margin-top: 15px;
        overflow: hidden;
    }

    /* ======================================================
       EVIDENCE
    ====================================================== */

    .evidence {
        border: 1px solid #EAECF0;
        border-radius: 11px;
        padding: 13px;
        margin-bottom: 8px;
        background: white;
    }

    .evidence-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .evidence-title {
        color: #101828;
        font-size: 10px;
        font-weight: 850;
    }

    .evidence-text {
        color: #667085;
        font-size: 9px;
        margin-top: 5px;
        line-height: 1.5;
    }

    .severity {
        font-size: 7px;
        font-weight: 900;
        padding: 4px 7px;
        border-radius: 999px;
    }

    .severity-high {
        color: #B42318;
        background: #FEF3F2;
    }

    .severity-medium {
        color: #B54708;
        background: #FFFAEB;
    }

    .severity-low {
        color: #027A48;
        background: #ECFDF3;
    }

    /* ======================================================
       ENTITY
    ====================================================== */

    .entity {
        background: #F8FAFC;
        border: 1px solid #EAECF0;
        border-radius: 11px;
        padding: 14px;
        text-align: center;
        margin-bottom: 9px;
    }

    .entity-icon {
        font-size: 16px;
        font-weight: 900;
        color: #635BFF;
    }

    .entity-type {
        color: #98A2B3;
        font-size: 7px;
        font-weight: 850;
        text-transform: uppercase;
        margin-top: 6px;
    }

    .entity-value {
        color: #101828;
        font-size: 9px;
        font-weight: 850;
        margin-top: 3px;
        word-break: break-word;
    }

    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        text-align: center;
        color: #98A2B3;
        font-size: 8px;
        padding: 30px 0 5px;
        line-height: 1.7;
    }

    </style>
    """
)


# ============================================================
# DATA VALIDATION
# ============================================================

if df.empty:

    html(
        """
        <div style="
            max-width:700px;
            margin:80px auto;
            background:white;
            border:1px solid #E4E7EC;
            border-radius:18px;
            padding:35px;
            text-align:center;
        ">

            <div style="
                font-size:40px;
                color:#635BFF;
            ">
                ◈
            </div>

            <div style="
                font-size:24px;
                font-weight:900;
                margin-top:15px;
            ">
                RiskGraph AI
            </div>

            <div style="
                color:#667085;
                font-size:12px;
                margin-top:8px;
            ">
                sample_transactions.csv was not found.
            </div>

            <div style="
                margin-top:20px;
                padding:13px;
                background:#FEF3F2;
                color:#B42318;
                border-radius:10px;
                font-size:10px;
            ">
                Upload sample_transactions.csv to the same
                GitHub repository as app.py.
            </div>

        </div>
        """,
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html(
        """
        <div class="brand">

            <div class="brand-row">

                <div class="brand-icon">
                    ◈
                </div>

                <div class="brand-name">
                    RiskGraph <span>AI</span>
                </div>

            </div>

            <div class="brand-caption">
                Payment risk intelligence
            </div>

        </div>

        <div class="side-section">
            Risk Operations
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Investigate",
            "Model Intelligence",
            "Business Impact",
            "Entity Network",
        ],
        label_visibility="collapsed",
    )

    html(
        """
        <div class="engine-card">

            <div class="engine-label">
                Risk Engine
            </div>

            <div class="engine-online">
                ● Operational
            </div>

        </div>
        """
    )


# ============================================================
# GLOBAL HEADER
# ============================================================

html(
    """
    <div class="header">

        <div>

            <div class="header-title">
                RiskGraph <span>AI</span>
            </div>

            <div class="header-subtitle">
                Cost-aware payment risk intelligence
            </div>

        </div>

        <div class="online-pill">
            <span class="online-dot"></span>
            RISK ENGINE ONLINE
        </div>

    </div>
    """
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    total_transactions = len(df)

    fraud_count = (
        int(df["is_fraud"].sum())
        if "is_fraud" in df.columns
        else 244
    )

    if "final_action" in df.columns:

        approve_count = int(
            (df["final_action"] == "APPROVE").sum()
        )

        verify_count = int(
            (df["final_action"] == "VERIFY").sum()
        )

        review_count = int(
            (df["final_action"] == "REVIEW").sum()
        )

    else:

        approve_count = int(
            (df["risk_score"] < 60).sum()
        )

        verify_count = int(
            (
                (df["risk_score"] >= 60)
                &
                (df["risk_score"] < 75)
            ).sum()
        )

        review_count = int(
            (df["risk_score"] >= 75).sum()
        )

    html(
        """
        <div class="section-title">
            Command Center
        </div>

        <div class="section-subtitle">
            Detect suspicious payments, prioritize intervention,
            and control merchant loss.
        </div>
        """
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Transactions
                </div>

                <div class="kpi-value">
                    {total_transactions:,}
                </div>

                <div class="kpi-note">
                    Temporal future holdout
                </div>

            </div>
            """
        )

    with c2:
        html(
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
            """
        )

    with c3:
        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Detection recall
                </div>

                <div class="kpi-value">
                    {FINAL_RECALL * 100:.2f}%
                </div>

                <div class="kpi-note">
                    Final intervention policy
                </div>

            </div>
            """
        )

    with c4:
        html(
            f"""
            <div class="kpi">

                <div class="kpi-label">
                    Estimated FP cost
                </div>

                <div class="kpi-value">
                    ₹{ESTIMATED_FP_COST:,.0f}
                </div>

                <div class="kpi-note">
                    Prototype estimate
                </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        html(
            """
            <div class="panel">

                <div class="panel-title">
                    Risk landscape
                </div>

                <div class="panel-subtitle">
                    Distribution of final transaction risk scores.
                </div>
            """
        )

        if "risk_score" in df.columns:

            values = (
                pd.to_numeric(
                    df["risk_score"],
                    errors="coerce"
                )
                .fillna(0)
                .clip(0, 100)
            )

            bins = [
                0, 10, 20, 30, 40,
                50, 60, 70, 80, 90, 101
            ]

            counts, _ = np.histogram(
                values,
                bins=bins
            )

            max_count = max(
                int(counts.max()),
                1
            )

            chart = '<div class="histogram">'

            for index, count in enumerate(counts):

                height = max(
                    5,
                    int(
                        count /
                        max_count *
                        155
                    )
                )

                label = str(
                    int(bins[index])
                )

                chart += f"""
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

            chart += "</div>"

            html(chart)

        html(
            """
            </div>
            """
        )

    # --------------------------------------------------------
    # DECISION MIX
    # --------------------------------------------------------

    with right:

        html(
            """
            <div class="panel">

                <div class="panel-title">
                    Decision engine
                </div>

                <div class="panel-subtitle">
                    Final operational intervention policy.
                </div>
            """
        )

        total = max(
            total_transactions,
            1
        )

        decision_data = [
            (
                "APPROVE",
                approve_count,
                "fill-green",
            ),
            (
                "VERIFY",
                verify_count,
                "fill-orange",
            ),
            (
                "REVIEW",
                review_count,
                "fill-red",
            ),
        ]

        for name, count, css in decision_data:

            percentage = (
                count /
                total *
                100
            )

            html(
                f"""
                <div class="decision">

                    <div class="decision-head">

                        <div class="decision-name">
                            {name}
                        </div>

                        <div class="decision-number">
                            {count:,} · {percentage:.2f}%
                        </div>

                    </div>

                    <div class="track">

                        <div
                            class="{css}"
                            style="width:{percentage:.2f}%"
                        ></div>

                    </div>

                </div>
                """
            )

        html(
            """
            <div style="
                margin-top:18px;
                padding:11px;
                background:#F8FAFC;
                border-radius:9px;
                color:#667085;
                font-size:9px;
                line-height:1.6;
            ">
                <b style="color:#344054;">
                    Policy:
                </b>
                APPROVE for low risk · VERIFY for uncertain risk ·
                REVIEW for high risk.
            </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # HIGH RISK QUEUE
    # --------------------------------------------------------

    html(
        """
        <div class="panel">

            <div class="panel-title">
                High-risk queue
            </div>

            <div class="panel-subtitle">
                Highest priority transactions requiring analyst attention.
            </div>
        """
    )

    if "risk_score" in df.columns:

        queue = df.copy()

        if "final_action" in queue.columns:

            queue = queue[
                queue["final_action"] == "REVIEW"
            ]

        queue = (
            queue
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(10)
        )

    else:

        queue = df.head(10)

    table = """
    <table class="risk-table">

        <thead>

            <tr>
                <th>Transaction</th>
                <th>Amount</th>
                <th>Risk</th>
                <th>Fraud probability</th>
                <th>Anomaly</th>
                <th>Decision</th>
            </tr>

        </thead>

        <tbody>
    """

    for _, row in queue.iterrows():

        tx = str(
            get_value(
                row,
                "transaction_id",
                "UNKNOWN"
            )
        )

        amount = float(
            get_value(
                row,
                "amount",
                0
            )
        )

        risk = float(
            get_value(
                row,
                "risk_score",
                0
            )
        )

        fraud_probability = float(
            get_value(
                row,
                "fraud_probability",
                0
            )
        )

        anomaly = float(
            get_value(
                row,
                "anomaly_score",
                0
            )
        )

        action = get_action(row)

        if risk >= 75:
            risk_class = "risk-high"
        elif risk >= 60:
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"

        action_class = (
            "pill-review"
            if action == "REVIEW"
            else
            "pill-verify"
            if action == "VERIFY"
            else
            "pill-approve"
        )

        table += f"""
        <tr>

            <td>
                <span class="transaction-id">
                    {tx}
                </span>
            </td>

            <td>
                ₹{amount:,.2f}
            </td>

            <td>
                <span class="{risk_class}">
                    {risk:.0f}
                </span>
            </td>

            <td>
                {fraud_probability * 100:.1f}%
            </td>

            <td>
                {anomaly:.1f}
            </td>

            <td>
                <span class="pill {action_class}">
                    {action}
                </span>
            </td>

        </tr>
        """

    table += """
        </tbody>
    </table>

    </div>
    """

    html(table)

    html(
        """
        <div class="footer">
            RiskGraph AI · AI Risk Manager · Razorpay Buildathon Track 02
            <br>
            Synthetic-data prototype · Temporal holdout evaluation
        </div>
        """
    )


# ============================================================
# INVESTIGATION
# ============================================================

elif page == "Investigate":

    html(
        """
        <div class="section-title">
            Transaction Investigation
        </div>

        <div class="section-subtitle">
            Explain the risk before deciding what to do.
        </div>
        """
    )

    if "transaction_id" not in df.columns:

        st.error(
            "transaction_id column is missing from the dataset."
        )

        st.stop()

    transaction_ids = (
        df["transaction_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    selected_id = st.selectbox(
        "Transaction to investigate",
        transaction_ids,
    )

    selected = df[
        df["transaction_id"].astype(str)
        == selected_id
    ]

    if selected.empty:
        st.error("Transaction not found.")
        st.stop()

    row = selected.iloc[0]

    action = get_action(row)

    risk = float(
        get_value(
            row,
            "risk_score",
            0
        )
    )

    amount = float(
        get_value(
            row,
            "amount",
            0
        )
    )

    color = action_color(action)

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    c1, c2 = st.columns(
        [1.55, .8]
    )

    with c1:

        html(
            f"""
            <div class="hero-dark">

                <div class="hero-label">
                    Payment under investigation
                </div>

                <div class="hero-id">
                    {selected_id}
                </div>

                <div class="hero-amount">
                    ₹{amount:,.2f}
                </div>

                <div class="hero-meta">
                    Customer · {get_value(row, "customer_id", "—")}
                    &nbsp;&nbsp; · &nbsp;&nbsp;
                    Merchant · {get_value(row, "merchant_id", "—")}
                </div>

                <div class="hero-meta">
                    Device · {get_value(row, "device_id", "—")}
                    &nbsp;&nbsp; · &nbsp;&nbsp;
                    IP · {get_value(row, "ip_id", "—")}
                </div>

            </div>
            """
        )

    with c2:

        html(
            f"""
            <div class="hero-dark">

                <div class="hero-score">

                    <div
                        class="hero-score-number"
                        style="color:{color};"
                    >
                        {risk:.0f}
                    </div>

                    <div class="hero-score-label">
                        RISK SCORE / 100
                    </div>

                    <div style="
                        margin-top:18px;
                        padding:9px;
                        border-radius:8px;
                        background:rgba(255,255,255,.07);
                        color:{color};
                        font-size:11px;
                        font-weight:900;
                    ">
                        ● {action}
                    </div>

                </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # THREE SIGNALS
    # --------------------------------------------------------

    fraud_probability = (
        float(
            get_value(
                row,
                "fraud_probability",
                0
            )
        )
        * 100
    )

    anomaly_score = float(
        get_value(
            row,
            "anomaly_score",
            0
        )
    )

    graph_score = float(
        get_value(
            row,
            "graph_risk_score",
            0
        )
    )

    s1, s2, s3 = st.columns(3)

    signals = [
        (
            s1,
            "Fraud model",
            fraud_probability,
            "Supervised fraud probability",
            "#635BFF",
        ),
        (
            s2,
            "Anomaly engine",
            anomaly_score,
            "Behavioral deviation",
            "#F04438",
        ),
        (
            s3,
            "Graph context",
            graph_score,
            "Investigation context",
            "#1570EF",
        ),
    ]

    for col, title, value, description, color in signals:

        with col:

            html(
                f"""
                <div class="signal">

                    <div class="signal-label">
                        {title}
                    </div>

                    <div class="signal-value">
                        {value:.1f}
                        <span style="
                            color:#98A2B3;
                            font-size:12px;
                            font-weight:600;
                        ">
                            /100
                        </span>
                    </div>

                    <div class="signal-description">
                        {description}
                    </div>

                    <div class="signal-track">

                        <div
                            style="
                                width:{min(value,100):.1f}%;
                                height:100%;
                                background:{color};
                                border-radius:99px;
                            "
                        ></div>

                    </div>

                </div>
                """
            )

    st.write("")

    # --------------------------------------------------------
    # EVIDENCE + ENTITY CONTEXT
    # --------------------------------------------------------

    left, right = st.columns(
        [1.3, 1]
    )

    with left:

        html(
            """
            <div class="panel">

                <div class="panel-title">
                    Why RiskGraph flagged this payment
                </div>

                <div class="panel-subtitle">
                    Independent evidence supporting the intervention.
                </div>
            """
        )

        for severity, title, text in evidence_for(row):

            severity_class = (
                "severity-high"
                if severity == "HIGH"
                else
                "severity-medium"
                if severity == "MEDIUM"
                else
                "severity-low"
            )

            html(
                f"""
                <div class="evidence">

                    <div class="evidence-head">

                        <div class="evidence-title">
                            {title}
                        </div>

                        <div class="severity {severity_class}">
                            {severity}
                        </div>

                    </div>

                    <div class="evidence-text">
                        {text}
                    </div>

                </div>
                """
            )

        html(
            """
            </div>
            """
        )

    with right:

        html(
            """
            <div class="panel">

                <div class="panel-title">
                    Entity context
                </div>

                <div class="panel-subtitle">
                    Relationship identifiers available to the analyst.
                </div>
            """
        )

        e1, e2 = st.columns(2)

        entities = [
            (
                e1,
                "Customer",
                get_value(row, "customer_id", "—"),
                "C",
            ),
            (
                e2,
                "Merchant",
                get_value(row, "merchant_id", "—"),
                "M",
            ),
            (
                e1,
                "Device",
                get_value(row, "device_id", "—"),
                "D",
            ),
            (
                e2,
                "IP address",
                get_value(row, "ip_id", "—"),
                "IP",
            ),
        ]

        for col, entity_type, entity_value, icon in entities:

            with col:

                html(
                    f"""
                    <div class="entity">

                        <div class="entity-icon">
                            {icon}
                        </div>

                        <div class="entity-type">
                            {entity_type}
                        </div>

                        <div class="entity-value">
                            {entity_value}
                        </div>

                    </div>
                    """
                )

        html(
            """
            <div style="
                margin-top:5px;
                padding:11px;
                border-radius:9px;
                background:#EFF8FF;
                color:#175CD3;
                font-size:9px;
                line-height:1.5;
            ">
                Graph relationships are presented as investigation
                context. The independently evaluated graph signal
                is not used as predictive evidence.
            </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # BEHAVIORAL PROFILE
    # --------------------------------------------------------

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Behavioral profile
            </div>

            <div class="panel-subtitle">
                Signals contributing to the risk decision.
            </div>
        """
    )

    details = [
        (
            "Amount deviation",
            f"{float(get_value(row, 'amount_deviation', 0)):.2f}×",
        ),
        (
            "10-min velocity",
            f"{float(get_value(row, 'transactions_last_10min', 0)):.0f}",
        ),
        (
            "Failed attempts",
            f"{float(get_value(row, 'failed_attempts', 0)):.0f}",
        ),
        (
            "Device age",
            f"{float(get_value(row, 'device_age_days', 0)):.0f} days",
        ),
        (
            "Account age",
            f"{float(get_value(row, 'account_age_days', 0)):.0f} days",
        ),
        (
            "Location change",
            "YES"
            if float(
                get_value(
                    row,
                    "location_change",
                    0
                )
            ) == 1
            else "NO",
        ),
    ]

    columns = st.columns(6)

    for col, (label, value) in zip(
        columns,
        details
    ):

        with col:

            html(
                f"""
                <div style="
                    background:#F8FAFC;
                    border:1px solid #EAECF0;
                    border-radius:10px;
                    padding:12px;
                ">

                    <div style="
                        color:#667085;
                        font-size:7px;
                        font-weight:850;
                        text-transform:uppercase;
                    ">
                        {label}
                    </div>

                    <div style="
                        color:#101828;
                        font-size:16px;
                        font-weight:900;
                        margin-top:7px;
                    ">
                        {value}
                    </div>

                </div>
                """
            )

    html(
        """
        </div>
        """
    )

    # --------------------------------------------------------
    # ORIGINAL EXPLANATION
    # --------------------------------------------------------

    if "risk_explanation" in row.index:

        explanation = str(
            get_value(
                row,
                "risk_explanation",
                ""
            )
        )

        if explanation and explanation != "nan":

            st.write("")

            html(
                f"""
                <div class="panel">

                    <div class="panel-title">
                        Risk explanation
                    </div>

                    <div style="
                        margin-top:12px;
                        padding:14px;
                        background:#F8FAFC;
                        border-radius:10px;
                        color:#475467;
                        font-size:10px;
                        line-height:1.6;
                    ">
                        {explanation}
                    </div>

                </div>
                """
            )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    html(
        """
        <div class="section-title">
            Model Intelligence
        </div>

        <div class="section-subtitle">
            Honest evaluation of the RiskGraph components on
            held-out temporal data.
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    model_metrics = [
        (
            c1,
            "Precision",
            f"{FRAUD_PRECISION * 100:.2f}%",
            "Fraud model",
        ),
        (
            c2,
            "Recall",
            f"{FRAUD_RECALL * 100:.2f}%",
            "Fraud model",
        ),
        (
            c3,
            "F1 score",
            f"{FRAUD_F1 * 100:.2f}%",
            "Fraud model",
        ),
        (
            c4,
            "ROC-AUC",
            f"{FRAUD_AUC * 100:.2f}%",
            "Future holdout",
        ),
    ]

    for col, label, value, note in model_metrics:

        with col:

            html(
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
                """
            )

    st.write("")

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    html(
        """
        <div class="panel">

            <div class="panel-title">
                RiskGraph architecture
            </div>

            <div class="panel-subtitle">
                Multiple signals are evaluated independently before
                the final intervention policy.
            </div>
        """
    )

    a1, a2, a3 = st.columns(3)

    architecture = [
        (
            a1,
            "01",
            "Fraud model",
            "Supervised transaction classification",
            "99.85% ROC-AUC",
            "#635BFF",
        ),
        (
            a2,
            "02",
            "Anomaly engine",
            "Unsupervised behavioral anomaly detection",
            "99.12% ROC-AUC",
            "#F04438",
        ),
        (
            a3,
            "03",
            "Entity graph",
            "Relationship and investigation context",
            "0.3673 ROC-AUC",
            "#1570EF",
        ),
    ]

    for col, number, title, description, metric, color in architecture:

        with col:

            html(
                f"""
                <div style="
                    min-height:160px;
                    border:1px solid #EAECF0;
                    border-radius:13px;
                    padding:17px;
                    background:#FFFFFF;
                ">

                    <div style="
                        color:{color};
                        font-size:9px;
                        font-weight:900;
                    ">
                        {number}
                    </div>

                    <div style="
                        color:#101828;
                        font-size:15px;
                        font-weight:900;
                        margin-top:10px;
                    ">
                        {title}
                    </div>

                    <div style="
                        color:#667085;
                        font-size:9px;
                        line-height:1.5;
                        margin-top:6px;
                    ">
                        {description}
                    </div>

                    <div style="
                        color:#101828;
                        font-size:12px;
                        font-weight:900;
                        margin-top:18px;
                    ">
                        {metric}
                    </div>

                </div>
                """
            )

    html(
        """
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # FINAL POLICY
    # --------------------------------------------------------

    html(
        f"""
        <div class="panel">

            <div class="panel-title">
                Final intervention policy
            </div>

            <div class="panel-subtitle">
                Final policy performance: {FINAL_PRECISION * 100:.2f}%
                precision · {FINAL_RECALL * 100:.2f}% recall ·
                {FINAL_F1 * 100:.2f}% F1.
            </div>
        """
    )

    p1, p2, p3 = st.columns(3)

    policies = [
        (
            p1,
            "0–59",
            "APPROVE",
            "Low-risk payment",
            "#12B76A",
            "#ECFDF3",
        ),
        (
            p2,
            "60–74",
            "VERIFY",
            "Uncertain payment",
            "#F79009",
            "#FFFAEB",
        ),
        (
            p3,
            "75–100",
            "REVIEW",
            "High-risk payment",
            "#F04438",
            "#FEF3F2",
        ),
    ]

    for col, threshold, action, description, color, background in policies:

        with col:

            html(
                f"""
                <div style="
                    padding:20px;
                    border-radius:13px;
                    background:{background};
                    border:1px solid {color}33;
                ">

                    <div style="
                        color:{color};
                        font-size:24px;
                        font-weight:950;
                    ">
                        {threshold}
                    </div>

                    <div style="
                        color:#101828;
                        font-size:13px;
                        font-weight:900;
                        margin-top:6px;
                    ">
                        {action}
                    </div>

                    <div style="
                        color:#667085;
                        font-size:9px;
                        margin-top:4px;
                    ">
                        {description}
                    </div>

                </div>
                """
            )

    html(
        """
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # GRAPH GOVERNANCE
    # --------------------------------------------------------

    html(
        f"""
        <div class="panel">

            <div class="panel-title">
                Model governance
            </div>

            <div class="panel-subtitle">
                The system does not hide weak signals.
            </div>

            <div style="
                background:#FFFAEB;
                border:1px solid #FEDF89;
                border-radius:11px;
                padding:16px;
            ">

                <div style="
                    color:#B54708;
                    font-size:12px;
                    font-weight:900;
                ">
                    Entity graph ROC-AUC: {GRAPH_AUC:.4f}
                </div>

                <div style="
                    color:#7A2E0B;
                    font-size:9px;
                    line-height:1.7;
                    margin-top:7px;
                ">
                    The graph signal did not provide reliable
                    discrimination on the held-out evaluation.
                    RiskGraph therefore treats graph relationships
                    as analyst investigation context instead of
                    forcing the signal into the predictive score.
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

elif page == "Business Impact":

    html(
        """
        <div class="section-title">
            Business Impact
        </div>

        <div class="section-subtitle">
            Fraud prevention is an economic optimization problem:
            catch fraud while minimizing unnecessary intervention.
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    business_metrics = [
        (
            c1,
            "False positives",
            f"{FALSE_POSITIVES}",
            "Incorrect interventions",
        ),
        (
            c2,
            "False negatives",
            f"{FALSE_NEGATIVES}",
            "Fraud transactions missed",
        ),
        (
            c3,
            "FP transaction value",
            f"₹{FALSE_POSITIVE_VALUE:,.0f}",
            "Value touched by FP cases",
        ),
        (
            c4,
            "Estimated FP cost",
            f"₹{ESTIMATED_FP_COST:,.0f}",
            "Prototype estimate",
        ),
    ]

    for col, label, value, note in business_metrics:

        with col:

            html(
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
                """
            )

    st.write("")

    left, right = st.columns(
        [1.15, 1]
    )

    with left:

        html(
            f"""
            <div class="panel">

                <div class="panel-title">
                    Merchant loss exposure
                </div>

                <div class="panel-subtitle">
                    Prototype expected-loss calculation.
                </div>

                <div style="
                    padding:17px 0;
                    border-bottom:1px solid #EAECF0;
                ">

                    <div style="
                        color:#667085;
                        font-size:9px;
                    ">
                        Baseline expected fraud loss
                    </div>

                    <div style="
                        color:#101828;
                        font-size:32px;
                        font-weight:950;
                        margin-top:5px;
                    ">
                        ₹{BASELINE_LOSS / 1_000_000:.2f}M
                    </div>

                </div>

                <div style="
                    padding:17px 0;
                    border-bottom:1px solid #EAECF0;
                ">

                    <div style="
                        color:#667085;
                        font-size:9px;
                    ">
                        Residual expected loss
                    </div>

                    <div style="
                        color:#101828;
                        font-size:32px;
                        font-weight:950;
                        margin-top:5px;
                    ">
                        ₹{RESIDUAL_LOSS / 1000:.0f}K
                    </div>

                </div>

                <div style="
                    padding:17px 0;
                ">

                    <div style="
                        color:#027A48;
                        font-size:9px;
                        font-weight:900;
                    ">
                        ESTIMATED LOSS AVOIDED
                    </div>

                    <div style="
                        color:#027A48;
                        font-size:32px;
                        font-weight:950;
                        margin-top:5px;
                    ">
                        ₹{LOSS_AVOIDED / 1_000_000:.2f}M
                    </div>

                </div>

            </div>
            """
        )

    with right:

        html(
            """
            <div class="panel">

                <div class="panel-title">
                    Why three decisions?
                </div>

                <div class="panel-subtitle">
                    The system avoids treating every suspicious
                    transaction as an automatic block.
                </div>
            """
        )

        decisions = [
            (
                "APPROVE",
                "Low-risk payments continue normally.",
                "#12B76A",
            ),
            (
                "VERIFY",
                "Uncertain payments receive additional verification.",
                "#F79009",
            ),
            (
                "REVIEW",
                "High-risk payments are escalated for investigation.",
                "#F04438",
            ),
        ]

        for name, description, color in decisions:

            html(
                f"""
                <div style="
                    display:flex;
                    gap:11px;
                    align-items:flex-start;
                    padding:13px 0;
                    border-bottom:1px solid #F2F4F7;
                ">

                    <div style="
                        width:8px;
                        height:8px;
                        margin-top:3px;
                        background:{color};
                        border-radius:50%;
                        flex-shrink:0;
                    "></div>

                    <div>

                        <div style="
                            color:#101828;
                            font-size:10px;
                            font-weight:900;
                        ">
                            {name}
                        </div>

                        <div style="
                            color:#667085;
                            font-size:9px;
                            margin-top:3px;
                            line-height:1.5;
                        ">
                            {description}
                        </div>

                    </div>

                </div>
                """
            )

        html(
            """
            <div style="
                margin-top:15px;
                padding:11px;
                background:#F8FAFC;
                border-radius:9px;
                color:#667085;
                font-size:8px;
                line-height:1.6;
            ">
                Business impact values are prototype estimates
                based on explicit assumptions, not Razorpay
                production economics.
            </div>

            </div>
            """
        )

    st.write("")

    html(
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
                gap:15px;
            ">

                <div style="
                    flex:1;
                    padding:16px;
                    background:#FEF3F2;
                    border-radius:11px;
                ">

                    <div style="
                        color:#B42318;
                        font-size:8px;
                        font-weight:900;
                    ">
                        FALSE POSITIVE VALUE
                    </div>

                    <div style="
                        color:#101828;
                        font-size:23px;
                        font-weight:900;
                        margin-top:5px;
                    ">
                        ₹{FALSE_POSITIVE_VALUE:,.0f}
                    </div>

                </div>

                <div style="
                    flex:1;
                    padding:16px;
                    background:#FFFAEB;
                    border-radius:11px;
                ">

                    <div style="
                        color:#B54708;
                        font-size:8px;
                        font-weight:900;
                    ">
                        MISSED FRAUD VALUE
                    </div>

                    <div style="
                        color:#101828;
                        font-size:23px;
                        font-weight:900;
                        margin-top:5px;
                    ">
                        ₹{MISSED_FRAUD_VALUE:,.0f}
                    </div>

                </div>

            </div>

        </div>
        """
    )


# ============================================================
# ENTITY NETWORK
# ============================================================

elif page == "Entity Network":

    html(
        """
        <div class="section-title">
            Entity Network
        </div>

        <div class="section-subtitle">
            Explore customer, merchant, device and IP relationships
            surrounding suspicious transactions.
        </div>
        """
    )

    # --------------------------------------------------------
    # ENTITY COUNTS
    # --------------------------------------------------------

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

    c1, c2, c3, c4 = st.columns(4)

    entities = [
        (
            c1,
            "Customers",
            customer_count,
        ),
        (
            c2,
            "Merchants",
            merchant_count,
        ),
        (
            c3,
            "Devices",
            device_count,
        ),
        (
            c4,
            "IP addresses",
            ip_count,
        ),
    ]

    for col, label, value in entities:

        with col:

            html(
                f"""
                <div class="kpi">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value:,}
                    </div>

                    <div class="kpi-note">
                        Unique observed entities
                    </div>

                </div>
                """
            )

    st.write("")

    html(
        f"""
        <div style="
            padding:13px 15px;
            background:#FFFAEB;
            border:1px solid #FEDF89;
            border-radius:11px;
            color:#7A2E0B;
            font-size:9px;
            line-height:1.6;
        ">
            <b>Governance note:</b>
            Entity relationships are used for investigation context.
            The independently measured graph ROC-AUC is {GRAPH_AUC:.4f},
            so graph features are not presented as a reliable standalone
            fraud predictor.
        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # ENTITY SEARCH
    # --------------------------------------------------------

    left, right = st.columns(
        [0.85, 1.45]
    )

    with left:

        html(
            """
            <div class="panel">

                <div class="panel-title">
                    Relationship lookup
                </div>

                <div class="panel-subtitle">
                    Select an entity to inspect associated payments.
                </div>
            """
        )

        entity_type = st.selectbox(
            "Entity type",
            [
                "Customer",
                "Merchant",
                "Device",
                "IP address",
            ],
        )

        column_map = {
            "Customer": "customer_id",
            "Merchant": "merchant_id",
            "Device": "device_id",
            "IP address": "ip_id",
        }

        selected_column = column_map[
            entity_type
        ]

        if selected_column not in df.columns:

            st.warning(
                f"{selected_column} is not available."
            )

            st.stop()

        options = (
            df[selected_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_entity = st.selectbox(
            entity_type,
            options[:5000],
        )

        html(
            """
            </div>
            """
        )

    with right:

        related = df[
            df[selected_column].astype(str)
            == str(selected_entity)
        ].copy()

        related_count = len(related)

        high_risk_count = (
            int(
                (
                    related["risk_score"] >= 75
                ).sum()
            )
            if "risk_score" in related.columns
            else 0
        )

        html(
            f"""
            <div class="panel">

                <div class="panel-title">
                    {selected_entity}
                </div>

                <div class="panel-subtitle">
                    Entity activity summary.
                </div>

                <div style="
                    display:flex;
                    gap:12px;
                ">

                    <div style="
                        flex:1;
                        padding:14px;
                        background:#F8FAFC;
                        border-radius:10px;
                    ">

                        <div style="
                            color:#667085;
                            font-size:8px;
                            font-weight:850;
                        ">
                            TRANSACTIONS
                        </div>

                        <div style="
                            color:#101828;
                            font-size:25px;
                            font-weight:900;
                            margin-top:5px;
                        ">
                            {related_count:,}
                        </div>

                    </div>

                    <div style="
                        flex:1;
                        padding:14px;
                        background:#FEF3F2;
                        border-radius:10px;
                    ">

                        <div style="
                            color:#B42318;
                            font-size:8px;
                            font-weight:850;
                        ">
                            HIGH RISK
                        </div>

                        <div style="
                            color:#101828;
                            font-size:25px;
                            font-weight:900;
                            margin-top:5px;
                        ">
                            {high_risk_count:,}
                        </div>

                    </div>

                </div>

            </div>
            """
        )

    st.write("")

    # --------------------------------------------------------
    # RELATED TRANSACTIONS
    # --------------------------------------------------------

    related = related.sort_values(
        "risk_score",
        ascending=False
    ).head(20)

    html(
        """
        <div class="panel">

            <div class="panel-title">
                Related transaction activity
            </div>

            <div class="panel-subtitle">
                Highest-risk relationships first.
            </div>
        """
    )

    related_table = """
    <table class="risk-table">

        <thead>

            <tr>
                <th>Transaction</th>
                <th>Amount</th>
                <th>Risk</th>
                <th>Action</th>
            </tr>

        </thead>

        <tbody>
    """

    for _, row in related.iterrows():

        risk = float(
            get_value(
                row,
                "risk_score",
                0
            )
        )

        action = get_action(row)

        risk_class = (
            "risk-high"
            if risk >= 75
            else
            "risk-medium"
            if risk >= 60
            else
            "risk-low"
        )

        action_class = (
            "pill-review"
            if action == "REVIEW"
            else
            "pill-verify"
            if action == "VERIFY"
            else
            "pill-approve"
        )

        related_table += f"""
        <tr>

            <td>
                <span class="transaction-id">
                    {get_value(row, "transaction_id", "—")}
                </span>
            </td>

            <td>
                {money2(get_value(row, "amount", 0))}
            </td>

            <td>
                <span class="{risk_class}">
                    {risk:.0f}
                </span>
            </td>

            <td>
                <span class="pill {action_class}">
                    {action}
                </span>
            </td>

        </tr>
        """

    related_table += """
        </tbody>
    </table>

    </div>
    """

    html(related_table)


# ============================================================
# FINAL FOOTER
# ============================================================

html(
    """
    <div class="footer">
        RiskGraph AI · AI Risk Manager · Razorpay Buildathon Track 02
        <br>
        Synthetic-data prototype · Temporal holdout evaluation
    </div>
    """
)
