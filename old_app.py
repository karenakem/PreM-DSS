import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import base64

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PreM-DSS | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ==================================================
# VISUAL DESIGN — PreM-DSS
# ==================================================

st.markdown(
    """
    <style>
    /* ---------- App shell ---------- */
    .stApp {
        background: #F6F8FC;
        color: #17233C;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 252, 0.92);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #F0F3F8;
        border-right: 1px solid #E0E6EF;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }

    [data-testid="stSidebar"] h1 {
        color: #102A56;
        font-size: 1.65rem;
        letter-spacing: -0.02em;
    }

    [data-testid="stSidebar"] p {
        color: #667085;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 9px;
        padding: 0.28rem 0.45rem;
        margin-bottom: 0.08rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #E5ECF8;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 {
        color: #102A56;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2.15rem !important;
        margin-bottom: 0.15rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
    }

    h3 {
        font-size: 1.12rem !important;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 3px 12px rgba(16, 42, 86, 0.055);
        min-height: 105px;
    }

    [data-testid="stMetricLabel"] {
        color: #667085;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #102A56;
        font-weight: 750;
    }

    /* ---------- Data / chart surfaces ---------- */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 14px;
        padding: 0.65rem;
        box-shadow: 0 2px 10px rgba(16, 42, 86, 0.035);
    }

    /* ---------- Inputs ---------- */
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        border-radius: 9px !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 9px;
        min-height: 2.7rem;
        font-weight: 650;
        border: 1px solid #2E7DE9;
    }

    .stButton > button[kind="primary"] {
        background: #2E7DE9;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #246AC8;
        border-color: #246AC8;
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border-width: 1px;
    }

    /* ---------- Dividers ---------- */
    hr {
        border-color: #E5EAF2 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* ---------- Custom brand/header ---------- */
    .prem-brand {
        padding: 0.2rem 0 0.65rem 0;
    }

    .prem-brand-title {
        color: #102A56;
        font-size: 1.72rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
    }

    .prem-brand-subtitle {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.35;
        margin-top: 0.35rem;
    }

    .prem-page-header {
        background: linear-gradient(90deg, #FFFFFF 0%, #F1F6FD 100%);
        border: 1px solid #E4EAF3;
        border-radius: 16px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 3px 14px rgba(16, 42, 86, 0.045);
    }

    .prem-page-title {
        color: #102A56;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin-bottom: 0.28rem;
    }

    .prem-page-subtitle {
        color: #667085;
        font-size: 0.98rem;
        line-height: 1.45;
    }

    .prem-eyebrow {
        color: #2E7DE9;
        font-size: 0.75rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    /* ---------- Dashboard KPI cards ---------- */
    .prem-kpi {
        border-radius: 14px;
        padding: 1rem 1.05rem;
        min-height: 112px;
        border: 1px solid rgba(16,42,86,0.06);
        box-shadow: 0 3px 12px rgba(16,42,86,0.04);
    }

    .prem-kpi.blue { background: #EAF5FF; }
    .prem-kpi.green { background: #EAF8F1; }
    .prem-kpi.red { background: #FFF0F0; }
    .prem-kpi.purple { background: #F2EEFF; }

    .prem-kpi-label {
        color: #5F6B7A;
        font-size: 0.83rem;
        font-weight: 650;
        margin-bottom: 0.32rem;
    }

    .prem-kpi-value {
        color: #102A56;
        font-size: 1.75rem;
        line-height: 1.1;
        font-weight: 800;
    }

    .prem-kpi-note {
        color: #667085;
        font-size: 0.76rem;
        margin-top: 0.28rem;
    }

    /* ---------- Small-screen handling ---------- */
    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .prem-page-title {
            font-size: 1.65rem;
        }
    }
    
    /* ---------- Professional sidebar navigation ---------- */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.18rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        position: relative;
        min-height: 42px;
        border-radius: 9px;
        padding: 0.58rem 0.65rem 0.58rem 2.75rem !important;
        margin: 0.02rem 0;
        cursor: pointer;
        transition: background 0.15s ease, box-shadow 0.15s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: #E4ECF8;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: #2E7DE9 !important;
        box-shadow: 0 3px 8px rgba(46,125,233,0.18);
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        color: #26364F !important;
        font-size: 0.91rem !important;
        font-weight: 540 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 650 !important;
    }

    /* Hide Streamlit radio control itself, but keep label clickable */
    [data-testid="stSidebar"] [role="radiogroup"] > label [data-testid="stRadio"] {
        display: none !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child:not(:last-child) {
        display: none !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label::before {
        content: "";
        position: absolute;
        left: 0.88rem;
        top: 50%;
        transform: translateY(-50%);
        width: 18px;
        height: 18px;
        background-color: #26364F;
        -webkit-mask-repeat: no-repeat;
        -webkit-mask-position: center;
        -webkit-mask-size: contain;
        mask-repeat: no-repeat;
        mask-position: center;
        mask-size: contain;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked)::before {
        background-color: #FFFFFF;
    }

    /* 1 Dashboard */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(1)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 3 2.5 11h2.2v10h5.8v-6h3v6h5.8V11h2.2L12 3Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 3 2.5 11h2.2v10h5.8v-6h3v6h5.8V11h2.2L12 3Z'/%3E%3C/svg%3E");
    }

    /* 2 Prediction */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(2)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M4 19h16v2H2V3h2v16Zm2-3.2 4.1-4.1 3 3 5.5-6.2 1.5 1.3-6.9 7.8-3.1-3.1-2.7 2.7L6 15.8Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M4 19h16v2H2V3h2v16Zm2-3.2 4.1-4.1 3 3 5.5-6.2 1.5 1.3-6.9 7.8-3.1-3.1-2.7 2.7L6 15.8Z'/%3E%3C/svg%3E");
    }

    /* 3 Risk */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(3)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 2 1 21h22L12 2Zm1 15h-2v2h2v-2Zm0-7h-2v5h2v-5Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 2 1 21h22L12 2Zm1 15h-2v2h2v-2Zm0-7h-2v5h2v-5Z'/%3E%3C/svg%3E");
    }

    /* 4 Live */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(4)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M2 13h4l2-7 4 13 3-9 2 3h5v2h-6l-.5-.5-2.8 8.4L8 9.5 7.5 15H2v-2Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M2 13h4l2-7 4 13 3-9 2 3h5v2h-6l-.5-.5-2.8 8.4L8 9.5 7.5 15H2v-2Z'/%3E%3C/svg%3E");
    }

    /* 5 Analysis */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(5)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M3 20h18v2H1V2h2v18Zm3-2H4v-6h2v6Zm5 0H8V8h3v10Zm5 0h-3V4h3v14Zm5 0h-3v-8h3v8Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M3 20h18v2H1V2h2v18Zm3-2H4v-6h2v6Zm5 0H8V8h3v10Zm5 0h-3V4h3v14Zm5 0h-3v-8h3v8Z'/%3E%3C/svg%3E");
    }

    /* 6 Validation */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(6)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 2 4 5v6c0 5.1 3.4 9.8 8 11 4.6-1.2 8-5.9 8-11V5l-8-3Zm-1 14-4-4 1.4-1.4L11 13.2l4.6-4.6L17 10l-6 6Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 2 4 5v6c0 5.1 3.4 9.8 8 11 4.6-1.2 8-5.9 8-11V5l-8-3Zm-1 14-4-4 1.4-1.4L11 13.2l4.6-4.6L17 10l-6 6Z'/%3E%3C/svg%3E");
    }

    /* 7 Performance */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(7)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 3a9 9 0 0 0-9 9c0 2.5 1 4.8 2.6 6.4l1.4-1.4A7 7 0 1 1 19 12c0 1.9-.8 3.7-2 5l1.4 1.4A9 9 0 0 0 12 3Zm5 5-6 3 2 2 4-5Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 3a9 9 0 0 0-9 9c0 2.5 1 4.8 2.6 6.4l1.4-1.4A7 7 0 1 1 19 12c0 1.9-.8 3.7-2 5l1.4 1.4A9 9 0 0 0 12 3Zm5 5-6 3 2 2 4-5Z'/%3E%3C/svg%3E");
    }

    /* 8 Data */
    [data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(8)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 2C7 2 3 3.8 3 6v12c0 2.2 4 4 9 4s9-1.8 9-4V6c0-2.2-4-4-9-4Zm0 2c4.3 0 7 .9 7 2s-2.7 2-7 2-7-.9-7-2 2.7-2 7-2Zm0 16c-4.3 0-7-.9-7-2v-2.2c1.7 1 4.2 1.7 7 1.7s5.3-.7 7-1.7V18c0 1.1-2.7 2-7 2Zm0-4.5c-4.3 0-7-.9-7-2v-2.2c1.7 1 4.2 1.7 7 1.7s5.3-.7 7-1.7v2.2c0 1.1-2.7 2-7 2Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M12 2C7 2 3 3.8 3 6v12c0 2.2 4 4 9 4s9-1.8 9-4V6c0-2.2-4-4-9-4Zm0 2c4.3 0 7 .9 7 2s-2.7 2-7 2-7-.9-7-2 2.7-2 7-2Zm0 16c-4.3 0-7-.9-7-2v-2.2c1.7 1 4.2 1.7 7 1.7s5.3-.7 7-1.7V18c0 1.1-2.7 2-7 2Zm0-4.5c-4.3 0-7-.9-7-2v-2.2c1.7 1 4.2 1.7 7 1.7s5.3-.7 7-1.7v2.2c0 1.1-2.7 2-7 2Z'/%3E%3C/svg%3E");
    }

    /* Dashboard hero */
    .prem-hero {
        min-height: 175px;
        border-radius: 16px;
        padding: 1.65rem 1.7rem;
        margin-bottom: 1.25rem;
        border: 1px solid #E4EAF3;
        background-size: cover;
        background-position: center right;
        box-shadow: 0 3px 14px rgba(16,42,86,0.05);
        display: flex;
        align-items: center;
    }

    .prem-hero-content {
        max-width: 58%;
    }

    .prem-hero-kicker {
        color: #2E7DE9;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.32rem;
    }

    .prem-hero-title {
        color: #102A56;
        font-size: 2.45rem;
        font-weight: 850;
        letter-spacing: -0.04em;
        line-height: 1.05;
        margin-bottom: 0.3rem;
    }

    .prem-hero-subtitle {
        color: #102A56;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.38rem;
    }

    .prem-hero-copy {
        color: #5F6B7A;
        font-size: 0.9rem;
        line-height: 1.45;
        max-width: 560px;
    }


    .prem-brand-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .prem-brand-title::before {
        content: "";
        width: 22px;
        height: 22px;
        display: inline-block;
        background-color: #102A56;
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M19.4 13a7.8 7.8 0 0 0 .1-1 7.8 7.8 0 0 0-.1-1l2.1-1.6-2-3.4-2.5 1a7.2 7.2 0 0 0-1.7-1L15 3.3h-4L10.6 6a7.2 7.2 0 0 0-1.7 1l-2.5-1-2 3.4L6.5 11a7.8 7.8 0 0 0-.1 1 7.8 7.8 0 0 0 .1 1l-2.1 1.6 2 3.4 2.5-1a7.2 7.2 0 0 0 1.7 1l.4 2.7h4l.4-2.7a7.2 7.2 0 0 0 1.7-1l2.5 1 2-3.4L19.4 13ZM13 15.5A3.5 3.5 0 1 1 13 8a3.5 3.5 0 0 1 0 7.5Z'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='black' d='M19.4 13a7.8 7.8 0 0 0 .1-1 7.8 7.8 0 0 0-.1-1l2.1-1.6-2-3.4-2.5 1a7.2 7.2 0 0 0-1.7-1L15 3.3h-4L10.6 6a7.2 7.2 0 0 0-1.7 1l-2.5-1-2 3.4L6.5 11a7.8 7.8 0 0 0-.1 1 7.8 7.8 0 0 0 .1 1l-2.1 1.6 2 3.4 2.5-1a7.2 7.2 0 0 0 1.7 1l.4 2.7h4l.4-2.7a7.2 7.2 0 0 0 1.7-1l2.5 1 2-3.4L19.4 13ZM13 15.5A3.5 3.5 0 1 1 13 8a3.5 3.5 0 0 1 0 7.5Z'/%3E%3C/svg%3E");
        -webkit-mask-repeat: no-repeat;
        -webkit-mask-position: center;
        -webkit-mask-size: contain;
        mask-repeat: no-repeat;
        mask-position: center;
        mask-size: contain;
    }


    /* Reliable menu icon treatment */
    [data-testid="stSidebar"] [role="radiogroup"] > label::before {
        display: none !important;
        content: none !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        padding-left: 0.75rem !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        font-family: Arial, "Segoe UI Symbol", sans-serif !important;
        font-size: 0.91rem !important;
        letter-spacing: 0 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: #2E7DE9 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
        color: white !important;
    }

</style>
    """,
    unsafe_allow_html=True
)


def get_banner_data_uri():
    """Load the local factory banner and return a browser-safe data URI."""
    banner_path = Path(__file__).resolve().parent / "prem_dss_factory_banner.png"
    if not banner_path.exists():
        return ""
    encoded = base64.b64encode(banner_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def page_header(title, subtitle, eyebrow="PreM-DSS"):
    st.markdown(
        f"""
        <div class="prem-page-header">
            <div class="prem-eyebrow">{eyebrow}</div>
            <div class="prem-page-title">{title}</div>
            <div class="prem-page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 2. LOAD DATA FROM SQLITE DATABASE
# ==================================================

connection = sqlite3.connect(
    "predictive_maintenance.db"
)

query = """
SELECT *
FROM machine_data
"""

df = pd.read_sql_query(
    query,
    connection
)

connection.close()


# ==================================================
# 3. TRAIN PREDICTION MODEL FOR APPLICATION
# ==================================================

feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[feature_columns].copy()

y = df["Machine failure"]


# One-hot encode product type
X_encoded = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)


# Rename columns to safe feature names
X_encoded = X_encoded.rename(
    columns={
        "Air temperature [K]": "Air_temperature",
        "Process temperature [K]": "Process_temperature",
        "Rotational speed [rpm]": "Rotational_speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool_wear"
    }
)


# Same 90/10 split used in our experiments
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# Train model
model.fit(
    X_train,
    y_train
)


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown(
    """
    <div class="prem-brand">
        <div class="prem-brand-title">PreM-DSS</div>
        <div class="prem-brand-subtitle">
            Predictive Maintenance<br>Decision Support System
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

menu_options = {
    "⌂   Dashboard": "Dashboard",
    "⌁   AI Failure Prediction": "AI Failure Prediction",
    "△   Risk Monitor": "Risk Monitor",
    "⌁   Live Monitoring": "Live Monitoring",
    "▥   Risk Factor Analysis": "Risk Factor Analysis",
    "✓   Validation & Limitations": "Validation & Limitations",
    "◔   Model Performance": "Model Performance",
    "▤   Machine Data": "Machine Data"
}

selected_menu = st.sidebar.radio(
    "Navigation",
    list(menu_options.keys()),
    key="main_navigation"
)

page = menu_options[selected_menu]

st.sidebar.divider()
st.sidebar.caption("AI4I Predictive Maintenance Project")


# ==================================================
# PAGE 1 — DASHBOARD
# ==================================================

if page == "Dashboard":

    banner_uri = get_banner_data_uri()

    if banner_uri:
        hero_background = (
            "linear-gradient(90deg, rgba(255,255,255,0.99) 0%, "
            "rgba(255,255,255,0.96) 37%, rgba(255,255,255,0.54) 60%, "
            "rgba(255,255,255,0.08) 100%), "
            f"url('{banner_uri}')"
        )
    else:
        hero_background = "linear-gradient(90deg, #FFFFFF 0%, #EAF2FC 100%)"

    st.markdown(
        f"""
        <div class="prem-hero" style="background-image:{hero_background};">
            <div class="prem-hero-content">
                <div class="prem-hero-kicker">Predictive Maintenance Decision Support System</div>
                <div class="prem-hero-title">PreM-DSS</div>
                <div class="prem-hero-subtitle">AI-powered maintenance intelligence</div>
                <div class="prem-hero-copy">
                    Use machine data and AI to predict failures, analyze risk,
                    monitor operating conditions, and support maintenance decisions.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ----------------------------------------------
    # SYSTEM OVERVIEW
    # ----------------------------------------------

    total_records = len(df)

    total_failures = int(
        df["Machine failure"].sum()
    )

    normal_operations = (
        total_records - total_failures
    )

    failure_rate = (
        total_failures / total_records
    ) * 100


    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="prem-kpi blue">
            <div class="prem-kpi-label">Machine Records</div>
            <div class="prem-kpi-value">{total_records:,}</div>
            <div class="prem-kpi-note">Total AI4I observations</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class="prem-kpi green">
            <div class="prem-kpi-label">Normal Operations</div>
            <div class="prem-kpi-value">{normal_operations:,}</div>
            <div class="prem-kpi-note">{(normal_operations / total_records) * 100:.2f}% of records</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""<div class="prem-kpi red">
            <div class="prem-kpi-label">Machine Failures</div>
            <div class="prem-kpi-value">{total_failures:,}</div>
            <div class="prem-kpi-note">Observed failure records</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""<div class="prem-kpi purple">
            <div class="prem-kpi-label">Failure Rate</div>
            <div class="prem-kpi-value">{failure_rate:.2f}%</div>
            <div class="prem-kpi-note">Overall dataset rate</div>
            </div>""",
            unsafe_allow_html=True
        )


    # ----------------------------------------------
    # FAILURE ANALYSIS
    # ----------------------------------------------

    st.divider()

    st.subheader("Failure Analysis")

    chart_col1, chart_col2 = st.columns(2)


    with chart_col1:

        st.write(
            "#### Machine Failure Distribution"
        )

        failure_distribution = (
            df["Machine failure"]
            .value_counts()
            .rename(
                index={
                    0: "Normal Operation",
                    1: "Machine Failure"
                }
            )
        )

        st.bar_chart(
            failure_distribution
        )


    with chart_col2:

        st.write(
            "#### Failure Mode Distribution"
        )

        failure_modes = pd.DataFrame(
            {
                "Failure Mode": [
                    "Tool Wear Failure",
                    "Heat Dissipation Failure",
                    "Power Failure",
                    "Overstrain Failure",
                    "Random Failure"
                ],

                "Count": [
                    df["TWF"].sum(),
                    df["HDF"].sum(),
                    df["PWF"].sum(),
                    df["OSF"].sum(),
                    df["RNF"].sum()
                ]
            }
        )

        failure_modes = failure_modes.set_index(
            "Failure Mode"
        )

        st.bar_chart(
            failure_modes
        )


    # ----------------------------------------------
    # OPERATIONAL RISK ANALYSIS
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Operational Risk Analysis"
    )

    analysis_col1, analysis_col2 = st.columns(2)


    with analysis_col1:

        st.write(
            "#### Failure Rate by Product Type"
        )

        product_analysis = (
            df.groupby("Type")
            .agg(
                Total_Records=(
                    "Machine failure",
                    "count"
                ),

                Failures=(
                    "Machine failure",
                    "sum"
                )
            )
            .reset_index()
        )

        product_analysis[
            "Failure Rate (%)"
        ] = (
            product_analysis["Failures"]
            / product_analysis["Total_Records"]
            * 100
        )

        product_analysis = (
            product_analysis.sort_values(
                "Failure Rate (%)",
                ascending=False
            )
        )

        st.bar_chart(
            product_analysis,
            x="Type",
            y="Failure Rate (%)"
        )

        st.dataframe(
            product_analysis,
            width="stretch",
            hide_index=True
        )


    with analysis_col2:

        st.write(
            "#### Average Conditions: Normal vs Failure"
        )

        condition_analysis = (
            df.groupby(
                "Machine failure"
            )[
                [
                    "Air temperature [K]",
                    "Process temperature [K]",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]"
                ]
            ]
            .mean()
            .round(2)
            .reset_index()
        )

        condition_analysis[
            "Machine failure"
        ] = (
            condition_analysis[
                "Machine failure"
            ]
            .replace(
                {
                    0: "Normal",
                    1: "Failure"
                }
            )
        )

        st.dataframe(
            condition_analysis,
            width="stretch",
            hide_index=True
        )


    # ----------------------------------------------
    # KEY OBSERVATIONS
    # ----------------------------------------------

    st.write(
        "#### Key Observations"
    )

    insight_col1, insight_col2, insight_col3 = (
        st.columns(3)
    )


    with insight_col1:

        st.info(
            """
            **Product Type Risk**

            Type L has the highest observed
            failure rate in the dataset at
            approximately **3.92%**.
            """
        )


    with insight_col2:

        st.info(
            """
            **Torque**

            Failure observations have higher
            average torque: approximately
            **50.17 Nm**, compared with
            **39.63 Nm** during normal operation.
            """
        )


    with insight_col3:

        st.info(
            """
            **Tool Wear**

            Failure observations have higher
            average tool wear: approximately
            **143.78 min**, compared with
            **106.69 min** during normal operation.
            """
        )


# ==================================================
# PAGE 2 — AI FAILURE PREDICTION
# ==================================================

elif page == "AI Failure Prediction":

    page_header(
        "AI Failure Prediction",
        "Enter machine operating conditions to estimate failure probability and receive maintenance decision support.",
        "Prediction"
    )

    st.divider()


    st.subheader(
        "Machine Operating Conditions"
    )

    input_col1, input_col2 = st.columns(2)


    with input_col1:

        product_type = st.selectbox(
            "Product Type",
            ["L", "M", "H"]
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=290.0,
            max_value=310.0,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=300.0,
            max_value=320.0,
            value=310.0,
            step=0.1
        )


    with input_col2:

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=10
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.5
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )


    st.divider()


    if st.button(
        "Analyze Machine",
        type="primary"
    ):

        input_data = pd.DataFrame(
            {
                "Air_temperature": [
                    air_temperature
                ],

                "Process_temperature": [
                    process_temperature
                ],

                "Rotational_speed": [
                    rotational_speed
                ],

                "Torque": [
                    torque
                ],

                "Tool_wear": [
                    tool_wear
                ],

                "Type_H": [
                    1 if product_type == "H" else 0
                ],

                "Type_L": [
                    1 if product_type == "L" else 0
                ],

                "Type_M": [
                    1 if product_type == "M" else 0
                ]
            }
        )


        input_data = input_data[
            X_train.columns
        ]


        failure_probability = (
            model.predict_proba(
                input_data
            )[0][1]
        )

        probability_percent = (
            failure_probability * 100
        )


        if probability_percent < 20:

            risk_level = "LOW"

        elif probability_percent < 40:

            risk_level = "MODERATE"

        elif probability_percent < 70:

            risk_level = "HIGH"

        else:

            risk_level = "CRITICAL"


        st.subheader(
            "Prediction Result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )


        with result_col1:

            st.metric(
                "Failure Probability",
                f"{probability_percent:.2f}%"
            )


        with result_col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        st.subheader(
            "Maintenance Decision"
        )


        if failure_probability >= 0.40:

            maintenance_alert = "YES"

            st.error(
                "MAINTENANCE ALERT — The predicted failure "
                "probability has reached or exceeded the "
                "prototype maintenance-alert threshold of 40%. "
                "Inspection and maintenance assessment are "
                "recommended."
            )

        else:

            maintenance_alert = "NO"

            if failure_probability >= 0.20:

                st.warning(
                    "ELEVATED RISK — The machine has not "
                    "reached the 40% maintenance-alert threshold, "
                    "but closer monitoring is recommended."
                )

            else:

                st.success(
                    "LOW RISK — The machine is below the "
                    "prototype maintenance-alert threshold. "
                    "Continue routine monitoring and scheduled "
                    "maintenance."
                )


        st.metric(
            "Maintenance Alert",
            maintenance_alert
        )


        st.caption(
            "Prototype maintenance-alert threshold: 40%. "
            "This threshold was selected from the Random Forest "
            "threshold analysis because it achieved the highest "
            "F1-score among the tested thresholds while improving "
            "failure recall."
        )

        # ------------------------------------------
        # HISTORICAL RISK INDICATORS
        # ------------------------------------------

        st.divider()

        st.subheader(
            "🔎 Historical Risk Indicators"
        )

        st.write(
            "These indicators compare the machine's current "
            "operating conditions with historical failure patterns "
            "observed in the AI4I dataset."
        )

        risk_indicators = []

        # High tool wear
        if tool_wear > 200:

            risk_indicators.append(
                "🔧 **High Tool Wear:** Tool wear is above "
                "200 minutes. Historical records in the 201+ "
                "minute range had a **15.49% observed failure rate**."
            )

        # Very high or very low torque
        if torque > 60:

            risk_indicators.append(
                "⚙️ **Very High Torque:** Torque is above "
                "60 Nm. Historical records above 60 Nm had a "
                "**41.95% observed failure rate**."
            )

        elif torque <= 20:

            risk_indicators.append(
                "⚙️ **Very Low Torque:** Torque is 20 Nm or "
                "lower. Historical records in this range had a "
                "**14.59% observed failure rate**."
            )

        # Very low or very high rotational speed
        if rotational_speed <= 1300:

            risk_indicators.append(
                "🔄 **Low Rotational Speed:** Speed is at or "
                "below 1300 rpm. Historical records in this range "
                "had a **22.69% observed failure rate**."
            )

        elif rotational_speed > 1900:

            risk_indicators.append(
                "🔄 **High Rotational Speed:** Speed is above "
                "1900 rpm. Historical records in this range had "
                "an **8.54% observed failure rate**."
            )

        # Product type
        if product_type == "L":

            risk_indicators.append(
                "🏷️ **Product Type L:** Type L had the highest "
                "historical failure rate among the three product "
                "types at **3.92%**."
            )

        if risk_indicators:

            for indicator in risk_indicators:

                st.warning(
                    indicator
                )

        else:

            st.success(
                "No major historical risk indicators from the "
                "current risk-factor analysis were detected for "
                "these operating conditions."
            )

        st.caption(
            "Historical risk indicators are descriptive associations "
            "from the AI4I dataset. They provide contextual decision "
            "support and should not be interpreted as proof that an "
            "operating condition caused a machine failure."
        )

        with st.expander(
                "View Machine Input"
        ):

            machine_input_summary = pd.DataFrame(
                {
                    "Parameter": [
                        "Product Type",
                        "Air Temperature",
                        "Process Temperature",
                        "Rotational Speed",
                        "Torque",
                        "Tool Wear"
                    ],
                    "Value": [
                        str(product_type),
                        f"{float(air_temperature):.1f} K",
                        f"{float(process_temperature):.1f} K",
                        f"{int(rotational_speed)} rpm",
                        f"{float(torque):.1f} Nm",
                        f"{int(tool_wear)} min"
                    ]
                }
            )

            st.dataframe(
                machine_input_summary,
                width="stretch",
                hide_index=True
            )


# ==================================================
# PAGE 3 — RISK MONITOR
# ==================================================

elif page == "Risk Monitor":

    page_header(
        "Machine Risk Monitor",
        "Monitor machine records, compare predicted risk, and identify observations requiring maintenance attention.",
        "Risk Monitoring"
    )

    st.divider()


    monitor_features = df[
        [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    ].copy()


    monitor_encoded = pd.get_dummies(
        monitor_features,
        columns=["Type"],
        dtype=int
    )


    monitor_encoded = monitor_encoded.rename(
        columns={
            "Air temperature [K]": "Air_temperature",
            "Process temperature [K]": "Process_temperature",
            "Rotational speed [rpm]": "Rotational_speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool_wear"
        }
    )


    monitor_encoded = monitor_encoded[
        X_train.columns
    ]


    monitor_probabilities = (
        model.predict_proba(
            monitor_encoded
        )[:, 1]
    )


    risk_df = df[
        [
            "UDI",
            "Product ID",
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    ].copy()


    risk_df[
        "Failure Probability (%)"
    ] = (
        monitor_probabilities * 100
    ).round(2)


    def assign_risk_level(probability):

        if probability < 20:
            return "LOW"

        elif probability < 40:
            return "MODERATE"

        elif probability < 70:
            return "HIGH"

        else:
            return "CRITICAL"


    risk_df[
        "Risk Level"
    ] = risk_df[
        "Failure Probability (%)"
    ].apply(
        assign_risk_level
    )


    risk_df[
        "Maintenance Alert"
    ] = risk_df[
        "Failure Probability (%)"
    ].apply(
        lambda probability:
        "YES"
        if probability >= 40
        else "NO"
    )


    total_monitored = len(
        risk_df
    )

    maintenance_alerts = (
        risk_df[
            "Maintenance Alert"
        ] == "YES"
    ).sum()

    critical_records = (
        risk_df[
            "Risk Level"
        ] == "CRITICAL"
    ).sum()

    high_risk_records = (
        risk_df[
            "Risk Level"
        ] == "HIGH"
    ).sum()


    st.subheader(
        "Risk Overview"
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(
            "Records Monitored",
            f"{total_monitored:,}"
        )


    with metric2:

        st.metric(
            "Maintenance Alerts",
            f"{maintenance_alerts:,}"
        )


    with metric3:

        st.metric(
            "High Risk",
            f"{high_risk_records:,}"
        )


    with metric4:

        st.metric(
            "Critical Risk",
            f"{critical_records:,}"
        )


    st.divider()

    st.subheader(
        "Risk Distribution"
    )


    risk_order = [
        "LOW",
        "MODERATE",
        "HIGH",
        "CRITICAL"
    ]


    risk_distribution = (
        risk_df[
            "Risk Level"
        ]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0
        )
    )


    st.bar_chart(
        risk_distribution
    )


    st.divider()

    st.subheader(
        "Maintenance Priority"
    )

    priority_col1, priority_col2 = st.columns(2)

    with priority_col1:

        st.error(
            f"""
            **Immediate Priority — CRITICAL**

            **{critical_records:,} records** have a predicted
            failure probability of **70% or higher**.

            These records should receive the highest priority
            for inspection and maintenance assessment.
            """
        )

    with priority_col2:

        st.warning(
            f"""
            **Inspection Priority — HIGH**

            **{high_risk_records:,} records** have a predicted
            failure probability between **40% and 69.99%**.

            These records have crossed the prototype maintenance
            alert threshold and should be reviewed.
            """
        )

    st.caption(
        "Priority categories are prototype decision-support rules "
        "derived from model probabilities. They are not validated "
        "industrial maintenance standards."
    )

    st.divider()

    st.subheader(
        "Machine Risk Records"
    )


    selected_risk = st.selectbox(
        "Filter by Risk Level",
        [
            "All",
            "Maintenance Alerts Only",
            "CRITICAL",
            "HIGH",
            "MODERATE",
            "LOW"
        ]
    )


    if selected_risk == "All":

        filtered_risk_df = (
            risk_df.copy()
        )

    elif selected_risk == "Maintenance Alerts Only":

        filtered_risk_df = risk_df[
            risk_df[
                "Maintenance Alert"
            ] == "YES"
        ].copy()

    else:

        filtered_risk_df = risk_df[
            risk_df[
                "Risk Level"
            ] == selected_risk
        ].copy()


    filtered_risk_df = (
        filtered_risk_df.sort_values(
            "Failure Probability (%)",
            ascending=False
        )
    )


    st.write(
        f"Showing **{len(filtered_risk_df):,}** records."
    )


    st.dataframe(
        filtered_risk_df,
        width="stretch",
        hide_index=True
    )


    st.info(
        """
        **Decision Support**

        Records with a predicted failure probability of
        **40% or higher** generate a prototype maintenance alert.

        Higher-risk records are displayed first so that
        maintenance personnel can prioritize inspection and
        further assessment.

        The alert threshold is based on the project's threshold
        analysis and is intended for prototype decision support,
        not as a validated industrial maintenance standard.
        """
    )


    # ----------------------------------------------
    # INDIVIDUAL MACHINE INSPECTION
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "🔎 Individual Machine Inspection"
    )

    st.write(
        "Select a machine record to review its operating "
        "conditions, predicted failure risk, and recommended "
        "maintenance action."
    )


    inspection_options = risk_df.sort_values(
        "Failure Probability (%)",
        ascending=False
    ).copy()


    inspection_options[
        "Selection Label"
    ] = inspection_options.apply(
        lambda row:
        f"UDI {int(row['UDI'])} | "
        f"{row['Product ID']} | "
        f"{row['Risk Level']} | "
        f"{float(row['Failure Probability (%)']):.2f}%",
        axis=1
    )


    selected_machine = st.selectbox(
        "Select Machine Record",
        inspection_options[
            "Selection Label"
        ].tolist()
    )


    selected_row = inspection_options[
        inspection_options[
            "Selection Label"
        ] == selected_machine
    ].iloc[0]
    selected_failure_probability = float(
        selected_row["Failure Probability (%)"]
    )

    selected_udi = int(
        selected_row["UDI"]
    )

    selected_rotational_speed = int(
        selected_row["Rotational speed [rpm]"]
    )

    selected_torque = float(
        selected_row["Torque [Nm]"]
    )

    selected_tool_wear = int(
        selected_row["Tool wear [min]"]
    )


    st.write(
        "#### Machine Identification"
    )

    id_col1, id_col2, id_col3 = st.columns(3)


    with id_col1:

        st.metric(
            "UDI",
            int(selected_row["UDI"])
        )


    with id_col2:

        st.metric(
            "Product ID",
            selected_row["Product ID"]
        )


    with id_col3:

        st.metric(
            "Product Type",
            selected_row["Type"]
        )


    st.write(
        "#### AI Risk Assessment"
    )

    risk_col1, risk_col2, risk_col3 = st.columns(3)


    with risk_col1:

        st.metric(
            "Failure Probability",
            f"{selected_failure_probability:.2f}%"
        )


    with risk_col2:

        st.metric(
            "Risk Level",
            selected_row["Risk Level"]
        )


    with risk_col3:

        st.metric(
            "Maintenance Alert",
            selected_row["Maintenance Alert"]
        )



    st.write(
        "#### Operating Conditions"
    )


    machine_conditions = pd.DataFrame(
        {
            "Parameter": [
                "Air Temperature",
                "Process Temperature",
                "Rotational Speed",
                "Torque",
                "Tool Wear"
            ],

            "Value": [
                f"{float(selected_row['Air temperature [K]']):.1f} K",
                f"{float(selected_row['Process temperature [K]']):.1f} K",
                f"{selected_rotational_speed} rpm",
                f"{selected_torque:.1f} Nm",
                f"{selected_tool_wear} min"
            ]
        }
    )


    st.dataframe(
        machine_conditions,
        width="stretch",
        hide_index=True
    )


    st.write(
        "#### Recommended Action"
    )


    selected_probability = selected_failure_probability


    if selected_probability >= 70:

        st.error(
            "CRITICAL PRIORITY — This record has a predicted "
            "failure probability of 70% or higher. Prioritize "
            "inspection and maintenance assessment."
        )


    elif selected_probability >= 40:

        st.warning(
            "HIGH PRIORITY — This record has crossed the "
            "40% prototype maintenance-alert threshold. "
            "Inspection and closer monitoring are recommended."
        )


    elif selected_probability >= 20:

        st.warning(
            "MODERATE RISK — This record has not crossed the "
            "maintenance-alert threshold, but closer monitoring "
            "is recommended."
        )


    else:

        st.success(
            "LOW RISK — This record is currently below the "
            "prototype maintenance-alert threshold. Continue "
            "routine monitoring and scheduled maintenance."
        )


    st.caption(
        "The recommendation is generated from the Random Forest "
        "failure probability and prototype decision thresholds. "
        "It should support, not replace, engineering inspection "
        "and maintenance judgment."
    )


# ==================================================
# PAGE 4 — LIVE MONITORING
# ==================================================

elif page == "Live Monitoring":

    page_header(
        "Live Monitoring",
        "Follow the simulated incoming machine-data stream, AI failure probabilities, risk levels, and maintenance alerts.",
        "Simulated Real-Time Monitoring"
    )
    st.info("**Historical stream replay:** AI4I observations are replayed sequentially to demonstrate a real-time monitoring architecture. This is not a connection to live industrial IoT sensors.")
    st.divider()

    try:
        realtime_connection = sqlite3.connect("predictive_maintenance.db")
        realtime_df = pd.read_sql_query("SELECT * FROM realtime_monitoring ORDER BY id ASC", realtime_connection)
        realtime_connection.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        realtime_df = pd.DataFrame()

    if realtime_df.empty:
        st.warning("No monitoring records are available. Run 17_realtime_database.py first.")
    else:
        total_stream_records = len(realtime_df)
        total_stream_alerts = int((realtime_df["maintenance_alert"] == "YES").sum())
        critical_stream_records = int((realtime_df["risk_level"] == "CRITICAL").sum())
        moderate_stream_records = int((realtime_df["risk_level"] == "MODERATE").sum())

        st.subheader("Monitoring Overview")
        a,b,c1,d = st.columns(4)
        a.metric("Records Processed", f"{total_stream_records:,}")
        b.metric("Maintenance Alerts", f"{total_stream_alerts:,}")
        c1.metric("Critical Risk", f"{critical_stream_records:,}")
        d.metric("Moderate Risk", f"{moderate_stream_records:,}")

        st.divider(); st.subheader("Latest Machine Reading")
        latest = realtime_df.iloc[-1]
        a,b,c1 = st.columns(3)
        a.metric("UDI", int(latest["udi"])); b.metric("Product ID", latest["product_id"]); c1.metric("Product Type", latest["product_type"])
        a,b,c1 = st.columns(3)
        a.metric("Failure Probability", f"{float(latest['failure_probability']):.2f}%"); b.metric("Risk Level", latest["risk_level"]); c1.metric("Maintenance Alert", latest["maintenance_alert"])
        conditions=pd.DataFrame({"Parameter":["Timestamp","Air Temperature","Process Temperature","Rotational Speed","Torque","Tool Wear"],"Value":[latest["timestamp"],f"{float(latest['air_temperature']):.1f} K",f"{float(latest['process_temperature']):.1f} K",f"{int(latest['rotational_speed'])} rpm",f"{float(latest['torque']):.1f} Nm",f"{int(latest['tool_wear'])} min"]})
        st.dataframe(conditions,width="stretch",hide_index=True)
        if latest["maintenance_alert"] == "YES": st.error("MAINTENANCE ALERT — The latest record crossed the 40% prototype threshold. Inspection and maintenance assessment are recommended.")
        elif latest["risk_level"] == "MODERATE": st.warning("MODERATE RISK — Below the alert threshold, but closer monitoring is recommended.")
        else: st.success("LOW RISK — Below the prototype maintenance-alert threshold. Continue monitoring.")

        st.divider(); st.subheader("Failure Risk Across Incoming Readings")
        chart=realtime_df[["id","failure_probability"]].rename(columns={"id":"Reading","failure_probability":"Failure Probability (%)"}).set_index("Reading")
        st.line_chart(chart)
        st.caption("AI-estimated failure probability for each stored stream observation. Prototype maintenance-alert threshold: 40%.")

        st.divider(); st.subheader("Stream Risk Distribution")
        dist=realtime_df["risk_level"].value_counts().reindex(["LOW","MODERATE","HIGH","CRITICAL"],fill_value=0)
        st.bar_chart(dist)

        st.divider(); st.subheader("Maintenance Alerts")
        alerts=realtime_df[realtime_df["maintenance_alert"]=="YES"].copy()
        if alerts.empty: st.success("No maintenance alerts have been generated in the current monitoring history.")
        else: st.dataframe(alerts[["timestamp","udi","product_id","product_type","failure_probability","risk_level","maintenance_alert","historical_failure"]].sort_values("failure_probability",ascending=False),width="stretch",hide_index=True)

        st.divider(); st.subheader("Recent Monitoring History")
        recent=realtime_df.sort_values("id",ascending=False).head(20)[["timestamp","udi","product_id","product_type","rotational_speed","torque","tool_wear","failure_probability","risk_level","maintenance_alert","historical_failure"]].copy()
        recent=recent.rename(columns={"timestamp":"Timestamp","udi":"UDI","product_id":"Product ID","product_type":"Type","rotational_speed":"Speed [rpm]","torque":"Torque [Nm]","tool_wear":"Tool Wear [min]","failure_probability":"Failure Probability (%)","risk_level":"Risk Level","maintenance_alert":"Maintenance Alert","historical_failure":"Historical Failure"})
        st.dataframe(recent,width="stretch",hide_index=True)
        st.caption("Historical Failure is retained only for comparison with the original AI4I record; it is not an input to the prediction model.")

        st.divider(); st.subheader("Business Intelligence Interpretation")
        st.write("Incoming observations are transformed into failure probabilities, risk levels, maintenance decisions, and persistent database records. In a real industrial implementation, the replay source could be replaced by machine sensors, an IoT platform, or another operational data source.")
        st.warning("**Prototype limitation:** These are historical AI4I observations replayed as a simulated stream. Because replayed records may include training observations, this page demonstrates the monitoring architecture rather than independent predictive performance.")


# ==================================================
# PAGE 5 — RISK FACTOR ANALYSIS
# ==================================================

elif page == "Risk Factor Analysis":

    page_header(
        "Risk Factor Analysis",
        "Explore historical operating conditions associated with higher machine failure rates in the AI4I dataset.",
        "Historical Analysis"
    )

    st.info(
        """
        This analysis is descriptive. The observed relationships
        show associations between operating conditions and machine
        failures in the dataset. They should not be interpreted as
        proof that a particular operating condition directly caused
        a failure.
        """
    )

    st.divider()


    # ----------------------------------------------
    # LOAD SAVED ANALYSIS RESULTS
    # ----------------------------------------------

    tool_wear_risk = pd.read_csv(
        "tool_wear_risk_analysis.csv"
    )

    torque_risk = pd.read_csv(
        "torque_risk_analysis.csv"
    )

    speed_risk = pd.read_csv(
        "speed_risk_analysis.csv"
    )

    product_risk = pd.read_csv(
        "product_type_risk_analysis.csv"
    )


    # ----------------------------------------------
    # KEY RISK INDICATORS
    # ----------------------------------------------

    st.subheader(
        "Key Historical Risk Indicators"
    )

    indicator1, indicator2, indicator3 = st.columns(3)


    with indicator1:

        st.metric(
            "Tool Wear 201+ min",
            "15.49%",
            help=(
                "Observed machine failure rate among records "
                "with tool wear above 200 minutes."
            )
        )


    with indicator2:

        st.metric(
            "Torque 61+ Nm",
            "41.95%",
            help=(
                "Observed machine failure rate among records "
                "with torque above 60 Nm."
            )
        )


    with indicator3:

        st.metric(
            "Speed ≤1300 rpm",
            "22.69%",
            help=(
                "Observed machine failure rate among records "
                "with rotational speed at or below 1300 rpm."
            )
        )


    st.caption(
        "These percentages are observed historical failure "
        "rates within the AI4I dataset, not predicted "
        "probabilities for an individual machine."
    )


    # ----------------------------------------------
    # TOOL WEAR ANALYSIS
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "1. Tool Wear and Failure Risk"
    )

    st.write(
        "Failure rates remain relatively low across most "
        "tool-wear ranges, but increase substantially once "
        "tool wear exceeds 200 minutes."
    )


    tool_col1, tool_col2 = st.columns(
        [1.4, 1]
    )


    with tool_col1:

        st.bar_chart(
            tool_wear_risk,
            x="Tool Wear Range",
            y="Failure Rate (%)"
        )


    with tool_col2:

        st.dataframe(
            tool_wear_risk,
            width="stretch",
            hide_index=True
        )


    st.warning(
        """
        **Historical insight:** Records with tool wear above
        200 minutes had an observed failure rate of **15.49%**,
        substantially higher than the approximately 2–3%
        observed across the lower tool-wear ranges.
        """
    )


    # ----------------------------------------------
    # TORQUE ANALYSIS
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "2. Torque and Failure Risk"
    )

    st.write(
        "Torque shows a non-linear risk pattern. Failure rates "
        "are higher at both very low and very high torque levels."
    )


    torque_col1, torque_col2 = st.columns(
        [1.4, 1]
    )


    with torque_col1:

        st.bar_chart(
            torque_risk,
            x="Torque Range",
            y="Failure Rate (%)"
        )


    with torque_col2:

        st.dataframe(
            torque_risk,
            width="stretch",
            hide_index=True
        )


    st.warning(
        """
        **Historical insight:** The observed failure rate was
        **14.59%** at 0–20 Nm and increased to **41.95%**
        for records above 60 Nm. The intermediate torque
        ranges had considerably lower failure rates.
        """
    )


    # ----------------------------------------------
    # ROTATIONAL SPEED ANALYSIS
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "3. Rotational Speed and Failure Risk"
    )

    st.write(
        "Rotational speed also shows higher observed failure "
        "rates at the extremes rather than a simple linear pattern."
    )


    speed_col1, speed_col2 = st.columns(
        [1.4, 1]
    )


    with speed_col1:

        st.bar_chart(
            speed_risk,
            x="Speed Range",
            y="Failure Rate (%)"
        )


    with speed_col2:

        st.dataframe(
            speed_risk,
            width="stretch",
            hide_index=True
        )


    st.warning(
        """
        **Historical insight:** Records at or below 1300 rpm
        had an observed failure rate of **22.69%**, while
        records above 1900 rpm had a failure rate of **8.54%**.
        The middle speed ranges showed much lower failure rates.
        """
    )


    # ----------------------------------------------
    # PRODUCT TYPE ANALYSIS
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "4. Product Type and Failure Risk"
    )

    st.write(
        "Failure rates also differ across the three product "
        "quality categories represented in the dataset."
    )


    product_col1, product_col2 = st.columns(
        [1.4, 1]
    )


    with product_col1:

        st.bar_chart(
            product_risk,
            x="Type",
            y="Failure Rate (%)"
        )


    with product_col2:

        st.dataframe(
            product_risk,
            width="stretch",
            hide_index=True
        )


    st.info(
        """
        **Historical insight:** Type L had the highest observed
        failure rate at **3.92%**, followed by Type M at
        **2.77%** and Type H at **2.09%**.
        """
    )


    # ----------------------------------------------
    # BI INTERPRETATION
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Business Intelligence Interpretation"
    )

    st.write(
        """
        The analysis indicates that machine failure risk is not
        associated with only one operating variable or with a
        simple increase in every variable.

        Particularly notable historical risk conditions include
        very high torque, very low rotational speed, high tool
        wear, and extreme operating ranges.

        These insights can support maintenance personnel in
        identifying operating conditions that deserve closer
        attention and complement the machine-learning predictions
        provided elsewhere in the application.
        """
    )


    st.success(
        """
        **Decision-support implication**

        Predictive probabilities identify which machine records
        may require attention, while risk-factor analysis provides
        additional historical context for understanding operating
        conditions associated with machine failures.
        """
    )


# ==================================================
# PAGE 6 — VALIDATION & LIMITATIONS
# ==================================================

elif page == "Validation & Limitations":

    page_header(
        "Validation & Limitations",
        "Evaluate the prototype maintenance-alert workflow using the held-out test set and review its limitations.",
        "Model Validation"
    )

    st.info(
        """
        **Why this page matters**

        The Risk Monitor demonstrates how the application can score
        and prioritize machine records, but it includes records from
        the full historical dataset.

        This validation page instead reports results from the
        **1,000-record held-out test set**, providing a more appropriate
        evaluation of how the model performed on unseen records.
        """
    )

    st.divider()

    # ----------------------------------------------
    # VALIDATION OVERVIEW
    # ----------------------------------------------

    st.subheader(
        "Held-Out Test Validation"
    )

    st.write(
        "The prototype maintenance alert uses the selected "
        "**40% failure-probability threshold**."
    )

    validation_col1, validation_col2, validation_col3, validation_col4 = (
        st.columns(4)
    )

    with validation_col1:

        st.metric(
            "True Negatives",
            "951",
            help="Normal records correctly left without an alert."
        )

    with validation_col2:

        st.metric(
            "False Positives",
            "15",
            help="Normal records that incorrectly generated an alert."
        )

    with validation_col3:

        st.metric(
            "False Negatives",
            "6",
            help="Actual failures that did not generate an alert."
        )

    with validation_col4:

        st.metric(
            "True Positives",
            "28",
            help="Actual failures correctly flagged by the system."
        )

    st.divider()

    # ----------------------------------------------
    # MAINTENANCE-ALERT PERFORMANCE
    # ----------------------------------------------

    st.subheader(
        "Maintenance-Alert Performance"
    )

    performance_col1, performance_col2, performance_col3 = (
        st.columns(3)
    )

    with performance_col1:

        st.metric(
            "Actual Failures",
            "34"
        )

    with performance_col2:

        st.metric(
            "Failures Detected",
            "28 / 34"
        )

    with performance_col3:

        st.metric(
            "Failure Recall",
            "82.35%"
        )

    st.success(
        """
        **Validation result**

        At the 40% prototype maintenance-alert threshold, the system
        detected **28 of the 34 actual failures** in the held-out test
        set, corresponding to **82.35% recall**.

        The system missed **6 actual failures** and generated
        **15 false maintenance alerts**.
        """
    )

    st.caption(
        "These results evaluate the 40% maintenance-alert decision "
        "rule. They are different from the Random Forest metrics shown "
        "at the standard 50% classification threshold."
    )

    st.divider()

    # ----------------------------------------------
    # LOAD DETAILED VALIDATION RESULTS
    # ----------------------------------------------

    st.subheader(
        "Validation Case Examples"
    )

    try:

        validation_results = pd.read_csv(
            "application_validation_results.csv"
        )

        validation_order = [
            "True Positive",
            "False Negative",
            "False Positive",
            "True Negative"
        ]

        selected_validation_case = st.selectbox(
            "Select Validation Outcome",
            validation_order
        )

        case_results = validation_results[
            validation_results[
                "Validation Result"
            ] == selected_validation_case
        ].copy()

        case_results = case_results.sort_values(
            "Failure Probability (%)",
            ascending=False
        )

        st.write(
            f"Showing **{len(case_results):,}** "
            f"{selected_validation_case} records."
        )

        display_columns = [
            "UDI",
            "Product ID",
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Machine failure",
            "Failure Probability (%)",
            "Predicted Alert",
            "Validation Result"
        ]

        st.dataframe(
            case_results[display_columns],
            width="stretch",
            hide_index=True
        )

        if selected_validation_case == "True Positive":

            st.success(
                "These are actual machine failures that the prototype "
                "correctly identified for maintenance attention."
            )

        elif selected_validation_case == "False Negative":

            st.error(
                "These are actual machine failures that the prototype "
                "missed. They demonstrate why the system should support, "
                "not replace, engineering inspection and maintenance "
                "judgment."
            )

        elif selected_validation_case == "False Positive":

            st.warning(
                "These records did not fail but generated maintenance "
                "alerts. In practice, false alarms can increase "
                "inspection effort and maintenance cost."
            )

        else:

            st.info(
                "These are normal records that were correctly left "
                "without a maintenance alert."
            )

    except (FileNotFoundError, OSError):

        st.warning(
            "The detailed validation file "
            "'application_validation_results.csv' could not be loaded. "
            "Run 14_application_validation.py to create it."
        )

    st.divider()

    # ----------------------------------------------
    # LIMITATIONS
    # ----------------------------------------------

    st.subheader(
        "Prototype Limitations"
    )

    st.warning(
        """
        **1. Some failures are still missed**

        At the selected 40% threshold, **6 of 34 actual failures**
        in the held-out test set did not generate a maintenance alert.
        """
    )

    st.warning(
        """
        **2. False alarms still occur**

        The prototype generated **15 false positive alerts** in the
        held-out test set. A real maintenance system would need to
        consider the cost of unnecessary inspection against the cost
        of missed failures.
        """
    )

    st.warning(
        """
        **3. The dataset is synthetic**

        AI4I 2020 is a synthetic predictive-maintenance dataset.
        Performance on this dataset does not establish performance
        on a real manufacturing plant or a different machine system.
        """
    )

    st.warning(
        """
        **4. The 40% threshold is experimental**

        The threshold was selected from the current project analysis
        because it produced the highest F1-score among the tested
        thresholds while improving failure recall. It is not an
        industrial maintenance standard.
        """
    )

    st.warning(
        """
        **5. Real deployment requires further validation**

        Before operational use, the model and alert threshold should
        be evaluated using additional unseen real-world data,
        maintenance costs, equipment-specific requirements, and
        engineering expertise.
        """
    )

    st.divider()

    st.subheader(
        "Business Intelligence Interpretation"
    )

    st.write(
        """
        The validation results demonstrate the trade-off involved in
        predictive maintenance decision support. Lowering the alert
        threshold from the conventional 50% level to the prototype
        40% level increased the number of actual failures detected,
        but also increased false alarms.

        Therefore, the application should not be interpreted as an
        autonomous maintenance decision maker. Its purpose is to
        transform machine operating data and model predictions into
        information that can help maintenance personnel prioritize
        inspection and make more informed decisions.
        """
    )

    st.success(
        """
        **Prototype conclusion**

        The application combines machine-learning prediction,
        threshold-based maintenance alerts, historical risk analysis,
        database-backed machine records, fleet-level monitoring, and
        held-out validation to support predictive-maintenance
        decision making.
        """
    )


# ==================================================
# PAGE 7 — MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    page_header(
        "Model Performance",
        "Compare the machine-failure prediction models evaluated using the same 90/10 train-test split.",
        "AI Evaluation"
    )

    st.divider()


    # ----------------------------------------------
    # MODEL COMPARISON
    # ----------------------------------------------

    st.subheader(
        "Model Comparison"
    )


    model_comparison = pd.DataFrame(
        {
            "Model": [
                "Decision Tree",
                "LightGBM",
                "Class-Weighted LightGBM",
                "Interpretable LightGBM",
                "Random Forest"
            ],

            "Accuracy (%)": [
                96.10,
                97.60,
                95.00,
                98.90,
                98.00
            ],

            "Precision (%)": [
                42.86,
                75.00,
                36.21,
                89.66,
                73.33
            ],

            "Recall (%)": [
                44.12,
                44.12,
                61.76,
                76.47,
                64.71
            ],

            "F1-score (%)": [
                43.48,
                55.56,
                45.65,
                82.54,
                68.75
            ]
        }
    )


    st.dataframe(
        model_comparison,
        width="stretch",
        hide_index=True
    )


    # ----------------------------------------------
    # DEPLOYED MODEL
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Deployed Prediction Model"
    )


    st.info(
        """
        **Random Forest** is currently used as the prediction
        engine in the prototype application.

        The interpretable LightGBM experiment achieved the
        strongest overall evaluation results, but Random Forest
        is used for the live Streamlit prototype because of
        runtime compatibility with the current application
        environment.
        """
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(
            "Accuracy",
            "98.00%"
        )


    with metric2:

        st.metric(
            "Precision",
            "73.33%"
        )


    with metric3:

        st.metric(
            "Recall",
            "64.71%"
        )


    with metric4:

        st.metric(
            "F1-score",
            "68.75%"
        )


    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Random Forest Confusion Matrix"
    )


    confusion_results = pd.DataFrame(
        {
            "Measure": [
                "True Negatives",
                "False Positives",
                "False Negatives",
                "True Positives"
            ],

            "Count": [
                958,
                8,
                12,
                22
            ],

            "Meaning": [
                "Normal machines correctly identified",
                "Normal machines incorrectly flagged as failures",
                "Actual failures missed by the model",
                "Actual failures correctly detected"
            ]
        }
    )


    st.dataframe(
        confusion_results,
        width="stretch",
        hide_index=True
    )


    st.warning(
        """
        At the standard 50% classification threshold,
        the test set contained 34 actual machine failures.
        Random Forest correctly detected 22 and missed 12.

        Because missed failures are important in predictive
        maintenance, threshold analysis was performed to improve
        failure detection.
        """
    )


    # ----------------------------------------------
    # THRESHOLD ANALYSIS
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Maintenance Alert Threshold Analysis"
    )

    st.write(
        "Different probability thresholds were tested to examine "
        "the trade-off between detecting actual failures and "
        "generating false maintenance alarms."
    )


    threshold_results = pd.DataFrame(
        {
            "Threshold (%)": [
                50,
                40,
                30,
                20,
                10
            ],

            "Failures Detected": [
                22,
                28,
                30,
                30,
                31
            ],

            "Failures Missed": [
                12,
                6,
                4,
                4,
                3
            ],

            "False Alarms": [
                8,
                15,
                30,
                57,
                93
            ],

            "Precision (%)": [
                73.33,
                65.12,
                50.00,
                34.48,
                25.00
            ],

            "Recall (%)": [
                64.71,
                82.35,
                88.24,
                88.24,
                91.18
            ],

            "F1-score (%)": [
                68.75,
                72.73,
                63.83,
                49.59,
                39.24
            ]
        }
    )


    st.dataframe(
        threshold_results,
        width="stretch",
        hide_index=True
    )


    st.success(
        """
        **Selected Prototype Threshold: 40%**

        The 40% threshold achieved the highest F1-score
        (**72.73%**) among the tested thresholds.

        It detected **28 of 34 actual failures**, giving
        **82.35% recall**, while generating **15 false alarms**.

        Therefore, 40% was selected as the prototype
        maintenance-alert threshold.
        """
    )


    st.caption(
        "The 40% threshold is an experimental prototype setting, "
        "not a validated industrial maintenance standard. "
        "A real deployment would require validation using "
        "additional unseen operational data and maintenance-cost "
        "information."
    )


    # ----------------------------------------------
    # LIGHTGBM FEATURE IMPORTANCE
    # ----------------------------------------------

    st.divider()

    st.subheader(
        "Additional LightGBM Analysis"
    )

    st.write(
        "The interpretable LightGBM experiment achieved the "
        "strongest overall performance. Its feature-importance "
        "analysis identified Torque, Tool Wear, and Rotational "
        "Speed as the most influential predictive variables."
    )


    try:

        st.image(
            "lightgbm_feature_importance.png",
            caption=(
                "Feature Importance from the "
                "Interpretable LightGBM Experiment"
            ),
            width="stretch"
        )

    except (FileNotFoundError, OSError):

        st.warning(
            "Feature importance image could not be loaded."
        )


# ==================================================
# PAGE 8 — MACHINE DATA
# ==================================================

elif page == "Machine Data":

    page_header(
        "Machine Data",
        "Explore machine operating records retrieved from the SQLite database.",
        "Database"
    )


    st.metric(
        "Database Records",
        f"{len(df):,}"
    )


    st.divider()


    selected_type = st.selectbox(
        "Filter by Product Type",
        [
            "All",
            "L",
            "M",
            "H"
        ]
    )


    if selected_type == "All":

        filtered_df = df

    else:

        filtered_df = df[
            df["Type"] == selected_type
        ]


    st.write(
        f"Showing **{len(filtered_df):,}** records."
    )


    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True
    )
