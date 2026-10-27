import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import base64
from datetime import datetime, timezone

APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "predictive_maintenance.db"

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

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

connection = sqlite3.connect(DATABASE_PATH)

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

# Create LightGBM model
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    class_weight="balanced",
    verbosity=-1,
    n_jobs=-1
)

# Train model
model.fit(
    X_train,
    y_train
)
# ==========================================
# LIGHTGBM MODEL VALIDATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Predict failure probabilities on unseen test data
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the standard 50% threshold
y_pred_50 = (y_prob >= 0.50).astype(int)

print("\n===== LIGHTGBM MODEL VALIDATION =====")

print("Accuracy:", accuracy_score(y_test, y_pred_50))
print("Precision:", precision_score(y_test, y_pred_50, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_50, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred_50, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_50))

# Evaluate our existing 40% maintenance-alert threshold
y_pred_40 = (y_prob >= 0.40).astype(int)

print("\n===== LIGHTGBM AT 40% ALERT THRESHOLD =====")
print("Precision:", precision_score(y_test, y_pred_40, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_40, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred_40, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_40))

# ==========================================
# LIGHTGBM THRESHOLD COMPARISON
# ==========================================

print("\n===== LIGHTGBM THRESHOLD COMPARISON =====")

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    predictions = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(
        f"Threshold: {threshold:.0%} | "
        f"Detected: {tp} | "
        f"Missed: {fn} | "
        f"False Alarms: {fp} | "
        f"Precision: {precision:.2%} | "
        f"Recall: {recall:.2%} | "
        f"F1: {f1:.2%}"
    )
# Shared prototype alert policy and dynamically computed evaluation.
# The 60% setting is exploratory; a separate validation set is needed
# before treating the threshold choice as independently validated.
ALERT_THRESHOLD = 0.60


def risk_category(probability_percent):
    if probability_percent < 20:
        return "LOW"
    if probability_percent < 40:
        return "MODERATE"
    if probability_percent < 70:
        return "HIGH"
    return "CRITICAL"


def evaluation_at(threshold):
    predicted = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    return {"threshold": threshold, "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0)}


alert_evaluation = evaluation_at(ALERT_THRESHOLD)
standard_evaluation = evaluation_at(0.50)
threshold_evaluations = [evaluation_at(t) for t in
                         [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]]

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
    "Dashboard": "Dashboard",
    "AI Failure Prediction": "AI Failure Prediction",
    "Risk Monitor": "Risk Monitor",
    "Live Monitoring": "Live Monitoring",
    "Risk Factor Analysis": "Risk Factor Analysis",
    "Validation & Limitations": "Validation & Limitations",
    "Model Performance": "Model Performance",
    "Machine Data": "Machine Data"
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

        condition_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]

        condition_analysis = (
            df[["Machine failure"] + condition_columns]
            .groupby("Machine failure")
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

        if failure_probability >= ALERT_THRESHOLD:

            maintenance_alert = "YES"

            st.error(
                "MAINTENANCE ALERT — The predicted failure "
                "probability has reached or exceeded the "
                "provisional maintenance-alert threshold of 60%. "
                "Inspection and maintenance assessment are "
                "recommended."
            )

        else:

            maintenance_alert = "NO"

            if failure_probability >= 0.20:

                st.warning(
                    "ELEVATED RISK — The machine has not "
                    "reached the 60% maintenance-alert threshold, "
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
            f"Provisional maintenance-alert threshold: {ALERT_THRESHOLD:.0%}. "
            f"In an exploratory test-set comparison, it detected "
            f"{alert_evaluation['tp']} of {alert_evaluation['tp'] + alert_evaluation['fn']} "
            f"failures with {alert_evaluation['fp']} false alarms. "
            "Threshold selection requires separate validation before deployment."
        )

        # ------------------------------------------
        # INDIVIDUAL LIGHTGBM PREDICTION EXPLANATION
        # ------------------------------------------

        st.divider()
        st.subheader("How LightGBM Reached This Prediction")

        st.write(
            "This explanation shows how each machine operating input "
            "influenced the model's predicted failure risk. "
            "It explains the AI prediction, not the physical cause of failure."
        )

        try:
            # Calculate SHAP-style contributions using LightGBM.
            contributions = model.booster_.predict(
                input_data,
                pred_contrib=True
            )

            contribution_row = contributions[0]

            feature_effects = pd.Series(
                contribution_row[:-1],
                index=X_train.columns,
                dtype=float
            )

            baseline_score = float(contribution_row[-1])

            # Combine the three product-type encoded features.
            readable_effects = pd.Series({
                "Product Type": feature_effects[
                    ["Type_H", "Type_L", "Type_M"]
                ].sum(),

                "Air Temperature": feature_effects[
                    "Air_temperature"
                ],

                "Process Temperature": feature_effects[
                    "Process_temperature"
                ],

                "Rotational Speed": feature_effects[
                    "Rotational_speed"
                ],

                "Torque": feature_effects[
                    "Torque"
                ],

                "Tool Wear": feature_effects[
                    "Tool_wear"
                ]
            })

            explanation_table = pd.DataFrame({
                "Operating Input": readable_effects.index,
                "Contribution (log-odds)": readable_effects.values
            })

            explanation_table["Effect on Predicted Risk"] = (
                explanation_table["Contribution (log-odds)"].apply(
                    lambda value:
                    "Increases" if value > 0.000001
                    else "Decreases" if value < -0.000001
                    else "Minimal"
                )
            )

            explanation_table["Absolute Impact"] = (
                explanation_table["Contribution (log-odds)"].abs()
            )

            explanation_table = explanation_table.sort_values(
                "Absolute Impact",
                ascending=False
            )

            st.write("#### Which Inputs Influenced the Prediction?")

            st.bar_chart(
                explanation_table.set_index("Operating Input")[
                    ["Contribution (log-odds)"]
                ]
            )

            st.dataframe(
                explanation_table.drop(
                    columns="Absolute Impact"
                ).round(4),
                width="stretch",
                hide_index=True
            )

            leading_input = explanation_table.iloc[0]

            st.info(
                f"**Largest model influence:** "
                f"{leading_input['Operating Input']} "
                f"({leading_input['Effect on Predicted Risk'].lower()} "
                "the predicted risk relative to the model baseline)."
            )

            with st.expander("How to Interpret This Explanation"):

                st.write(
                    "LightGBM starts with a baseline model score "
                    "and adds the contribution of each input. "
                    "Positive contributions increase predicted failure "
                    "risk, while negative contributions decrease it."
                )

                st.write(
                    "The contribution values are measured in "
                    "**log-odds**, not percentages. "
                    "They describe how the trained model behaves."
                )

                st.write(
                    f"Baseline score: {baseline_score:.4f}"
                )

                st.write(
                    f"Sum of contributions: "
                    f"{readable_effects.sum():+.4f}"
                )

                st.write(
                    f"Final raw model score: "
                    f"{baseline_score + readable_effects.sum():.4f}"
                )

                st.caption(
                    "These explanations do not prove causation. "
                    "A high predicted failure probability does not "
                    "guarantee that a machine will fail."
                )

        except (AttributeError, ValueError, TypeError, KeyError, IndexError) as exc:
            st.warning(
                "The prediction was calculated, but its individual "
                f"feature contributions could not be displayed: {exc}"
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

        return risk_category(probability)


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
        if probability >= ALERT_THRESHOLD * 100
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

            These records warrant closer monitoring; those at or above
            the 60% alert threshold generate a maintenance alert.
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
        **60% or higher** generate a prototype maintenance alert.

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


    elif selected_probability >= 60:

        st.warning(
            "MAINTENANCE ALERT — This record has crossed the "
            "60% provisional maintenance-alert threshold. "
            "Inspection and closer monitoring are recommended."
        )

    elif selected_probability >= 40:

        st.warning(
            "HIGH RISK — Below the 60% maintenance-alert threshold. "
            "Closer monitoring is recommended."
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
        "The maintenance recommendation is based on the LightGBM "
        "predicted failure probability and the application's "
        "prototype decision rules, including a provisional 60% "
        "maintenance-alert threshold. The recommendation is intended "
        "to support, not replace, engineering inspection and "
        "maintenance judgment."
    )


# ==================================================
# PAGE 4 — LIVE MONITORING
# ==================================================

elif page == "Live Monitoring":
    page_header(
        "Simulated Factory Control Room",
        "Automatic historical observation replay with LightGBM risk predictions and engineer review.",
        "Factory Simulation"
    )
    st.info(
        "SIMULATION ONLY — These are historical machine observations, not live factory "
        "sensors or verified time-series measurements of one machine. Each replay step "
        "represents a different observation, and the time axis is simulation step number."
    )

    # Prefer the existing historical monitoring records; fall back to the AI4I table
    # if the monitoring table is missing. Neither source is live sensor telemetry.
    try:
        with sqlite3.connect(DATABASE_PATH) as replay_connection:
            replay_df = pd.read_sql_query(
                "SELECT * FROM realtime_monitoring ORDER BY id ASC", replay_connection
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        replay_df = pd.DataFrame()

    required_replay_columns = [
        "product_type", "air_temperature", "process_temperature",
        "rotational_speed", "torque", "tool_wear"
    ]
    if replay_df.empty or not all(c in replay_df.columns for c in required_replay_columns):
        replay_df = df[[
            "UDI", "Product ID", "Type", "Air temperature [K]",
            "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]"
        ]].rename(columns={
            "UDI": "id", "Product ID": "product_id", "Type": "product_type",
            "Air temperature [K]": "air_temperature",
            "Process temperature [K]": "process_temperature",
            "Rotational speed [rpm]": "rotational_speed",
            "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear"
        }).copy()
        st.caption("Replay source: AI4I machine_data table (historical observations).")
    else:
        st.caption("Replay source: realtime_monitoring SQLite table (stored historical observations).")

    replay_df = replay_df.dropna(subset=required_replay_columns).reset_index(drop=True)
    if replay_df.empty:
        st.error("No complete historical observations are available for simulation.")
    else:
        # Match exactly the feature names and ordering used during model training.
        replay_features = pd.DataFrame({
            "Air_temperature": replay_df["air_temperature"].astype(float),
            "Process_temperature": replay_df["process_temperature"].astype(float),
            "Rotational_speed": replay_df["rotational_speed"].astype(float),
            "Torque": replay_df["torque"].astype(float),
            "Tool_wear": replay_df["tool_wear"].astype(float),
            "Type_H": (replay_df["product_type"] == "H").astype(int),
            "Type_L": (replay_df["product_type"] == "L").astype(int),
            "Type_M": (replay_df["product_type"] == "M").astype(int),
        })[X_train.columns]
        replay_probabilities = model.predict_proba(replay_features)[:, 1]
        replay_df["failure_probability"] = replay_probabilities * 100
        replay_df["risk_level"] = [risk_category(p) for p in replay_df["failure_probability"]]
        replay_df["maintenance_alert"] = [
            "YES" if p >= ALERT_THRESHOLD else "NO" for p in replay_probabilities
        ]

        # All state belongs to the current Streamlit browser session.
        if "factory_step" not in st.session_state:
            st.session_state.factory_step = 0
        if "factory_running" not in st.session_state:
            st.session_state.factory_running = False
        # Create a separate review table; never modify historical machine records.
        with sqlite3.connect(DATABASE_PATH) as review_db:
            review_db.execute("""
                CREATE TABLE IF NOT EXISTS engineer_decisions (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reviewed_at_utc TEXT NOT NULL,
                    replay_step INTEGER NOT NULL,
                    product_id TEXT NOT NULL,
                    failure_probability_percent REAL NOT NULL,
                    maintenance_alert TEXT NOT NULL,
                    engineer_decision TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
            """)
        st.session_state.factory_step = min(
            st.session_state.factory_step, len(replay_df) - 1
        )

        st.subheader("Simulation Controls")
        control1, control2, control3, control4 = st.columns([1, 1, 1, 2])
        with control1:
            if st.button("▶ Start", use_container_width=True, type="primary"):
                if st.session_state.factory_step >= len(replay_df) - 1:
                    st.session_state.factory_step = 0
                st.session_state.factory_running = True
                st.session_state.factory_skip_first_tick = True
        with control2:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.factory_running = False
        with control3:
            if st.button("↺ Restart", use_container_width=True):
                st.session_state.factory_running = False
                st.session_state.factory_step = 0
                st.session_state.factory_skip_first_tick = False
        with control4:
            speed_seconds = st.selectbox(
                "Seconds per observation", [2, 3, 5], index=1,
                help="Automatic replay speed; one historical observation per interval."
            )

        # Prevent Start from skipping the first observation on its own rerun.
        # Subsequent timed fragment reruns advance the replay.
        if "factory_skip_first_tick" not in st.session_state:
            st.session_state.factory_skip_first_tick = False


        # Streamlit fragments rerun this part automatically without a button click.
        # A stopped simulation still redraws but never advances its observation index.
        @st.fragment(run_every=f"{speed_seconds}s")
        def factory_replay_panel():
            if st.session_state.factory_running:
                if st.session_state.factory_skip_first_tick:
                    st.session_state.factory_skip_first_tick = False
                elif st.session_state.factory_step < len(replay_df) - 1:
                    st.session_state.factory_step += 1
                else:
                    st.session_state.factory_running = False

            position = st.session_state.factory_step
            current = replay_df.iloc[position]
            visible = replay_df.iloc[:position + 1].copy()
            status = "RUNNING" if st.session_state.factory_running else "PAUSED"
            st.write(
                f"**Simulation status:** {status}  |  "
                f"**Observation:** {position + 1:,} / {len(replay_df):,}  |  "
                f"**Alert threshold:** {ALERT_THRESHOLD:.0%}"
            )
            st.progress((position + 1) / len(replay_df))

            st.subheader("Current Machine Observation")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Product ID", str(current.get("product_id", current.get("id", "—"))))
            m2.metric("Product Type", str(current["product_type"]))
            m3.metric("Failure Probability", f"{current['failure_probability']:.2f}%")
            m4.metric("Risk Level", str(current["risk_level"]))

            s1, s2, s3 = st.columns(3)
            s1.metric("Air Temperature", f"{current['air_temperature']:.1f} K")
            s2.metric("Process Temperature", f"{current['process_temperature']:.1f} K")
            s3.metric("Rotational Speed", f"{current['rotational_speed']:.0f} rpm")
            s4, s5 = st.columns(2)
            s4.metric("Torque", f"{current['torque']:.1f} Nm")
            s5.metric("Tool Wear", f"{current['tool_wear']:.0f} min")

            if current["maintenance_alert"] == "YES":
                st.error(
                    "🚨 MAINTENANCE ALERT — LightGBM predicted failure probability "
                    "has reached the provisional 60% threshold. Engineer review recommended."
                )
            elif current["failure_probability"] >= 20:
                st.warning("Elevated predicted risk — continue monitoring and assess operating conditions.")
            else:
                st.success("No maintenance alert for this observation.")

            st.subheader("Monitoring Charts (Replay Step, Not Actual Time)")
            chart_history = visible.tail(100).copy()
            chart_history.index = range(position + 2 - len(chart_history), position + 2)
            chart_history.index.name = "Replay Step"
            st.line_chart(chart_history[["failure_probability"]].rename(
                columns={"failure_probability": "Failure Probability (%)"}
            ))
            left_chart, right_chart = st.columns(2)
            with left_chart:
                st.line_chart(chart_history[["air_temperature", "process_temperature"]].rename(
                    columns={"air_temperature": "Air Temp (K)",
                             "process_temperature": "Process Temp (K)"}
                ))
                st.line_chart(chart_history[["rotational_speed"]].rename(
                    columns={"rotational_speed": "Rotational Speed (rpm)"}
                ))
            with right_chart:
                st.line_chart(chart_history[["torque"]].rename(columns={"torque": "Torque (Nm)"}))
                st.line_chart(chart_history[["tool_wear"]].rename(
                    columns={"tool_wear": "Tool Wear (min)"}
                ))

            st.subheader("Maintenance Engineer Review")
            st.caption(
                "The AI recommends inspection; an engineer makes the final decision. "
                "Pause the simulation before submitting a review."
            )
            with st.form("factory_engineer_review"):
                engineer_decision = st.selectbox(
                    "Engineer decision",
                    ["Inspect machine", "Continue monitoring", "False alarm suspected",
                     "Schedule maintenance"]
                )
                engineer_note = st.text_area("Reason / inspection note")
                review_submitted = st.form_submit_button("Record Engineer Decision")
            if review_submitted:
                if st.session_state.factory_running:
                    st.warning("Pause the simulation first, then submit your decision.")
                elif not engineer_note.strip():
                    st.warning("Enter a reason before recording the decision.")
                else:
                    try:
                        with sqlite3.connect(DATABASE_PATH) as review_db:
                            review_db.execute("""
                                INSERT INTO engineer_decisions
                                (reviewed_at_utc, replay_step, product_id,
                                 failure_probability_percent, maintenance_alert,
                                 engineer_decision, reason)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                position + 1,
                                str(current.get("product_id", current.get("id", "—"))),
                                round(float(current["failure_probability"]), 2),
                                str(current["maintenance_alert"]),
                                engineer_decision,
                                engineer_note.strip()
                            ))
                        st.success("Engineer decision saved to the SQLite database.")
                    except sqlite3.Error as exc:
                        st.error(f"Could not save the engineer decision: {exc}")

            st.subheader("Recent Replay History")
            st.dataframe(
                visible.tail(15).iloc[::-1][[
                    "product_type", "air_temperature", "process_temperature",
                    "rotational_speed", "torque", "tool_wear",
                    "failure_probability", "risk_level", "maintenance_alert"
                ]].round(2),
                width="stretch", hide_index=True
            )
            try:
                with sqlite3.connect(DATABASE_PATH) as review_db:
                    review_table = pd.read_sql_query("""
                        SELECT review_id AS "Review ID",
                               reviewed_at_utc AS "Reviewed At (UTC)",
                               replay_step AS "Replay Step",
                               product_id AS "Product ID",
                               failure_probability_percent AS "Failure Probability (%)",
                               maintenance_alert AS "Maintenance Alert",
                               engineer_decision AS "Engineer Decision",
                               reason AS "Reason"
                        FROM engineer_decisions
                        ORDER BY review_id DESC
                    """, review_db)
                if not review_table.empty:
                    st.subheader("Engineer Decision Log (Saved in SQLite)")
                    st.dataframe(review_table, width="stretch", hide_index=True)
                    st.download_button(
                        "Download Engineer Decisions (CSV)",
                        review_table.to_csv(index=False).encode("utf-8"),
                        file_name="prem_dss_engineer_decisions.csv", mime="text/csv"
                    )
            except (sqlite3.Error, pd.errors.DatabaseError) as exc:
                st.error(f"Could not load engineer decision history: {exc}")
            st.caption(
                "LightGBM scores are computed from the stored observations and displayed "
                "as the replay advances. "
                "The 60% alert threshold is exploratory, not independently validated. "
                "This replay is a demonstration, not real-time sensor monitoring or "
                "independent evaluation of model performance. Engineer decisions "
                "are saved in the local SQLite engineer_decisions table."
            )


        factory_replay_panel()


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
        APP_DIR / "tool_wear_risk_analysis.csv"
    )

    torque_risk = pd.read_csv(
        APP_DIR / "torque_risk_analysis.csv"
    )

    speed_risk = pd.read_csv(
        APP_DIR / "speed_risk_analysis.csv"
    )

    product_risk = pd.read_csv(
        APP_DIR / "product_type_risk_analysis.csv"
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
    page_header("Validation & Limitations",
                "Held-out test performance of the current LightGBM model and provisional 60% alert rule.",
                "Model Validation")
    st.info("The 1,000-record test set was not used to fit the model. "
            "However, comparing thresholds on this same test set is exploratory; "
            "a separate validation set is required for unbiased threshold selection.")
    st.subheader("Maintenance Alert Evaluation — 60% Threshold")
    a, b, c, d = st.columns(4)
    a.metric("True Negatives", alert_evaluation["tn"])
    b.metric("False Positives", alert_evaluation["fp"])
    c.metric("False Negatives", alert_evaluation["fn"])
    d.metric("True Positives", alert_evaluation["tp"])
    a, b, c = st.columns(3)
    a.metric("Failures Detected", f"{alert_evaluation['tp']} / {alert_evaluation['tp'] + alert_evaluation['fn']}")
    b.metric("Recall", f"{alert_evaluation['recall']:.2%}")
    c.metric("F1-score", f"{alert_evaluation['f1']:.2%}")
    st.subheader("Validation Case Examples")
    validation_results = df.loc[X_test.index, ["UDI", "Product ID", "Type",
                                               "Air temperature [K]", "Process temperature [K]",
                                               "Rotational speed [rpm]",
                                               "Torque [Nm]", "Tool wear [min]", "Machine failure"]].copy()
    validation_results["Failure Probability (%)"] = (y_prob * 100).round(2)
    validation_results["Predicted Alert"] = ["YES" if p >= ALERT_THRESHOLD else "NO" for p in y_prob]
    actual = y_test.to_numpy()
    predicted = (y_prob >= ALERT_THRESHOLD).astype(int)
    validation_results["Validation Result"] = [
        "True Positive" if a == 1 and p == 1 else
        "False Negative" if a == 1 else
        "False Positive" if p == 1 else "True Negative"
        for a, p in zip(actual, predicted)]
    case_type = st.selectbox("Select Validation Outcome",
                             ["True Positive", "False Negative", "False Positive", "True Negative"])
    cases = validation_results[validation_results["Validation Result"] == case_type]
    st.write(f"Showing **{len(cases)}** {case_type} records.")
    st.dataframe(cases.sort_values("Failure Probability (%)", ascending=False),
                 width="stretch", hide_index=True)
    st.subheader("Prototype Limitations")
    st.warning(f"The provisional 60% rule missed {alert_evaluation['fn']} actual failures "
               f"and generated {alert_evaluation['fp']} false alarms in this test set.")
    st.warning("AI4I 2020 is a synthetic dataset. These results do not demonstrate "
               "reliability on real industrial equipment or live sensor data.")
    st.warning("The threshold was compared using the test set, so its apparent "
               "performance may be optimistic. Choose thresholds on validation data "
               "and reserve an untouched test set for final evaluation.")
    st.info("AI predictions are decision-support signals, not automatic maintenance orders. "
            "A qualified engineer should review alerts before acting.")


# ==================================================
# PAGE 7 — MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":
    page_header("Model Performance",
                "Metrics computed from the currently deployed LightGBM model.",
                "AI Evaluation")
    st.subheader("Deployed Prediction Model: LightGBM")
    st.info("The application uses a class-weighted LightGBM classifier. "
            "The metrics below are calculated from the current 90/10 split, "
            "not copied from earlier Random Forest experiments.")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy (50%)", f"{standard_evaluation['accuracy']:.2%}")
    b.metric("Precision (50%)", f"{standard_evaluation['precision']:.2%}")
    c.metric("Recall (50%)", f"{standard_evaluation['recall']:.2%}")
    d.metric("F1-score (50%)", f"{standard_evaluation['f1']:.2%}")
    st.subheader("LightGBM Confusion Matrix — 50% Threshold")
    st.dataframe(pd.DataFrame({"Measure": ["True Negatives", "False Positives",
                                           "False Negatives", "True Positives"],
                               "Count": [standard_evaluation[k] for k in ["tn", "fp", "fn", "tp"]]}),
                 width="stretch", hide_index=True)
    st.subheader("Exploratory Threshold Comparison")
    threshold_table = pd.DataFrame([{
        "Threshold (%)": int(round(r["threshold"] * 100)),
        "Failures Detected": r["tp"], "Failures Missed": r["fn"],
        "False Alarms": r["fp"],
        "Precision (%)": round(r["precision"] * 100, 2),
        "Recall (%)": round(r["recall"] * 100, 2),
        "F1-score (%)": round(r["f1"] * 100, 2)}
        for r in threshold_evaluations])
    st.dataframe(threshold_table, width="stretch", hide_index=True)
    st.info(f"The application currently uses a provisional {ALERT_THRESHOLD:.0%} "
            "maintenance-alert threshold. Its test-set comparison is exploratory "
            "and should not be treated as independent threshold validation.")
    st.subheader("LightGBM Feature Importance — Current Model")
    importance = pd.DataFrame({"Feature": X_train.columns,
                               "Importance": model.feature_importances_})
    st.bar_chart(importance.set_index("Feature").sort_values("Importance", ascending=False))
    st.caption("Built-in LightGBM feature importance is model-specific and does not "
               "establish causation.")


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
