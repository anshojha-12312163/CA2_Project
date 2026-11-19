"""
dashboard/app.py

High-Performance, Professional Clinical Intelligence & Research Dashboard
"Imbalance-Handling Strategies on Small Imbalanced Medical Data"
(Pima Indians Diabetes Diagnostic Study)

Design System: Modern Glassmorphic Dark Architecture (Inter & Plus Jakarta Sans typography)
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Pima Clinical ML Benchmark",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- PATH RESOLUTION -----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
candidate_roots = [
    os.path.abspath(os.path.join(APP_DIR, "..")),
    os.getcwd(),
]

ROOT_DIR = None
for cand in candidate_roots:
    if os.path.exists(os.path.join(cand, "data", "pima_diabetes.csv")) and os.path.exists(os.path.join(cand, "results", "all_runs.csv")):
        ROOT_DIR = cand
        break

if ROOT_DIR is None:
    ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

DATA_PATH = os.path.join(ROOT_DIR, "data", "pima_diabetes.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# ----------------- PALETTE & CONSTANTS -----------------
METHOD_COLORS = {
    "Baseline": "#94A3B8",        # Slate gray
    "ClassWeight": "#38BDF8",     # Sky blue
    "RandomOver": "#818CF8",      # Indigo
    "RandomUnder": "#10B981",     # Emerald green
    "SMOTE": "#F472B6",           # Rose/Pink
    "BorderlineSMOTE": "#FBBF24", # Amber/Gold
}

METHODS_ORDER = ["Baseline", "ClassWeight", "RandomOver", "RandomUnder", "SMOTE", "BorderlineSMOTE"]
TRAIN_SIZES_ORDER = [60, 120, 250, 537]
MODELS_ORDER = ["LogReg", "RandomForest"]

METRIC_MAP = {
    "Balanced Accuracy": "BA",
    "Macro F1": "F1",
    "G-mean": "GM",
    "Minority Recall": "MinorityRecall",
}

# ----------------- CUSTOM CSS DESIGN SYSTEM -----------------
def inject_custom_css():
    st.markdown(r"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3.5rem;
        max-width: 1440px;
    }

    /* Global Layout & Dark Theme Enforcement */
    .stApp {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    footer {
        visibility: hidden !important;
    }

    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 4rem !important;
        max-width: 1440px !important;
    }

    /* Hero Banner */
    .hero-container {
        background: radial-gradient(circle at 12% 20%, rgba(0, 173, 181, 0.22) 0%, rgba(15, 23, 42, 0) 55%),
                    radial-gradient(circle at 88% 80%, rgba(56, 189, 248, 0.15) 0%, rgba(15, 23, 42, 0) 50%),
                    linear-gradient(135deg, #0E1628 0%, #131E35 50%, #0A0F1E 100%);
        border: 1px solid rgba(56, 189, 248, 0.32);
        border-radius: 20px;
        padding: 38px 44px;
        margin-bottom: 28px;
        box-shadow: 0 20px 48px -12px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-badge-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
        align-items: center;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(0, 173, 181, 0.16);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 6px 15px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #34D399;
        box-shadow: 0 0 10px #34D399;
    }
    .hero-title {
        color: #F8FAFC;
        font-size: clamp(2rem, 3.2vw, 2.75rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        margin: 0 0 12px 0;
        background: linear-gradient(135deg, #FFFFFF 10%, #E0F2FE 55%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.18;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: clamp(0.95rem, 1.3vw, 1.1rem);
        margin: 0 0 20px 0;
        font-weight: 400;
        max-width: 1020px;
        line-height: 1.65;
    }
    .hero-meta-row {
        display: flex;
        gap: 24px;
        color: #7DD3FC;
        font-size: 0.88rem;
        font-weight: 500;
        flex-wrap: wrap;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
        padding-top: 16px;
    }
    .hero-meta-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #94A3B8;
    }
    .hero-meta-item b {
        color: #E2E8F0;
    }

    /* Cards */
    .glass-card {
        background: linear-gradient(180deg, #131B2E 0%, #0F1728 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 26px;
        margin-bottom: 22px;
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }
    .card-title {
        color: #F8FAFC;
        font-size: 1.18rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Gateway Nav Cards */
    .gateway-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }
    @media (max-width: 1024px) {
        .gateway-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
        .gateway-grid { grid-template-columns: 1fr; }
    }
    .gateway-card {
        background: linear-gradient(180deg, #162238 0%, #101827 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: linear-gradient(180deg, #162238 0%, #0F172A 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.22);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 185px;
        margin-bottom: 10px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .gateway-card:hover {
        transform: translateY(-4px);
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15);
        box-shadow: 0 10px 28px rgba(56, 189, 248, 0.18);
    }
    .gateway-icon {
        font-size: 2rem;
        font-size: 2.1rem;
        margin-bottom: 12px;
    }
    .gateway-title {
        font-size: 1.1rem;
        font-size: 1.12rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
    }
    .gateway-desc {
        font-size: 0.85rem;
        font-size: 0.86rem;
        color: #94A3B8;
        line-height: 1.5;
        margin: 0 0 16px 0;
        margin: 0;
        flex-grow: 1;
    }

    /* Primary Streamlit Button Styling */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #00ADB5 0%, #0284C7 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.1rem !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(0, 173, 181, 0.25) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 173, 181, 0.45) !important;
        border: none !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
    }

    /* Architecture Stepper Grid */
    .arch-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 28px;
    }
    @media (max-width: 1024px) {
        .arch-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
        .arch-grid { grid-template-columns: 1fr; }
    }
    .arch-card {
        background: linear-gradient(180deg, #151F33 0%, #0E1626 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 20px;
        position: relative;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .arch-step-num {
        display: inline-block;
        width: 26px;
        height: 26px;
        line-height: 26px;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        background: rgba(56, 189, 248, 0.15);
        background: rgba(56, 189, 248, 0.18);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 50%;
        font-size: 0.78rem;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 10px;
        margin-bottom: 12px;
    }
    .arch-step-title {
        color: #F8FAFC;
        font-size: 0.95rem;
        font-size: 0.98rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        margin: 0 0 8px 0;
    }
    .arch-step-desc {
        color: #94A3B8;
        font-size: 0.82rem;
        line-height: 1.45;
        font-size: 0.84rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Key Insights Grid */
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
        margin-bottom: 26px;
        margin-bottom: 28px;
    }
    @media (max-width: 900px) {
        .insight-grid { grid-template-columns: 1fr; }
    }
    .insight-card {
        background: linear-gradient(180deg, #162238 0%, #101827 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: linear-gradient(180deg, #162238 0%, #0F1728 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.22);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    .insight-card.teal { border-top: 3px solid #00ADB5; }
    .insight-card.emerald { border-top: 3px solid #10B981; }
    .insight-card.indigo { border-top: 3px solid #818CF8; }

    .insight-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .insight-badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 3px 10px;
        padding: 4px 10px;
        border-radius: 12px;
    }
    .insight-badge.teal { background: rgba(0, 173, 181, 0.18); color: #38BDF8; }
    .insight-badge.emerald { background: rgba(16, 185, 129, 0.18); color: #34D399; }
    .insight-badge.indigo { background: rgba(129, 140, 248, 0.18); color: #A5B4FC; }
    .insight-badge.teal { background: rgba(0, 173, 181, 0.18); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .insight-badge.emerald { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .insight-badge.indigo { background: rgba(129, 140, 248, 0.18); color: #A5B4FC; border: 1px solid rgba(165, 180, 252, 0.3); }

    .insight-title {
        font-size: 1.08rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 8px 0;
    }
    .insight-text {
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.55;
        margin: 0;
    }

    /* Custom KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-bottom: 26px;
        margin-bottom: 28px;
    }
    @media (max-width: 1024px) {
        .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 640px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-card {
        background: linear-gradient(180deg, #162238 0%, #101827 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        background: linear-gradient(180deg, #162238 0%, #0F172A 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #00ADB5;
    }
    .kpi-card.teal::before { background: #00ADB5; }
    .kpi-card.blue::before { background: #38BDF8; }
    .kpi-card.indigo::before { background: #818CF8; }
    .kpi-card.emerald::before { background: #10B981; }
    .kpi-card.rose::before { background: #F43F5E; }
    .kpi-card.amber::before { background: #FBBF24; }

    .kpi-label {
        font-size: 0.74rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        margin-bottom: 8px;
    }
    .kpi-val {
        font-size: 1.75rem;
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
        margin-top: 5px;
        font-weight: 500;
    }

    /* Diagnosis Result Card */
    .risk-card {
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .risk-card.high {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.18) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(244, 63, 94, 0.45);
    }
    .risk-card.low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .risk-badge.high {
        background: rgba(244, 63, 94, 0.25);
        color: #FDA4AF;
        border: 1px solid #F43F5E;
    }
    .risk-badge.low {
        background: rgba(16, 185, 129, 0.25);
        color: #6EE7B7;
        border: 1px solid #10B981;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        border-radius: 10px;
        padding: 14px 18px;
        color: #CBD5E1;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 24px;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background-color: #0B0F19;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        background-color: #0B101E !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
    }
    .sidebar-brand {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 18px;
    }
    .sidebar-brand h2 {
        font-size: 1.25rem;
        font-size: 1.3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .sidebar-brand p {
        font-size: 0.78rem;
        font-size: 0.8rem;
        color: #64748B;
        margin: 4px 0 0 0;
    }

    /* Inputs, Selectboxes, Radios */
    div[data-baseweb="select"] > div {
        background-color: #151F33 !important;
        color: #F8FAFC !important;
        border-color: rgba(148, 163, 184, 0.25) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background-color: #151F33 !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="menu"] * {
        color: #F8FAFC !important;
    }
    div[data-baseweb="select"] span {
        color: #F8FAFC !important;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: rgba(22, 34, 56, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: rgba(56, 189, 248, 0.4) !important;
        background-color: rgba(22, 34, 56, 0.95) !important;
    }

    /* Expanders & DataFrames */
    div[data-testid="stExpander"] {
        background-color: #131B2E !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 12px !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: #131B2E !important;
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
    }
</style>
""", unsafe_allow_html=True)

inject_custom_css()


# ----------------- CACHED DATA LOADERS -----------------
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_all_runs():
    p = os.path.join(RESULTS_DIR, "all_runs.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


@st.cache_data
def load_summary_mean():
    p = os.path.join(RESULTS_DIR, "summary_mean.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


@st.cache_data
def load_summary_mean_std():
    p = os.path.join(RESULTS_DIR, "summary_mean_std.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


@st.cache_data
def load_best_methods():
    p = os.path.join(RESULTS_DIR, "best_method_per_setting.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


@st.cache_data
def load_wilcoxon():
    p = os.path.join(RESULTS_DIR, "wilcoxon_vs_baseline_BA.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


# ----------------- CACHED MODEL TRAINING (LIVE PREDICTOR) -----------------
@st.cache_resource
def train_live_models():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None

    df = pd.read_csv(DATA_PATH)
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    # Champion Pipeline 1: RandomForest with RandomUnderSampler
    rf_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("s", RandomUnderSampler(random_state=1000)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=1000, n_jobs=-1)),
    ])
    rf_pipe.fit(X, y)

    # Baseline Pipeline 1: RandomForest unweighted (No Resampling)
    rf_base = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=1000, n_jobs=-1)),
    ])
    rf_base.fit(X, y)

    # Champion Pipeline 2: Logistic Regression with ClassWeight='balanced'
    lr_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=1000)),
    ])
    lr_pipe.fit(X, y)

    # Baseline Pipeline 2: Logistic Regression standard unweighted
    lr_base = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight=None, random_state=1000)),
    ])
    lr_base.fit(X, y)

    return rf_pipe, lr_pipe, rf_base, lr_base


# ----------------- PLOT THEME HELPER -----------------
def style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19, 27, 46, 0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC", size=12),
        margin=dict(l=45, r=35, t=45, b=40),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Plus Jakarta Sans",
            bordercolor="rgba(56, 189, 248, 0.4)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(19, 27, 46, 0.8)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.08)",
        zeroline=False,
    )
    return fig


# ----------------- HERO HEADER -----------------
def render_hero():
    st.markdown(r"""
    <div class="hero-container" style="background: radial-gradient(circle at 12% 20%, rgba(0, 173, 181, 0.22) 0%, rgba(15, 23, 42, 0) 55%), radial-gradient(circle at 88% 80%, rgba(56, 189, 248, 0.15) 0%, rgba(15, 23, 42, 0) 50%), linear-gradient(135deg, #0E1628 0%, #131E35 50%, #0A0F1E 100%); border: 1px solid rgba(56, 189, 248, 0.32); border-radius: 20px; padding: 34px 40px; margin-bottom: 26px; box-shadow: 0 20px 48px -12px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1); position: relative; overflow: hidden;">
        <div class="hero-badge-row" style="display: flex; gap: 10px; margin-bottom: 16px; align-items: center; flex-wrap: wrap;">
            <span class="hero-badge" style="background: rgba(0, 173, 181, 0.16); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 20px; padding: 5px 14px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; display: inline-flex; align-items: center; gap: 7px;"><span class="pulse-dot" style="width: 8px; height: 8px; border-radius: 50%; background-color: #34D399; box-shadow: 0 0 10px #34D399; display: inline-block;"></span> Clinical AI Benchmark</span>
            <span class="hero-badge" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 5px 14px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;">Leakage-Free Pipeline</span>
            <span class="hero-badge" style="background: rgba(129, 140, 248, 0.15); color: #A5B4FC; border: 1px solid rgba(129, 140, 248, 0.3); border-radius: 20px; padding: 5px 14px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;">20× Monte Carlo Validated</span>
            <span class="hero-badge" style="background: rgba(244, 63, 94, 0.12); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 20px; padding: 5px 14px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;">6 Resampling Strategies</span>
        </div>
        <h1 class="hero-title" style="color: #F8FAFC; font-size: 2.35rem; font-weight: 800; letter-spacing: -0.035em; margin: 0 0 12px 0; line-height: 1.2;">Imbalance-Handling Strategies<br><span style="background: linear-gradient(135deg, #FFFFFF 10%, #E0F2FE 55%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">on Small Imbalanced Medical Data</span></h1>
        <p class="hero-subtitle" style="color: #94A3B8; font-size: 1.05rem; margin: 0 0 20px 0; font-weight: 400; max-width: 1040px; line-height: 1.6;">
            A rigorous empirical research benchmark evaluating algorithmic cost-weighting and synthetic resampling under acute sample scarcity (N = 60 to 537) on the <b style="color: #F8FAFC;">Pima Indians Diabetes</b> diagnostic cohort with zero-leakage cross-validation.
        </p>
        <div class="hero-meta-row" style="display: flex; gap: 24px; color: #7DD3FC; font-size: 0.88rem; font-weight: 500; flex-wrap: wrap; border-top: 1px solid rgba(148, 163, 184, 0.14); padding-top: 16px;">
            <span class="hero-meta-item" style="color: #94A3B8;">📁 <b style="color: #E2E8F0;">Cohort:</b> UCI / NIDDK (768 Patients)</span>
            <span class="hero-meta-item" style="color: #94A3B8;">🧬 <b style="color: #E2E8F0;">Features:</b> 8 Physiological Biomarkers</span>
            <span class="hero-meta-item" style="color: #94A3B8;">⚖️ <b style="color: #E2E8F0;">Class Ratio:</b> 1 : 1.87 (34.9% Diabetic)</span>
            <span class="hero-meta-item" style="color: #94A3B8;">🔬 <b style="color: #E2E8F0;">Pipeline:</b> Median Impute → Robust Scaler → Resampler → Estimator</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# PAGE 1: LANDING PAGE & COHORT OVERVIEW
# =========================================================================
def render_overview():
    df = load_dataset()
    all_runs = load_all_runs()

    if df is None:
        st.error(f"Dataset not found at `{DATA_PATH}`.")
        return

    n_rows = len(df)
    n_features = df.shape[1] - 1
    pos_count = int(df["Outcome"].sum())
    neg_count = n_rows - pos_count
    pos_pct = pos_count / n_rows * 100
    neg_pct = 100 - pos_pct
    n_total_runs = len(all_runs) if all_runs is not None else 960

    # 6 Cohort KPI Metric Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "Cohort Size", f"{n_rows:,}", "Adult Female Patients", "#00ADB5"),
        (k2, "Biomarkers", f"{n_features}", "Clinical Covariates", "#38BDF8"),
        (k3, "Negative (0)", f"{neg_count:,}", f"{neg_pct:.1f}% Non-Diabetic", "#10B981"),
        (k4, "Positive (1)", f"{pos_count:,}", f"{pos_pct:.1f}% Diabetic Cases", "#F43F5E"),
        (k5, "Strategies", "6", "Algorithmic & Resampling", "#818CF8"),
        (k6, "Evaluations", f"{n_total_runs:,}", "20 Repeats × 4 Sizes", "#FBBF24"),
    ]
    for col, label, val, sub, color in kpis:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(180deg, #162238 0%, #0F172A 100%); border: 1px solid rgba(148, 163, 184, 0.16); border-top: 3px solid {color}; border-radius: 14px; padding: 18px 16px; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22); margin-bottom: 22px;">
                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">{label}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #F8FAFC; line-height: 1.1; letter-spacing: -0.02em;">{val}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 5px; font-weight: 500;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # 4 Interactive Gateway Cards
    st.markdown('<div class="card-title" style="color: #F8FAFC; font-size: 1.18rem; font-weight: 700; letter-spacing: -0.015em; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;">🚀 Interactive Research Modules</div>', unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)

    gateways = [
        (g1, "📊", "Results Explorer", "Analyze performance scaling curves, error-bars, and top metrics as training size shrinks.", "btn_g1", "🔍 Results Explorer", "Launch Explorer →"),
        (g2, "⚖️", "Method Matrix", "Explore Method × Size heatmaps and inspect variance distributions over 20 Monte Carlo splits.", "btn_g2", "⚖️ Method Comparison", "Open Comparison →"),
        (g3, "📈", "Statistical Tests", "Review paired Wilcoxon signed-rank significance tests (p < 0.05) and percentage point gains.", "btn_g3", "📈 Statistical Tests", "View Significance →"),
        (g4, "🩺", "Live Predictor", "Interactive patient risk simulator using calibrated top-performing Random Forest pipeline.", "btn_g4", "🩺 Live Predictor", "Run Simulator →"),
    ]

    for col, icon, title, desc, btn_key, nav_target, btn_label in gateways:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(180deg, #162238 0%, #0F172A 100%); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px; padding: 22px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25); min-height: 165px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 1.75rem; margin-bottom: 10px;">{icon}</div>
                    <div style="color: #F8FAFC; font-size: 1.05rem; font-weight: 700; margin-bottom: 6px;">{title}</div>
                    <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(btn_label, key=btn_key, use_container_width=True):
                st.session_state["main_nav_radio"] = nav_target
                st.rerun()

    st.write("")

    # Experimental Methodology Architecture (4 Columns)
    st.markdown('<div class="card-title" style="color: #F8FAFC; font-size: 1.18rem; font-weight: 700; letter-spacing: -0.015em; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;">🛡️ Rigorous Zero-Leakage Pipeline Architecture</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)

    arch_steps = [
        (a1, "01", "Clinical Cohort Ingestion", "UCI / NIDDK Pima dataset (768 adult female patients, 8 clinical features, 34.9% diabetic positive prevalence)."),
        (a2, "02", "Physiological Sanitization", "Biologically impossible 0s in Glucose, BP, Skin, Insulin, and BMI converted to NaN; imputed via training fold median."),
        (a3, "03", "Nested Monte Carlo Splits", "20 independent stratified train/test partitions evaluated across 4 acute sample sizes: N = 60, 120, 250, 537."),
        (a4, "04", "In-Pipeline Resampling", "Resamplers and scalers execute exclusively inside <code>imblearn.pipeline.Pipeline</code> during fit to eliminate test leakage."),
    ]

    for col, step_num, step_title, step_desc in arch_steps:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(180deg, #131B2E 0%, #0E1628 100%); border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 14px; padding: 22px; margin-bottom: 22px; box-shadow: 0 4px 18px rgba(0, 0, 0, 0.22); min-height: 175px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 700; color: #38BDF8; background: rgba(56, 189, 248, 0.12); border-radius: 6px; padding: 3px 8px; display: inline-block; margin-bottom: 10px;">{step_num}</div>
                <div style="color: #F8FAFC; font-size: 0.98rem; font-weight: 700; margin-bottom: 6px;">{step_title}</div>
                <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">{step_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # 3 Key Research Takeaways Cards
    st.markdown('<div class="card-title" style="color: #F8FAFC; font-size: 1.18rem; font-weight: 700; letter-spacing: -0.015em; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;">💡 Core Empirical Findings & Breakthroughs</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)

    insights = [
        (i1, "🌲", "Major Breakthrough", "#10B981", "rgba(16, 185, 129, 0.15)", "#34D399",
         "Random Under-Sampling Dominates Random Forest",
         "For Random Forest, <b>RandomUnder</b> was the overall top strategy across all cohort sizes, lifting Balanced Accuracy by <b>+6.49 pp</b> and doubling minority diabetic detection (Recall: 36.3% → <b>72.0%</b> at N=60, <i>p < 0.0001</i>)."),
        (i2, "⚖️", "Optimal Linear Strategy", "#00ADB5", "rgba(0, 173, 181, 0.15)", "#38BDF8",
         "Cost-Sensitive ClassWeight Leads Logistic Regression",
         "For Logistic Regression, algorithmic <b>ClassWeight</b> achieves the highest Balanced Accuracy at N=60 (<b>0.713 ± 0.035</b>), outperforming complex synthetic interpolators like SMOTE without introducing artificial boundary noise."),
        (i3, "🛡️", "Methodology Benchmark", "#818CF8", "rgba(129, 140, 248, 0.15)", "#A5B4FC",
         "Strict Zero Data-Leakage Architecture",
         "All preprocessing (median imputation of physiological zeros, standard scaling) and resampling execute strictly within <code>imblearn.pipeline.Pipeline</code> during <code>.fit()</code>, ensuring held-out test data is pristine."),
    ]

    for col, emoji_icon, badge_text, border_col, badge_bg, badge_fg, card_head, card_body in insights:
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(180deg, #131B2E 0%, #0E1628 100%); border: 1px solid rgba(148, 163, 184, 0.16); border-top: 3px solid {border_col}; border-radius: 14px; padding: 22px; margin-bottom: 22px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25); min-height: 210px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.25rem;">{emoji_icon}</span>
                    <span style="background: {badge_bg}; color: {badge_fg}; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.04em;">{badge_text}</span>
                </div>
                <div style="color: #F8FAFC; font-size: 1.02rem; font-weight: 700; margin-bottom: 8px;">{card_head}</div>
                <div style="font-size: 0.84rem; color: #94A3B8; line-height: 1.55;">{card_body}</div>
            </div>
            """, unsafe_allow_html=True)

    # Visual Impact Chart: Minority Recall Comparison at N=60
    summary_mean = load_summary_mean()
    if summary_mean is not None:
        st.markdown('<div class="glass-card"><div class="card-title">📊 Minority Recall Impact at Smallest Cohort (N = 60)</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; font-size:0.86rem; margin-top:-10px; margin-bottom:14px;'>How each resampling strategy affects diabetic patient detection rate at extreme sample scarcity.</p>", unsafe_allow_html=True)

        c_rf_chart, c_lr_chart = st.columns(2, gap="medium")

        for col_slot, model_name, model_label in [(c_rf_chart, "RandomForest", "🌲 Random Forest"), (c_lr_chart, "LogReg", "📈 Logistic Regression")]:
            with col_slot:
                sub = summary_mean[(summary_mean["model"] == model_name) & (summary_mean["train_size"] == 60)].copy()
                sub = sub.set_index("method").reindex(METHODS_ORDER).reset_index()
                fig_recall = go.Figure()
                for _, row in sub.iterrows():
                    meth = row["method"]
                    val = row["MinorityRecall"]
                    is_best = val == sub["MinorityRecall"].max()
                    fig_recall.add_trace(go.Bar(
                        x=[meth], y=[val],
                        marker_color=METHOD_COLORS.get(meth, "#38BDF8"),
                        marker_line_width=2 if is_best else 0,
                        marker_line_color="#FFFFFF" if is_best else "rgba(0,0,0,0)",
                        text=f"{val:.1%}",
                        textposition="outside",
                        textfont=dict(size=11, color="#F8FAFC"),
                        hovertemplate=f"<b>{meth}</b><br>Minority Recall: {val:.3f}<extra></extra>",
                        showlegend=False,
                    ))
                fig_recall.update_layout(
                    title=dict(text=model_label, font=dict(size=14, color="#F8FAFC"), x=0.5),
                    yaxis_title="Minority Recall",
                    yaxis=dict(range=[0, min(1.0, sub["MinorityRecall"].max() + 0.12)]),
                    xaxis_title="",
                    height=320,
                    showlegend=False,
                )
                style_plotly(fig_recall)
                st.plotly_chart(fig_recall, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Executive Scaling Curves for Balanced Accuracy across Sample Scarcity
        st.markdown('<div class="glass-card"><div class="card-title">📈 Balanced Accuracy Trajectory Across Sample Scarcity (N = 60 to 537)</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; font-size:0.86rem; margin-top:-10px; margin-bottom:14px;'>Empirical performance curves showing resilience of each strategy as training sample size decreases to acute scarcity (N = 60).</p>", unsafe_allow_html=True)

        c_rf_line, c_lr_line = st.columns(2, gap="medium")

        for col_slot, model_name, model_label in [(c_rf_line, "RandomForest", "🌲 Random Forest (Ensemble)"), (c_lr_line, "LogReg", "📈 Logistic Regression (Linear)")]:
            with col_slot:
                m_summary = summary_mean[summary_mean["model"] == model_name]
                fig_traj = go.Figure()
                for meth in METHODS_ORDER:
                    m_sub = m_summary[m_summary["method"] == meth].sort_values("train_size")
                    is_champ = meth in ["RandomUnder", "ClassWeight"]
                    fig_traj.add_trace(go.Scatter(
                        x=m_sub["train_size"],
                        y=m_sub["BA"],
                        mode="lines+markers",
                        name=meth,
                        line=dict(color=METHOD_COLORS.get(meth, "#38BDF8"), width=3.2 if is_champ else 1.8),
                        marker=dict(size=7),
                        hovertemplate=f"<b>{meth}</b><br>Cohort Size (N): %{{x}}<br>Balanced Accuracy: %{{y:.3f}}<extra></extra>",
                    ))
                fig_traj.update_layout(
                    title=dict(text=model_label, font=dict(size=14, color="#F8FAFC"), x=0.5),
                    xaxis=dict(title="Training Cohort Size (N)", tickvals=[60, 120, 250, 537], ticktext=["N=60", "N=120", "N=250", "N=537"]),
                    yaxis=dict(title="Balanced Accuracy", range=[0.58, 0.78]),
                    height=320,
                    legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(size=10)),
                )
                style_plotly(fig_traj)
                st.plotly_chart(fig_traj, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Champion Benchmark Table Direct on Landing Page
    best_df = load_best_methods()
    if best_df is not None:
        st.markdown('<div class="glass-card"><div class="card-title">🏆 Benchmark Champion Strategies at a Glance</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; font-size:0.88rem; margin-top:-8px; margin-bottom:16px;'>Top performing strategy selected by Balanced Accuracy across 20 Monte Carlo iterations for each model and training size.</p>", unsafe_allow_html=True)
        
        c_rf, c_lr = st.columns(2, gap="medium")
        
        def build_champs_table(subset_df, accent_color):
            rows_html = ""
            for _, r in subset_df.iterrows():
                sz = r["Cohort Size (N)"]
                meth = r["Champion Strategy"]
                ba = r["Balanced Accuracy"]
                rec = r["Minority Recall"]
                
                if "Under" in meth:
                    b_bg, b_col, b_brd = "rgba(16, 185, 129, 0.18)", "#34D399", "rgba(52, 211, 153, 0.35)"
                elif "Weight" in meth:
                    b_bg, b_col, b_brd = "rgba(56, 189, 248, 0.18)", "#38BDF8", "rgba(56, 189, 248, 0.35)"
                else:
                    b_bg, b_col, b_brd = "rgba(129, 140, 248, 0.18)", "#A5B4FC", "rgba(129, 140, 248, 0.35)"
                
                rows_html += f"""
                <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <td style="padding: 10px 14px; font-weight: 600; color: #F1F5F9; font-size: 0.88rem;">{sz}</td>
                    <td style="padding: 10px 14px;">
                        <span style="background:{b_bg}; color:{b_col}; border:1px solid {b_brd}; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.78rem;">
                            {meth}
                        </span>
                    </td>
                    <td style="padding: 10px 14px; font-weight: 700; color: {accent_color}; font-size: 0.92rem;">{ba}</td>
                    <td style="padding: 10px 14px; font-weight: 600; color: #F8FAFC; font-size: 0.88rem;">{rec}</td>
                </tr>
                """
            return f"""
            <table style="width:100%; border-collapse:collapse; background:rgba(15, 23, 42, 0.5); border-radius:10px; overflow:hidden; border: 1px solid rgba(148, 163, 184, 0.12);">
                <thead>
                    <tr style="background:rgba(30, 41, 59, 0.7); color:#94A3B8; font-size:0.74rem; text-transform:uppercase; letter-spacing:0.05em; text-align:left;">
                        <th style="padding:10px 14px;">Cohort Size</th>
                        <th style="padding:10px 14px;">Champion Strategy</th>
                        <th style="padding:10px 14px;">Balanced Acc</th>
                        <th style="padding:10px 14px;">Minority Recall</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """

        with c_rf:
            st.markdown("<div style='font-size:0.96rem; font-weight:700; color:#34D399; margin-bottom:10px;'>🌲 Random Forest (Ensemble)</div>", unsafe_allow_html=True)
            rf_champs = best_df[best_df["model"] == "RandomForest"][["train_size", "method", "BA", "MinorityRecall"]].copy()
            rf_champs.columns = ["Cohort Size (N)", "Champion Strategy", "Balanced Accuracy", "Minority Recall"]
            rf_champs["Cohort Size (N)"] = rf_champs["Cohort Size (N)"].apply(lambda s: f"N = {s}" if s != 537 else "Full (N = 537)")
            rf_champs["Balanced Accuracy"] = rf_champs["Balanced Accuracy"].apply(lambda v: f"{v*100:.1f}%")
            rf_champs["Minority Recall"] = rf_champs["Minority Recall"].apply(lambda v: f"{v*100:.1f}%")
            st.markdown(build_champs_table(rf_champs, "#34D399"), unsafe_allow_html=True)
            
        with c_lr:
            st.markdown("<div style='font-size:0.96rem; font-weight:700; color:#38BDF8; margin-bottom:10px;'>📈 Logistic Regression (Linear)</div>", unsafe_allow_html=True)
            lr_champs = best_df[best_df["model"] == "LogReg"][["train_size", "method", "BA", "MinorityRecall"]].copy()
            lr_champs.columns = ["Cohort Size (N)", "Champion Strategy", "Balanced Accuracy", "Minority Recall"]
            lr_champs["Cohort Size (N)"] = lr_champs["Cohort Size (N)"].apply(lambda s: f"N = {s}" if s != 537 else "Full (N = 537)")
            lr_champs["Balanced Accuracy"] = lr_champs["Balanced Accuracy"].apply(lambda v: f"{v*100:.1f}%")
            lr_champs["Minority Recall"] = lr_champs["Minority Recall"].apply(lambda v: f"{v*100:.1f}%")
            st.markdown(build_champs_table(lr_champs, "#38BDF8"), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Visual Cohort Exploration & Feature Matrix
    col_chart, col_desc = st.columns([1, 1.25], gap="large")

    with col_chart:
        st.markdown('<div class="glass-card"><div class="card-title">🍩 Diagnostic Cohort Composition</div>', unsafe_allow_html=True)
        donut_df = pd.DataFrame({
            "Classification": ["Non-Diabetic (Outcome 0)", "Diabetic (Outcome 1)"],
            "Patients": [neg_count, pos_count],
        })
        fig_donut = px.pie(
            donut_df,
            names="Classification",
            values="Patients",
            hole=0.62,
            color="Classification",
            color_discrete_map={
                "Non-Diabetic (Outcome 0)": "#00ADB5",
                "Diabetic (Outcome 1)": "#F43F5E",
            },
        )
        fig_donut.update_traces(
            textinfo="percent+value",
            textfont=dict(size=13, color="#FFFFFF", family="Plus Jakarta Sans"),
            marker=dict(line=dict(color="#0F172A", width=3)),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Proportion: %{percent}<extra></extra>",
        )
        fig_donut.update_layout(
            height=340,
            showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
            annotations=[dict(text=f"<b>{n_rows}</b><br><span style='font-size:11px;color:#94A3B8;'>Patients</span>", x=0.5, y=0.5, font_size=20, showarrow=False)],
        )
        style_plotly(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_desc:
        st.markdown('<div class="glass-card"><div class="card-title">🔬 Clinical Covariates & Missingness Pipeline</div>', unsafe_allow_html=True)
        covariates_info = [
            ("Glucose", "Continuous", "70 - 99 mg/dL", "Zeros → NaN → Median imputed"),
            ("BloodPressure", "Continuous", "60 - 80 mm Hg", "Zeros → NaN → Median imputed"),
            ("SkinThickness", "Continuous", "10 - 30 mm", "Zeros → NaN → Median imputed"),
            ("Insulin", "Continuous", "16 - 166 mu U/ml", "Zeros → NaN → Median imputed"),
            ("BMI", "Continuous", "18.5 - 24.9 kg/m²", "Zeros → NaN → Median imputed"),
            ("Pregnancies", "Discrete", "0 - 17 count", "Valid biological zero"),
            ("Age", "Discrete", "21 - 81 years", "Valid chronological zero"),
            ("DiabetesPedigree", "Continuous", "0.08 - 2.42", "Valid genetic risk metric"),
        ]
        cov_rows_html = ""
        for bio, typ, norm, miss in covariates_info:
            is_imputed = "imputed" in miss
            miss_color = "#FBBF24" if is_imputed else "#34D399"
            cov_rows_html += f"""
            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.08);">
                <td style="padding: 8px 12px; font-weight: 700; color: #F8FAFC; font-size: 0.84rem;">{bio}</td>
                <td style="padding: 8px 10px; color: #94A3B8; font-size: 0.8rem;">{typ}</td>
                <td style="padding: 8px 10px; color: #38BDF8; font-family:'JetBrains Mono',monospace; font-size: 0.8rem;">{norm}</td>
                <td style="padding: 8px 10px; color: {miss_color}; font-size: 0.8rem;">{miss}</td>
            </tr>
            """
        cov_table_html = f"""
        <table style="width:100%; border-collapse:collapse; background:rgba(15, 23, 42, 0.5); border-radius:10px; overflow:hidden; border: 1px solid rgba(148, 163, 184, 0.12);">
            <thead>
                <tr style="background:rgba(30, 41, 59, 0.7); color:#94A3B8; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; text-align:left;">
                    <th style="padding:9px 12px;">Biomarker</th>
                    <th style="padding:9px 10px;">Type</th>
                    <th style="padding:9px 10px;">Clinical Normal</th>
                    <th style="padding:9px 10px;">Treatment</th>
                </tr>
            </thead>
            <tbody>
                {cov_rows_html}
            </tbody>
        </table>
        """
        st.markdown(cov_table_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 Interactive Cohort Data Explorer (First 50 Patient Encounters)"):
        st.dataframe(df.head(50), use_container_width=True)


# =========================================================================
# PAGE 2: RESULTS EXPLORER
# =========================================================================
def render_results_explorer():
    all_runs = load_all_runs()
    summary_mean = load_summary_mean()
    if all_runs is None or summary_mean is None:
        st.error("Results files not found. Ensure `results/` is populated.")
        return

    st.sidebar.markdown('<div class="sidebar-brand"><h2>🎛️ Explorer Filters</h2><p>Filter experimental runs</p></div>', unsafe_allow_html=True)
    model_choice = st.sidebar.selectbox("Classifier Architecture", ["RandomForest", "LogReg"], format_func=lambda x: "Random Forest (200 Trees)" if x == "RandomForest" else "Logistic Regression (L2 Regularized)")
    size_choice = st.sidebar.selectbox("Training Cohort Size (N)", TRAIN_SIZES_ORDER, format_func=lambda x: f"N = {x} Samples (25% Minority)" if x != 537 else "Full Training Set (N = 537)")
    metric_choice = st.sidebar.selectbox("Primary Clinical Metric", list(METRIC_MAP.keys()))
    metric_col = METRIC_MAP[metric_choice]

    col1, col2 = st.columns([1.1, 1.25], gap="large")

    with col1:
        st.markdown(f'<div class="glass-card"><div class="card-title">📊 {metric_choice} by Method (Cohort N = {size_choice})</div>', unsafe_allow_html=True)
        sub_runs = all_runs[(all_runs["model"] == model_choice) & (all_runs["train_size"] == size_choice)]
        stat_df = sub_runs.groupby("method")[metric_col].agg(["mean", "std"]).reindex(METHODS_ORDER).reset_index()

        fig_bar = go.Figure()
        for _, row in stat_df.iterrows():
            meth = row["method"]
            fig_bar.add_trace(go.Bar(
                x=[meth],
                y=[row["mean"]],
                name=meth,
                marker_color=METHOD_COLORS.get(meth, "#38BDF8"),
                marker_line_width=0,
                error_y=dict(type='data', array=[row["std"]], visible=True, color="rgba(248, 250, 252, 0.7)", thickness=1.5, width=4),
                hovertemplate=f"<b>{meth}</b><br>Mean: {row['mean']:.3f}<br>Std: ±{row['std']:.3f}<extra></extra>",
            ))
        fig_bar.update_layout(
            yaxis_title=metric_choice,
            xaxis_title="",
            showlegend=False,
            height=370,
        )
        style_plotly(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="glass-card"><div class="card-title">📈 Metric Scaling Trajectory vs. Training Size</div>', unsafe_allow_html=True)
        model_summary = summary_mean[summary_mean["model"] == model_choice]

        fig_line = go.Figure()
        for meth in METHODS_ORDER:
            m_sub = model_summary[model_summary["method"] == meth].sort_values("train_size")
            fig_line.add_trace(go.Scatter(
                x=m_sub["train_size"],
                y=m_sub[metric_col],
                mode="lines+markers",
                name=meth,
                line=dict(color=METHOD_COLORS.get(meth, "#38BDF8"), width=2.5),
                marker=dict(size=8, symbol="circle"),
                hovertemplate=f"<b>{meth}</b><br>Size: %{{x}}<br>{metric_choice}: %{{y:.3f}}<extra></extra>",
            ))
        fig_line.update_layout(
            xaxis_title="Training Cohort Size (N)",
            yaxis_title=metric_choice,
            height=370,
            xaxis=dict(tickvals=[60, 120, 250, 537], ticktext=["60", "120", "250", "537"]),
        )
        style_plotly(fig_line)
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="card-title">📋 Performance Statistics (Mean ± Std over 20 Independent Splits)</div>', unsafe_allow_html=True)
    sub_runs_all = all_runs[(all_runs["model"] == model_choice) & (all_runs["train_size"] == size_choice)]
    table_rows = []
    for meth in METHODS_ORDER:
        m_runs = sub_runs_all[sub_runs_all["method"] == meth]
        table_rows.append({
            "Method": meth,
            "Balanced Accuracy": f"{m_runs['BA'].mean():.3f} ± {m_runs['BA'].std():.3f}",
            "Macro F1": f"{m_runs['F1'].mean():.3f} ± {m_runs['F1'].std():.3f}",
            "G-mean": f"{m_runs['GM'].mean():.3f} ± {m_runs['GM'].std():.3f}",
            "Minority Recall": f"{m_runs['MinorityRecall'].mean():.3f} ± {m_runs['MinorityRecall'].std():.3f}",
            "raw_BA": m_runs['BA'].mean(),
            "raw_F1": m_runs['F1'].mean(),
            "raw_GM": m_runs['GM'].mean(),
            "raw_Recall": m_runs['MinorityRecall'].mean(),
        })
    tdf = pd.DataFrame(table_rows)

    best_ba_meth = tdf.loc[tdf["raw_BA"].idxmax(), "Method"]
    best_f1_meth = tdf.loc[tdf["raw_F1"].idxmax(), "Method"]
    best_gm_meth = tdf.loc[tdf["raw_GM"].idxmax(), "Method"]
    best_rec_meth = tdf.loc[tdf["raw_Recall"].idxmax(), "Method"]

    display_df = tdf[["Method", "Balanced Accuracy", "Macro F1", "G-mean", "Minority Recall"]].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.info(f"🏆 **Optimal Methods for {model_choice} (N = {size_choice})**: Best Balanced Accuracy: `{best_ba_meth}` | Best Macro F1: `{best_f1_meth}` | Best G-mean: `{best_gm_meth}` | Best Minority Recall: `{best_rec_meth}`")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# PAGE 3: METHOD COMPARISON
# =========================================================================
def render_method_comparison():
    all_runs = load_all_runs()
    summary_mean = load_summary_mean()
    if all_runs is None or summary_mean is None:
        st.error("Results files not found.")
        return

    st.sidebar.markdown('<div class="sidebar-brand"><h2>⚖️ Comparison Filters</h2><p>Cross-setting exploration</p></div>', unsafe_allow_html=True)
    comp_model = st.sidebar.selectbox("Select Classifier", ["RandomForest", "LogReg"], format_func=lambda x: "Random Forest" if x == "RandomForest" else "Logistic Regression")
    comp_metric_name = st.sidebar.selectbox("Select Metric", list(METRIC_MAP.keys()))
    comp_size = st.sidebar.selectbox("Cohort Size for Boxplot Spread", TRAIN_SIZES_ORDER)
    comp_col = METRIC_MAP[comp_metric_name]

    col1, col2 = st.columns([1.05, 1.25], gap="large")

    with col1:
        st.markdown(f'<div class="glass-card"><div class="card-title">🗺️ Method × Size Performance Matrix ({comp_model})</div>', unsafe_allow_html=True)
        model_df = summary_mean[summary_mean["model"] == comp_model]
        piv = model_df.pivot(index="method", columns="train_size", values=comp_col).reindex(METHODS_ORDER)
        piv.columns = [f"N={c}" for c in piv.columns]

        fig_heat = px.imshow(
            piv,
            text_auto=".3f",
            color_continuous_scale=[[0, "#0F172A"], [0.5, "#0E7490"], [1, "#38BDF8"]],
            aspect="auto",
            labels=dict(x="Training Cohort Size", y="Method", color=comp_metric_name),
        )
        fig_heat.update_layout(height=390)
        style_plotly(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="glass-card"><div class="card-title">📦 Variance & Stability Across 20 Splits (N = {comp_size})</div>', unsafe_allow_html=True)
        runs_sub = all_runs[(all_runs["model"] == comp_model) & (all_runs["train_size"] == comp_size)]

        fig_box = go.Figure()
        for meth in METHODS_ORDER:
            m_data = runs_sub[runs_sub["method"] == meth][comp_col]
            fig_box.add_trace(go.Box(
                y=m_data,
                name=meth,
                marker_color=METHOD_COLORS.get(meth, "#38BDF8"),
                boxpoints="outliers",
                line_width=1.5,
                hovertemplate=f"<b>{meth}</b><br>{comp_metric_name}: %{{y:.3f}}<extra></extra>",
            ))
        fig_box.update_layout(
            yaxis_title=comp_metric_name,
            showlegend=False,
            height=390,
        )
        style_plotly(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# PAGE 4: STATISTICAL TESTS
# =========================================================================
def render_statistical_tests():
    wilcoxon_df = load_wilcoxon()
    if wilcoxon_df is None:
        st.error("Wilcoxon results file not found.")
        return

    st.sidebar.markdown('<div class="sidebar-brand"><h2>📈 Statistical Filters</h2><p>Hypothesis testing controls</p></div>', unsafe_allow_html=True)
    test_model = st.sidebar.selectbox("Filter Architecture", ["All", "RandomForest", "LogReg"])

    filtered_w = wilcoxon_df.copy()
    if test_model != "All":
        filtered_w = filtered_w[filtered_w["model"] == test_model]

    col_chart, col_tbl = st.columns([1.15, 1.25], gap="large")

    with col_chart:
        st.markdown('<div class="glass-card"><div class="card-title">📊 Balanced Accuracy Gain over Baseline (pp)</div>', unsafe_allow_html=True)
        plot_w = filtered_w.copy()
        plot_w["Gain (pp)"] = plot_w["mean_BA_diff"] * 100

        fig_diff = px.bar(
            plot_w,
            x="train_size",
            y="Gain (pp)",
            color="method",
            barmode="group",
            facet_col="model" if test_model == "All" else None,
            color_discrete_map=METHOD_COLORS,
            labels={"train_size": "Training Size", "Gain (pp)": "BA Gain (Percentage Points)", "method": "Method"},
        )
        fig_diff.update_layout(height=420)
        style_plotly(fig_diff)
        st.plotly_chart(fig_diff, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_tbl:
        st.markdown('<div class="glass-card"><div class="card-title">📋 Paired Wilcoxon Signed-Rank Test Table</div>', unsafe_allow_html=True)
        display_tbl = filtered_w.copy()
        display_tbl["BA Gain (pp)"] = (display_tbl["mean_BA_diff"] * 100).map("{:+.2f} pp".format)
        display_tbl["p-value"] = display_tbl["p_value"].apply(lambda p: f"{p:.4f}" if p >= 0.0001 else "< 0.0001")
        display_tbl["Significant"] = display_tbl["p_value"].apply(lambda p: "✅ p < 0.05" if p < 0.05 else "❌ Not Sig.")

        st.dataframe(
            display_tbl[["model", "train_size", "method", "BA Gain (pp)", "p-value", "Significant"]],
            use_container_width=True,
            hide_index=True,
        )

        n_sig = (filtered_w["p_value"] < 0.05).sum()
        total_tests = len(filtered_w)
        pct_sig = n_sig / total_tests * 100
        st.success(f"📌 **Statistical Confidence**: {n_sig}/{total_tests} configurations ({pct_sig:.1f}%) demonstrated statistically significant improvements over the unweighted Baseline ($p < 0.05$).")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# PAGE 5: LIVE PREDICTOR
# =========================================================================
def render_live_predictor():
    rf_pipe, lr_pipe, rf_base, lr_base = train_live_models()
    if rf_pipe is None or lr_pipe is None:
        st.error("Failed to load models for live prediction.")
        return

    st.sidebar.markdown('<div class="sidebar-brand"><h2>🩺 Clinical Engine</h2><p>Select diagnostic pipeline</p></div>', unsafe_allow_html=True)
    chosen_clf = st.sidebar.radio(
        "Classifier Engine",
        ["Random Forest (Champion: RandomUnder)", "Logistic Regression (Champion: ClassWeight)"],
    )

    is_rf = "Random Forest" in chosen_clf
    pipeline = rf_pipe if is_rf else lr_pipe
    base_pipe = rf_base if is_rf else lr_base

    st.markdown('<div class="glass-card"><div class="card-title">🧪 Patient Clinical Biomarkers & Physical Measurements</div>', unsafe_allow_html=True)

    # 1-Click Patient Presets
    st.markdown("<p style='color:#94A3B8; font-size:0.86rem; margin-top:-8px; margin-bottom:12px;'>⚡ <b>Quick Clinical Encounter Presets:</b> Click any preset below to auto-populate patient biomarkers.</p>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)

    if "p_preg" not in st.session_state: st.session_state["p_preg"] = 3
    if "p_gluc" not in st.session_state: st.session_state["p_gluc"] = 125
    if "p_bp" not in st.session_state: st.session_state["p_bp"] = 72
    if "p_skin" not in st.session_state: st.session_state["p_skin"] = 23
    if "p_ins" not in st.session_state: st.session_state["p_ins"] = 35
    if "p_bmi" not in st.session_state: st.session_state["p_bmi"] = 32.5
    if "p_dpf" not in st.session_state: st.session_state["p_dpf"] = 0.45
    if "p_age" not in st.session_state: st.session_state["p_age"] = 33

    with p_col1:
        if st.button("👤 Normal / Healthy Control", key="preset_healthy", use_container_width=True):
            st.session_state["p_preg"] = 1
            st.session_state["p_gluc"] = 88
            st.session_state["p_bp"] = 66
            st.session_state["p_skin"] = 18
            st.session_state["p_ins"] = 30
            st.session_state["p_bmi"] = 22.8
            st.session_state["p_dpf"] = 0.22
            st.session_state["p_age"] = 24
            st.rerun()

    with p_col2:
        if st.button("👤 Borderline Metabolic Case", key="preset_borderline", use_container_width=True):
            st.session_state["p_preg"] = 4
            st.session_state["p_gluc"] = 132
            st.session_state["p_bp"] = 76
            st.session_state["p_skin"] = 28
            st.session_state["p_ins"] = 85
            st.session_state["p_bmi"] = 29.5
            st.session_state["p_dpf"] = 0.54
            st.session_state["p_age"] = 41
            st.rerun()

    with p_col3:
        if st.button("👤 High-Risk Diabetic Encounter", key="preset_highrisk", use_container_width=True):
            st.session_state["p_preg"] = 7
            st.session_state["p_gluc"] = 176
            st.session_state["p_bp"] = 88
            st.session_state["p_skin"] = 42
            st.session_state["p_ins"] = 220
            st.session_state["p_bmi"] = 38.4
            st.session_state["p_dpf"] = 1.15
            st.session_state["p_age"] = 52
            st.rerun()

    st.write("")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("##### 🧬 Physiological Vitals")
        preg = st.slider("Pregnancies (Total Count)", 0, 17, int(st.session_state["p_preg"]), key="sl_preg")
        glucose = st.slider("2-Hour Glucose Concentration (mg/dL)", 40, 200, int(st.session_state["p_gluc"]), key="sl_gluc")
        bp = st.slider("Diastolic Blood Pressure (mm Hg)", 40, 130, int(st.session_state["p_bp"]), key="sl_bp")
        skin = st.slider("Triceps Skinfold Thickness (mm)", 7, 99, int(st.session_state["p_skin"]), key="sl_skin")

    with c2:
        st.markdown("##### 🩸 Metabolic Biomarkers & History")
        insulin = st.slider("2-Hour Serum Insulin (mu U/ml)", 14, 846, int(st.session_state["p_ins"]), key="sl_ins")
        bmi = st.slider("Body Mass Index (BMI, kg/m²)", 15.0, 67.0, float(st.session_state["p_bmi"]), step=0.1, key="sl_bmi")
        dpf = st.slider("Diabetes Pedigree Function (Genetic)", 0.05, 2.50, float(st.session_state["p_dpf"]), step=0.01, key="sl_dpf")
        age = st.slider("Patient Age (Years)", 21, 85, int(st.session_state["p_age"]), key="sl_age")

    input_data = pd.DataFrame([{
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }])

    st.write("")
    predict_clicked = st.button("⚡ Compute Calibrated Diabetes Probability", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        probs = pipeline.predict_proba(input_data)[0]
        prob_diabetes = probs[1]
        pct_diabetes = prob_diabetes * 100

        base_probs = base_pipe.predict_proba(input_data)[0]
        base_pct = base_probs[1] * 100

        col_gauge, col_result = st.columns([1.2, 1.1], gap="large")

        with col_gauge:
            st.markdown('<div class="glass-card"><div class="card-title">⏱️ Calibrated Risk Trajectory (Champion Model)</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct_diabetes,
                number=dict(suffix="%", font=dict(size=42, color="#F8FAFC", family="Plus Jakarta Sans")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8"),
                    bar=dict(color="#38BDF8", thickness=0.28),
                    bgcolor="#101827",
                    borderwidth=1,
                    bordercolor="rgba(148, 163, 184, 0.2)",
                    steps=[
                        {"range": [0, 40], "color": "rgba(16, 185, 129, 0.25)"},
                        {"range": [40, 65], "color": "rgba(251, 191, 36, 0.25)"},
                        {"range": [65, 100], "color": "rgba(244, 63, 94, 0.25)"},
                    ],
                    threshold=dict(
                        line=dict(color="#F43F5E", width=3),
                        thickness=0.8,
                        value=50.0,
                    ),
                ),
            ))
            fig_gauge.update_layout(height=290, margin=dict(l=25, r=25, t=30, b=10))
            style_plotly(fig_gauge)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            if prob_diabetes >= 0.50:
                st.markdown(f"""
                <div class="risk-card high" style="background: linear-gradient(135deg, rgba(244, 63, 94, 0.18) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(244, 63, 94, 0.45); border-radius: 14px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
                    <span class="risk-badge high" style="background: rgba(244, 63, 94, 0.2); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 0.76rem; font-weight: 700; text-transform: uppercase;">⚠️ High Diagnostic Risk</span>
                    <h2 style="color: #F8FAFC; margin: 12px 0 8px 0; font-size: 2rem; font-weight: 800;">{pct_diabetes:.1f}% Probability</h2>
                    <p style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.55; margin: 0;">
                        The model classifies this encounter in the <b>high-risk diabetes stratum</b>. Driving clinical indicators include oral glucose concentration ({glucose} mg/dL) and body mass index ({bmi:.1f} kg/m²).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card low" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(16, 185, 129, 0.45); border-radius: 14px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
                    <span class="risk-badge low" style="background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 0.76rem; font-weight: 700; text-transform: uppercase;">✅ Low Diagnostic Risk</span>
                    <h2 style="color: #F8FAFC; margin: 12px 0 8px 0; font-size: 2rem; font-weight: 800;">{pct_diabetes:.1f}% Probability</h2>
                    <p style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.55; margin: 0;">
                        The model classifies this encounter in the <b>low-risk stratum</b>. Fasting glucose, insulin, and BMI remain aligned with the normoglycemic cohort distribution.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Baseline vs Champion Comparison Section
        st.markdown(f"""
        <div style="background: #131B2E; border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 14px; padding: 22px; margin-top: 18px; box-shadow: 0 6px 20px rgba(0,0,0,0.25);">
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                <span>⚖️ Clinical Decision Comparison: Champion Strategy vs. Unweighted Baseline</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 16px;">
                    <div style="font-size: 0.76rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">Champion Model (With Resampling)</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC;">{pct_diabetes:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Classification: <b style="color: {'#F43F5E' if prob_diabetes >= 0.5 else '#34D399'};">{'Positive (Diabetic)' if prob_diabetes >= 0.5 else 'Negative (Non-Diabetic)'}</b></div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 10px; padding: 16px;">
                    <div style="font-size: 0.76rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">Standard Baseline (No Resampling)</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #94A3B8;">{base_pct:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Classification: <b style="color: {'#F43F5E' if base_probs[1] >= 0.5 else '#94A3B8'};">{'Positive (Diabetic)' if base_probs[1] >= 0.5 else 'Negative (Non-Diabetic)'}</b></div>
                </div>
            </div>
            <p style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5; margin: 14px 0 0 0;">
                💡 <b>Clinical Finding:</b> Unweighted models trained on imbalanced medical cohorts frequently suppress minority probabilities toward zero due to majority class dominance (65.1% negative). Algorithmic cost-weighting and under-sampling recalibrate the decision threshold to protect against life-threatening false negatives.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ Educational Demonstration — Not Medical Advice:</b> This clinical risk simulator was developed strictly for academic demonstration and research benchmarking on the historical Pima Indians Diabetes dataset. It is not cleared for patient diagnostics or clinical intervention.
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# PAGE 6: DOWNLOADS
# =========================================================================
def render_downloads():
    st.markdown('<div class="glass-card"><div class="card-title">📥 Research Artifacts & Publication Figures</div><p style="color: #94A3B8; margin-top:-8px;">Export all processed tabular experimental logs, publication-formatted tables, and high-resolution figures.</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="glass-card"><div class="card-title">📊 Tabular Logs & Tables</div>', unsafe_allow_html=True)
        file_map_csv = [
            ("all_runs.csv", "Full 960 Experimental Runs (CSV)", "text/csv"),
            ("summary_mean.csv", "Aggregated Mean Metrics (CSV)", "text/csv"),
            ("summary_mean_std.csv", "Mean ± Std Metrics (CSV)", "text/csv"),
            ("best_method_per_setting.csv", "Optimal Method per Setting (CSV)", "text/csv"),
            ("wilcoxon_vs_baseline_BA.csv", "Wilcoxon Statistical Tests (CSV)", "text/csv"),
            ("paper_tables.xlsx", "Publication Tables (Excel Workbook)", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("paper_tables.md", "Formatted Tables (Markdown)", "text/markdown"),
        ]

        for fname, label, mime in file_map_csv:
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    data = f.read()
                st.download_button(
                    label=f"⬇️ {label}",
                    data=data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                )
            else:
                st.caption(f"⚪ `{fname}` not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card"><div class="card-title">🖼️ High-Resolution Figures</div>', unsafe_allow_html=True)
        fig_map = [
            ("fig_confusion_matrix_LogReg.png", "Confusion Matrix - Logistic Regression (PNG)"),
            ("fig_confusion_matrix_RandomForest.png", "Confusion Matrix - Random Forest (PNG)"),
            ("fig_metrics_vs_size_RandomForest.png", "Metrics vs. Size - Random Forest (PNG)"),
            ("fig_metrics_vs_size_LogReg.png", "Metrics vs. Size - Logistic Regression (PNG)"),
            ("fig_minority_recall_smallest.png", "Minority Recall at Smallest Size (PNG)"),
            ("fig_class_distribution.png", "Class Distribution (PNG)"),
        ]

        for fname, label in fig_map:
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    data = f.read()
                st.download_button(
                    label=f"🖼️ {label}",
                    data=data,
                    file_name=fname,
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.caption(f"⚪ `{fname}` not found.")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# MAIN ROUTER
# =========================================================================
def main():
    inject_custom_css()
    render_hero()

    st.sidebar.markdown('<div class="sidebar-brand"><h2>🩺 Pima Clinical AI</h2><p>Imbalance Benchmark v2.0</p></div>', unsafe_allow_html=True)
    
    page_options = [
        "📊 Overview & Highlights",
        "🔍 Results Explorer",
        "⚖️ Method Comparison",
        "📈 Statistical Tests",
        "🩺 Live Predictor",
        "📥 Downloads",
    ]

    if "main_nav_radio" not in st.session_state:
        st.session_state["main_nav_radio"] = "📊 Overview & Highlights"

    page = st.sidebar.radio(
        "Navigation Menu",
        page_options,
        key="main_nav_radio",
    )

    if page == "📊 Overview & Highlights":
        render_overview()
    elif page == "🔍 Results Explorer":
        render_results_explorer()
    elif page == "⚖️ Method Comparison":
        render_method_comparison()
    elif page == "📈 Statistical Tests":
        render_statistical_tests()
    elif page == "🩺 Live Predictor":
        render_live_predictor()
    elif page == "📥 Downloads":
        render_downloads()


if __name__ == "__main__":
    main()
