"""
Customer Segmentation & Churn Engine
======================================
A decision intelligence platform that mirrors what Uber, Netflix,
Salesforce, and HubSpot run in production for customer retention.

Five pages:
  1. Segmentation Explorer  — UMAP clusters, segment profiles, bootstrap stability
  2. Churn Risk Dashboard   — Per-segment models, calibrated probabilities, SHAP
  3. Uplift Intelligence    — Persuadable identification, ROI ranking, causal ML
  4. Retention Actions      — Batch LLM plans + multi-turn AI Customer Assistant
  5. Audit & Analytics      — Audit trail, outcome tracking, feedback loop
"""

import os
import sys
import json
import uuid
import warnings
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add src to path
SRC_PATH = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_PATH)

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")
PLAYBOOK_PATH = os.path.join(os.path.dirname(__file__), "data", "playbook.json")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Design system CSS ───────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    /* ── BASE ──────────────────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: #F4F6FF !important;
    }
    .main .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1360px;
    }
    /* Body text — consistent 15px everywhere */
    p, li, td, th { font-size: 15px !important; line-height: 1.65 !important; }
    label { font-size: 14px !important; font-weight: 500 !important; }

    /* ── SIDEBAR — vibrant indigo gradient ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(175deg, #312E81 0%, #4F46E5 55%, #7C3AED 100%) !important;
        border-right: none !important;
        box-shadow: 6px 0 30px rgba(79,70,229,0.22) !important;
    }
    section[data-testid="stSidebar"] > div { background: transparent !important; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span:not([data-testid]),
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div { color: #DDD6FE !important; font-size: 14px !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important; background: none !important;
        border-bottom: none !important; border-left: none !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important;
        font-size: 10px !important; padding: 0 !important; box-shadow: none !important;
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.18) !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #C4B5FD !important; font-weight: 500 !important; font-size: 14px !important;
        padding: 0.45rem 0.9rem !important; border-radius: 10px !important; transition: all 0.15s !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        color: #FFFFFF !important; background: rgba(255,255,255,0.14) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        color: #FFFFFF !important; background: rgba(255,255,255,0.22) !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] input[type="password"],
    section[data-testid="stSidebar"] input[type="number"] {
        background: rgba(255,255,255,0.14) !important;
        border: 1.5px solid rgba(255,255,255,0.32) !important;
        border-radius: 10px !important; color: #FFFFFF !important; font-size: 14px !important;
    }
    section[data-testid="stSidebar"] input::placeholder { color: rgba(255,255,255,0.45) !important; }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background: rgba(255,255,255,0.14) !important;
        border-color: rgba(255,255,255,0.24) !important; color: #FFFFFF !important;
    }

    /* ── HEADINGS ──────────────────────────────────────────────────────────── */
    h1 {
        font-size: 30px !important; font-weight: 800 !important;
        color: #1E1B4B !important; letter-spacing: -0.03em !important;
        line-height: 1.2 !important; padding-bottom: 0.7rem !important;
        margin-bottom: 1.25rem !important; border-bottom: 4px solid #4F46E5 !important;
    }
    /* h2 — colored box, white text, rounded — the signature "heading pop" */
    h2 {
        font-size: 17px !important; font-weight: 700 !important;
        color: #FFFFFF !important; letter-spacing: 0 !important;
        background: linear-gradient(110deg, #4338CA 0%, #7C3AED 100%) !important;
        padding: 0.55rem 1.1rem !important; border-radius: 12px !important;
        margin: 1.5rem 0 0.9rem !important; border: none !important;
        box-shadow: 0 4px 16px rgba(67,56,202,0.28) !important;
        display: block !important;
    }
    /* h3 — left-bordered pill label */
    h3 {
        font-size: 13px !important; font-weight: 700 !important;
        color: #4F46E5 !important; text-transform: uppercase !important;
        letter-spacing: 0.09em !important; line-height: 1.4 !important;
        background: #EEF2FF !important; padding: 0.3rem 0.85rem !important;
        border-radius: 0 8px 8px 0 !important;
        border-left: 4px solid #4F46E5 !important; margin-bottom: 0.5rem !important;
    }

    /* ── BUTTONS ───────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%) !important;
        color: #FFFFFF !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        font-size: 14px !important; padding: 0.55rem 1.5rem !important;
        box-shadow: 0 3px 12px rgba(79,70,229,0.4), 0 1px 0 rgba(255,255,255,0.12) inset !important;
        transition: all 0.15s ease !important; letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%) !important;
        box-shadow: 0 8px 24px rgba(79,70,229,0.5) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #34D399 0%, #059669 100%) !important;
        box-shadow: 0 3px 12px rgba(5,150,105,0.4) !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #6EE7B7 0%, #34D399 100%) !important;
        box-shadow: 0 8px 24px rgba(5,150,105,0.5) !important;
    }

    /* ── DROPDOWNS — big, white, clearly bordered ───────────────────────────── */
    [data-baseweb="select"] > div:first-child {
        background: #FFFFFF !important; border: 2px solid #818CF8 !important;
        border-radius: 11px !important; min-height: 48px !important;
        font-size: 15px !important; color: #1E1B4B !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
        padding: 0.1rem 0.25rem !important;
    }
    [data-baseweb="select"] > div:first-child:hover { border-color: #4F46E5 !important; }
    [data-baseweb="select"]:focus-within > div:first-child {
        border-color: #4F46E5 !important; box-shadow: 0 0 0 4px rgba(99,102,241,0.18) !important;
    }
    [data-baseweb="popover"] ul, [data-baseweb="menu"] {
        background: #FFFFFF !important; border: 2px solid #C7D2FE !important;
        border-radius: 12px !important; box-shadow: 0 12px 40px rgba(79,70,229,0.18) !important;
        font-size: 15px !important;
    }
    [data-baseweb="option"]:hover, [data-baseweb="option"][aria-selected="true"] {
        background: #EEF2FF !important; color: #4338CA !important; font-weight: 600 !important;
    }

    /* Text/number inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important; border: 2px solid #818CF8 !important;
        border-radius: 11px !important; min-height: 44px !important;
        font-size: 15px !important; color: #1E1B4B !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,0.18) !important; outline: none !important;
    }
    [data-testid="stMultiSelect"] > div > div {
        background: #FFFFFF !important; border: 2px solid #818CF8 !important;
        border-radius: 11px !important; min-height: 48px !important;
    }
    [data-baseweb="tag"] {
        background: #EEF2FF !important; color: #4338CA !important;
        border: 1.5px solid #C7D2FE !important; border-radius: 8px !important;
        font-size: 13px !important; font-weight: 700 !important; padding: 0.1rem 0.5rem !important;
    }

    /* ── METRIC CARDS ──────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #FFFFFF !important; border-radius: 16px !important;
        padding: 1.25rem 1.4rem !important;
        border: 2px solid #E0E7FF !important; border-top: 5px solid #6366F1 !important;
        box-shadow: 0 4px 18px rgba(79,70,229,0.10) !important;
        transition: box-shadow 0.2s, transform 0.15s !important;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 8px 28px rgba(79,70,229,0.18) !important; transform: translateY(-3px) !important;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 11px !important; font-weight: 800 !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important; color: #7C3AED !important;
    }
    [data-testid="stMetricValue"] > div {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 30px !important; font-weight: 700 !important;
        color: #1E1B4B !important; letter-spacing: -0.03em !important;
    }
    [data-testid="stMetricDelta"] > div { font-size: 14px !important; font-weight: 600 !important; }

    /* ── TABS ──────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important; border-bottom: 2.5px solid #C7D2FE !important; gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px !important; font-weight: 600 !important; color: #6B7280 !important;
        background: transparent !important; border-bottom: 3px solid transparent !important;
        margin-bottom: -2.5px !important; padding: 0.65rem 1.3rem !important;
        border-radius: 8px 8px 0 0 !important; transition: all 0.15s !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #4F46E5 !important; background: #EEF2FF !important; }
    .stTabs [aria-selected="true"] {
        color: #4F46E5 !important; border-bottom: 3px solid #4F46E5 !important;
        background: transparent !important; font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

    /* ── CARDS / BORDERED CONTAINERS ───────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important; border: 2px solid #E0E7FF !important;
        border-radius: 16px !important; box-shadow: 0 4px 18px rgba(79,70,229,0.08) !important;
        overflow: hidden !important;
    }

    /* ── EXPANDERS ─────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 2px solid #DDD6FE !important; border-radius: 14px !important;
        background: #FFFFFF !important; box-shadow: 0 3px 12px rgba(79,70,229,0.07) !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 15px !important; font-weight: 600 !important; color: #1E1B4B !important;
        padding: 0.85rem 1.1rem !important; background: #F5F3FF !important; transition: background 0.15s !important;
    }
    [data-testid="stExpander"] summary:hover { background: #EEF2FF !important; color: #4F46E5 !important; }
    [data-testid="stExpander"][open] > summary {
        background: #EEF2FF !important; color: #4F46E5 !important;
        border-bottom: 2px solid #DDD6FE !important;
    }

    /* ── ALERTS ─────────────────────────────────────────────────────────────── */
    [data-testid="stAlert"] { border-radius: 12px !important; font-size: 15px !important; }

    /* ── TOGGLE ─────────────────────────────────────────────────────────────── */
    [role="switch"][aria-checked="true"] { background-color: #4F46E5 !important; }

    /* ── DATAFRAMES — pop the headers ──────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important; overflow: hidden !important;
        border: 2px solid #DDD6FE !important;
        box-shadow: 0 4px 16px rgba(79,70,229,0.09) !important;
    }
    /* Attempt to style dataframe headers via canvas font override */
    [data-testid="stDataFrame"] canvas { border-radius: 14px !important; }

    /* ── CHARTS — full width, no clipping ──────────────────────────────────── */
    [data-testid="stPlotlyChart"] { border-radius: 14px !important; overflow: hidden !important; }
    [data-testid="stPlotlyChart"] > div { border-radius: 14px !important; }

    /* ── CHAT — fully light, bigger input ──────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important; background: #FFFFFF !important;
        border: 2px solid #DDD6FE !important; box-shadow: 0 3px 12px rgba(79,70,229,0.07) !important;
        margin-bottom: 0.9rem !important; font-size: 15px !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stChatInputContainer"], [data-testid="stChatInput"] {
        background: #FFFFFF !important; border-top: 2px solid #DDD6FE !important;
        padding-top: 0.75rem !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important; border: 2px solid #818CF8 !important;
        border-radius: 12px !important; color: #1E1B4B !important;
        font-size: 15px !important; min-height: 80px !important;
        font-family: 'Inter', sans-serif !important; padding: 0.75rem 1rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4F46E5 !important; box-shadow: 0 0 0 4px rgba(99,102,241,0.18) !important;
    }

    /* ── CAPTIONS ──────────────────────────────────────────────────────────── */
    [data-testid="stCaptionContainer"] p, .stCaption, small {
        color: #7C3AED !important; font-size: 13px !important; font-style: italic !important;
    }

    /* ── DIVIDERS ──────────────────────────────────────────────────────────── */
    hr { border-color: #DDD6FE !important; margin: 1.75rem 0 !important; }

    /* ── SLIDER ─────────────────────────────────────────────────────────────── */
    [data-testid="stSlider"] [role="slider"] {
        background: #4F46E5 !important; border: 3px solid #6366F1 !important;
    }

    /* ── HIDE STREAMLIT CHROME ─────────────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    </style>
    """, unsafe_allow_html=True)


# ─── Colour palette (consistent across all charts) ──────────────────────────
SEGMENT_COLORS = {
    "Champions": "#2ECC71",
    "Loyal Customers": "#3498DB",
    "At-Risk": "#E74C3C",
    "Price Sensitive": "#F39C12",
    "Lapsed": "#95A5A6",
}

CUSTOMER_TYPE_COLORS = {
    "Persuadable": "#27AE60",
    "Sure Thing": "#2980B9",
    "Lost Cause": "#E74C3C",
    "Sleeping Dog": "#7F8C8D",
}

RISK_COLORS = {
    "High Risk": "#E74C3C",
    "Medium Risk": "#F39C12",
    "Low Risk": "#2ECC71",
}


# ─── Data Loading (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(os.path.join(PROCESSED_PATH, "uplift.parquet"))
    return df


@st.cache_data
def load_stability():
    path = os.path.join(MODELS_PATH, "stability.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_segment_models():
    path = os.path.join(MODELS_PATH, "segment_models.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {}


@st.cache_data
def load_playbook() -> dict:
    if os.path.exists(PLAYBOOK_PATH):
        with open(PLAYBOOK_PATH) as f:
            return json.load(f)
    return {}


def build_segment_profiles(df):
    cols = [
        "EngagementScore",
        "RecencySignal",
        "StickinessIndex",
        "SpendTrend",
        "SupportRiskScore",
        "DiscountSensitivity",
        "TenureStability",
        "Churn",
    ]
    available = [c for c in cols if c in df.columns]
    profile = df.groupby("Segment")[available].mean().round(3)
    profile["CustomerCount"] = df.groupby("Segment").size()
    profile["ChurnRate"] = df.groupby("Segment")["Churn"].mean().round(3)
    return profile


# ─── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.markdown("""
    <div style="padding:1.5rem 0.75rem 1.25rem; border-bottom:1px solid rgba(255,255,255,0.15); margin-bottom:1.1rem;">
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.35rem;">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="lg1" x1="0" y1="0" x2="38" y2="38" gradientUnits="userSpaceOnUse">
                        <stop offset="0%" stop-color="#A78BFA"/>
                        <stop offset="100%" stop-color="#6366F1"/>
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2.5" result="blur"/>
                        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                </defs>
                <rect width="38" height="38" rx="10" fill="url(#lg1)" filter="url(#glow)"/>
                <rect x="8" y="24" width="5" height="8" rx="2" fill="white" opacity="0.9"/>
                <rect x="16.5" y="17" width="5" height="15" rx="2" fill="white"/>
                <rect x="25" y="10" width="5" height="22" rx="2" fill="white" opacity="0.75"/>
                <path d="M10.5 20 L19 13 L27.5 7" stroke="#FCD34D" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="27.5" cy="7" r="2.5" fill="#FCD34D"/>
            </svg>
            <div>
                <div style="font-size:1.05rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em; font-family:'Inter',sans-serif; line-height:1.1;">Churn Engine</div>
                <div style="font-size:0.65rem; color:#C4B5FD; font-weight:600; text-transform:uppercase; letter-spacing:0.12em; font-family:'Inter',sans-serif;">Decision Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigate",
        [
            "Segmentation Explorer",
            "Churn Risk Dashboard",
            "Uplift Intelligence",
            "Retention Actions",
            "Audit & Analytics",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Summary**")
    st.sidebar.metric("Total Customers", f"{len(df):,}")
    st.sidebar.metric("Segments", df["Segment"].nunique())
    st.sidebar.metric(
        "Persuadables",
        f"{(df['CustomerType'] == 'Persuadable').sum():,}",
        help="Customers who will churn AND respond to intervention",
    )
    st.sidebar.metric(
        "High Risk",
        f"{(df['RiskTier'] == 'High Risk').sum():,}",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Architecture**")
    st.sidebar.markdown(
        "- K-Means++ segmentation\n"
        "- GMM soft assignments\n"
        "- UMAP visualization\n"
        "- Bootstrap ARI stability\n"
        "- Per-segment XGBoost\n"
        "- Isotonic calibration\n"
        "- CausalML uplift (T+S Learner)\n"
        "- Llama 3.3 retention actions (Groq)"
    )

    return page


# ─── UMAP caption lookup (changes dynamically with Colour by dropdown) ───────
_UMAP_CAPTIONS = {
    "Segment": (
        "Each dot is one customer. Dots close together behave similarly — "
        "similar purchase frequency, spend, engagement, and recency. "
        "Colors show the 5 behavioral segments discovered by K-Means++."
    ),
    "Churn": (
        "Red dots represent customers who actually churned; blue stayed. "
        "Dense red clusters reveal which behavioral regions carry the highest "
        "real-world churn concentration."
    ),
    "RiskTier": (
        "Green = Low Risk · Orange = Medium Risk · Red = High Risk. "
        "Customers in red zones are the highest priority for retention intervention."
    ),
    "CustomerType": (
        "Green = Persuadable (target for intervention) · Blue = Sure Thing (stays anyway) · "
        "Red = Lost Cause (won't respond to intervention) · Gray = Sleeping Dog (do not contact — "
        "intervention may trigger churn)."
    ),
    "EngagementScore": (
        "Darker red = less engaged customer. Disengaged clusters overlapping with "
        "churned customers confirm that falling engagement is a leading churn indicator."
    ),
    "ChurnProbability": (
        "Darker red = higher predicted churn probability from the XGBoost model. "
        "Compare this view with Segment view to see which cohorts carry the most model-predicted risk."
    ),
    "UpliftScore": (
        "Green = high uplift (responds to intervention) · Red = low/negative uplift (won't respond). "
        "Only customers in green zones have positive expected ROI from a retention campaign."
    ),
}


# ─── Page 1: Segmentation Explorer ──────────────────────────────────────────
def page_segmentation(df):
    st.title("Customer Segmentation Explorer")
    st.markdown(
        "Customers are segmented into behavioral cohorts using **K-Means++**, "
        "validated with **Gaussian Mixture Models** (soft probability assignments), "
        "and visualized in 2D using **UMAP**. Segment stability is validated via "
        "**bootstrap Adjusted Rand Index** across 100 resamplings — the same "
        "validation approach used in production ML systems."
    )
    st.info(
        "**What are these charts?** These are NOT geographic maps. "
        "**UMAP** (Uniform Manifold Approximation and Projection) is a mathematical technique that "
        "takes 13 behavioral features per customer — purchase frequency, spend trends, "
        "satisfaction scores, app engagement — and compresses them into a 2D scatter plot "
        "so you can visually see which customers behave similarly. "
        "Companies like Netflix, Spotify, and Uber use UMAP to visualize customer segments. "
        "The X and Y axes have no real-world label — they represent behavioral distance in feature space.",
        icon="ℹ️",
    )

    stability = load_stability()

    # ── KPIs ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, (seg, color) in enumerate(SEGMENT_COLORS.items()):
        count = (df["Segment"] == seg).sum()
        churn_rate = df[df["Segment"] == seg]["Churn"].mean() if count > 0 else 0
        [col1, col2, col3, col4, col5][i].metric(
            seg, f"{count:,}", f"{churn_rate:.1%} churn"
        )

    st.markdown("---")

    # ── UMAP Scatter — full width ────────────────────────────────────────────
    st.subheader("Customer Behavioral Space (2D Projection)")
    color_by = st.selectbox(
        "Colour by",
        [
            "Segment",
            "Churn",
            "RiskTier",
            "CustomerType",
            "EngagementScore",
            "ChurnProbability",
            "UpliftScore",
        ],
        index=0,
    )

    if color_by == "Segment":
        fig = px.scatter(
            df,
            x="UMAP_1",
            y="UMAP_2",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.7,
            hover_data=["CustomerID", "Churn", "ChurnProbability"],
            title="Customer Behavioral Space (UMAP 2D)",
        )
    elif color_by in ["EngagementScore", "ChurnProbability", "UpliftScore"]:
        fig = px.scatter(
            df,
            x="UMAP_1",
            y="UMAP_2",
            color=color_by,
            color_continuous_scale="RdYlGn_r",
            opacity=0.7,
            hover_data=["CustomerID", "Segment"],
            title=f"Customer Behavioral Space — coloured by {color_by}",
        )
    elif color_by == "RiskTier":
        fig = px.scatter(
            df,
            x="UMAP_1",
            y="UMAP_2",
            color="RiskTier",
            color_discrete_map=RISK_COLORS,
            opacity=0.7,
            hover_data=["CustomerID", "Segment", "ChurnProbability"],
            title="Customer Behavioral Space — coloured by Risk Tier",
        )
    elif color_by == "CustomerType":
        fig = px.scatter(
            df,
            x="UMAP_1",
            y="UMAP_2",
            color="CustomerType",
            color_discrete_map=CUSTOMER_TYPE_COLORS,
            opacity=0.7,
            hover_data=["CustomerID", "Segment"],
            title="Customer Behavioral Space — coloured by Customer Type",
        )
    else:
        fig = px.scatter(
            df,
            x="UMAP_1",
            y="UMAP_2",
            color=color_by,
            opacity=0.7,
            title=f"UMAP — {color_by}",
        )

    fig.update_traces(marker=dict(size=5, line=dict(width=0.3, color='white')))
    fig.update_layout(
        height=620,
        template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
        st.caption(_UMAP_CAPTIONS.get(color_by, ""))

    st.markdown("---")

    # ── Segment Stability — full width below UMAP ────────────────────────────
    st.subheader("Segment Stability Score")
    st.markdown(
        "Tests whether the same 5 segments emerge from **100 different random data samples** "
        "— proving the clusters are real behavioral patterns, not a random artifact of the training data."
    )
    if stability:
        ari = stability["mean_ari"]
        grade = stability["grade"]
        color = "#4F46E5" if ari >= 0.85 else "#F59E0B" if ari >= 0.70 else "#EF4444"

        s1, s2, s3 = st.columns(3)
        s1.metric("Mean ARI", f"{ari:.3f}", help="Adjusted Rand Index across 100 bootstrap resamplings. Above 0.70 = stable.")
        s2.metric("Std ARI", f"{stability['std_ari']:.3f}", help="Lower = more consistent across resamplings.")
        s3.metric("Stability Grade", grade)

        ari_scores = stability.get("ari_scores", [])
        if ari_scores:
            fig_ari = px.histogram(
                x=ari_scores, nbins=25,
                labels={"x": "ARI Score", "y": "Bootstrap Resamplings"},
                title="Bootstrap Stability Distribution — 100 Resamplings of the Dataset",
                color_discrete_sequence=[color],
            )
            fig_ari.add_vline(x=ari, line_dash="dash", line_color="#1E1B4B",
                              annotation_text=f"Mean = {ari:.3f}", annotation_position="top right")
            fig_ari.update_layout(
                height=420,
                template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"),
                showlegend=False, paper_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            with st.container(border=True):
                st.plotly_chart(fig_ari, use_container_width=True)
                st.caption(
                    "A tight cluster near 1.0 means the segments are highly reproducible. "
                    "ARI = 1.0 is a perfect match; ARI > 0.85 is production-grade stability."
                )
    else:
        st.warning("Stability data not available. Run pipeline first.")

    st.markdown("---")

    # ── Segment Profiles Heatmap ─────────────────────────────────────────────
    st.subheader("Segment Behavioral Profiles")
    st.caption(
        "Each row is a behavioral metric; each column is a segment. "
        "Darker red = higher value. The ChurnRate row shows actual observed churn — "
        "use this to understand which segments are most at risk and why."
    )
    profiles = build_segment_profiles(df)

    heat_cols = [
        "EngagementScore",
        "RecencySignal",
        "StickinessIndex",
        "SpendTrend",
        "SupportRiskScore",
        "DiscountSensitivity",
        "TenureStability",
        "ChurnRate",
    ]
    heat_data = profiles[[c for c in heat_cols if c in profiles.columns]]

    fig_heat = px.imshow(
        heat_data.T,
        color_continuous_scale="RdYlGn_r",
        title="Segment Profile Heatmap — darker red = higher value for that metric",
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(
        height=480,
        template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    with st.container(border=True):
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── GMM Segment Confidence ───────────────────────────────────────────────
    st.subheader("Segment Assignment Confidence")
    st.caption(
        "K-Means forces every customer into exactly one segment. "
        "GMM (Gaussian Mixture Models) goes further — it gives each customer a confidence score "
        "for their assigned segment. A score near 1.0 means the model is highly certain "
        "about where this customer belongs; a score near 0.5 means they sit on the boundary "
        "between two segments and need closer attention."
    )
    gmm_cols = [c for c in df.columns if c.startswith("GMM_Prob_Seg")]
    if gmm_cols:
        confidence = df[gmm_cols].max(axis=1)
        conf_df = pd.DataFrame({"Confidence": confidence, "Segment": df["Segment"]})

        # Per-segment breakdown: % of customers in each confidence tier
        tiers = []
        for seg in conf_df["Segment"].unique():
            seg_conf = conf_df[conf_df["Segment"] == seg]["Confidence"]
            tiers.append({"Segment": seg, "Tier": "High (≥90%)",   "Customers": (seg_conf >= 0.90).sum()})
            tiers.append({"Segment": seg, "Tier": "Medium (80–90%)", "Customers": ((seg_conf >= 0.80) & (seg_conf < 0.90)).sum()})
            tiers.append({"Segment": seg, "Tier": "Boundary (<80%)", "Customers": (seg_conf < 0.80).sum()})
        tier_df = pd.DataFrame(tiers)

        fig_conf = px.bar(
            tier_df,
            x="Segment", y="Customers", color="Tier",
            barmode="stack",
            color_discrete_map={
                "High (≥90%)": "#4F46E5",
                "Medium (80–90%)": "#F59E0B",
                "Boundary (<80%)": "#EF4444",
            },
            title="Segment Assignment Confidence — Customers by Confidence Tier per Segment",
            labels={"Customers": "Number of Customers"},
        )
        fig_conf.update_layout(
            height=480,
            template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"),
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        with st.container(border=True):
            st.plotly_chart(fig_conf, use_container_width=True)
            pct_certain = (confidence >= 0.80).mean()
            st.caption(
                f"{pct_certain:.0%} of all customers have ≥80% confidence in their segment. "
                "Indigo = clearly placed, amber = borderline, red = sits between two segments and warrants review."
            )


# ─── Page 2: Churn Risk Dashboard ───────────────────────────────────────────
def page_churn_risk(df):
    st.title("Churn Risk Dashboard")
    st.markdown(
        "Per-segment XGBoost classifiers with **isotonic probability calibration**. "
        "Calibrated probabilities are used for ROI calculations — a raw 0.7 score "
        "doesn't mean 70% of customers churn, but a calibrated 0.7 does. "
        "This matches how Salesforce Einstein and HubSpot score customer health."
    )

    seg_models = load_segment_models()

    # ── Model Performance ───────────────────────────────────────────────────
    st.subheader("Per-Segment Model Performance")
    st.caption(
        "A separate XGBoost model was trained for each customer segment. "
        "CV AUC measures how well the model separates churners from non-churners (1.0 = perfect, 0.5 = random). "
        "Brier Score measures probability calibration quality — lower is better."
    )
    metrics_data = []
    for seg, model_dict in seg_models.items():
        if model_dict and "metrics" in model_dict:
            m = model_dict["metrics"]
            metrics_data.append(
                {
                    "Segment": seg,
                    "Customers": m.get("n_customers", 0),
                    "Churn Rate": f"{m.get('churn_rate', 0):.1%}",
                    "CV AUC": f"{m.get('cv_auc', 0):.3f}",
                    "CV AP": f"{m.get('cv_ap', 0):.3f}",
                    "Brier Score": f"{m.get('brier_score', 0):.4f}",
                }
            )
    if metrics_data:
        st.dataframe(
            pd.DataFrame(metrics_data).set_index("Segment"),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Risk Distribution ───────────────────────────────────────────────────
    st.subheader("Churn Probability Distribution")
    st.caption(
        "Bars show how many customers fall at each predicted churn probability. "
        "A spike near 1.0 means the model is confidently identifying high-risk customers. "
        "Filter by segment to see the risk distribution for individual cohorts."
    )
    seg_filter = st.multiselect(
        "Filter by segment",
        df["Segment"].unique().tolist(),
        default=df["Segment"].unique().tolist(),
    )
    df_filtered = df[df["Segment"].isin(seg_filter)]

    fig_hist = px.histogram(
        df_filtered,
        x="ChurnProbability",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        nbins=40,
        barmode="overlay",
        opacity=0.75,
        title="Calibrated Churn Probability Distribution by Segment",
        labels={"ChurnProbability": "Churn Probability"},
    )
    fig_hist.update_layout(
        height=480, template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"),
        paper_bgcolor="#FFFFFF", margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    with st.container(border=True):
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    st.subheader("Risk Tier Breakdown")
    st.caption(
        "Customers bucketed into Low / Medium / High Risk based on predicted churn probability. "
        "High Risk = churn probability above 60%. Use this to size your retention budget by segment."
    )
    risk_seg = df_filtered.groupby(["Segment", "RiskTier"]).size().reset_index(name="Count")
    fig_risk = px.bar(
        risk_seg, x="Segment", y="Count", color="RiskTier",
        color_discrete_map=RISK_COLORS,
        title="Risk Tier Distribution by Segment",
        barmode="group",
    )
    fig_risk.update_layout(
        height=480, template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"),
        paper_bgcolor="#FFFFFF", margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    with st.container(border=True):
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # ── SHAP Feature Importance per Segment ─────────────────────────────────
    st.subheader("Top Churn Drivers by Segment")
    st.caption(
        "This chart answers: 'Why is this segment churning?' — not just 'Who is churning?'. "
        "Each bar shows how much a feature contributes to the churn prediction for the selected segment. "
        "Different segments churn for different reasons, which is why one global model performs worse "
        "than 5 dedicated segment models."
    )
    selected_seg = st.selectbox("Select segment", list(seg_models.keys()))
    if selected_seg in seg_models and seg_models[selected_seg]:
        mean_abs_shap = seg_models[selected_seg]["mean_abs_shap"]
        shap_df = mean_abs_shap.reset_index()
        shap_df.columns = ["Feature", "Importance"]
        shap_df = shap_df.head(15)

        fig_shap = px.bar(
            shap_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Reds",
            title=f"Top Churn Drivers — {selected_seg} Segment",
        )
        fig_shap.update_layout(
            height=520,
            template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        with st.container(border=True):
            st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")

    # ── Customer Risk Table ──────────────────────────────────────────────────
    st.subheader("Customer Risk Table")
    st.caption(
        "All customers ranked by predicted churn probability. "
        "Use the filters to drill into specific risk tiers or segments. "
        "In a production CRM (Salesforce, HubSpot), this list would feed directly "
        "into a campaign queue for the retention team."
    )

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        risk_tier_filter = st.multiselect(
            "Filter by Risk Tier",
            ["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk"],
        )
    with table_col2:
        show_n = st.slider("Rows to display", min_value=25, max_value=500, value=100, step=25)

    table_df = df_filtered[df_filtered["RiskTier"].isin(risk_tier_filter)].nlargest(
        show_n, "ChurnProbability"
    )

    display_cols = [
        "CustomerID",
        "Segment",
        "ChurnProbability",
        "RiskTier",
        "CustomerType",
        "NetROI",
        "HourSpendOnApp",
        "DaySinceLastOrder",
        "SatisfactionScore",
        "Complain",
    ]
    display_cols = [c for c in display_cols if c in table_df.columns]

    st.dataframe(
        table_df[display_cols]
        .reset_index(drop=True)
        .style.background_gradient(subset=["ChurnProbability"], cmap="Reds"),
        use_container_width=True,
        height=420,
    )
    st.caption(f"Showing {len(table_df):,} customers · sorted by churn probability descending")


# ─── Page 3: Uplift Intelligence ────────────────────────────────────────────
def page_uplift(df):
    st.title("Uplift Intelligence")
    st.markdown(
        "**Uplift modeling** (causal ML) identifies which customers will **respond** "
        "to a retention intervention — not just who will churn. Targeting the wrong "
        "customers wastes retention budget. This is the approach used by "
        "**Uber (CausalML)**, **Netflix**, and **Salesforce** in production."
    )

    # ── CLV / Cost Controls ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Business Parameters")
    avg_clv = st.sidebar.number_input(
        "Avg Customer Lifetime Value ($)", value=500, step=50
    )
    intervention_cost = st.sidebar.number_input(
        "Intervention Cost ($)", value=15, step=5
    )

    # ── The Four Quadrants ──────────────────────────────────────────────────
    st.subheader("The Four Uplift Quadrants")
    col1, col2, col3, col4 = st.columns(4)
    type_counts = df["CustomerType"].value_counts()

    col1.metric(
        "Persuadables",
        f"{type_counts.get('Persuadable', 0):,}",
        help="High churn risk + responds to intervention — TARGET THESE",
    )
    col2.metric(
        "Sure Things",
        f"{type_counts.get('Sure Thing', 0):,}",
        help="Low churn risk — would stay anyway, no action needed",
    )
    col3.metric(
        "Lost Causes",
        f"{type_counts.get('Lost Cause', 0):,}",
        help="High churn risk + won't respond — don't waste budget",
    )
    col4.metric(
        "Sleeping Dogs",
        f"{type_counts.get('Sleeping Dog', 0):,}",
        help="Low churn risk — do NOT intervene, risk triggering churn",
    )

    st.markdown("---")

    # ── Uplift Scatter ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Churn Probability vs Uplift Score")
        fig_scatter = px.scatter(
            df,
            x="ChurnProbability",
            y="UpliftScore",
            color="CustomerType",
            color_discrete_map=CUSTOMER_TYPE_COLORS,
            opacity=0.6,
            hover_data=["CustomerID", "Segment", "NetROI"],
            title="Uplift Quadrant Map",
            labels={
                "ChurnProbability": "Churn Probability",
                "UpliftScore": "Uplift Score",
            },
        )
        fig_scatter.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="gray",
            annotation_text="Uplift threshold",
        )
        fig_scatter.add_vline(
            x=0.30,
            line_dash="dash",
            line_color="gray",
            annotation_text="Churn threshold",
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(height=450, template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.subheader("ROI Analysis")
        persuadables = df[df["CustomerType"] == "Persuadable"].copy()
        persuadables["NetROI_calc"] = (
            persuadables["UpliftScore"] * avg_clv - intervention_cost
        )

        total_spend = len(persuadables) * intervention_cost
        total_retained_value = (persuadables["UpliftScore"] * avg_clv).sum()
        net_campaign_roi = total_retained_value - total_spend

        st.metric("Persuadables to Target", f"{len(persuadables):,}")
        st.metric("Total Intervention Spend", f"${total_spend:,.0f}")
        st.metric("Expected Retained Value", f"${total_retained_value:,.0f}")
        st.metric(
            "Net Campaign ROI",
            f"${net_campaign_roi:,.0f}",
            delta=f"{'+' if net_campaign_roi > 0 else ''}{net_campaign_roi / max(total_spend, 1):.1%} ROI",
        )

        # ROI by segment
        roi_by_seg = (
            persuadables.groupby("Segment")
            .agg(
                Count=("CustomerID", "count"),
                AvgUplift=("UpliftScore", "mean"),
                TotalROI=("NetROI_calc", "sum"),
            )
            .round(2)
        )
        st.dataframe(roi_by_seg, use_container_width=True)

    st.markdown("---")

    # ── Priority Intervention List ───────────────────────────────────────────
    st.subheader("Priority Intervention List (Persuadables ranked by ROI)")
    st.markdown(
        "Only **Persuadables** are shown — customers with both high churn risk "
        "and positive response to intervention. Sorted by Net ROI descending."
    )

    priority_df = df[df["CustomerType"] == "Persuadable"].copy()
    priority_df["NetROI_calc"] = (
        priority_df["UpliftScore"] * avg_clv - intervention_cost
    )
    priority_df = priority_df.nlargest(100, "NetROI_calc")

    display_cols = [
        "CustomerID",
        "Segment",
        "ChurnProbability",
        "UpliftScore",
        "NetROI_calc",
        "HourSpendOnApp",
        "DaySinceLastOrder",
        "SatisfactionScore",
        "Complain",
    ]
    display_cols = [c for c in display_cols if c in priority_df.columns]

    st.dataframe(
        priority_df[display_cols]
        .reset_index(drop=True)
        .rename(columns={"NetROI_calc": "NetROI ($)"})
        .style.background_gradient(subset=["ChurnProbability"], cmap="Reds")
        .background_gradient(subset=["UpliftScore"], cmap="Greens"),
        use_container_width=True,
        height=400,
    )

    # ── Uplift Distribution ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Uplift Score Distribution by Segment")
    fig_up = px.box(
        df,
        x="Segment",
        y="UpliftScore",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        title="Uplift Score Distribution — which segments are most responsive to intervention?",
    )
    fig_up.update_layout(height=380, template="plotly_white", font=dict(family="Inter, sans-serif", color="#334155"), showlegend=False)
    st.plotly_chart(fig_up, use_container_width=True)


# ─── Retention Actions helpers ───────────────────────────────────────────────

def _render_action_card(action: dict, agentic_mode: bool, db_action_id: str = None):
    """Render a single retention action card with optional trace and feedback."""
    import database as db

    cid = action.get("customer_id", "N/A")
    seg = action.get("segment", "N/A")
    churn_p = action.get("churn_probability", 0)
    uplift = action.get("uplift_score", 0)
    roi = action.get("net_roi", 0)

    with st.container(border=True):
        st.markdown(
            f"**Customer {cid}** &nbsp;|&nbsp; {seg} &nbsp;|&nbsp; "
            f"Churn: **{churn_p:.1%}** &nbsp;|&nbsp; Uplift: **{uplift:+.3f}** &nbsp;|&nbsp; ROI: **${roi:.0f}**"
        )
        # Show agent reasoning trace if available
        trace = action.get("trace", [])
        if agentic_mode and trace:
            with st.expander(f"🔍 Agent reasoning — {len(trace)} tool call(s)", expanded=False):
                for t in trace:
                    st.markdown(f"**Round {t['round']} → `{t['tool']}`**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Arguments sent")
                        st.json(t["args"])
                    with c2:
                        st.caption("Data returned")
                        st.json(t["result"])
                    st.divider()

        if action.get("error"):
            st.error(f"Error: {action['error']}")
            return

        if action.get("do_not_intervene_reason"):
            st.warning(f"No intervention recommended: {action['do_not_intervene_reason']}")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Intervention", action.get("intervention_type", "N/A"))
        col2.metric("Channel", action.get("channel", "N/A"))
        col3.metric("Timing", action.get("timing", "N/A"))

        col4, col5 = st.columns(2)
        col4.metric("Estimated Cost", action.get("intervention_cost_estimate", "N/A"))
        col5.metric("Model Confidence", action.get("confidence", "N/A"))

        st.markdown(f"**Why at risk:** {action.get('primary_risk_reason', '')}")
        st.markdown(f"**Will they respond?** {action.get('customer_receptivity', '')}")
        st.markdown("**Suggested Message** *(copy-paste ready)*")
        st.code(action.get("message_framing", ""), language=None)
        st.markdown(f"**Expected outcome:** {action.get('expected_outcome', '')}")

        # Feedback buttons (CSM marks outcome — writes to DB audit trail)
        if db_action_id and db.is_available():
            st.markdown("**Mark outcome after executing this intervention:**")
            fb_col1, fb_col2, fb_col3 = st.columns(3)
            if fb_col1.button("✅ Customer Retained", key=f"ret_{cid}"):
                db.save_feedback(db_action_id, str(cid), "retained")
                st.success("Outcome logged: Retained")
            if fb_col2.button("❌ Customer Churned", key=f"chu_{cid}"):
                db.save_feedback(db_action_id, str(cid), "churned")
                st.warning("Outcome logged: Churned")
            if fb_col3.button("⏳ Still Pending", key=f"pen_{cid}"):
                db.save_feedback(db_action_id, str(cid), "pending")
                st.info("Outcome logged: Pending")


def _render_batch_tab(df, api_key, avg_clv, top_n, agentic_mode, playbook):
    """Batch Generator tab — generate N action plans at once."""
    import database as db

    persuadables = df[df["CustomerType"] == "Persuadable"].nlargest(50, "NetROI").copy()
    st.subheader(f"Top Persuadable Customers — {len(persuadables)} available")
    st.caption(
        "Only Persuadables are shown — customers with both high churn risk and positive "
        "response to intervention. These are the only customers with positive expected ROI."
    )
    preview_cols = ["CustomerID", "Segment", "ChurnProbability", "UpliftScore", "NetROI", "SatisfactionScore", "Complain"]
    preview_cols = [c for c in preview_cols if c in persuadables.columns]
    st.dataframe(persuadables[preview_cols].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    mode_label = "Agentic (tool-calling)" if agentic_mode else "Standard (single prompt)"
    if st.button(f"Generate {top_n} Retention Action Plans  [{mode_label}]", type="primary"):
        seg_profiles = build_segment_profiles(df)

        if agentic_mode:
            from agent_loop import generate_batch_agentic
            with st.spinner(f"Agentic mode: agent is calling tools for {top_n} customers..."):
                actions = generate_batch_agentic(
                    df_uplift=persuadables,
                    df_full=df,
                    playbook=playbook,
                    segment_profiles=seg_profiles,
                    api_key=api_key,
                    top_n=top_n,
                    avg_clv=avg_clv,
                )
        else:
            from retention_llm import generate_batch_retention_actions
            with st.spinner(f"Generating {top_n} retention action plans via Llama 3.3..."):
                actions = generate_batch_retention_actions(
                    df_uplift=persuadables,
                    segment_profiles=seg_profiles,
                    api_key=api_key,
                    top_n=top_n,
                    avg_clv=avg_clv,
                )

        st.success(f"Generated {len(actions)} plans.")

        # CSV export
        export_rows = [{
            "CustomerID": a.get("customer_id"), "Segment": a.get("segment"),
            "ChurnProbability": a.get("churn_probability"), "UpliftScore": a.get("uplift_score"),
            "NetROI": a.get("net_roi"), "InterventionType": a.get("intervention_type"),
            "Channel": a.get("channel"), "Timing": a.get("timing"),
            "Cost": a.get("intervention_cost_estimate"), "Confidence": a.get("confidence"),
            "WhyAtRisk": a.get("primary_risk_reason"), "SuggestedMessage": a.get("message_framing"),
            "ExpectedOutcome": a.get("expected_outcome"),
        } for a in actions]
        st.download_button(
            label="Export to CSV (CRM handoff)",
            data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
            file_name="retention_actions.csv",
            mime="text/csv",
            help="Import into Salesforce, HubSpot, or Marketo",
        )

        st.markdown("---")
        for action in actions:
            db_action_id = db.save_retention_action(action, agentic_mode=agentic_mode)
            _render_action_card(action, agentic_mode=agentic_mode, db_action_id=db_action_id)


def _render_chat_tab(df, api_key, playbook):
    """AI Customer Assistant tab — multi-turn conversational agent."""
    import database as db
    from agent_loop import run_agentic_loop, SYSTEM_PROMPT_CHAT

    st.subheader("AI Customer Assistant")
    st.caption(
        "Ask anything about your customers. The agent calls tools to look up real data "
        "before answering. Unlike the batch generator, this is conversational — ask follow-up questions."
    )
    st.markdown(
        "**Try:** *'Tell me about customer 50001'* · "
        "*'Which At-Risk customers have the highest ROI?'* · "
        "*'Is customer 50234 worth a $25 discount?'*"
    )

    # Initialise session state for this tab
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent_api_messages" not in st.session_state:
        st.session_state.agent_api_messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    # Load from DB if we have a session_id but no messages yet (page reload)
    session_id = st.session_state.get("session_id", "local")
    if not st.session_state.chat_messages and db.is_available():
        loaded = db.load_conversation_messages(session_id)
        if loaded:
            st.session_state.chat_messages = loaded

    # Clear chat button
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.chat_messages = []
        st.session_state.agent_api_messages = []
        st.session_state.conversation_id = None
        st.rerun()

    # Render existing messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            trace = msg.get("trace", [])
            if trace:
                with st.expander(f"🔍 Agent used {len(trace)} tool(s)", expanded=False):
                    for t in trace:
                        st.markdown(f"**Round {t['round']} → `{t['tool']}`**")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Arguments")
                            st.json(t["args"])
                        with c2:
                            st.caption("Result")
                            st.json(t["result"])
                        st.divider()
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about a customer, segment, or intervention strategy..."):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build API message list
        if not st.session_state.agent_api_messages:
            st.session_state.agent_api_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_CHAT}
            ]
            # Create DB conversation on first message
            if db.is_available():
                conv_id = db.create_conversation(session_id)
                st.session_state.conversation_id = conv_id

        st.session_state.agent_api_messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        if db.is_available() and st.session_state.conversation_id:
            db.save_message(st.session_state.conversation_id, "user", prompt)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking — calling tools..."):
                result = run_agentic_loop(
                    messages=st.session_state.agent_api_messages,
                    df=df,
                    playbook=playbook,
                    api_key=api_key,
                )

            trace = result.get("trace", [])
            response = result.get("response") or "I wasn't able to generate a response. Please try again."

            if result.get("error"):
                st.error(f"Agent error: {result['error']}")
                return

            # Show tool trace
            if trace:
                with st.expander(f"🔍 Agent used {len(trace)} tool(s) to answer this", expanded=False):
                    for t in trace:
                        st.markdown(f"**Round {t['round']} → `{t['tool']}`**")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Arguments")
                            st.json(t["args"])
                        with c2:
                            st.caption("Result")
                            st.json(t["result"])
                        st.divider()

            st.markdown(response)

        # Persist to session state and DB
        assistant_msg = {"role": "assistant", "content": response, "trace": trace}
        st.session_state.chat_messages.append(assistant_msg)
        st.session_state.agent_api_messages = result.get("messages", st.session_state.agent_api_messages)

        if db.is_available() and st.session_state.conversation_id:
            db.save_message(
                st.session_state.conversation_id, "assistant", response, tool_calls=trace
            )


# ─── Page 4: Retention Actions ──────────────────────────────────────────────
def page_retention_actions(df):
    st.title("LLM-Powered Retention Actions")
    st.markdown(
        "Two modes: **Batch Generator** produces structured action plans for your top Persuadables. "
        "**AI Customer Assistant** is a multi-turn conversational agent — ask anything about "
        "specific customers and it calls tools to retrieve real data before answering. "
        "Both use Llama 3.3 70B via Groq. This mirrors Salesforce Einstein Copilot's architecture."
    )

    # ── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Groq API Key")

    # Read from Streamlit Secrets first, then environment, then text input
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:
        _secret_key = os.getenv("GROQ_API_KEY", "")

    api_key_input = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_... (or set in Streamlit Secrets)",
        help="Set GROQ_API_KEY in Streamlit Secrets to avoid pasting it here each session.",
    )
    api_key = api_key_input or _secret_key

    if _secret_key and not api_key_input:
        st.sidebar.caption("✅ Key loaded from Secrets")

    avg_clv = st.sidebar.number_input("Customer Lifetime Value ($)", value=500, step=50)
    top_n = st.sidebar.slider("Batch: customers to generate", 1, 20, 5)
    agentic_mode = st.sidebar.toggle(
        "Agentic Mode",
        value=False,
        help="When ON: agent decides what data to retrieve using tool calls. Shows full reasoning trace. When OFF: classic single-prompt generation.",
    )

    playbook = load_playbook()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📋 Batch Generator", "💬 AI Customer Assistant"])

    with tab1:
        if not api_key:
            st.warning("Add your Groq API key in the sidebar to generate retention plans. Get one free at console.groq.com.")
        else:
            _render_batch_tab(df, api_key, avg_clv, top_n, agentic_mode, playbook)

    with tab2:
        if not api_key:
            st.warning("Add your Groq API key in the sidebar to use the AI Customer Assistant.")
        else:
            _render_chat_tab(df, api_key, playbook)


# ─── Page 5: Audit & Analytics ──────────────────────────────────────────────
def page_analytics():
    import database as db

    st.title("Audit & Analytics")
    st.markdown(
        "Full audit trail of every LLM-generated retention action, with outcome tracking. "
        "CSMs mark results after executing interventions — this data closes the feedback loop "
        "and would retrain the uplift model quarterly in a production system."
    )

    if not db.is_available():
        st.warning("Database not connected — audit trail is not persisting yet.")
        st.markdown("---")
        st.markdown("### How to connect the database (3 steps)")
        st.markdown("""
**Step 1** — Create a free Supabase project at [supabase.com](https://supabase.com)

**Step 2** — Copy the connection string:
`Settings → Database → Connection string → URI mode`

It looks like: `postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres`

**Step 3** — Add to Streamlit Secrets (Settings → Secrets in your Streamlit Cloud app):
```toml
DATABASE_URL = "postgresql://postgres:..."
GROQ_API_KEY = "gsk_..."
```

Tables are created automatically on first connection. No SQL needed.

Once connected, every retention action generated on the Retention Actions page will be logged here,
and CSMs can mark outcomes (Retained / Churned / Pending) directly on each action card.
        """)
        return

    summary = db.get_audit_summary()
    if not summary:
        st.info("No retention actions have been generated yet. Go to Retention Actions and generate some plans first.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────────
    outcomes = summary.get("outcomes", {})
    total = summary.get("total_actions", 0)
    retained = outcomes.get("retained", 0)
    churned = outcomes.get("churned", 0)
    pending = outcomes.get("pending", 0) + (total - retained - churned - outcomes.get("pending", 0))

    st.subheader("Campaign Summary")
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1], gap="medium")
    m1.metric("Total Actions", f"{total:,}")
    m2.metric("Retained", f"{retained:,}", f"{retained/total:.0%}" if total else "—")
    m3.metric("Churned", f"{churned:,}", f"{churned/total:.0%}" if total else "—")
    m4.metric("Pending", f"{total - retained - churned:,}")

    st.markdown("---")

    # ── Outcome by intervention type ─────────────────────────────────────────
    by_type = summary.get("by_intervention_type", [])
    by_seg = summary.get("by_segment", [])

    if by_type:
        st.subheader("Retention Rate by Intervention Type")
        st.caption("Which intervention types are actually retaining customers? Mark outcomes on the Retention Actions page to populate this.")
        type_df = pd.DataFrame([
            {
                "Intervention": r["intervention_type"] or "Unknown",
                "Total Actions": r["total"],
                "With Feedback": r["with_feedback"],
                "Retention Rate": f"{r['retention_rate']:.0%}" if r["retention_rate"] is not None else "No feedback yet",
            }
            for r in by_type
        ])
        st.dataframe(type_df.set_index("Intervention"), use_container_width=True, height=280)

        st.markdown("---")

        st.subheader("Retention Rate by Segment")
        st.caption("Which segments respond best to AI-generated interventions?")
        seg_df = pd.DataFrame([
            {
                "Segment": r["segment"] or "Unknown",
                "Total Actions": r["total"],
                "With Feedback": r["with_feedback"],
                "Retention Rate": f"{r['retention_rate']:.0%}" if r["retention_rate"] is not None else "No feedback yet",
            }
            for r in by_seg
        ])
        st.dataframe(seg_df.set_index("Segment"), use_container_width=True, height=280)

    st.markdown("---")

    # ── Full action log ──────────────────────────────────────────────────────
    st.subheader("Full Action Log")
    st.caption(
        "Every retention action ever generated, with the latest outcome. "
        "In production this would feed the quarterly model retraining pipeline."
    )
    all_actions = db.get_all_retention_actions(limit=200)
    if all_actions:
        log_df = pd.DataFrame(all_actions)
        log_df["agentic_mode"] = log_df["agentic_mode"].map({True: "Agentic", False: "Standard"})
        log_df["outcome"] = log_df["outcome"].fillna("pending")
        log_df["churn_probability"] = log_df["churn_probability"].round(3)
        st.dataframe(
            log_df[["customer_id", "segment", "churn_probability", "intervention_type",
                    "channel", "agentic_mode", "outcome", "generated_at"]].rename(columns={
                "customer_id": "CustomerID", "segment": "Segment",
                "churn_probability": "ChurnProb", "intervention_type": "Intervention",
                "channel": "Channel", "agentic_mode": "Mode",
                "outcome": "Outcome", "generated_at": "GeneratedAt",
            }),
            use_container_width=True,
            height=400,
        )
        st.caption(f"Showing {len(log_df):,} most recent actions · sorted newest first")
    else:
        st.info("No logged actions yet.")


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    _inject_css()

    # Check if pipeline has been run
    uplift_path = os.path.join(PROCESSED_PATH, "uplift.parquet")
    if not os.path.exists(uplift_path):
        st.error(
            "Pipeline artifacts not found. Run `python src/pipeline.py` first to build "
            "all models and cached data."
        )
        st.code("python src/pipeline.py")
        st.stop()

    # Initialise DB — try Streamlit Secrets first, then environment variable
    import database as db
    _db_url = ""
    try:
        _db_url = st.secrets.get("DATABASE_URL", "") or os.getenv("DATABASE_URL", "")
    except Exception:
        _db_url = os.getenv("DATABASE_URL", "")
    db.initialize(database_url=_db_url or None)

    # Show DB status in sidebar so we can always see connection state
    st.sidebar.markdown("---")
    if db.is_available():
        st.sidebar.success("DB connected", icon="🗄️")
    elif _db_url:
        st.sidebar.error("DB connection failed — check logs", icon="🗄️")
    else:
        st.sidebar.caption("🗄️ No DATABASE_URL set — audit trail disabled")

    # Generate a stable session ID for this browser session
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    df = load_data()
    page = render_sidebar(df)

    if page == "Segmentation Explorer":
        page_segmentation(df)
    elif page == "Churn Risk Dashboard":
        page_churn_risk(df)
    elif page == "Uplift Intelligence":
        page_uplift(df)
    elif page == "Retention Actions":
        page_retention_actions(df)
    elif page == "Audit & Analytics":
        page_analytics()


if __name__ == "__main__":
    main()
