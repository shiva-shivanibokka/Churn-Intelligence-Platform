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
    st.sidebar.markdown("## 📊")
    st.sidebar.title("Churn Engine")
    st.sidebar.caption("Decision Intelligence Platform")
    st.sidebar.markdown("---")

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

    # ── UMAP Scatter ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
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

        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            height=500,
            template="plotly_white",
            paper_bgcolor="#FAFAFA",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)
            st.caption(_UMAP_CAPTIONS.get(color_by, ""))

    with col_right:
        st.subheader("Segment Stability Score")
        st.caption(
            "Tests whether the same 5 segments emerge from 100 different random "
            "data samples — proving the clusters are real, not a random artifact."
        )
        if stability:
            ari = stability["mean_ari"]
            grade = stability["grade"]
            color = (
                "#2ECC71" if ari >= 0.85 else "#F39C12" if ari >= 0.70 else "#E74C3C"
            )

            st.metric(
                "Mean ARI",
                f"{ari:.3f}",
                help="Adjusted Rand Index across 100 bootstrap resamplings. Above 0.70 = stable.",
            )
            st.metric("Std ARI", f"{stability['std_ari']:.3f}")
            st.metric("Stability Grade", grade)

            ari_scores = stability.get("ari_scores", [])
            if ari_scores:
                fig_ari = px.histogram(
                    x=ari_scores,
                    nbins=20,
                    labels={"x": "ARI Score"},
                    title="Stability Score Distribution (100 bootstraps)",
                    color_discrete_sequence=[color],
                )
                fig_ari.add_vline(
                    x=ari,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Mean={ari:.3f}",
                )
                fig_ari.update_layout(
                    height=260,
                    template="plotly_white",
                    showlegend=False,
                    paper_bgcolor="#FAFAFA",
                )
                with st.container(border=True):
                    st.plotly_chart(fig_ari, use_container_width=True)
                    st.caption(
                        "A tight distribution near 1.0 means the segments are highly reproducible. "
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
        height=400,
        template="plotly_white",
        paper_bgcolor="#FAFAFA",
        margin=dict(l=10, r=10, t=40, b=10),
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
        # Confidence = max GMM probability across all segments for each customer
        confidence = df[gmm_cols].max(axis=1)
        conf_df = pd.DataFrame({"Confidence": confidence, "Segment": df["Segment"]})

        fig_conf = px.histogram(
            conf_df,
            x="Confidence",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            title="How confident is the model about each customer's segment assignment?",
            labels={"Confidence": "Assignment Confidence (0 = uncertain, 1 = certain)"},
        )
        fig_conf.add_vline(
            x=0.80,
            line_dash="dash",
            line_color="black",
            annotation_text="80% confidence",
            annotation_position="top right",
        )
        fig_conf.update_layout(
            height=340,
            template="plotly_white",
            paper_bgcolor="#FAFAFA",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        with st.container(border=True):
            st.plotly_chart(fig_conf, use_container_width=True)
            pct_certain = (confidence >= 0.80).mean()
            st.caption(
                f"**{pct_certain:.0%} of customers score ≥80% confidence** — the large spike at 1.0 "
                "is expected and correct: it means the 5 segments are very well-separated in behavioral space, "
                "so GMM can place most customers into one segment with near-certainty. "
                "This validates the segmentation quality. Customers below 0.80 sit on a boundary "
                "between two segments and deserve closer manual review."
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
    col1, col2 = st.columns(2)

    with col1:
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
            opacity=0.7,
            title="Calibrated Churn Probability by Segment",
            labels={"ChurnProbability": "Churn Probability"},
        )
        fig_hist.update_layout(
            height=380, template="plotly_white", paper_bgcolor="#FAFAFA"
        )
        with st.container(border=True):
            st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Risk Tier Breakdown")
        st.caption(
            "Customers are bucketed into Low / Medium / High Risk based on their predicted "
            "churn probability. High Risk = churn probability above 60%. "
            "Use this to size your retention budget by segment."
        )
        risk_seg = (
            df_filtered.groupby(["Segment", "RiskTier"])
            .size()
            .reset_index(name="Count")
        )
        fig_risk = px.bar(
            risk_seg,
            x="Segment",
            y="Count",
            color="RiskTier",
            color_discrete_map=RISK_COLORS,
            title="Risk Tier Distribution by Segment",
            barmode="group",
        )
        fig_risk.update_layout(
            height=380, template="plotly_white", paper_bgcolor="#FAFAFA"
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
            height=400,
            template="plotly_white",
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="#FAFAFA",
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
        fig_scatter.update_layout(height=450, template="plotly_white")
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
    fig_up.update_layout(height=380, template="plotly_white", showlegend=False)
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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Actions Generated", f"{total:,}")
    m2.metric("Marked Retained ✅", f"{retained:,}", f"{retained/total:.0%}" if total else "0%")
    m3.metric("Marked Churned ❌", f"{churned:,}", f"{churned/total:.0%}" if total else "0%")
    m4.metric("Pending Outcome ⏳", f"{total - retained - churned:,}")

    st.markdown("---")

    # ── Outcome by intervention type ─────────────────────────────────────────
    by_type = summary.get("by_intervention_type", [])
    by_seg = summary.get("by_segment", [])

    if by_type:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Retention Rate by Intervention Type")
            st.caption("Which intervention types are actually retaining customers?")
            type_df = pd.DataFrame([
                {
                    "Intervention": r["intervention_type"] or "Unknown",
                    "Total Actions": r["total"],
                    "Retention Rate": f"{r['retention_rate']:.0%}" if r["retention_rate"] is not None else "No feedback yet",
                }
                for r in by_type
            ])
            st.dataframe(type_df.set_index("Intervention"), use_container_width=True)

        with col2:
            st.subheader("Retention Rate by Segment")
            st.caption("Which segments respond best to AI-generated interventions?")
            seg_df = pd.DataFrame([
                {
                    "Segment": r["segment"] or "Unknown",
                    "Total Actions": r["total"],
                    "Retention Rate": f"{r['retention_rate']:.0%}" if r["retention_rate"] is not None else "No feedback yet",
                }
                for r in by_seg
            ])
            st.dataframe(seg_df.set_index("Segment"), use_container_width=True)

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
