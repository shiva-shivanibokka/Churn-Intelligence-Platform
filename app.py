"""
Customer Segmentation & Churn Engine
======================================
A decision intelligence platform that mirrors what Uber, Netflix,
Salesforce, and HubSpot run in production for customer retention.

Four pages:
  1. Segmentation Explorer  — UMAP clusters, segment profiles, bootstrap stability
  2. Churn Risk Dashboard   — Per-segment models, calibrated probabilities, SHAP
  3. Uplift Intelligence    — Persuadable identification, ROI ranking, causal ML
  4. Retention Actions      — LLM-generated intervention strategies per customer
"""

import os
import sys
import json
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
                f"{pct_certain:.0%} of customers have ≥80% confidence in their segment — "
                "the peak near 1.0 shows most customers have a clear behavioral home. "
                "Customers below 0.80 sit on segment boundaries and may need manual review."
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


# ─── Page 4: Retention Actions ──────────────────────────────────────────────
def page_retention_actions(df):
    st.title("LLM-Powered Retention Actions")
    st.markdown(
        "For each Persuadable customer, Llama 3.3 (via Groq) analyzes their **SHAP risk factors**, "
        "**segment profile**, **churn probability**, and **uplift score** to generate "
        "a structured retention strategy: intervention type, channel, timing, message "
        "framing, and estimated ROI. This mirrors Salesforce Einstein Copilot's "
        "CSM playbook generation."
    )

    # ── API Key Input ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Groq API Key")
    api_key = st.sidebar.text_input(
        "Groq API Key (free)",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com — required for LLM retention action generation",
    )
    avg_clv = st.sidebar.number_input("Customer Lifetime Value ($)", value=500, step=50)
    top_n = st.sidebar.slider("Customers to generate actions for", 1, 20, 5)

    # ── Persuadable Preview ─────────────────────────────────────────────────
    persuadables = df[df["CustomerType"] == "Persuadable"].nlargest(50, "NetROI").copy()

    st.subheader(f"Top Persuadable Customers ({len(persuadables)} shown)")
    preview_cols = [
        "CustomerID",
        "Segment",
        "ChurnProbability",
        "UpliftScore",
        "NetROI",
        "SatisfactionScore",
        "Complain",
        "HourSpendOnApp",
    ]
    preview_cols = [c for c in preview_cols if c in persuadables.columns]
    st.dataframe(
        persuadables[preview_cols].reset_index(drop=True), use_container_width=True
    )

    st.markdown("---")

    # ── How this works in production ─────────────────────────────────────────
    if not api_key:
        st.info(
            "Enter your free Groq API key in the sidebar to generate retention action plans. "
            "Get one at **console.groq.com** — takes 2 minutes, no credit card required."
        )
        st.markdown("---")
        st.markdown("### How this works in production (Salesforce / HubSpot pattern)")
        st.markdown(
            """
| Step | What happens | Tool used |
|------|-------------|-----------|
| 1. Score | Churn model scores all customers nightly | XGBoost (this engine) |
| 2. Filter | Only Persuadables are passed downstream | Uplift model (this engine) |
| 3. Generate | LLM writes a personalized intervention plan per customer | Llama 3.3 / GPT-4 |
| 4. Review | A CSM (Customer Success Manager) reviews and approves | Salesforce inbox / HubSpot task |
| 5. Execute | Approved message is sent via the selected channel | Marketo / Outreach / Intercom |
| 6. Track | Open rate, reply rate, and churn outcome are logged | CRM analytics |
| 7. Feedback | Logged outcomes retrain the uplift model quarterly | MLOps pipeline |

This page implements Steps 1–3. Steps 4–7 would connect to a CRM via API in a production system.
            """
        )
        return

    # ── Generate Actions ─────────────────────────────────────────────────────
    if st.button("Generate Retention Actions", type="primary"):
        from retention_llm import generate_batch_retention_actions

        seg_profiles = build_segment_profiles(df)

        with st.spinner(
            f"Generating {top_n} retention action plans via Llama 3.3 (Groq)..."
        ):
            actions = generate_batch_retention_actions(
                df_uplift=persuadables,
                segment_profiles=seg_profiles,
                api_key=api_key,
                top_n=top_n,
                avg_clv=avg_clv,
            )

        st.success(f"Generated {len(actions)} retention action plans.")

        # ── CSV Export (mirrors CRM handoff in production) ───────────────────
        export_rows = []
        for a in actions:
            export_rows.append({
                "CustomerID": a.get("customer_id"),
                "Segment": a.get("segment"),
                "ChurnProbability": a.get("churn_probability"),
                "UpliftScore": a.get("uplift_score"),
                "NetROI": a.get("net_roi"),
                "InterventionType": a.get("intervention_type"),
                "Channel": a.get("channel"),
                "Timing": a.get("timing"),
                "Cost": a.get("intervention_cost_estimate"),
                "Confidence": a.get("confidence"),
                "WhyAtRisk": a.get("primary_risk_reason"),
                "WillRespond": a.get("customer_receptivity"),
                "SuggestedMessage": a.get("message_framing"),
                "ExpectedOutcome": a.get("expected_outcome"),
            })
        export_df = pd.DataFrame(export_rows)
        st.download_button(
            label="Export to CSV (CRM handoff)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="retention_actions.csv",
            mime="text/csv",
            help="Download this file to import into Salesforce, HubSpot, or Marketo",
        )

        st.markdown("---")

        for action in actions:
            cid = action.get("customer_id", "N/A")
            seg = action.get("segment", "N/A")
            churn_p = action.get("churn_probability", 0)
            uplift = action.get("uplift_score", 0)
            roi = action.get("net_roi", 0)

            with st.expander(
                f"Customer {cid} | {seg} | Churn: {churn_p:.1%} | "
                f"Uplift: {uplift:+.3f} | ROI: ${roi:.0f}",
                expanded=True,
            ):
                if action.get("error"):
                    st.error(f"Error: {action['error']}")
                elif action.get("do_not_intervene_reason"):
                    st.warning(
                        f"No intervention recommended: {action['do_not_intervene_reason']}"
                    )
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Intervention", action.get("intervention_type", "N/A"))
                    col2.metric("Channel", action.get("channel", "N/A"))
                    col3.metric("Timing", action.get("timing", "N/A"))

                    col4, col5 = st.columns(2)
                    col4.metric("Estimated Cost", action.get("intervention_cost_estimate", "N/A"))
                    col5.metric("Model Confidence", action.get("confidence", "N/A"))

                    st.markdown(f"**Why at risk:** {action.get('primary_risk_reason', '')}")
                    st.markdown(f"**Will they respond?** {action.get('customer_receptivity', '')}")

                    st.markdown("**Suggested Message** *(copy-paste ready for CSM)*")
                    st.code(action.get("message_framing", ""), language=None)

                    st.markdown(f"**Expected outcome:** {action.get('expected_outcome', '')}")


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


if __name__ == "__main__":
    main()
