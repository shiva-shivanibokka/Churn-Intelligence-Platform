"""
Customer Segmentation Engine
==============================
Implements production-grade segmentation with:
- K-Means++ for business-interpretable hard cluster assignments
- Gaussian Mixture Models (GMM) for soft probability assignments
  (each customer gets a probability distribution across segments — not a hard label)
- PaCMAP for 2D visualization of high-dimensional behavioral space
- Bootstrap cluster stability analysis via Adjusted Rand Index (ARI)
  across 100 resamplings — a production validation step that most
  portfolio implementations skip entirely

Architecture mirrors Salesforce Einstein's customer health segmentation:
segments are named with business-readable labels and profiled as heatmaps
so non-technical stakeholders can act on them.
"""

import logging
import os
import warnings

import joblib
import numpy as np
import pacmap
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Segment business labels — named for non-technical stakeholders
# (the same pattern Salesforce uses for "Health Score" tiers)
SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk",
    3: "Price Sensitive",
    4: "Lapsed",
}

SEGMENT_COLORS = {
    "Champions": "#2ECC71",
    "Loyal Customers": "#3498DB",
    "At-Risk": "#E74C3C",
    "Price Sensitive": "#F39C12",
    "Lapsed": "#95A5A6",
}


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    StandardScaler normalization. Required before distance-based clustering.
    Returns both the scaled array and the fitted scaler (needed for inference).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """
    Elbow method + Silhouette score sweep to find optimal number of clusters.
    Returns a dict of k -> metrics for display in the Streamlit UI.
    """
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
        db = davies_bouldin_score(X_scaled, labels)
        results[k] = {
            "inertia": inertia,
            "silhouette": sil,
            "davies_bouldin": db,
            "labels": labels,
        }
    return results


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 5) -> tuple[KMeans, np.ndarray]:
    """
    Fit K-Means++ with the chosen number of clusters.
    K-Means++ initialization reduces convergence time and avoids poor local minima.
    """
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=42,
    )
    labels = km.fit_predict(X_scaled)
    return km, labels


def fit_gmm(
    X_scaled: np.ndarray, n_components: int = 5
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    Fit Gaussian Mixture Model for soft probabilistic segment assignments.

    Unlike K-Means, GMM assigns each customer a probability distribution
    across all segments — "60% Champion, 30% Loyal, 10% At-Risk."
    This is used in the uplift model to weight predictions by segment membership
    probability, matching how Salesforce handles ambiguous health score boundaries.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=5,
        max_iter=200,
        random_state=42,
    )
    gmm.fit(X_scaled)
    hard_labels = gmm.predict(X_scaled)
    soft_probs = gmm.predict_proba(X_scaled)  # shape: (n_customers, n_segments)
    return gmm, hard_labels, soft_probs


def bootstrap_stability(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    n_bootstrap: int = 100,
    sample_frac: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Bootstrap Cluster Stability Analysis via Adjusted Rand Index (ARI).

    Procedure:
    1. Fit K-Means on the full dataset → reference labels
    2. For each bootstrap iteration:
       a. Sample 80% of the data (with replacement)
       b. Fit K-Means on the sample
       c. Compute ARI between reference labels (on the sampled rows) and
          new labels from the bootstrap fit
    3. Report mean ARI, std, and stability grade

    ARI Interpretation:
    - ARI > 0.85: Highly stable — segments are robust across data samples
    - ARI 0.70-0.85: Moderately stable — segments are consistent
    - ARI < 0.70: Unstable — segments are sensitive to data, reduce k

    This is a production validation step that proves segments are not
    artifacts of a specific random seed or data sample.
    """
    n_samples = len(X_scaled)

    # Seeded generator, not the global RNG.
    #
    # np.random.choice draws from global state, so this function returned a
    # different mean ARI on every run while the README quoted one to three
    # decimals and called it reproducible. Every KMeans here is already seeded;
    # the resampling was the one unseeded step, so it was the only reason the
    # headline number moved.
    rng = np.random.default_rng(seed)

    # Reference model on full dataset
    ref_km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    ref_labels = ref_km.fit_predict(X_scaled)

    ari_scores = []
    for i in range(n_bootstrap):
        # Sample indices with replacement
        sample_idx = rng.choice(
            n_samples, size=int(n_samples * sample_frac), replace=True
        )
        X_sample = X_scaled[sample_idx]

        # Fit on sample
        boot_km = KMeans(
            n_clusters=n_clusters, init="k-means++", n_init=5, random_state=i
        )
        boot_labels = boot_km.fit_predict(X_sample)

        # ARI against reference labels on the same sampled rows
        ref_sub = ref_labels[sample_idx]
        ari = adjusted_rand_score(ref_sub, boot_labels)
        ari_scores.append(ari)

    ari_arr = np.array(ari_scores)
    mean_ari = ari_arr.mean()

    if mean_ari >= 0.85:
        grade = "Highly Stable"
    elif mean_ari >= 0.70:
        grade = "Moderately Stable"
    else:
        grade = "Unstable — consider reducing k"

    return {
        "ari_scores": ari_arr.tolist(),
        "mean_ari": float(mean_ari),
        "std_ari": float(ari_arr.std()),
        "min_ari": float(ari_arr.min()),
        "max_ari": float(ari_arr.max()),
        "grade": grade,
        "n_bootstrap": n_bootstrap,
    }


def fit_pacmap(X_scaled: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """
    PaCMAP dimensionality reduction to 2D for interactive cluster visualization.

    Chosen over UMAP for the same purpose: no numba/TensorFlow dependency, and
    it works against any NumPy version.

    This used to be called `fit_umap` and took a `min_dist` argument, which is a
    UMAP parameter that PaCMAP has no equivalent for — it was accepted and
    silently dropped. The name outlived the swap and spread: the dashboard still
    announced a "UMAP 2D Projection" and the glossary defined Uniform Manifold
    Approximation and Projection to visitors looking at a PaCMAP embedding.

    The emitted columns stay `UMAP_1` / `UMAP_2`, and so do `umap_1` / `umap_2`
    in Postgres. Those are internal identifiers nobody reading the dashboard
    sees, and renaming them means altering a live table for no gain — but they
    are PaCMAP coordinates, and this is the note saying so.
    """
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled, init="pca")
    return embedding


def label_segments(
    km_labels: np.ndarray, df: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:
    """
    Assign business-readable segment names to K-Means cluster labels.

    The mapping is determined by cluster profile:
    - Champions: high engagement, low recency (bought recently), low support risk
    - Loyal Customers: high tenure, moderate engagement
    - At-Risk: declining engagement, high recency signal (haven't bought recently)
    - Price Sensitive: high discount sensitivity, low cashback
    - Lapsed: very high recency (long since last order), very low engagement

    Clusters are ranked by EngagementScore (descending) and RecencySignal (ascending)
    to assign labels systematically, making the naming deterministic.
    """
    df = df.copy()
    df["RawSegment"] = km_labels

    # Profile each cluster by key features
    profile = df.groupby("RawSegment")[
        [
            "EngagementScore",
            "RecencySignal",
            "SupportRiskScore",
            "StickinessIndex",
            "DiscountSensitivity",
            "TenureStability",
        ]
    ].mean()

    # Rank by engagement (desc) to assign Champion → Loyal → ...
    # Break ties by recency (asc = bought more recently)
    profile["rank_score"] = (
        profile["EngagementScore"] * 0.4
        + (1 - profile["RecencySignal"]) * 0.3  # lower recency signal = bought recently
        + profile["StickinessIndex"] * 0.2
        + (1 - profile["SupportRiskScore"]) * 0.1
    )

    sorted_clusters = profile["rank_score"].sort_values(ascending=False).index.tolist()
    label_map = {}
    available_labels = list(SEGMENT_LABELS.values())

    for i, cluster_id in enumerate(sorted_clusters):
        if i < len(available_labels):
            label_map[cluster_id] = available_labels[i]
        else:
            label_map[cluster_id] = f"Segment {cluster_id}"

    df["Segment"] = df["RawSegment"].map(label_map)
    return df, label_map


def build_segment_profiles(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Build segment-level aggregate profiles for heatmap visualization.
    This is the "business output" step — converting cluster indices into
    actionable descriptions that a CSM or marketing team can act on.
    """
    profile_cols = [
        "EngagementScore",
        "RecencySignal",
        "StickinessIndex",
        "SpendTrend",
        "SupportRiskScore",
        "DiscountSensitivity",
        "TenureStability",
        "WarehouseFriction",
        "Churn",
    ]
    available = [c for c in profile_cols if c in df.columns]
    profile = df.groupby("Segment")[available].mean().round(3)
    profile["CustomerCount"] = df.groupby("Segment").size()
    profile["ChurnRate"] = df.groupby("Segment")["Churn"].mean().round(3)
    return profile


def run_segmentation(df: pd.DataFrame, feature_cols: list, n_clusters: int = 5) -> dict:
    """
    Full segmentation pipeline. Returns all artifacts needed for the Streamlit UI.
    """
    logger.info("Scaling %d features for %d customers...", len(feature_cols), len(df))
    X_scaled, scaler = scale_features(df[feature_cols])

    logger.info("Finding optimal k...")
    k_metrics = find_optimal_k(X_scaled)

    logger.info("Fitting K-Means++ with k=%d...", n_clusters)
    km, km_labels = fit_kmeans(X_scaled, n_clusters)

    logger.info("Fitting GMM with %d components...", n_clusters)
    gmm, gmm_labels, gmm_probs = fit_gmm(X_scaled, n_clusters)

    logger.info("Running bootstrap stability analysis (100 iterations)...")
    stability = bootstrap_stability(X_scaled, n_clusters=n_clusters, n_bootstrap=100)
    logger.info("Stability: mean ARI=%.3f (%s)", stability["mean_ari"], stability["grade"])

    logger.info("Fitting PaCMAP for visualization...")
    umap_embedding = fit_pacmap(X_scaled)

    logger.info("Labeling segments...")
    df_out, label_map = label_segments(km_labels, df, feature_cols)

    # Add GMM soft probabilities per segment
    for i in range(n_clusters):
        df_out[f"GMM_Prob_Seg{i}"] = gmm_probs[:, i]

    # Add the 2D embedding (PaCMAP — see fit_pacmap on the column names)
    df_out["UMAP_1"] = umap_embedding[:, 0]
    df_out["UMAP_2"] = umap_embedding[:, 1]

    # Build profiles
    profiles = build_segment_profiles(df_out, feature_cols)

    # Save artifacts
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(km, os.path.join(MODELS_PATH, "kmeans.pkl"))
    joblib.dump(gmm, os.path.join(MODELS_PATH, "gmm.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))
    # Without this the mapping from a KMeans cluster index to a business segment
    # name lived only inside this function's return value, so api/serve.py — which
    # loads kmeans.pkl and has to name the cluster it predicts — had nothing to
    # look the name up in, and fell back to whichever segment happened to be first.
    joblib.dump(label_map, os.path.join(MODELS_PATH, "label_map.pkl"))

    out_path = os.path.join(PROCESSED_PATH, "segmented.parquet")
    df_out.to_parquet(out_path, index=False)

    logger.info("Done. Saved segmented data to %s", out_path)

    return {
        "df": df_out,
        "km": km,
        "gmm": gmm,
        "scaler": scaler,
        "k_metrics": k_metrics,
        "stability": stability,
        "umap_embedding": umap_embedding,
        "profiles": profiles,
        "label_map": label_map,
        "n_clusters": n_clusters,
    }


if __name__ == "__main__":
    from features import build_pipeline, get_feature_sets

    df = build_pipeline(save=True)
    feature_sets = get_feature_sets()
    results = run_segmentation(df, feature_sets["clustering"])

    print("\nSegment Profiles:")
    print(results["profiles"])
    print("\nStability Report:")
    print(f"  Mean ARI: {results['stability']['mean_ari']:.3f}")
    print(f"  Grade:    {results['stability']['grade']}")
