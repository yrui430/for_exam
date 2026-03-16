"""
Trader classification: style clustering and tier assignment.

Two-dimensional organization:
1. Style clusters (horizontal): group traders by behavioral similarity
2. Performance tiers (vertical): rank traders by composite quality score
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Tuple, Dict

from .config import Config


# Features used for style clustering
STYLE_FEATURES = [
    "daily_vol",
    "return_autocorr",
    "skewness",
    "kurtosis",
    "win_rate",
    "tail_ratio",
    "activity_rate",
    "sharpe_stability",
]


def classify_traders(
    qualified: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Classify traders by style (clusters) and performance (tiers).

    Returns:
        classified: DataFrame with added 'cluster', 'tier', 'cluster_label' columns
        cluster_info: Dict with cluster statistics
    """
    cp = config.classification
    np.random.seed(config.random_seed)

    classified = qualified.copy()

    # --- Style Clustering ---
    available_features = [f for f in STYLE_FEATURES if f in classified.columns]
    X = classified[available_features].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle edge cases
    n_traders = len(classified)
    n_clusters = min(cp.n_clusters, n_traders)

    if n_traders < 4:
        classified["cluster"] = 0
        classified["cluster_label"] = "single_group"
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.random_seed, n_init=10)
        classified["cluster"] = kmeans.fit_predict(X_scaled)

        # PCA for visualization
        if X_scaled.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=config.random_seed)
            pca_coords = pca.fit_transform(X_scaled)
            classified["pca_1"] = pca_coords[:, 0]
            classified["pca_2"] = pca_coords[:, 1]

        # Label clusters based on dominant characteristics vs population
        classified["cluster_label"] = classified["cluster"].map(
            lambda c: _label_cluster(
                classified[classified["cluster"] == c],
                classified,
                available_features,
            )
        )

    # --- Performance Tiering ---
    score_rank = classified["composite_score"].rank(pct=True, ascending=True)

    classified["tier"] = "watch"
    classified.loc[score_rank >= (1 - cp.core_top_pct), "tier"] = "core"
    satellite_lower = 1 - cp.core_top_pct - cp.satellite_pct
    classified.loc[
        (score_rank >= satellite_lower) & (score_rank < (1 - cp.core_top_pct)),
        "tier"
    ] = "satellite"

    # --- Cluster Info ---
    cluster_info = {}
    for c in classified["cluster"].unique():
        cdf = classified[classified["cluster"] == c]
        cluster_info[int(c)] = {
            "label": cdf["cluster_label"].iloc[0],
            "count": len(cdf),
            "avg_sharpe": float(cdf["sharpe"].mean()),
            "avg_vol": float(cdf["daily_vol"].mean()),
            "avg_win_rate": float(cdf["win_rate"].mean()),
            "avg_autocorr": float(cdf["return_autocorr"].mean()),
            "tier_distribution": cdf["tier"].value_counts().to_dict(),
        }

    return classified, cluster_info


def _label_cluster(cluster_df: pd.DataFrame, all_df: pd.DataFrame, features: list) -> str:
    """
    Assign a descriptive label to a cluster by comparing against
    the full qualified population medians.
    """
    profile = cluster_df[features].mean()
    global_median = all_df[features].median()

    vol = profile.get("daily_vol", 0)
    autocorr = profile.get("return_autocorr", 0)
    activity = profile.get("activity_rate", 0)
    win_rate = profile.get("win_rate", 0)
    skew = profile.get("skewness", 0)
    kurt = profile.get("kurtosis", 0)
    stability = profile.get("sharpe_stability", 0)

    labels = []

    # Volatility relative to population
    vol_med = global_median.get("daily_vol", vol)
    if vol > vol_med * 1.5:
        labels.append("high_vol")
    elif vol < vol_med * 0.6:
        labels.append("low_vol")

    # Trend / mean-reversion
    if autocorr > 0.08:
        labels.append("trend")
    elif autocorr < -0.08:
        labels.append("mean_rev")

    # Activity
    act_med = global_median.get("activity_rate", activity)
    if activity > act_med * 1.2:
        labels.append("freq")
    elif activity < act_med * 0.7:
        labels.append("infreq")

    # Win rate
    wr_med = global_median.get("win_rate", win_rate)
    if win_rate > wr_med * 1.1:
        labels.append("high_wr")
    elif win_rate < wr_med * 0.9:
        labels.append("low_wr")

    # Consistency
    stab_med = global_median.get("sharpe_stability", stability)
    if stability < stab_med * 0.7:
        labels.append("stable")
    elif stability > stab_med * 1.5:
        labels.append("erratic")

    # Tail shape
    if skew > 0.5:
        labels.append("pos_skew")
    elif skew < -0.5:
        labels.append("neg_skew")

    if not labels:
        labels.append("balanced")

    return "_".join(labels)


def get_classification_summary(classified: pd.DataFrame, cluster_info: Dict) -> str:
    """Generate text summary of classification results."""
    lines = [
        "=== Classification Summary ===",
        "",
        "--- Tier Distribution ---",
    ]

    for tier in ["core", "satellite", "watch"]:
        tdf = classified[classified["tier"] == tier]
        if len(tdf) > 0:
            lines.append(
                f"  {tier.upper()}: {len(tdf)} traders, "
                f"avg_sharpe={tdf['sharpe'].mean():.2f}, "
                f"avg_score={tdf['composite_score'].mean():.3f}"
            )

    lines.append("")
    lines.append("--- Style Clusters ---")
    for c, info in sorted(cluster_info.items()):
        lines.append(
            f"  Cluster {c} ({info['label']}): "
            f"{info['count']} traders, "
            f"sharpe={info['avg_sharpe']:.2f}, "
            f"vol={info['avg_vol']:.4f}"
        )
        tier_dist = info["tier_distribution"]
        lines.append(f"    Tiers: {tier_dist}")

    return "\n".join(lines)
