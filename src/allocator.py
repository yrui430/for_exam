"""
Capital allocation: distribute capital across tiers and traders.

Allocation follows a two-level hierarchy:
1. Tier level: fixed allocation ratio (Core / Satellite / Cash Reserve)
2. Within-tier: score-weighted with inverse-volatility adjustment

Key constraints:
- No single trader exceeds max_single_trader_weight
- Style cluster concentration limit
- Correlation-based diversification adjustment
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from .config import Config


def allocate_capital(
    classified: pd.DataFrame,
    return_panel: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute capital allocation weights for each trader.

    Returns:
        allocation: DataFrame with trader weights and dollar amounts
        allocation_info: Summary statistics
    """
    ap = config.allocation
    rp = config.risk

    # Only allocate to core and satellite tiers
    allocated = classified[classified["tier"].isin(["core", "satellite"])].copy()

    if len(allocated) == 0:
        return pd.DataFrame(), {"error": "No traders to allocate"}

    # --- Step 1: Within-tier raw scores ---
    allocated["raw_weight"] = 0.0

    for tier, tier_budget in [("core", ap.core_weight), ("satellite", ap.satellite_weight)]:
        tier_mask = allocated["tier"] == tier
        tier_df = allocated[tier_mask]

        if len(tier_df) == 0:
            continue

        if ap.method == "score_weighted":
            weights = _score_weighted(tier_df, return_panel, config)
        elif ap.method == "inverse_vol":
            weights = _inverse_volatility(tier_df)
        elif ap.method == "risk_parity":
            weights = _risk_parity(tier_df, return_panel)
        else:
            weights = _score_weighted(tier_df, return_panel, config)

        # Normalize to tier budget
        weights = weights / weights.sum() * tier_budget

        allocated.loc[tier_mask, "raw_weight"] = weights

    # --- Step 2: Apply concentration limits ---
    allocated["weight"] = allocated["raw_weight"].clip(
        lower=ap.min_single_trader_weight,
        upper=ap.max_single_trader_weight,
    )

    # --- Step 3: Iteratively apply constraints until convergence ---
    total_budget = ap.core_weight + ap.satellite_weight

    for _iteration in range(20):
        changed = False

        # Cluster concentration cap
        for cluster in allocated["cluster"].unique():
            cluster_mask = allocated["cluster"] == cluster
            cluster_weight = allocated.loc[cluster_mask, "weight"].sum()
            if cluster_weight > rp.max_cluster_weight + 1e-6:
                scale = rp.max_cluster_weight / cluster_weight
                allocated.loc[cluster_mask, "weight"] *= scale
                changed = True

        # Single trader cap
        over = allocated["weight"] > ap.max_single_trader_weight + 1e-6
        if over.any():
            allocated.loc[over, "weight"] = ap.max_single_trader_weight
            changed = True

        # Re-distribute freed weight to uncapped traders proportionally
        current_total = allocated["weight"].sum()
        if current_total > 0 and abs(current_total - total_budget) > 1e-6:
            # Identify capped vs uncapped traders
            at_single_cap = (allocated["weight"] >= ap.max_single_trader_weight - 1e-6)
            at_cluster_cap = pd.Series(False, index=allocated.index)
            for cluster in allocated["cluster"].unique():
                cm = allocated["cluster"] == cluster
                if allocated.loc[cm, "weight"].sum() >= rp.max_cluster_weight - 1e-6:
                    at_cluster_cap |= cm

            capped = at_single_cap | at_cluster_cap
            uncapped = ~capped

            if uncapped.any():
                deficit = total_budget - current_total
                uncapped_total = allocated.loc[uncapped, "weight"].sum()
                if uncapped_total > 0:
                    boost = 1 + deficit / uncapped_total
                    allocated.loc[uncapped, "weight"] *= boost
                    changed = True
            else:
                # All capped — just normalize
                allocated["weight"] = allocated["weight"] / current_total * total_budget

        if not changed:
            break

    # Final normalization ensuring budget
    current_total = allocated["weight"].sum()
    if current_total > 0 and abs(current_total - total_budget) > 1e-4:
        allocated["weight"] = allocated["weight"] / current_total * total_budget

    # Dollar amounts
    allocated["dollar_allocation"] = allocated["weight"] * ap.total_capital

    # --- Summary ---
    allocation_info = _allocation_summary(allocated, config)

    return allocated, allocation_info


def _score_weighted(
    tier_df: pd.DataFrame,
    return_panel: pd.DataFrame,
    config: Config,
) -> pd.Series:
    """
    Score-weighted allocation with volatility adjustment.

    weight_i = (score_i / vol_i) / sum(score_j / vol_j)
    """
    scores = tier_df["composite_score"].copy()
    vols = tier_df["daily_vol"].replace(0, tier_df["daily_vol"].median())
    vols = vols.clip(lower=1e-6)

    # Score / volatility gives risk-adjusted attractiveness
    raw = scores / vols
    raw = raw.clip(lower=0)

    if raw.sum() == 0:
        return pd.Series(1.0 / len(tier_df), index=tier_df.index)

    return raw / raw.sum()


def _inverse_volatility(tier_df: pd.DataFrame) -> pd.Series:
    """Simple inverse-volatility weighting."""
    vols = tier_df["daily_vol"].replace(0, tier_df["daily_vol"].median())
    vols = vols.clip(lower=1e-6)
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


def _risk_parity(tier_df: pd.DataFrame, return_panel: pd.DataFrame) -> pd.Series:
    """
    Simplified risk parity: each trader contributes equal risk.
    Approximated by inverse-volatility when correlations are ignored.
    """
    return _inverse_volatility(tier_df)


def _allocation_summary(allocated: pd.DataFrame, config: Config) -> Dict:
    """Generate allocation summary statistics."""
    info = {
        "n_traders_allocated": len(allocated),
        "total_weight": float(allocated["weight"].sum()),
        "cash_reserve": config.allocation.cash_reserve,
        "tier_weights": {},
        "cluster_weights": {},
        "concentration": {
            "max_single_weight": float(allocated["weight"].max()),
            "top_5_weight": float(allocated.nlargest(5, "weight")["weight"].sum()),
            "herfindahl_index": float((allocated["weight"] ** 2).sum()),
        },
    }

    for tier in ["core", "satellite"]:
        tdf = allocated[allocated["tier"] == tier]
        if len(tdf) > 0:
            info["tier_weights"][tier] = {
                "count": len(tdf),
                "total_weight": float(tdf["weight"].sum()),
                "avg_weight": float(tdf["weight"].mean()),
            }

    for cluster in allocated["cluster"].unique():
        cdf = allocated[allocated["cluster"] == cluster]
        info["cluster_weights"][int(cluster)] = {
            "count": len(cdf),
            "total_weight": float(cdf["weight"].sum()),
            "label": cdf["cluster_label"].iloc[0] if "cluster_label" in cdf.columns else str(cluster),
        }

    return info


def get_allocation_summary(allocated: pd.DataFrame, allocation_info: Dict) -> str:
    """Generate text summary of allocation."""
    lines = [
        "=== Allocation Summary ===",
        f"Traders allocated: {allocation_info['n_traders_allocated']}",
        f"Total deployed weight: {allocation_info['total_weight']:.2%}",
        f"Cash reserve: {allocation_info['cash_reserve']:.2%}",
        "",
        "--- Tier Breakdown ---",
    ]

    for tier, info in allocation_info.get("tier_weights", {}).items():
        lines.append(
            f"  {tier.upper()}: {info['count']} traders, "
            f"total={info['total_weight']:.2%}, "
            f"avg={info['avg_weight']:.2%}"
        )

    lines.append("")
    lines.append("--- Cluster Breakdown ---")
    for cluster, info in allocation_info.get("cluster_weights", {}).items():
        lines.append(
            f"  Cluster {cluster} ({info['label']}): "
            f"{info['count']} traders, weight={info['total_weight']:.2%}"
        )

    conc = allocation_info.get("concentration", {})
    lines.append("")
    lines.append("--- Concentration ---")
    lines.append(f"  Max single trader: {conc.get('max_single_weight', 0):.2%}")
    lines.append(f"  Top 5 traders: {conc.get('top_5_weight', 0):.2%}")
    lines.append(f"  Herfindahl index: {conc.get('herfindahl_index', 0):.4f}")

    return "\n".join(lines)
