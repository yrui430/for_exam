"""
Trader screening: multi-criteria filtering to identify traders worth following.

Filtering philosophy:
- We are conservative: false negatives (missing a good trader) are acceptable,
  but false positives (following a bad trader) are costly.
- Each criterion targets a specific failure mode.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .config import Config


def apply_screening(
    features: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply multi-criteria screening to trader features.

    Returns:
        qualified: DataFrame of traders passing all screens
        rejected: DataFrame of traders failing at least one screen
        rejection_reasons: Dict mapping trader -> list of failure reasons
    """
    sp = config.screening
    reasons = {t: [] for t in features.index}

    # 1. Minimum observation period
    mask_obs = features["date_span_days"] >= sp.min_observation_days
    for t in features[~mask_obs].index:
        reasons[t].append(f"date_span={features.loc[t, 'date_span_days']}d < {sp.min_observation_days}d")

    # 2. Minimum active days
    mask_active = features["n_valid_returns"] >= sp.min_active_days
    for t in features[~mask_active].index:
        reasons[t].append(f"active_days={features.loc[t, 'n_valid_returns']} < {sp.min_active_days}")

    # 3. Data quality: max gap
    mask_gap = features["max_gap_days"] <= sp.max_gap_days
    for t in features[~mask_gap].index:
        reasons[t].append(f"max_gap={features.loc[t, 'max_gap_days']}d > {sp.max_gap_days}d")

    # 4. Minimum average equity (account size)
    mask_equity = features["avg_equity"] >= sp.min_avg_equity
    for t in features[~mask_equity].index:
        reasons[t].append(f"avg_equity=${features.loc[t, 'avg_equity']:.0f} < ${sp.min_avg_equity}")

    # 5. Minimum Sharpe ratio
    mask_sharpe = features["sharpe"] >= sp.min_sharpe
    for t in features[~mask_sharpe].index:
        reasons[t].append(f"sharpe={features.loc[t, 'sharpe']:.2f} < {sp.min_sharpe}")

    # 6. Maximum drawdown
    mask_dd = features["max_drawdown"] >= -sp.max_drawdown_threshold
    for t in features[~mask_dd].index:
        reasons[t].append(f"max_dd={features.loc[t, 'max_drawdown']:.2%} > {sp.max_drawdown_threshold:.0%}")

    # 7. Worst single day loss
    mask_tail = features["worst_day"] >= -sp.max_single_day_loss
    for t in features[~mask_tail].index:
        reasons[t].append(f"worst_day={features.loc[t, 'worst_day']:.2%}")

    # 8. Minimum win rate
    mask_wr = features["win_rate"] >= sp.min_win_rate
    for t in features[~mask_wr].index:
        reasons[t].append(f"win_rate={features.loc[t, 'win_rate']:.2%} < {sp.min_win_rate:.0%}")

    # 9. Minimum profit factor
    mask_pf = features["profit_factor"] >= sp.min_profit_factor
    for t in features[~mask_pf].index:
        reasons[t].append(f"profit_factor={features.loc[t, 'profit_factor']:.2f} < {sp.min_profit_factor}")

    # Combine all masks
    all_pass = mask_obs & mask_active & mask_gap & mask_equity & mask_sharpe & mask_dd & mask_tail & mask_wr & mask_pf

    qualified = features[all_pass].copy()
    rejected = features[~all_pass].copy()

    # Clean up reasons dict
    rejection_reasons = {t: r for t, r in reasons.items() if len(r) > 0}

    # Add a composite score for ranking
    qualified = _compute_composite_score(qualified, config)

    return qualified, rejected, rejection_reasons


def _compute_composite_score(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Compute a composite score for ranking qualified traders.

    Score components (weighted):
    - Sharpe ratio: 25%
    - Sortino ratio: 15%
    - Calmar ratio: 10%
    - Profit factor: 15%
    - Stability (inverse sharpe_stability): 15%
    - Win rate: 10%
    - Tail ratio: 10%
    """
    df = df.copy()

    def _rank_normalize(series: pd.Series) -> pd.Series:
        """Rank-normalize to [0, 1]."""
        ranked = series.rank(pct=True, method="average")
        return ranked

    scores = pd.DataFrame(index=df.index)
    scores["sharpe_score"] = _rank_normalize(df["sharpe"]) * 0.25
    scores["sortino_score"] = _rank_normalize(df["sortino"]) * 0.15
    scores["calmar_score"] = _rank_normalize(df["calmar"]) * 0.10
    scores["pf_score"] = _rank_normalize(df["profit_factor"]) * 0.15
    # Lower sharpe_stability = more consistent = better
    scores["stability_score"] = _rank_normalize(-df["sharpe_stability"]) * 0.15
    scores["wr_score"] = _rank_normalize(df["win_rate"]) * 0.10
    scores["tail_score"] = _rank_normalize(df["tail_ratio"]) * 0.10

    df["composite_score"] = scores.sum(axis=1)

    return df


def get_screening_summary(
    features: pd.DataFrame,
    qualified: pd.DataFrame,
    rejected: pd.DataFrame,
    rejection_reasons: Dict,
) -> str:
    """Generate a text summary of screening results."""
    lines = [
        f"=== Screening Summary ===",
        f"Total traders evaluated: {len(features)}",
        f"Qualified: {len(qualified)} ({len(qualified)/len(features)*100:.1f}%)",
        f"Rejected: {len(rejected)} ({len(rejected)/len(features)*100:.1f}%)",
        f"",
        f"--- Rejection Breakdown ---",
    ]

    # Count each rejection reason type
    reason_counts = {}
    for trader, r_list in rejection_reasons.items():
        for r in r_list:
            key = r.split("=")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count} traders")

    lines.append("")
    lines.append("--- Qualified Trader Stats ---")
    for col in ["sharpe", "sortino", "max_drawdown", "win_rate", "total_return", "composite_score"]:
        if col in qualified.columns:
            lines.append(
                f"  {col}: mean={qualified[col].mean():.3f}, "
                f"median={qualified[col].median():.3f}, "
                f"std={qualified[col].std():.3f}"
            )

    return "\n".join(lines)
