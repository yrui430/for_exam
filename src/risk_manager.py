"""
Risk management: portfolio-level and per-trader risk controls.

Implements:
1. Per-trader trailing stop loss
2. Portfolio-level drawdown circuit breaker
3. Daily loss limit
4. Correlation monitoring
5. Drawdown-based position reduction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .config import Config


def compute_correlation_matrix(
    return_panel: pd.DataFrame,
    traders: List[str],
    min_overlap: int = 30,
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix for selected traders.

    Only uses overlapping observation windows with at least min_overlap days.
    """
    available = [t for t in traders if t in return_panel.columns]
    sub = return_panel[available].dropna(how="all")

    corr = sub.corr(min_periods=min_overlap)
    return corr


def identify_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.70,
) -> List[Tuple[str, str, float]]:
    """Find pairs of traders with correlation above threshold."""
    pairs = []
    traders = corr_matrix.columns.tolist()

    for i in range(len(traders)):
        for j in range(i + 1, len(traders)):
            c = corr_matrix.iloc[i, j]
            if not np.isnan(c) and abs(c) > threshold:
                pairs.append((traders[i], traders[j], float(c)))

    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


def apply_risk_adjustments(
    allocated: pd.DataFrame,
    return_panel: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply risk-based adjustments to the allocation.

    Checks:
    1. Highly correlated pairs -> reduce combined weight
    2. Cluster concentration -> already handled in allocator
    3. Flag potential issues

    Returns:
        adjusted: DataFrame with risk-adjusted weights
        risk_report: Dict with risk findings
    """
    rp = config.risk
    adjusted = allocated.copy()

    traders = adjusted.index.tolist()
    corr_matrix = compute_correlation_matrix(return_panel, traders)
    correlated_pairs = identify_correlated_pairs(corr_matrix, rp.correlation_warning_threshold)

    # Reduce weight of highly correlated traders
    if len(correlated_pairs) > rp.max_correlated_pairs:
        # For excess correlated pairs, reduce the lower-scored trader
        for pair in correlated_pairs[rp.max_correlated_pairs:]:
            t1, t2, c = pair
            if t1 in adjusted.index and t2 in adjusted.index:
                # Reduce the lower-scored one
                if adjusted.loc[t1, "composite_score"] < adjusted.loc[t2, "composite_score"]:
                    adjusted.loc[t1, "weight"] *= 0.7
                else:
                    adjusted.loc[t2, "weight"] *= 0.7

    # Re-normalize weights
    total_budget = config.allocation.core_weight + config.allocation.satellite_weight
    current = adjusted["weight"].sum()
    if current > 0:
        adjusted["weight"] = adjusted["weight"] / current * total_budget
    adjusted["dollar_allocation"] = adjusted["weight"] * config.allocation.total_capital

    risk_report = {
        "correlated_pairs": correlated_pairs[:10],  # Top 10
        "n_correlated_pairs": len(correlated_pairs),
        "avg_correlation": float(
            corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        ) if len(traders) > 1 else 0.0,
        "cluster_concentrations": {
            int(c): float(adjusted[adjusted["cluster"] == c]["weight"].sum())
            for c in adjusted["cluster"].unique()
        },
    }

    return adjusted, risk_report


class PortfolioRiskMonitor:
    """
    Real-time risk monitoring for the portfolio simulation.

    Tracks:
    - Per-trader cumulative PnL and drawdown
    - Portfolio-level equity curve and drawdown
    - Stop-loss triggers
    """

    def __init__(self, config: Config, traders: List[str], weights: Dict[str, float]):
        self.config = config
        self.rp = config.risk
        self.traders = traders
        self.initial_weights = weights.copy()
        self.active_weights = weights.copy()

        # Tracking state
        self.trader_peak_equity = {t: 1.0 for t in traders}
        self.trader_cum_return = {t: 0.0 for t in traders}
        self.stopped_traders = {}  # trader -> stop_date
        self.portfolio_peak = 1.0
        self.portfolio_equity = 1.0
        self.alerts = []

    def update(self, date, trader_returns: Dict[str, float]) -> Dict:
        """
        Process one day of returns and apply risk rules.

        Returns dict with actions taken.
        """
        actions = {"date": date, "stops": [], "reduces": [], "alerts": []}

        # Update per-trader tracking
        for trader in self.traders:
            if trader in self.stopped_traders:
                # Check cooldown
                stop_date = self.stopped_traders[trader]
                if (date - stop_date).days < self.rp.stop_loss_cooldown_days:
                    continue
                else:
                    # Re-activate with reduced weight
                    del self.stopped_traders[trader]
                    self.active_weights[trader] = self.initial_weights.get(trader, 0) * 0.5
                    self.trader_peak_equity[trader] = 1.0
                    self.trader_cum_return[trader] = 0.0

            ret = trader_returns.get(trader, 0.0)
            if np.isnan(ret):
                ret = 0.0

            self.trader_cum_return[trader] += ret
            trader_equity = 1.0 + self.trader_cum_return[trader]

            if trader_equity > self.trader_peak_equity[trader]:
                self.trader_peak_equity[trader] = trader_equity

            # Per-trader drawdown
            if self.trader_peak_equity[trader] > 0:
                trader_dd = (trader_equity - self.trader_peak_equity[trader]) / self.trader_peak_equity[trader]
            else:
                trader_dd = 0.0

            # Check per-trader stop loss
            if trader_dd < self.rp.per_trader_stop_loss:
                self.stopped_traders[trader] = date
                self.active_weights[trader] = 0.0
                actions["stops"].append(trader)

            # Check drawdown reduction threshold
            elif trader_dd < self.rp.drawdown_reduce_threshold:
                self.active_weights[trader] = (
                    self.initial_weights.get(trader, 0) * self.rp.drawdown_reduce_factor
                )
                actions["reduces"].append(trader)

        # Portfolio-level return
        port_ret = sum(
            self.active_weights.get(t, 0) * trader_returns.get(t, 0.0)
            for t in self.traders
            if t not in self.stopped_traders and not np.isnan(trader_returns.get(t, 0.0))
        )

        self.portfolio_equity *= (1 + port_ret)
        if self.portfolio_equity > self.portfolio_peak:
            self.portfolio_peak = self.portfolio_equity

        # Portfolio drawdown
        port_dd = (self.portfolio_equity - self.portfolio_peak) / self.portfolio_peak

        # Portfolio circuit breaker
        if port_dd < self.rp.portfolio_max_drawdown:
            actions["alerts"].append(f"CIRCUIT BREAKER: Portfolio DD={port_dd:.2%}")
            # Reduce all weights by 50%
            for t in self.traders:
                if t not in self.stopped_traders:
                    self.active_weights[t] *= 0.5

        # Daily loss limit
        if port_ret < self.rp.daily_loss_limit:
            actions["alerts"].append(f"DAILY LOSS LIMIT: {port_ret:.2%}")

        actions["portfolio_equity"] = self.portfolio_equity
        actions["portfolio_dd"] = port_dd
        actions["portfolio_return"] = port_ret
        actions["n_active"] = len(self.traders) - len(self.stopped_traders)

        return actions
