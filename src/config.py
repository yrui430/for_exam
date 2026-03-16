"""
Configuration and assumptions for the multi-trader copy trading strategy.
"""
import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "笔试题数据包.csv")
REPORT_DIR = os.path.join(BASE_DIR, "report")


@dataclass
class Assumptions:
    """Key assumptions for the copy trading strategy."""
    # --- Execution Model ---
    # We follow position changes (not raw trades) to reduce frequency and slippage
    follow_mode: str = "position_change"
    # Estimated execution delay in seconds (time to detect + place order)
    execution_delay_sec: float = 5.0
    # One-way transaction cost (taker fee + estimated slippage)
    transaction_cost_bps: float = 8.0  # 5 bps fee + 3 bps slippage
    # Maximum leverage allowed for the copy portfolio
    max_leverage: float = 3.0
    # Whether unified risk control overrides individual trader signals
    unified_risk_control: bool = True


@dataclass
class ScreeningParams:
    """Parameters for trader screening."""
    # Minimum number of observation days to be considered
    min_observation_days: int = 60
    # Minimum number of active trading days (with equity change)
    min_active_days: int = 30
    # Minimum Sharpe ratio (annualized, daily frequency)
    min_sharpe: float = 0.5
    # Maximum drawdown threshold (absolute value)
    max_drawdown_threshold: float = 0.50
    # Minimum equity to ensure reasonable account size ($)
    min_avg_equity: float = 1000.0
    # Maximum single-day loss that indicates unacceptable tail risk
    max_single_day_loss: float = 0.40
    # Minimum win rate for daily returns
    min_win_rate: float = 0.35
    # Minimum profit factor
    min_profit_factor: float = 1.0
    # Maximum gap days allowed between observations (data quality)
    max_gap_days: int = 14


@dataclass
class ClassificationParams:
    """Parameters for trader classification."""
    n_clusters: int = 4  # Number of style clusters
    # Tier thresholds (percentiles within qualified traders)
    core_top_pct: float = 0.30   # Top 30% -> Core
    satellite_pct: float = 0.40  # Next 40% -> Satellite
    # Bottom 30% -> Watch list (not allocated in initial deployment)


@dataclass
class AllocationParams:
    """Parameters for capital allocation."""
    total_capital: float = 100_000.0  # Total portfolio capital ($)
    # Capital allocation across tiers
    core_weight: float = 0.60
    satellite_weight: float = 0.30
    cash_reserve: float = 0.10
    # Per-trader concentration limits
    max_single_trader_weight: float = 0.10  # No single trader > 10%
    min_single_trader_weight: float = 0.01  # Minimum 1% to be meaningful
    # Allocation method: "inverse_vol", "risk_parity", "score_weighted"
    method: str = "score_weighted"
    # Rebalance frequency in days
    rebalance_interval_days: int = 7


@dataclass
class RiskParams:
    """Parameters for risk management."""
    # Per-trader trailing stop loss (on cumulative return since entry)
    per_trader_stop_loss: float = -0.15
    # Portfolio-level maximum drawdown before full de-risk
    portfolio_max_drawdown: float = -0.10
    # Portfolio daily loss limit
    daily_loss_limit: float = -0.03
    # Correlation threshold: flag if pairwise correlation exceeds this
    correlation_warning_threshold: float = 0.70
    # Maximum number of highly correlated trader pairs allowed
    max_correlated_pairs: int = 3
    # Maximum weight in any single style cluster
    max_cluster_weight: float = 0.40
    # Cooldown days after a trader is stopped out before re-entry
    stop_loss_cooldown_days: int = 14
    # Drawdown recovery: reduce allocation if trailing drawdown exceeds threshold
    drawdown_reduce_threshold: float = -0.08
    drawdown_reduce_factor: float = 0.50


@dataclass
class Config:
    """Master configuration."""
    assumptions: Assumptions = field(default_factory=Assumptions)
    screening: ScreeningParams = field(default_factory=ScreeningParams)
    classification: ClassificationParams = field(default_factory=ClassificationParams)
    allocation: AllocationParams = field(default_factory=AllocationParams)
    risk: RiskParams = field(default_factory=RiskParams)
    # Annualization factor for daily data
    annualization_factor: float = 365.0
    # Random seed for reproducibility
    random_seed: int = 42
