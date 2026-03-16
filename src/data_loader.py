"""
Data loading, cleaning, and daily return computation.

Key design decisions:
- Returns are computed between consecutive observations only.
- Deposits/withdrawals are stripped out so returns reflect pure trading PnL.
- Days with zero or near-zero equity are handled to avoid division errors.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from .config import Config, DATA_PATH


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and perform basic type conversions."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(["trader", "date"]).reset_index(drop=True)
    return df


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deposit-adjusted daily returns for each trader.

    daily_trading_pnl = (equity_t - equity_{t-1}) - (net_deposit_t - net_deposit_{t-1})
    daily_return = daily_trading_pnl / equity_{t-1}

    Returns are only computed between consecutive observation rows.
    If the gap between observations > max_gap_days, return is set to NaN.
    """
    df = df.copy()
    df = df.sort_values(["trader", "date"])

    # Compute differences within each trader group
    df["equity_prev"] = df.groupby("trader")["equity"].shift(1)
    df["deposit_prev"] = df.groupby("trader")["net_deposit"].shift(1)
    df["date_prev"] = df.groupby("trader")["date"].shift(1)

    # Trading PnL = equity change - deposit change
    df["deposit_change"] = df["net_deposit"] - df["deposit_prev"]
    df["trading_pnl"] = (df["equity"] - df["equity_prev"]) - df["deposit_change"]

    # Gap in days between observations
    df["day_gap"] = (df["date"] - df["date_prev"]).dt.days

    # Daily return (annualized to per-day by dividing by gap if gap > 1)
    # Use per-observation return (not annualized per day) for accuracy
    mask = (df["equity_prev"] > 0) & (df["equity_prev"].notna())
    df["daily_return"] = np.nan
    df.loc[mask, "daily_return"] = df.loc[mask, "trading_pnl"] / df.loc[mask, "equity_prev"]

    # For multi-day gaps, compute the per-day return (geometric decomposition)
    multi_day = mask & (df["day_gap"] > 1)
    if multi_day.any():
        gross = 1 + df.loc[multi_day, "daily_return"]
        gross = gross.clip(lower=0.001)  # prevent negative base for fractional exponent
        per_day = np.sign(df.loc[multi_day, "daily_return"]) * (
            np.abs(gross) ** (1.0 / df.loc[multi_day, "day_gap"]) - 1
        )
        df.loc[multi_day, "daily_return"] = per_day

    # Cap extreme returns to reduce noise from micro-accounts
    df["daily_return"] = df["daily_return"].clip(-0.99, 10.0)

    return df


def build_daily_panel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Build a date x trader panel of daily returns.

    Returns:
        panel: DataFrame with date index, trader columns, values = daily_return
        calendar: Full date range
    """
    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    # Pivot to panel
    pivot = df.pivot_table(
        index="date", columns="trader", values="daily_return", aggfunc="first"
    )
    pivot = pivot.reindex(calendar)

    return pivot, calendar


def build_equity_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build a date x trader panel of equity values (forward-filled)."""
    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    pivot = df.pivot_table(
        index="date", columns="trader", values="equity", aggfunc="first"
    )
    pivot = pivot.reindex(calendar).ffill()
    return pivot


def prepare_data(config: Config) -> Dict:
    """
    Full data preparation pipeline.

    Returns a dict with:
        - raw_df: raw data
        - returns_df: per-observation returns
        - return_panel: date x trader return panel
        - equity_panel: date x trader equity panel (forward-filled)
        - calendar: full date range
        - trader_list: list of all trader addresses
    """
    raw_df = load_raw_data()
    returns_df = compute_daily_returns(raw_df)
    return_panel, calendar = build_daily_panel(returns_df)
    equity_panel = build_equity_panel(raw_df)

    return {
        "raw_df": raw_df,
        "returns_df": returns_df,
        "return_panel": return_panel,
        "equity_panel": equity_panel,
        "calendar": calendar,
        "trader_list": raw_df["trader"].unique().tolist(),
    }
