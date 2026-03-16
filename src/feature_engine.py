"""
Feature engineering: compute per-trader metrics for screening and classification.

Three categories of features:
1. Performance: return, Sharpe, Sortino, Calmar, win rate, profit factor
2. Risk: max drawdown, volatility, VaR, CVaR, tail ratio
3. Style/Behavior: return autocorrelation, skewness, kurtosis, activity rate

Optimized: pre-groups trader data to avoid repeated DataFrame filtering.
"""
import pandas as pd
import numpy as np
from typing import Dict

from .config import Config


def _compute_single_trader(
    rets: np.ndarray,
    equity: np.ndarray,
    dates: np.ndarray,
    pnl_last: float,
    ann: float,
) -> dict:
    """Compute all features for a single trader given pre-extracted arrays."""
    n_obs = len(equity)
    n_valid_rets = len(rets)

    if n_obs < 2:
        return None

    # Date span and activity
    date_span = int((dates[-1] - dates[0]) / np.timedelta64(1, "D"))
    if date_span < 1:
        date_span = 1
    activity_rate = n_obs / (date_span + 1)

    # --- Performance metrics ---
    total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0.0
    ann_return = (1 + total_return) ** (ann / max(date_span, 1)) - 1 if total_return > -1 else -1.0

    # Sharpe / Sortino
    if n_valid_rets >= 5:
        ret_mean = rets.mean()
        ret_std = rets.std(ddof=1)
        sharpe = float(ret_mean / ret_std * np.sqrt(ann)) if ret_std > 1e-10 else 0.0

        downside = rets[rets < 0]
        if len(downside) >= 2:
            ds_std = downside.std(ddof=1)
            sortino = float(ret_mean / ds_std * np.sqrt(ann)) if ds_std > 1e-10 else 10.0
        else:
            sortino = 10.0

        vol = float(ret_std * np.sqrt(ann))
        daily_vol = float(ret_std)
    else:
        sharpe = sortino = vol = daily_vol = 0.0

    # Drawdown
    if len(equity) >= 2:
        peak = np.maximum.accumulate(equity)
        mask = peak > 0
        dd_arr = np.zeros_like(equity, dtype=np.float64)
        dd_arr[mask] = (equity[mask] - peak[mask]) / peak[mask]
        max_dd = float(dd_arr.min())
    else:
        max_dd = 0.0

    # Calmar
    if abs(max_dd) > 1e-10 and date_span >= 5 and total_return > -1:
        ann_ret_c = (1 + total_return) ** (ann / date_span) - 1
        calmar = float(ann_ret_c / abs(max_dd))
    else:
        calmar = 0.0

    # Win rate and profit factor
    if n_valid_rets > 0:
        win_rate = float((rets > 0).sum() / n_valid_rets)
        gains = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        profit_fac = float(gains / losses) if losses > 1e-10 else 10.0
    else:
        win_rate = profit_fac = 0.0

    # VaR / CVaR
    if n_valid_rets >= 10:
        var_5 = float(np.percentile(rets, 5))
        below_var = rets[rets <= var_5]
        cvar_5 = float(below_var.mean()) if len(below_var) > 0 else var_5
        p95 = float(np.percentile(rets, 95))
        tail_ratio = abs(p95) / abs(var_5) if abs(var_5) > 1e-10 else 5.0
    else:
        var_5 = cvar_5 = 0.0
        tail_ratio = 1.0

    worst_day = float(rets.min()) if n_valid_rets > 0 else 0.0
    best_day = float(rets.max()) if n_valid_rets > 0 else 0.0

    # --- Style / Behavior metrics ---
    if n_valid_rets >= 10:
        # Manual autocorrelation (faster than pd.Series.autocorr)
        r1, r2 = rets[:-1], rets[1:]
        m1, m2 = r1.mean(), r2.mean()
        denom = np.sqrt(((r1 - m1) ** 2).sum() * ((r2 - m2) ** 2).sum())
        autocorr = float(((r1 - m1) * (r2 - m2)).sum() / denom) if denom > 1e-15 else 0.0

        # Skewness and kurtosis (manual for speed)
        n = float(n_valid_rets)
        m = rets.mean()
        s = rets.std(ddof=1)
        if s > 1e-15 and n > 2:
            z = (rets - m) / s
            skew = float((n / ((n - 1) * (n - 2))) * (z ** 3).sum())
            if n > 3:
                kurt = float((n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (z ** 4).sum()
                             - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            else:
                kurt = 0.0
        else:
            skew = kurt = 0.0
    else:
        autocorr = skew = kurt = 0.0

    avg_equity = float(equity.mean())

    # Rolling Sharpe stability
    if n_valid_rets >= 30:
        window = 20
        min_p = 10
        # Simple rolling computation
        cum = np.cumsum(rets)
        cum_sq = np.cumsum(rets ** 2)
        rolling_sharpes = []
        for i in range(min_p - 1, n_valid_rets):
            start = max(0, i - window + 1)
            length = i - start + 1
            if length < min_p:
                continue
            s_mean = (cum[i] - (cum[start - 1] if start > 0 else 0)) / length
            s_sq = (cum_sq[i] - (cum_sq[start - 1] if start > 0 else 0)) / length
            s_var = s_sq - s_mean ** 2
            if s_var > 1e-15:
                rolling_sharpes.append(s_mean / np.sqrt(s_var))
        sharpe_stability = float(np.std(rolling_sharpes)) if len(rolling_sharpes) > 2 else 10.0
    else:
        sharpe_stability = 10.0

    # Max gap
    if len(dates) >= 2:
        gaps = np.diff(dates).astype("timedelta64[D]").astype(np.int64)
        max_gap = int(gaps.max())
        median_gap = float(np.median(gaps))
    else:
        max_gap = 999
        median_gap = 999.0

    return {
        "n_observations": n_obs,
        "n_valid_returns": n_valid_rets,
        "date_span_days": date_span,
        "activity_rate": activity_rate,
        "avg_equity": avg_equity,
        "final_pnl": pnl_last,
        "max_gap_days": max_gap,
        "median_gap_days": median_gap,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_fac,
        "max_drawdown": max_dd,
        "volatility": vol,
        "daily_vol": daily_vol,
        "var_5pct": var_5,
        "cvar_5pct": cvar_5,
        "tail_ratio": tail_ratio,
        "worst_day": worst_day,
        "best_day": best_day,
        "return_autocorr": autocorr,
        "skewness": skew,
        "kurtosis": kurt,
        "sharpe_stability": sharpe_stability,
    }


def compute_trader_features(
    returns_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Compute comprehensive features for each trader.

    Optimized by pre-grouping and extracting arrays once per trader.
    """
    ann = config.annualization_factor

    # Pre-sort
    returns_df = returns_df.sort_values(["trader", "date"])

    # Group once and iterate
    grouped = returns_df.groupby("trader")
    records = {}

    for trader, tdf in grouped:
        rets = tdf["daily_return"].dropna().values
        equity = tdf["equity"].values.astype(np.float64)
        dates = tdf["date"].values
        pnl_last = float(tdf["pnl"].iloc[-1])

        result = _compute_single_trader(rets, equity, dates, pnl_last, ann)
        if result is not None:
            records[trader] = result

    features = pd.DataFrame.from_dict(records, orient="index")
    features.index.name = "trader"
    return features
