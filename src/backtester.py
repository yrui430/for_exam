"""
Portfolio backtester: simulate the copy-trading portfolio over history.

Supports both Cython-accelerated and pure Python simulation.
Implements walk-forward validation with in-sample/out-of-sample split.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from .config import Config

# Try to import Cython module; fall back to pure Python
try:
    from .fast_backtest import fast_portfolio_simulate
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


def _pure_python_simulate(
    return_matrix: np.ndarray,
    weights: np.ndarray,
    per_trader_stop_loss: float,
    portfolio_max_drawdown: float,
    daily_loss_limit: float,
    drawdown_reduce_threshold: float,
    drawdown_reduce_factor: float,
    transaction_cost_bps: float,
    rebalance_interval: int,
    cooldown_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pure Python fallback for portfolio simulation."""
    n_days, n_traders = return_matrix.shape
    tc = transaction_cost_bps / 10000.0

    port_equity = np.ones(n_days)
    port_returns = np.zeros(n_days)
    active_weights = np.zeros((n_days, n_traders))

    current_weights = weights.copy()
    initial_weights = weights.copy()
    trader_cum_ret = np.zeros(n_traders)
    trader_peak = np.ones(n_traders)
    stop_day = np.full(n_traders, -9999, dtype=np.int32)

    port_peak = 1.0
    port_val = 1.0
    n_stops = 0

    for i in range(n_days):
        daily_ret = 0.0

        for j in range(n_traders):
            if stop_day[j] > -9999:
                day_since_stop = i - stop_day[j]
                if day_since_stop < cooldown_days:
                    current_weights[j] = 0.0
                    active_weights[i, j] = 0.0
                    continue
                else:
                    stop_day[j] = -9999
                    current_weights[j] = initial_weights[j] * 0.5
                    trader_cum_ret[j] = 0.0
                    trader_peak[j] = 1.0

            trader_ret = return_matrix[i, j]
            if np.isnan(trader_ret):
                trader_ret = 0.0

            trader_cum_ret[j] += trader_ret
            trader_equity = 1.0 + trader_cum_ret[j]

            if trader_equity > trader_peak[j]:
                trader_peak[j] = trader_equity

            trader_dd = 0.0
            if trader_peak[j] > 0:
                trader_dd = (trader_equity - trader_peak[j]) / trader_peak[j]

            if trader_dd < per_trader_stop_loss:
                stop_day[j] = i
                current_weights[j] = 0.0
                n_stops += 1
                active_weights[i, j] = 0.0
                continue

            if trader_dd < drawdown_reduce_threshold:
                current_weights[j] = initial_weights[j] * drawdown_reduce_factor

            active_weights[i, j] = current_weights[j]
            daily_ret += current_weights[j] * trader_ret

        # Transaction cost on rebalance
        if rebalance_interval > 0 and i > 0 and i % rebalance_interval == 0:
            cost = 0.0
            for j in range(n_traders):
                cost += abs(active_weights[i, j] - active_weights[i - 1, j]) * tc
            daily_ret -= cost

        port_val *= (1.0 + daily_ret)
        port_equity[i] = port_val
        port_returns[i] = daily_ret

        if port_val > port_peak:
            port_peak = port_val

        port_dd = (port_val - port_peak) / port_peak
        if port_dd < portfolio_max_drawdown:
            current_weights *= 0.5

    return port_equity, port_returns, active_weights, n_stops


def run_backtest(
    allocated: pd.DataFrame,
    return_panel: pd.DataFrame,
    config: Config,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> Dict:
    """
    Run portfolio backtest simulation.

    Args:
        allocated: DataFrame with trader weights
        return_panel: date x trader return panel
        config: Configuration
        start_date: Optional start date for backtest window
        end_date: Optional end date for backtest window

    Returns:
        Dict with backtest results including equity curve, metrics, etc.
    """
    rp = config.risk
    ap = config.allocation

    # Get trader list and weights
    traders = allocated.index.tolist()
    weight_dict = allocated["weight"].to_dict()

    # Subset return panel to allocated traders
    available = [t for t in traders if t in return_panel.columns]
    if not available:
        return {"error": "No trader data available for backtest"}

    panel = return_panel[available].copy()

    if start_date is not None:
        panel = panel[panel.index >= start_date]
    if end_date is not None:
        panel = panel[panel.index <= end_date]

    # Build arrays
    weights = np.array([weight_dict.get(t, 0.0) for t in available])
    return_matrix = panel.fillna(0.0).values.astype(np.float64)
    weights = weights.astype(np.float64)

    # Choose simulation engine
    simulate_fn = fast_portfolio_simulate if CYTHON_AVAILABLE else _pure_python_simulate

    port_equity, port_returns, active_weights, n_stops = simulate_fn(
        return_matrix=return_matrix,
        weights=weights,
        per_trader_stop_loss=rp.per_trader_stop_loss,
        portfolio_max_drawdown=rp.portfolio_max_drawdown,
        daily_loss_limit=rp.daily_loss_limit,
        drawdown_reduce_threshold=rp.drawdown_reduce_threshold,
        drawdown_reduce_factor=rp.drawdown_reduce_factor,
        transaction_cost_bps=config.assumptions.transaction_cost_bps,
        rebalance_interval=ap.rebalance_interval_days,
        cooldown_days=rp.stop_loss_cooldown_days,
    )

    # Compute metrics
    metrics = _compute_backtest_metrics(port_equity, port_returns, config)
    metrics["n_stop_loss_events"] = int(n_stops)
    metrics["engine"] = "cython" if CYTHON_AVAILABLE else "python"

    return {
        "equity_curve": pd.Series(port_equity, index=panel.index, name="portfolio_equity"),
        "daily_returns": pd.Series(port_returns, index=panel.index, name="daily_return"),
        "active_weights": pd.DataFrame(active_weights, index=panel.index, columns=available),
        "metrics": metrics,
        "traders": available,
        "dates": panel.index,
    }


def run_walk_forward(
    allocated: pd.DataFrame,
    return_panel: pd.DataFrame,
    config: Config,
    in_sample_ratio: float = 0.6,
) -> Dict:
    """
    Walk-forward validation: train on first portion, test on remainder.

    This validates whether the screening and allocation decisions made on
    historical data hold up in the out-of-sample period.
    """
    dates = return_panel.index
    split_idx = int(len(dates) * in_sample_ratio)
    split_date = dates[split_idx]

    # In-sample backtest
    is_result = run_backtest(
        allocated, return_panel, config,
        end_date=split_date,
    )

    # Out-of-sample backtest
    oos_result = run_backtest(
        allocated, return_panel, config,
        start_date=split_date,
    )

    return {
        "in_sample": is_result,
        "out_of_sample": oos_result,
        "split_date": split_date,
    }


def _compute_backtest_metrics(
    equity: np.ndarray,
    returns: np.ndarray,
    config: Config,
) -> Dict:
    """Compute comprehensive backtest performance metrics."""
    ann = config.annualization_factor
    n = len(returns)

    if n < 2:
        return {}

    # Filter out zero returns for stats
    valid_rets = returns[~np.isnan(returns)]
    if len(valid_rets) < 2:
        return {}

    # Total and annualized return
    total_return = float(equity[-1] / equity[0] - 1)
    ann_return = float((equity[-1] / equity[0]) ** (ann / n) - 1) if equity[0] > 0 else 0.0

    # Volatility
    vol = float(np.std(valid_rets, ddof=1) * np.sqrt(ann))
    daily_vol = float(np.std(valid_rets, ddof=1))

    # Sharpe
    sharpe = float(np.mean(valid_rets) / np.std(valid_rets, ddof=1) * np.sqrt(ann)) if np.std(valid_rets) > 1e-10 else 0.0

    # Sortino
    downside = valid_rets[valid_rets < 0]
    if len(downside) > 1:
        sortino = float(np.mean(valid_rets) / np.std(downside, ddof=1) * np.sqrt(ann))
    else:
        sortino = 10.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    # Calmar
    calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    # Win rate
    win_rate = float((valid_rets > 0).sum() / len(valid_rets))

    # Profit factor
    gains = valid_rets[valid_rets > 0].sum()
    losses = abs(valid_rets[valid_rets < 0].sum())
    profit_factor = float(gains / losses) if losses > 1e-10 else 10.0

    # VaR / CVaR
    var_5 = float(np.percentile(valid_rets, 5))
    cvar_5 = float(valid_rets[valid_rets <= var_5].mean()) if (valid_rets <= var_5).any() else var_5

    # Max consecutive losses
    is_loss = valid_rets < 0
    max_consec_loss = 0
    current_streak = 0
    for loss in is_loss:
        if loss:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Drawdown duration
    dd_duration = 0
    max_dd_duration = 0
    for d in dd:
        if d < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "volatility": vol,
        "daily_vol": daily_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "var_5pct": var_5,
        "cvar_5pct": cvar_5,
        "max_consecutive_losses": max_consec_loss,
        "max_drawdown_duration_days": max_dd_duration,
        "n_trading_days": n,
    }


def get_backtest_summary(result: Dict) -> str:
    """Generate text summary of backtest results."""
    m = result["metrics"]
    lines = [
        "=== Backtest Results ===",
        f"Engine: {m.get('engine', 'python')}",
        f"Period: {result['dates'][0].strftime('%Y-%m-%d')} to {result['dates'][-1].strftime('%Y-%m-%d')}",
        f"Trading days: {m['n_trading_days']}",
        "",
        "--- Returns ---",
        f"Total return: {m['total_return']:.2%}",
        f"Annualized return: {m['ann_return']:.2%}",
        f"Volatility (ann.): {m['volatility']:.2%}",
        "",
        "--- Risk-Adjusted ---",
        f"Sharpe ratio: {m['sharpe']:.3f}",
        f"Sortino ratio: {m['sortino']:.3f}",
        f"Calmar ratio: {m['calmar']:.3f}",
        "",
        "--- Risk ---",
        f"Max drawdown: {m['max_drawdown']:.2%}",
        f"Max DD duration: {m['max_drawdown_duration_days']} days",
        f"VaR (5%): {m['var_5pct']:.2%}",
        f"CVaR (5%): {m['cvar_5pct']:.2%}",
        "",
        "--- Trading ---",
        f"Win rate: {m['win_rate']:.2%}",
        f"Profit factor: {m['profit_factor']:.2f}",
        f"Max consecutive losses: {m['max_consecutive_losses']}",
        f"Stop-loss events: {m.get('n_stop_loss_events', 'N/A')}",
    ]
    return "\n".join(lines)
