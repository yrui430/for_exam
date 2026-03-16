# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated portfolio simulation loop.

Speeds up the day-by-day iteration through returns,
weight application, drawdown tracking, and stop-loss checks.
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs, isnan, NAN

ctypedef np.float64_t DTYPE_t


def fast_portfolio_simulate(
    np.ndarray[DTYPE_t, ndim=2] return_matrix,  # (n_days, n_traders)
    np.ndarray[DTYPE_t, ndim=1] weights,         # (n_traders,)
    double per_trader_stop_loss,
    double portfolio_max_drawdown,
    double daily_loss_limit,
    double drawdown_reduce_threshold,
    double drawdown_reduce_factor,
    double transaction_cost_bps,
    int rebalance_interval,
    int cooldown_days,
):
    """
    Fast portfolio simulation with risk management.

    Returns:
        portfolio_equity: array of daily portfolio equity values
        portfolio_returns: array of daily returns
        trader_active: 2D array of active flags per day per trader
        n_stops: total number of stop-loss events
    """
    cdef int n_days = return_matrix.shape[0]
    cdef int n_traders = return_matrix.shape[1]
    cdef int i, j, day_since_stop

    # Output arrays
    cdef np.ndarray[DTYPE_t, ndim=1] port_equity = np.ones(n_days, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] port_returns = np.zeros(n_days, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] active_weights = np.zeros((n_days, n_traders), dtype=np.float64)

    # State variables
    cdef np.ndarray[DTYPE_t, ndim=1] current_weights = weights.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] initial_weights = weights.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] trader_cum_ret = np.zeros(n_traders, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] trader_peak = np.ones(n_traders, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] stop_day = np.full(n_traders, -9999, dtype=np.int32)

    cdef double port_peak = 1.0
    cdef double port_val = 1.0
    cdef double daily_ret, trader_ret, trader_equity, trader_dd, port_dd
    cdef double cost_adjustment
    cdef int n_stops = 0
    cdef int n_rebalances = 0

    # Transaction cost as a fraction
    cdef double tc = transaction_cost_bps / 10000.0

    for i in range(n_days):
        daily_ret = 0.0

        for j in range(n_traders):
            # Check if trader is stopped
            if stop_day[j] > -9999:
                day_since_stop = i - stop_day[j]
                if day_since_stop < cooldown_days:
                    current_weights[j] = 0.0
                    active_weights[i, j] = 0.0
                    continue
                else:
                    # Reactivate with halved weight
                    stop_day[j] = -9999
                    current_weights[j] = initial_weights[j] * 0.5
                    trader_cum_ret[j] = 0.0
                    trader_peak[j] = 1.0

            trader_ret = return_matrix[i, j]
            if isnan(trader_ret):
                trader_ret = 0.0

            # Update trader tracking
            trader_cum_ret[j] += trader_ret
            trader_equity = 1.0 + trader_cum_ret[j]

            if trader_equity > trader_peak[j]:
                trader_peak[j] = trader_equity

            # Trader drawdown
            if trader_peak[j] > 0:
                trader_dd = (trader_equity - trader_peak[j]) / trader_peak[j]
            else:
                trader_dd = 0.0

            # Per-trader stop loss
            if trader_dd < per_trader_stop_loss:
                stop_day[j] = i
                current_weights[j] = 0.0
                n_stops += 1
                active_weights[i, j] = 0.0
                continue

            # Drawdown reduction
            if trader_dd < drawdown_reduce_threshold:
                current_weights[j] = initial_weights[j] * drawdown_reduce_factor

            active_weights[i, j] = current_weights[j]

            # Accumulate portfolio return
            daily_ret += current_weights[j] * trader_ret

        # Apply transaction cost on rebalance days
        if rebalance_interval > 0 and i > 0 and i % rebalance_interval == 0:
            # Estimate turnover as sum of weight changes
            cost_adjustment = 0.0
            for j in range(n_traders):
                if i > 0:
                    cost_adjustment += fabs(active_weights[i, j] - active_weights[i-1, j]) * tc
            daily_ret -= cost_adjustment
            n_rebalances += 1

        # Portfolio equity update
        port_val *= (1.0 + daily_ret)
        port_equity[i] = port_val
        port_returns[i] = daily_ret

        # Portfolio drawdown tracking
        if port_val > port_peak:
            port_peak = port_val

        port_dd = (port_val - port_peak) / port_peak

        # Portfolio circuit breaker
        if port_dd < portfolio_max_drawdown:
            for j in range(n_traders):
                current_weights[j] *= 0.5

    return port_equity, port_returns, active_weights, n_stops
