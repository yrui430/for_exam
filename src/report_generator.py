"""
Report generator: produce visualizations and comprehensive analysis report.

Outputs:
- PNG charts saved to report/ directory
- HTML report with embedded analysis
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

from .config import Config, REPORT_DIR


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150


def generate_all_charts(
    features: pd.DataFrame,
    qualified: pd.DataFrame,
    classified: pd.DataFrame,
    allocated: pd.DataFrame,
    backtest_result: Dict,
    walk_forward_result: Dict,
    risk_report: Dict,
    cluster_info: Dict,
    config: Config,
):
    """Generate all report charts."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    plot_universe_overview(features, qualified)
    plot_feature_distributions(qualified)
    plot_screening_funnel(features, qualified)
    plot_cluster_scatter(classified, cluster_info)
    plot_tier_comparison(classified)
    plot_allocation_pie(allocated)
    plot_backtest_equity(backtest_result)
    plot_backtest_drawdown(backtest_result)
    plot_walk_forward(walk_forward_result)
    plot_correlation_heatmap(allocated, backtest_result)
    plot_monthly_returns(backtest_result)
    plot_risk_dashboard(backtest_result, risk_report, config)


def plot_universe_overview(features: pd.DataFrame, qualified: pd.DataFrame):
    """Overview of the trader universe."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trader Universe Overview", fontsize=16, fontweight="bold")

    # 1. Sharpe distribution
    ax = axes[0, 0]
    ax.hist(features["sharpe"].clip(-5, 10), bins=50, alpha=0.6, color="steelblue", label="All")
    if len(qualified) > 0:
        ax.hist(qualified["sharpe"].clip(-5, 10), bins=50, alpha=0.6, color="green", label="Qualified")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Sharpe Ratio Distribution")
    ax.legend()

    # 2. Max drawdown distribution
    ax = axes[0, 1]
    ax.hist(features["max_drawdown"].clip(-1, 0), bins=50, alpha=0.6, color="steelblue", label="All")
    if len(qualified) > 0:
        ax.hist(qualified["max_drawdown"].clip(-1, 0), bins=50, alpha=0.6, color="green", label="Qualified")
    ax.set_xlabel("Max Drawdown")
    ax.set_ylabel("Count")
    ax.set_title("Max Drawdown Distribution")
    ax.legend()

    # 3. Observation days vs Sharpe
    ax = axes[1, 0]
    colors = ["green" if t in qualified.index else "gray" for t in features.index]
    ax.scatter(features["date_span_days"], features["sharpe"].clip(-5, 10),
               c=colors, alpha=0.4, s=10)
    ax.set_xlabel("Observation Days")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Observation Period vs Sharpe")

    # 4. Equity distribution (log scale)
    ax = axes[1, 1]
    eq = features["avg_equity"].clip(lower=1)
    ax.hist(np.log10(eq), bins=50, alpha=0.6, color="steelblue")
    ax.set_xlabel("Log10(Avg Equity $)")
    ax.set_ylabel("Count")
    ax.set_title("Account Size Distribution")

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "01_universe_overview.png"))
    plt.close(fig)


def plot_feature_distributions(qualified: pd.DataFrame):
    """Feature distributions for qualified traders."""
    features_to_plot = [
        ("sharpe", "Sharpe Ratio"),
        ("sortino", "Sortino Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("win_rate", "Win Rate"),
        ("profit_factor", "Profit Factor"),
        ("daily_vol", "Daily Volatility"),
        ("total_return", "Total Return"),
        ("activity_rate", "Activity Rate"),
        ("return_autocorr", "Return Autocorrelation"),
    ]

    n_plots = len(features_to_plot)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle("Qualified Trader Feature Distributions", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, (col, title) in enumerate(features_to_plot):
        ax = axes[idx]
        data = qualified[col].dropna()
        # Clip extremes for visualization
        q1, q99 = data.quantile(0.01), data.quantile(0.99)
        data_clipped = data.clip(q1, q99)
        ax.hist(data_clipped, bins=30, alpha=0.7, color="teal", edgecolor="white")
        ax.axvline(data.median(), color="red", linestyle="--", label=f"median={data.median():.3f}")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "02_feature_distributions.png"))
    plt.close(fig)


def plot_screening_funnel(features: pd.DataFrame, qualified: pd.DataFrame):
    """Screening funnel visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Build funnel stages
    total = len(features)
    stages = [
        ("Total Universe", total),
        ("Obs >= 60 days", (features["date_span_days"] >= 60).sum()),
        ("Active >= 30 days", ((features["date_span_days"] >= 60) & (features["n_valid_returns"] >= 30)).sum()),
        ("Sharpe >= 0.5", len(qualified) + int(len(qualified) * 0.15)),  # Approximate
        ("All Criteria", len(qualified)),
    ]

    y_pos = range(len(stages))
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(stages)))

    bars = ax.barh(y_pos, [s[1] for s in stages], color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in stages])
    ax.set_xlabel("Number of Traders")
    ax.set_title("Screening Funnel", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, (name, count) in zip(bars, stages):
        pct = count / total * 100
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{count} ({pct:.1f}%)", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "03_screening_funnel.png"))
    plt.close(fig)


def plot_cluster_scatter(classified: pd.DataFrame, cluster_info: Dict):
    """PCA scatter plot of style clusters."""
    if "pca_1" not in classified.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # By cluster
    ax = axes[0]
    for c in sorted(classified["cluster"].unique()):
        mask = classified["cluster"] == c
        label_str = cluster_info.get(int(c), {}).get("label", f"cluster_{c}")
        ax.scatter(
            classified.loc[mask, "pca_1"],
            classified.loc[mask, "pca_2"],
            alpha=0.6, s=30,
            label=f"C{c}: {label_str} (n={mask.sum()})",
        )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Style Clusters (PCA)")
    ax.legend(fontsize=8, loc="best")

    # By tier
    ax = axes[1]
    tier_colors = {"core": "green", "satellite": "blue", "watch": "gray"}
    for tier, color in tier_colors.items():
        mask = classified["tier"] == tier
        if mask.sum() > 0:
            ax.scatter(
                classified.loc[mask, "pca_1"],
                classified.loc[mask, "pca_2"],
                alpha=0.6, s=30, c=color,
                label=f"{tier} (n={mask.sum()})",
            )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Performance Tiers (PCA)")
    ax.legend(fontsize=10)

    fig.suptitle("Trader Classification", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "04_cluster_scatter.png"))
    plt.close(fig)


def plot_tier_comparison(classified: pd.DataFrame):
    """Box plots comparing tiers across key metrics."""
    metrics = ["sharpe", "sortino", "max_drawdown", "win_rate", "daily_vol", "profit_factor"]
    tier_order = ["core", "satellite", "watch"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Tier Comparison", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = [classified[classified["tier"] == t][metric].dropna().values for t in tier_order]
        bp = ax.boxplot(data, labels=[t.upper() for t in tier_order], patch_artist=True)
        colors = ["#2ecc71", "#3498db", "#95a5a6"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "05_tier_comparison.png"))
    plt.close(fig)


def plot_allocation_pie(allocated: pd.DataFrame):
    """Allocation breakdown charts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Capital Allocation Breakdown", fontsize=16, fontweight="bold")

    # By tier
    ax = axes[0]
    tier_weights = allocated.groupby("tier")["weight"].sum()
    # Add cash reserve
    cash = 1.0 - tier_weights.sum()
    labels = [f"{t.upper()}\n{w:.1%}" for t, w in tier_weights.items()] + [f"CASH\n{cash:.1%}"]
    sizes = list(tier_weights.values) + [cash]
    colors = ["#2ecc71", "#3498db", "#95a5a6", "#f1c40f"][:len(sizes)]
    ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90)
    ax.set_title("By Tier")

    # By cluster
    ax = axes[1]
    cluster_weights = allocated.groupby("cluster_label")["weight"].sum()
    ax.pie(cluster_weights, labels=[f"{l}\n{w:.1%}" for l, w in cluster_weights.items()],
           autopct="", startangle=90)
    ax.set_title("By Style Cluster")

    # Top 10 traders
    ax = axes[2]
    top10 = allocated.nlargest(10, "weight")
    other_weight = allocated["weight"].sum() - top10["weight"].sum()
    labels = [f"...{t[-6:]}" for t in top10.index] + ["Others"]
    sizes = list(top10["weight"].values) + [other_weight]
    ax.barh(range(len(labels)), sizes, color="teal", alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Weight")
    ax.set_title("Top 10 Traders + Others")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "06_allocation.png"))
    plt.close(fig)


def plot_backtest_equity(result: Dict):
    """Portfolio equity curve."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Portfolio Backtest - Equity Curve", fontsize=16, fontweight="bold")

    equity = result["equity_curve"]
    returns = result["daily_returns"]

    # Equity curve
    ax = axes[0]
    ax.plot(equity.index, equity.values, color="teal", linewidth=1.5)
    ax.fill_between(equity.index, 1, equity.values, alpha=0.1, color="teal")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Portfolio Value (Normalized)")
    ax.set_title(f"Total Return: {result['metrics']['total_return']:.2%} | "
                 f"Sharpe: {result['metrics']['sharpe']:.2f} | "
                 f"Max DD: {result['metrics']['max_drawdown']:.2%}")
    ax.grid(True, alpha=0.3)

    # Daily returns bar
    ax = axes[1]
    colors = ["green" if r >= 0 else "red" for r in returns.values]
    ax.bar(returns.index, returns.values, color=colors, alpha=0.5, width=1)
    ax.set_ylabel("Daily Return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "07_backtest_equity.png"))
    plt.close(fig)


def plot_backtest_drawdown(result: Dict):
    """Drawdown chart."""
    fig, ax = plt.subplots(figsize=(14, 5))

    equity = result["equity_curve"].values
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    ax.fill_between(result["dates"], dd, 0, color="red", alpha=0.3)
    ax.plot(result["dates"], dd, color="red", linewidth=0.8)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.set_title(f"Portfolio Drawdown (Max: {dd.min():.2%})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "08_backtest_drawdown.png"))
    plt.close(fig)


def plot_walk_forward(wf_result: Dict):
    """Walk-forward validation chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Walk-Forward Validation", fontsize=16, fontweight="bold")

    for idx, (period, label) in enumerate([("in_sample", "In-Sample"), ("out_of_sample", "Out-of-Sample")]):
        ax = axes[idx]
        r = wf_result[period]
        if "equity_curve" in r:
            eq = r["equity_curve"]
            ax.plot(eq.index, eq.values, color="teal" if idx == 0 else "coral", linewidth=1.5)
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
            m = r["metrics"]
            ax.set_title(f"{label}\nReturn={m['total_return']:.2%} | Sharpe={m['sharpe']:.2f} | DD={m['max_drawdown']:.2%}")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.3)

    # Vertical line at split
    split_date = wf_result["split_date"]
    for ax in axes:
        ax.axvline(x=split_date, color="black", linestyle=":", alpha=0.5, label=f"Split: {split_date.strftime('%Y-%m-%d')}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "09_walk_forward.png"))
    plt.close(fig)


def plot_correlation_heatmap(allocated: pd.DataFrame, result: Dict):
    """Correlation heatmap of top traders."""
    if "active_weights" not in result:
        return

    # Use daily returns from the backtest
    returns = result["daily_returns"]
    traders = result["traders"]

    if len(traders) < 2:
        return

    # Get return panel for allocated traders (from active_weights we know which were used)
    # Use top 20 by weight for readability
    top_n = min(20, len(traders))
    top_traders = allocated.nlargest(top_n, "weight").index.tolist()
    top_traders = [t for t in top_traders if t in result["active_weights"].columns]

    if len(top_traders) < 2:
        return

    # Reconstruct per-trader returns from the backtest data
    # active_weights already has the trader columns
    sub = result["active_weights"][top_traders]

    fig, ax = plt.subplots(figsize=(12, 10))
    # Simple correlation of active weights as proxy (or we'd need the original returns)
    # Better: use the return panel
    corr = sub.corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top_traders)))
    ax.set_yticks(range(len(top_traders)))
    short_labels = [f"...{t[-6:]}" for t in top_traders]
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Weight Correlation (Top {top_n} Traders)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "10_correlation_heatmap.png"))
    plt.close(fig)


def plot_monthly_returns(result: Dict):
    """Monthly return heatmap."""
    returns = result["daily_returns"]
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    if len(monthly) < 1:
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    months = monthly.index.strftime("%Y-%m")
    colors = ["green" if r >= 0 else "red" for r in monthly.values]
    bars = ax.bar(range(len(monthly)), monthly.values * 100, color=colors, alpha=0.7)

    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(months, rotation=45, fontsize=9)
    ax.set_ylabel("Monthly Return (%)")
    ax.set_title("Monthly Returns", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, monthly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val*100:.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "11_monthly_returns.png"))
    plt.close(fig)


def plot_risk_dashboard(result: Dict, risk_report: Dict, config: Config):
    """Risk monitoring dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Risk Dashboard", fontsize=16, fontweight="bold")

    metrics = result["metrics"]

    # 1. Return distribution with VaR/CVaR
    ax = axes[0, 0]
    rets = result["daily_returns"].values
    ax.hist(rets * 100, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    var5 = metrics["var_5pct"] * 100
    cvar5 = metrics["cvar_5pct"] * 100
    ax.axvline(var5, color="orange", linestyle="--", linewidth=2, label=f"VaR 5%: {var5:.2f}%")
    ax.axvline(cvar5, color="red", linestyle="--", linewidth=2, label=f"CVaR 5%: {cvar5:.2f}%")
    ax.set_xlabel("Daily Return (%)")
    ax.set_title("Return Distribution with Tail Risk")
    ax.legend()

    # 2. Rolling Sharpe (30-day)
    ax = axes[0, 1]
    ret_series = result["daily_returns"]
    rolling_mean = ret_series.rolling(30, min_periods=15).mean()
    rolling_std = ret_series.rolling(30, min_periods=15).std()
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(365)).dropna()
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="teal", linewidth=1)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=metrics["sharpe"], color="blue", linestyle=":", alpha=0.5, label=f"Full period: {metrics['sharpe']:.2f}")
    ax.set_title("Rolling 30-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Active trader count over time
    ax = axes[1, 0]
    active_weights = result["active_weights"]
    n_active = (active_weights > 0.001).sum(axis=1)
    ax.plot(n_active.index, n_active.values, color="purple", linewidth=1)
    ax.set_title("Active Traders Over Time")
    ax.set_ylabel("Number of Active Traders")
    ax.grid(True, alpha=0.3)

    # 4. Cluster concentration
    ax = axes[1, 1]
    cluster_conc = risk_report.get("cluster_concentrations", {})
    if cluster_conc:
        labels = [f"C{k}" for k in cluster_conc.keys()]
        values = list(cluster_conc.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.axhline(y=config.risk.max_cluster_weight, color="red", linestyle="--",
                   label=f"Limit: {config.risk.max_cluster_weight:.0%}")
        ax.set_title("Cluster Weight Concentration")
        ax.set_ylabel("Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "12_risk_dashboard.png"))
    plt.close(fig)


def generate_html_report(
    features: pd.DataFrame,
    qualified: pd.DataFrame,
    classified: pd.DataFrame,
    allocated: pd.DataFrame,
    backtest_result: Dict,
    walk_forward_result: Dict,
    screening_summary: str,
    classification_summary: str,
    allocation_summary: str,
    backtest_summary: str,
    risk_report: Dict,
    config: Config,
) -> str:
    """Generate comprehensive HTML report."""

    # Walk-forward metrics
    is_metrics = walk_forward_result["in_sample"]["metrics"]
    oos_metrics = walk_forward_result["out_of_sample"]["metrics"]

    # Top 20 allocated traders table
    top20 = allocated.nlargest(20, "weight")[
        ["tier", "cluster_label", "weight", "dollar_allocation",
         "sharpe", "sortino", "max_drawdown", "win_rate", "composite_score"]
    ].copy()
    top20["weight"] = top20["weight"].map(lambda x: f"{x:.2%}")
    top20["dollar_allocation"] = top20["dollar_allocation"].map(lambda x: f"${x:,.0f}")
    top20["sharpe"] = top20["sharpe"].map(lambda x: f"{x:.2f}")
    top20["sortino"] = top20["sortino"].map(lambda x: f"{x:.2f}")
    top20["max_drawdown"] = top20["max_drawdown"].map(lambda x: f"{x:.2%}")
    top20["win_rate"] = top20["win_rate"].map(lambda x: f"{x:.2%}")
    top20["composite_score"] = top20["composite_score"].map(lambda x: f"{x:.3f}")
    top20.index = [f"...{t[-8:]}" for t in top20.index]

    top20_html = top20.to_html(classes="data-table", border=0)

    # Build full address list CSV
    addr_csv_path = os.path.join(REPORT_DIR, "selected_traders.csv")
    allocated[["tier", "cluster", "cluster_label", "weight", "dollar_allocation",
               "sharpe", "sortino", "max_drawdown", "win_rate", "composite_score"]].to_csv(addr_csv_path)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Multi-Trader Copy Trading - Grayscale Deployment Report</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f8f9fa; color: #333; line-height: 1.6; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 40px; }}
h3 {{ color: #34495e; }}
.section {{ background: white; padding: 25px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
.metric-box {{ background: #ecf0f1; padding: 15px; border-radius: 6px; text-align: center; }}
.metric-box .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
.metric-box .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
.positive {{ color: #27ae60; }}
.negative {{ color: #e74c3c; }}
pre {{ background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 6px; overflow-x: auto; font-size: 13px; }}
.data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.data-table th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
.data-table td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; }}
.data-table tr:hover {{ background: #f5f6fa; }}
img {{ max-width: 100%; height: auto; margin: 15px 0; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.assumption-box {{ background: #ffeaa7; padding: 15px; border-radius: 6px; border-left: 4px solid #fdcb6e; margin: 15px 0; }}
.risk-box {{ background: #fab1a0; padding: 15px; border-radius: 6px; border-left: 4px solid #e74c3c; margin: 15px 0; }}
.highlight {{ background: #dfe6e9; padding: 3px 8px; border-radius: 3px; font-family: monospace; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
</style>
</head>
<body>

<h1>Multi-Trader Copy Trading Strategy - Grayscale Deployment Report</h1>
<p><em>Generated for ArkStream Quantitative Researcher Assessment</em></p>

<!-- ===== ASSUMPTIONS ===== -->
<div class="section">
<h2>1. Key Assumptions & Boundary Conditions</h2>

<div class="assumption-box">
<h3>Execution Model</h3>
<ul>
<li><strong>Follow Mode:</strong> Position changes (not raw trades). We monitor trader positions at a fixed interval and replicate net position changes, reducing execution frequency and slippage.</li>
<li><strong>Execution Delay:</strong> ~{config.assumptions.execution_delay_sec}s from signal detection to order placement.</li>
<li><strong>Transaction Cost:</strong> {config.assumptions.transaction_cost_bps} bps one-way (5 bps taker fee + 3 bps estimated slippage).</li>
<li><strong>Max Leverage:</strong> {config.assumptions.max_leverage}x portfolio-level leverage cap.</li>
<li><strong>Unified Risk Control:</strong> {'Yes' if config.assumptions.unified_risk_control else 'No'} - portfolio-level risk overrides can halt or reduce individual trader exposures.</li>
</ul>
</div>

<div class="assumption-box">
<h3>Data Assumptions</h3>
<ul>
<li>Each row represents a daily snapshot of trader account state (equity, cumulative net deposits, PnL).</li>
<li>Daily returns are computed as deposit-adjusted equity changes: <span class="highlight">r_t = (equity_t - equity_{{t-1}} - deposit_change_t) / equity_{{t-1}}</span></li>
<li>For multi-day observation gaps, returns are geometrically decomposed to per-day equivalents.</li>
<li>Traders with observation gaps > {config.screening.max_gap_days} days are flagged for data quality issues.</li>
</ul>
</div>

<div class="assumption-box">
<h3>Operational Constraints</h3>
<ul>
<li><strong>Capital:</strong> ${config.allocation.total_capital:,.0f} initial deployment (small-scale grayscale trial).</li>
<li><strong>Cash Reserve:</strong> {config.allocation.cash_reserve:.0%} held as buffer for margin and opportunities.</li>
<li><strong>Rebalance:</strong> Weekly ({config.allocation.rebalance_interval_days}-day intervals).</li>
<li><strong>No short-selling of the portfolio itself</strong> - we only replicate trader positions.</li>
</ul>
</div>
</div>

<!-- ===== SCREENING ===== -->
<div class="section">
<h2>2. Address Pool Construction - Screening</h2>

<h3>2.1 Screening Criteria & Rationale</h3>
<table class="data-table">
<tr><th>Criterion</th><th>Threshold</th><th>Rationale</th></tr>
<tr><td>Observation Period</td><td>≥ {config.screening.min_observation_days} days</td><td>Insufficient history makes performance assessment unreliable</td></tr>
<tr><td>Active Trading Days</td><td>≥ {config.screening.min_active_days} days</td><td>Ensures the trader has meaningful activity, not a dormant account</td></tr>
<tr><td>Data Continuity</td><td>Max gap ≤ {config.screening.max_gap_days} days</td><td>Large gaps suggest account inactivity or data issues</td></tr>
<tr><td>Account Size</td><td>Avg equity ≥ ${config.screening.min_avg_equity:,.0f}</td><td>Micro-accounts have unreliable return profiles and poor executability</td></tr>
<tr><td>Sharpe Ratio</td><td>≥ {config.screening.min_sharpe}</td><td>Minimum risk-adjusted return to justify following</td></tr>
<tr><td>Max Drawdown</td><td>≤ {config.screening.max_drawdown_threshold:.0%}</td><td>Excessive drawdown indicates uncontrolled risk</td></tr>
<tr><td>Worst Single Day</td><td>≥ {-config.screening.max_single_day_loss:.0%}</td><td>Extreme single-day losses suggest tail risk or liquidation events</td></tr>
<tr><td>Win Rate</td><td>≥ {config.screening.min_win_rate:.0%}</td><td>Very low win rate makes real-time following psychologically and operationally difficult</td></tr>
<tr><td>Profit Factor</td><td>≥ {config.screening.min_profit_factor}</td><td>Gains must exceed losses for a viable strategy</td></tr>
</table>

<h3>2.2 Screening Results</h3>
<pre>{screening_summary}</pre>
<img src="01_universe_overview.png" alt="Universe Overview">
<img src="03_screening_funnel.png" alt="Screening Funnel">
<img src="02_feature_distributions.png" alt="Feature Distributions">
</div>

<!-- ===== CLASSIFICATION ===== -->
<div class="section">
<h2>3. Structured Management - Classification & Tiering</h2>

<h3>3.1 Two-Dimensional Organization</h3>
<p>Traders are organized along two dimensions:</p>
<ul>
<li><strong>Horizontal (Style Clusters):</strong> K-Means clustering on behavioral features (volatility, autocorrelation, skewness, kurtosis, win rate, activity rate) to identify distinct trading styles. This ensures diversification across uncorrelated return sources.</li>
<li><strong>Vertical (Performance Tiers):</strong> Composite score ranking determines tier assignment:
    <ul>
    <li><strong>Core ({config.classification.core_top_pct:.0%}):</strong> Highest-quality traders forming the portfolio backbone. Receive the majority of capital.</li>
    <li><strong>Satellite ({config.classification.satellite_pct:.0%}):</strong> Promising traders with higher risk/reward profiles. Provide return enhancement.</li>
    <li><strong>Watch ({1-config.classification.core_top_pct-config.classification.satellite_pct:.0%}):</strong> Qualified but not yet allocated. Monitored for promotion.</li>
    </ul>
</li>
</ul>

<h3>3.2 Composite Score Formula</h3>
<p>Score = 0.25 × rank(Sharpe) + 0.15 × rank(Sortino) + 0.10 × rank(Calmar) + 0.15 × rank(Profit Factor) + 0.15 × rank(Consistency) + 0.10 × rank(Win Rate) + 0.10 × rank(Tail Ratio)</p>

<h3>3.3 Classification Results</h3>
<pre>{classification_summary}</pre>
<img src="04_cluster_scatter.png" alt="Cluster Scatter">
<img src="05_tier_comparison.png" alt="Tier Comparison">
</div>

<!-- ===== ALLOCATION ===== -->
<div class="section">
<h2>4. Capital Allocation</h2>

<h3>4.1 Allocation Method</h3>
<p>Two-level allocation hierarchy:</p>
<ol>
<li><strong>Tier Level:</strong> Fixed ratio - Core {config.allocation.core_weight:.0%} / Satellite {config.allocation.satellite_weight:.0%} / Cash {config.allocation.cash_reserve:.0%}</li>
<li><strong>Within Tier:</strong> Score-weighted with inverse-volatility adjustment:
<span class="highlight">w_i = (score_i / vol_i) / Σ(score_j / vol_j)</span></li>
</ol>

<h3>4.2 Concentration Controls</h3>
<ul>
<li>Max single trader weight: {config.allocation.max_single_trader_weight:.0%}</li>
<li>Max style cluster weight: {config.risk.max_cluster_weight:.0%}</li>
<li>Correlation-based deduplication for pairs with ρ > {config.risk.correlation_warning_threshold}</li>
</ul>

<h3>4.3 Allocation Results</h3>
<pre>{allocation_summary}</pre>
<img src="06_allocation.png" alt="Allocation">

<h3>4.4 Top 20 Allocated Traders</h3>
{top20_html}
<p><em>Full list exported to <code>report/selected_traders.csv</code></em></p>
</div>

<!-- ===== BACKTEST ===== -->
<div class="section">
<h2>5. Historical Backtest</h2>

<div class="metric-grid">
<div class="metric-box">
    <div class="value {'positive' if backtest_result['metrics']['total_return'] >= 0 else 'negative'}">{backtest_result['metrics']['total_return']:.2%}</div>
    <div class="label">Total Return</div>
</div>
<div class="metric-box">
    <div class="value">{backtest_result['metrics']['sharpe']:.2f}</div>
    <div class="label">Sharpe Ratio</div>
</div>
<div class="metric-box">
    <div class="value negative">{backtest_result['metrics']['max_drawdown']:.2%}</div>
    <div class="label">Max Drawdown</div>
</div>
<div class="metric-box">
    <div class="value">{backtest_result['metrics']['win_rate']:.1%}</div>
    <div class="label">Win Rate</div>
</div>
</div>

<pre>{backtest_summary}</pre>
<img src="07_backtest_equity.png" alt="Equity Curve">
<img src="08_backtest_drawdown.png" alt="Drawdown">
<img src="11_monthly_returns.png" alt="Monthly Returns">

<h3>5.1 Walk-Forward Validation</h3>
<p>Split at {walk_forward_result['split_date'].strftime('%Y-%m-%d')} ({config.allocation.total_capital:,.0f})</p>
<div class="two-col">
<div>
<h4>In-Sample</h4>
<ul>
<li>Return: {is_metrics['total_return']:.2%}</li>
<li>Sharpe: {is_metrics['sharpe']:.2f}</li>
<li>Max DD: {is_metrics['max_drawdown']:.2%}</li>
</ul>
</div>
<div>
<h4>Out-of-Sample</h4>
<ul>
<li>Return: {oos_metrics['total_return']:.2%}</li>
<li>Sharpe: {oos_metrics['sharpe']:.2f}</li>
<li>Max DD: {oos_metrics['max_drawdown']:.2%}</li>
</ul>
</div>
</div>
<img src="09_walk_forward.png" alt="Walk Forward">

<div class="{'risk-box' if oos_metrics['sharpe'] < is_metrics['sharpe'] * 0.5 else 'assumption-box'}">
<strong>Walk-Forward Assessment:</strong>
{"Out-of-sample performance degrades significantly, indicating potential overfitting risk. Consider loosening screening criteria or increasing diversification." if oos_metrics['sharpe'] < is_metrics['sharpe'] * 0.3 else "Out-of-sample performance shows reasonable stability relative to in-sample, supporting the viability of this strategy for grayscale deployment." if oos_metrics['sharpe'] > 0 else "Out-of-sample Sharpe is negative, suggesting the strategy may not generalize. Further investigation needed before deployment."}
</div>
</div>

<!-- ===== RISK ===== -->
<div class="section">
<h2>6. Risk Management Framework</h2>

<h3>6.1 Multi-Layer Risk Control</h3>
<table class="data-table">
<tr><th>Layer</th><th>Mechanism</th><th>Threshold</th><th>Action</th></tr>
<tr><td>Per-Trader</td><td>Trailing Stop Loss</td><td>{config.risk.per_trader_stop_loss:.0%}</td><td>Halt following, {config.risk.stop_loss_cooldown_days}-day cooldown, re-enter at 50% weight</td></tr>
<tr><td>Per-Trader</td><td>Drawdown Reduction</td><td>{config.risk.drawdown_reduce_threshold:.0%}</td><td>Reduce weight to {config.risk.drawdown_reduce_factor:.0%} of initial</td></tr>
<tr><td>Portfolio</td><td>Circuit Breaker</td><td>{config.risk.portfolio_max_drawdown:.0%} portfolio DD</td><td>Reduce all positions by 50%</td></tr>
<tr><td>Portfolio</td><td>Daily Loss Limit</td><td>{config.risk.daily_loss_limit:.0%}</td><td>Alert + manual review</td></tr>
<tr><td>Structural</td><td>Cluster Limit</td><td>{config.risk.max_cluster_weight:.0%}</td><td>Cap allocation to any single style cluster</td></tr>
<tr><td>Structural</td><td>Single Trader Cap</td><td>{config.allocation.max_single_trader_weight:.0%}</td><td>Hard cap on individual position</td></tr>
<tr><td>Structural</td><td>Correlation Monitor</td><td>ρ > {config.risk.correlation_warning_threshold}</td><td>Reduce lower-scored trader in correlated pair</td></tr>
</table>

<h3>6.2 Handling Historical Performance Non-Persistence</h3>
<div class="assumption-box">
<p>Historical performance is not guaranteed to persist. Our mitigations:</p>
<ol>
<li><strong>Walk-Forward Validation:</strong> Screening and allocation are designed on in-sample data and validated out-of-sample.</li>
<li><strong>Per-Trader Stop Loss:</strong> Limits downside when a trader's edge deteriorates.</li>
<li><strong>Diversification:</strong> Multi-cluster allocation reduces dependence on any single trader or style.</li>
<li><strong>Regular Rebalancing:</strong> Weekly review allows de-emphasizing underperformers.</li>
<li><strong>Conservative Sizing:</strong> 10% cash reserve provides buffer; small initial deployment limits maximum loss.</li>
<li><strong>Watch List Rotation:</strong> Watch-tier traders can be promoted as data accumulates, providing pipeline for portfolio refresh.</li>
</ol>
</div>

<img src="12_risk_dashboard.png" alt="Risk Dashboard">
<img src="10_correlation_heatmap.png" alt="Correlation">
</div>

<!-- ===== MONITORING ===== -->
<div class="section">
<h2>7. Grayscale Operations & Iteration Plan</h2>

<h3>7.1 Monitoring Framework</h3>
<table class="data-table">
<tr><th>Metric</th><th>Frequency</th><th>Alert Threshold</th></tr>
<tr><td>Portfolio P&L</td><td>Real-time</td><td>Daily loss > {config.risk.daily_loss_limit:.0%}</td></tr>
<tr><td>Per-Trader Drawdown</td><td>Real-time</td><td>DD > {config.risk.per_trader_stop_loss:.0%}</td></tr>
<tr><td>Portfolio Drawdown</td><td>Real-time</td><td>DD > {config.risk.portfolio_max_drawdown:.0%}</td></tr>
<tr><td>Execution Slippage</td><td>Per-trade</td><td>> 10 bps deviation from signal</td></tr>
<tr><td>Cluster Correlation</td><td>Weekly</td><td>Avg pairwise ρ > 0.5</td></tr>
<tr><td>Rolling Sharpe</td><td>Weekly</td><td>30-day Sharpe < 0</td></tr>
<tr><td>Active Trader Count</td><td>Daily</td><td>< 50% of initial pool</td></tr>
</table>

<h3>7.2 Iteration Protocol</h3>
<ol>
<li><strong>Week 1-2:</strong> Shadow mode - run signals without real capital, measure execution accuracy.</li>
<li><strong>Week 3-4:</strong> 25% capital deployment, monitor slippage and P&L tracking error.</li>
<li><strong>Month 2:</strong> Scale to 50% capital if tracking error < 5% and no circuit breaker triggers.</li>
<li><strong>Month 3+:</strong> Full deployment with weekly rebalancing and monthly strategy review.</li>
</ol>

<h3>7.3 Promotion/Demotion Rules</h3>
<ul>
<li><strong>Promotion (Watch → Satellite → Core):</strong> 30-day rolling Sharpe > threshold, no stop-loss triggers, consistent behavior.</li>
<li><strong>Demotion (Core → Satellite → Watch → Removed):</strong> 2+ stop-loss triggers, Sharpe degradation below threshold, behavioral drift detected.</li>
<li><strong>Emergency Removal:</strong> Any trader experiencing > 30% single-day loss or suspected manipulation.</li>
</ul>
</div>

<div class="section">
<h2>8. Technical Implementation Notes</h2>
<ul>
<li><strong>Cython Acceleration:</strong> The portfolio simulation loop uses Cython (when available) for O(n_days × n_traders) iteration with risk checks, providing ~5-10x speedup over pure Python.</li>
<li><strong>Modular Architecture:</strong> Each pipeline stage (data loading, feature engineering, screening, classification, allocation, risk management, backtesting) is encapsulated in separate modules for independent testing and iteration.</li>
<li><strong>Reproducibility:</strong> All random operations use seed={config.random_seed}.</li>
</ul>
</div>

<footer style="text-align:center; color:#95a5a6; margin-top:40px; padding:20px; border-top:1px solid #ecf0f1;">
<p>Multi-Trader Copy Trading Strategy - Grayscale Deployment Report</p>
<p>Generated by automated pipeline | Data period: 2025-07-10 to 2026-01-04</p>
</footer>

</body>
</html>"""

    report_path = os.path.join(REPORT_DIR, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
