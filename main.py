"""
Multi-Trader Copy Trading Strategy - Main Pipeline

Orchestrates the full workflow:
1. Data loading & preprocessing
2. Feature engineering
3. Trader screening
4. Classification & tiering
5. Capital allocation
6. Risk management adjustments
7. Portfolio backtesting (with walk-forward validation)
8. Report generation

Usage:
    python main.py
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, REPORT_DIR
from src.data_loader import prepare_data
from src.feature_engine import compute_trader_features
from src.screener import apply_screening, get_screening_summary
from src.classifier import classify_traders, get_classification_summary
from src.allocator import allocate_capital, get_allocation_summary
from src.risk_manager import apply_risk_adjustments
from src.backtester import run_backtest, run_walk_forward, get_backtest_summary, CYTHON_AVAILABLE
from src.report_generator import generate_all_charts, generate_html_report


def main():
    """Run the complete pipeline."""
    start_time = time.time()
    config = Config()

    print("=" * 60)
    print("  Multi-Trader Copy Trading Strategy Pipeline")
    print("=" * 60)
    print(f"  Cython acceleration: {'ENABLED' if CYTHON_AVAILABLE else 'DISABLED (using Python fallback)'}")
    print()

    # --- Step 1: Data Loading ---
    print("[1/8] Loading and preprocessing data...")
    t0 = time.time()
    data = prepare_data(config)
    print(f"  Loaded {len(data['trader_list'])} traders, "
          f"{len(data['calendar'])} calendar days")
    print(f"  Date range: {data['calendar'][0].strftime('%Y-%m-%d')} to "
          f"{data['calendar'][-1].strftime('%Y-%m-%d')}")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    # --- Step 2: Feature Engineering ---
    print("[2/8] Computing trader features...")
    t0 = time.time()
    features = compute_trader_features(data["returns_df"], data["raw_df"], config)
    print(f"  Computed {len(features.columns)} features for {len(features)} traders")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    # --- Step 3: Screening ---
    print("[3/8] Applying screening criteria...")
    t0 = time.time()
    qualified, rejected, rejection_reasons = apply_screening(features, config)
    screening_summary = get_screening_summary(features, qualified, rejected, rejection_reasons)
    print(f"  Qualified: {len(qualified)} / {len(features)} traders")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    if len(qualified) == 0:
        print("ERROR: No traders passed screening. Adjusting criteria...")
        # Relax criteria and retry
        config.screening.min_sharpe = 0.2
        config.screening.max_drawdown_threshold = 0.70
        config.screening.min_win_rate = 0.30
        qualified, rejected, rejection_reasons = apply_screening(features, config)
        screening_summary = get_screening_summary(features, qualified, rejected, rejection_reasons)
        print(f"  After relaxation: {len(qualified)} qualified")
        if len(qualified) == 0:
            print("FATAL: Still no traders qualified. Exiting.")
            return
    print()

    # --- Step 4: Classification ---
    print("[4/8] Classifying traders (clustering + tiering)...")
    t0 = time.time()
    classified, cluster_info = classify_traders(qualified, config)
    classification_summary = get_classification_summary(classified, cluster_info)
    print(f"  Clusters: {len(cluster_info)}")
    for tier in ["core", "satellite", "watch"]:
        n = (classified["tier"] == tier).sum()
        print(f"  {tier.upper()}: {n} traders")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    # --- Step 5: Capital Allocation ---
    print("[5/8] Allocating capital...")
    t0 = time.time()
    allocated, allocation_info = allocate_capital(
        classified, data["return_panel"], config
    )
    allocation_summary = get_allocation_summary(allocated, allocation_info)
    print(f"  Allocated to {len(allocated)} traders")
    print(f"  Total weight deployed: {allocated['weight'].sum():.2%}")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    # --- Step 6: Risk Adjustments ---
    print("[6/8] Applying risk adjustments...")
    t0 = time.time()
    allocated, risk_report = apply_risk_adjustments(
        allocated, data["return_panel"], config
    )
    print(f"  Correlated pairs (ρ > {config.risk.correlation_warning_threshold}): "
          f"{risk_report['n_correlated_pairs']}")
    print(f"  Avg pairwise correlation: {risk_report['avg_correlation']:.3f}")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    # --- Step 7: Backtesting ---
    print("[7/8] Running backtest simulation...")
    t0 = time.time()

    # Full backtest
    backtest_result = run_backtest(allocated, data["return_panel"], config)
    backtest_summary = get_backtest_summary(backtest_result)

    # Walk-forward validation
    wf_result = run_walk_forward(allocated, data["return_panel"], config)

    print(f"  Full period: {backtest_result['metrics']['total_return']:.2%} return, "
          f"Sharpe={backtest_result['metrics']['sharpe']:.2f}")
    print(f"  In-sample Sharpe: {wf_result['in_sample']['metrics']['sharpe']:.2f}")
    print(f"  Out-of-sample Sharpe: {wf_result['out_of_sample']['metrics']['sharpe']:.2f}")
    print(f"  Stop-loss events: {backtest_result['metrics'].get('n_stop_loss_events', 0)}")
    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    # --- Step 8: Report Generation ---
    print("[8/8] Generating report and charts...")
    t0 = time.time()
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Generate charts
    generate_all_charts(
        features=features,
        qualified=qualified,
        classified=classified,
        allocated=allocated,
        backtest_result=backtest_result,
        walk_forward_result=wf_result,
        risk_report=risk_report,
        cluster_info=cluster_info,
        config=config,
    )

    # Generate HTML report
    report_path = generate_html_report(
        features=features,
        qualified=qualified,
        classified=classified,
        allocated=allocated,
        backtest_result=backtest_result,
        walk_forward_result=wf_result,
        screening_summary=screening_summary,
        classification_summary=classification_summary,
        allocation_summary=allocation_summary,
        backtest_summary=backtest_summary,
        risk_report=risk_report,
        config=config,
    )

    # Also save key data files
    features.to_csv(os.path.join(REPORT_DIR, "all_trader_features.csv"))
    classified.to_csv(os.path.join(REPORT_DIR, "classified_traders.csv"))
    allocated[["tier", "cluster", "cluster_label", "weight", "dollar_allocation",
               "sharpe", "sortino", "max_drawdown", "win_rate", "composite_score"]].to_csv(
        os.path.join(REPORT_DIR, "final_allocation.csv")
    )

    print(f"  Elapsed: {time.time() - t0:.2f}s")
    print()

    # --- Summary ---
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"  Pipeline complete in {total_time:.2f}s")
    print(f"  Report: {report_path}")
    print(f"  Charts: {REPORT_DIR}/")
    print("=" * 60)
    print()

    # Print key results
    print(screening_summary)
    print()
    print(classification_summary)
    print()
    print(allocation_summary)
    print()
    print(backtest_summary)


if __name__ == "__main__":
    main()
