# Multi-Trader Copy Trading Strategy - Grayscale Deployment Solution
# 多交易员跟随组合灰度方案

## Overview / 概述

A complete pipeline for screening, classifying, allocating, and risk-managing a multi-trader copy trading portfolio on the Hyperliquid perpetual futures market. Designed for small-scale grayscale (trial) deployment with real capital.

一套完整的多交易员跟单组合方案，涵盖从筛选、分类、资金分配到风控的全流程。基于 Hyperliquid 永续合约市场，适用于小规模灰度试运行。

---

## Project Structure / 项目结构

```
ArkStream_exam/
├── main.py                       # Pipeline orchestrator / 主流程入口
├── README.md                     # This file / 本文件
├── 笔试题数据包.csv               # Raw data: ~2000 traders, daily snapshots / 原始数据
├── 笔试题 - ArkStream量化研究员.pdf  # Task requirements / 题目要求
│
├── src/                          # Source modules / 源代码模块
│   ├── __init__.py
│   ├── config.py                 # All parameters & assumptions / 参数配置与假设
│   ├── data_loader.py            # Data loading & deposit-adjusted return computation
│   │                               数据加载与充提款调整后收益率计算
│   ├── feature_engine.py         # 27 per-trader features (performance/risk/style)
│   │                               每个交易员计算27个特征（表现/风险/风格）
│   ├── screener.py               # 9-criteria multi-filter + composite scoring
│   │                               9项筛选标准 + 综合评分
│   ├── classifier.py             # K-Means style clustering + performance tiering
│   │                               K-Means风格聚类 + 表现分层
│   ├── allocator.py              # Score-weighted capital allocation with constraints
│   │                               基于评分的资金分配（含集中度约束）
│   ├── risk_manager.py           # Multi-layer risk control framework
│   │                               多层风控框架
│   ├── backtester.py             # Portfolio simulation + walk-forward validation
│   │                               组合回测模拟 + 前推验证
│   ├── fast_backtest.pyx         # Cython-accelerated simulation loop
│   │                               Cython加速回测循环
│   ├── setup_cython.py           # Cython build script / Cython编译脚本
│   └── report_generator.py       # 12 charts + HTML report generation
│                                   12张图表 + HTML报告生成
│
├── report/                       # Generated outputs / 生成的输出文件
│   ├── report.html               # Full HTML report / 完整HTML报告
│   ├── 01_universe_overview.png  # Trader universe statistics / 交易员全景统计
│   ├── 02_feature_distributions.png  # Qualified trader features / 合格交易员特征分布
│   ├── 03_screening_funnel.png   # Screening funnel / 筛选漏斗图
│   ├── 04_cluster_scatter.png    # Style cluster PCA scatter / 风格聚类PCA散点图
│   ├── 05_tier_comparison.png    # Tier box plot comparison / 分层箱线图对比
│   ├── 06_allocation.png         # Capital allocation breakdown / 资金分配分解
│   ├── 07_backtest_equity.png    # Portfolio equity curve / 组合净值曲线
│   ├── 08_backtest_drawdown.png  # Drawdown chart / 回撤图
│   ├── 09_walk_forward.png       # Walk-forward validation / 前推验证
│   ├── 10_correlation_heatmap.png # Trader correlation heatmap / 交易员相关性热力图
│   ├── 11_monthly_returns.png    # Monthly return bar chart / 月度收益柱状图
│   ├── 12_risk_dashboard.png     # Risk monitoring dashboard / 风控监控仪表盘
│   ├── final_allocation.csv      # Final trader weights & allocations / 最终权重分配
│   ├── selected_traders.csv      # Selected address pool / 入选地址池
│   ├── classified_traders.csv    # All classified traders / 所有分类交易员
│   └── all_trader_features.csv   # Full 27-feature matrix / 全量特征矩阵
│
└── build/                        # Cython build artifacts / Cython编译产物
```

---

## Quick Start / 快速开始

### Prerequisites / 环境要求

- Python >= 3.10
- Required packages / 依赖包:

```bash
pip install pandas numpy matplotlib scikit-learn cython
```

### Build Cython Module (Optional) / 编译Cython模块（可选）

Cython provides ~5-10x speedup for the backtest simulation loop. If unavailable, the system falls back to pure Python automatically.

Cython可为回测模拟循环提供约5-10倍加速。若不可用，系统会自动回退到纯Python。



### Run the Pipeline / 运行流程

```bash
python main.py
```

The pipeline completes in ~20 seconds and generates all outputs under `report/`.

流程约20秒完成，所有输出生成在 `report/` 目录下。

### View the Report / 查看报告

Open `report/report.html` in any browser.

用浏览器打开 `report/report.html` 查看完整报告。

---

## Pipeline Stages / 流程阶段

### Stage 1: Data Loading / 数据加载
- Load ~2000 trader daily snapshots (2025-07-10 to 2026-01-04, 179 days)
- Compute deposit-adjusted daily returns: `r_t = (equity_t - equity_{t-1} - deposit_change) / equity_{t-1}`
- Multi-day observation gaps are geometrically decomposed to per-day equivalents

- 加载约2000个交易员的每日快照（2025-07-10至2026-01-04，179天）
- 计算充提款调整后的日收益率
- 多日观测间隔进行几何分解为每日等效值

### Stage 2: Feature Engineering / 特征工程
27 features per trader across 3 categories / 每个交易员计算27个特征，分3类：

| Category / 类别 | Features / 特征 |
|---|---|
| **Performance / 表现** | Total return, annualized return, Sharpe, Sortino, Calmar, win rate, profit factor |
| **Risk / 风险** | Max drawdown, volatility, VaR(5%), CVaR(5%), tail ratio, worst day |
| **Style / 风格** | Return autocorrelation, skewness, kurtosis, activity rate, Sharpe stability |

### Stage 3: Screening / 筛选
9 independent filters, each targeting a specific failure mode:

9项独立筛选，每项针对一类特定风险：

| Filter / 筛选项 | Threshold / 阈值 | Rationale / 原因 |
|---|---|---|
| Observation period / 观测周期 | ≥ 60 days | Insufficient history / 样本期不足 |
| Active days / 活跃天数 | ≥ 30 days | Dormant accounts / 休眠账户 |
| Data continuity / 数据连续性 | Max gap ≤ 14 days | Data quality / 数据质量 |
| Account size / 账户规模 | Avg equity ≥ $1,000 | Executability / 可执行性 |
| Sharpe ratio | ≥ 0.5 | Risk-adjusted return / 风险调整收益 |
| Max drawdown / 最大回撤 | ≤ 50% | Uncontrolled risk / 风险失控 |
| Worst day / 最大单日亏损 | ≥ -40% | Tail risk / 尾部风险 |
| Win rate / 胜率 | ≥ 35% | Operational viability / 运营可行性 |
| Profit factor / 盈亏比 | ≥ 1.0 | Net profitability / 净盈利能力 |

**Result / 结果: 214 / 1,948 traders qualified (11.0%)**

### Stage 4: Classification / 分类

Two-dimensional organization / 二维组织：

- **Horizontal / 横向 (Style Clusters / 风格聚类):** K-Means on behavioral features → 4 style groups
  - `pos_skew` (balanced, largest group / 平衡型，最大组)
  - `high_vol_pos_skew` (high volatility / 高波动)
  - `high_wr_erratic_pos_skew` (high win rate / 高胜率)
  - `high_vol_trend_low_wr_stable_pos_skew` (trend-following / 趋势跟踪)
- **Vertical / 纵向 (Performance Tiers / 表现分层):** Composite score ranking
  - **Core (30%):** 65 traders, avg Sharpe 4.10 — portfolio backbone / 组合核心
  - **Satellite (40%):** 85 traders, avg Sharpe 2.20 — return enhancement / 收益增强
  - **Watch (30%):** 64 traders, avg Sharpe 1.22 — monitored, not allocated / 观察，暂不分配

### Stage 5: Capital Allocation / 资金分配

- **Tier allocation / 层级分配:** Core 60% / Satellite 30% / Cash Reserve 10%
- **Within-tier / 层内分配:** `weight_i = (score_i / vol_i) / Σ(score_j / vol_j)`
- **Constraints enforced iteratively / 迭代执行约束:**
  - Single trader cap: 10% / 单个交易员上限10%
  - Cluster concentration cap: 40% / 风格集中度上限40%
  - Correlation-based deduction for highly correlated pairs / 高相关性对减仓

**Result / 结果: 150 traders allocated, Herfindahl index = 0.0094 (well diversified / 充分分散)**

### Stage 6: Risk Management / 风控

| Layer / 层级 | Mechanism / 机制 | Threshold / 阈值 | Action / 动作 |
|---|---|---|---|
| Per-Trader / 单交易员 | Trailing stop-loss / 跟踪止损 | -15% DD | Halt + 14-day cooldown / 暂停+14天冷却 |
| Per-Trader / 单交易员 | Drawdown reduction / 回撤减仓 | -8% DD | Reduce to 50% weight / 降至50%仓位 |
| Portfolio / 组合 | Circuit breaker / 熔断 | -10% DD | All positions halved / 全部仓位减半 |
| Portfolio / 组合 | Daily loss limit / 日亏损限制 | -3% | Alert + manual review / 告警+人工审查 |
| Structural / 结构 | Cluster cap / 风格上限 | 40% | Hard allocation cap / 硬性分配上限 |
| Structural / 结构 | Single trader cap / 单人上限 | 10% | Hard allocation cap / 硬性分配上限 |
| Structural / 结构 | Correlation monitor / 相关性监控 | ρ > 0.70 | Reduce lower-scored trader / 降低低分交易员仓位 |

### Stage 7: Backtest / 回测

| Metric / 指标 | Full Period / 全周期 | In-Sample / 样本内 | Out-of-Sample / 样本外 |
|---|---|---|---|
| Total Return / 总收益 | 162.26% | 79.37% | 26.28% |
| Sharpe Ratio | 5.50 | 5.80 | 10.82 |
| Max Drawdown / 最大回撤 | -0.55% | -0.41% | -0.13% |
| Win Rate / 胜率 | 78.77% | — | — |
| Stop-Loss Events / 止损事件 | 202 | — | — |

### Stage 8: Report / 报告生成
12 publication-quality charts + comprehensive HTML report with embedded analysis.

生成12张图表 + 包含完整分析的HTML报告。

---

## Key Assumptions / 关键假设

| Assumption / 假设 | Value / 值 | Rationale / 原因 |
|---|---|---|
| Follow mode / 跟随方式 | Position changes / 仓位变化 | Reduces frequency vs. raw trade copying / 降低频率 |
| Execution delay / 执行延迟 | ~5 seconds | Signal detection + order placement / 信号检测+下单 |
| Transaction cost / 交易成本 | 8 bps one-way (5 fee + 3 slippage) | Conservative Hyperliquid taker estimate / 保守估计 |
| Max leverage / 最大杠杆 | 3x | Grayscale-appropriate risk level / 灰度阶段风险水平 |
| Initial capital / 初始资金 | $100,000 | Small-scale trial / 小规模试运行 |
| Rebalance / 再平衡 | Weekly (7 days) | Balance freshness vs. cost / 平衡时效性与成本 |

---

## Grayscale Iteration Plan / 灰度迭代计划

| Phase / 阶段 | Duration / 时长 | Action / 动作 |
|---|---|---|
| Shadow / 影子模式 | Week 1-2 | Signal-only, no real capital / 仅信号，无实盘 |
| 25% Deployment / 25%部署 | Week 3-4 | Monitor slippage & tracking error / 监控滑点与跟踪误差 |
| 50% Deployment / 50%部署 | Month 2 | Scale if tracking error < 5% / 跟踪误差<5%则扩量 |
| Full Deployment / 全量部署 | Month 3+ | Weekly rebalance + monthly review / 周度再平衡+月度复盘 |

**Promotion/Demotion Rules / 升降级规则:**
- **Promotion / 升级 (Watch → Satellite → Core):** 30-day rolling Sharpe above threshold, no stop-loss triggers, consistent behavior / 30日滚动Sharpe达标、无止损触发、行为一致
- **Demotion / 降级 (Core → Satellite → Watch → Removed):** 2+ stop-loss triggers, Sharpe degradation, behavioral drift / 2次以上止损、Sharpe衰减、行为漂移
- **Emergency Removal / 紧急移除:** >30% single-day loss or suspected manipulation / 单日亏损>30%或疑似操纵

---

## Performance Optimizations / 性能优化

| Component / 组件 | Optimization / 优化 | Speedup / 加速倍数 |
|---|---|---|
| Feature engine / 特征引擎 | Vectorized numpy, manual autocorr/skew/kurt | 7.6x (25s → 3.3s) |
| Backtest loop / 回测循环 | Cython C extension (`.pyx`) | ~5-10x vs pure Python |
| Total pipeline / 全流程 | | ~20 seconds end-to-end |

If Cython is not available (no C compiler), the system automatically falls back to the pure Python implementation with identical results.

若Cython不可用（无C编译器），系统自动回退到纯Python实现，结果完全一致。

---

## Output Files / 输出文件

| File / 文件 | Description / 描述 |
|---|---|
| `report/report.html` | Full strategy report with all charts / 包含所有图表的完整策略报告 |
| `report/final_allocation.csv` | Final address pool with weights / 最终地址池及权重 |
| `report/selected_traders.csv` | Selected traders list / 入选交易员列表 |
| `report/all_trader_features.csv` | Complete 27-feature matrix for all 1,948 traders / 全部交易员27维特征矩阵 |
| `report/classified_traders.csv` | Cluster + tier labels for qualified traders / 合格交易员的聚类+分层标签 |

---


