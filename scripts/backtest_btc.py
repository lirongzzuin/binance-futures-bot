#!/usr/bin/env python
"""
Standalone BTC backtest runner — no API keys required.

Generates a synthetic year of BTC/USDT 5-min data that models
actual BTC 2024 price behaviour, then runs the full backtest
pipeline (filters → regime → signals → risk → exits → analysis).

Usage:
    python scripts/backtest_btc.py [--days 365] [--capital 10000] [--seed 42]

Exit code 0 = quality gate passed, 1 = failed.
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on path when run from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest.analyzer import BacktestAnalyzer
from src.backtest.engine import BacktestEngine
from src.backtest.synthetic import SyntheticBTCData
from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic BTC backtest (Binance)")
    parser.add_argument("--days",    type=int,   default=365,     help="Days of data to generate")
    parser.add_argument("--capital", type=float, default=10_000., help="Starting capital (USDT)")
    parser.add_argument("--seed",    type=int,   default=42,      help="RNG seed for reproducibility")
    parser.add_argument("--quiet",   action="store_true",         help="Suppress deep analysis output")
    args = parser.parse_args()

    # ── Settings ──────────────────────────────────────────────────────────────
    cfg = Settings(
        TRADING_MODE="dry-run",
        BINANCE_API_KEY="",
        BINANCE_API_SECRET="",
        BACKTEST_INITIAL_CAPITAL=args.capital,
        MAX_LEVERAGE_CAP=10,
        MAX_DRAWDOWN_PCT=0.50,
    )

    # ── Generate synthetic data (3 symbols for trade volume) ─────────────────
    print(f"\n{'=' * 60}")
    print("  SYNTHETIC 5-MIN BACKTEST  (BTC / ETH / SOL) — Binance")
    print(f"{'=' * 60}")

    gen = SyntheticBTCData()
    symbol_seeds = [("BTCUSDT", args.seed), ("ETHUSDT", args.seed + 1), ("SOLUSDT", args.seed + 2)]
    data = {}
    for sym, seed in symbol_seeds:
        df = gen.generate(
            start_price=42_000.0,
            n_days=args.days,
            interval_min=5,
            seed=seed,
        )
        data[sym] = df
        summary = gen.summary(df)
        print(f"  {sym}: {summary['bars']:,} bars | "
              f"${summary['start_price']:,.0f} → ${summary['end_price']:,.0f} "
              f"({summary['start_date'][:10]})")
    print()

    # ── Run backtest ──────────────────────────────────────────────────────────
    print(f"  Running backtest ({len(data)} symbols | capital: ${args.capital:,.0f} | lev cap: {cfg.MAX_LEVERAGE_CAP}×)…")
    engine = BacktestEngine(cfg)
    results = engine.run(data, oos_split=0.80)

    if not results or "all" not in results:
        print("  ERROR: Backtest produced no results.")
        return 1

    # ── Quality gate ──────────────────────────────────────────────────────────
    from src.backtest.report import QualityGate
    qg = QualityGate(cfg)
    passed, gate_results = qg.evaluate(results["all"])

    # ── Summary output ────────────────────────────────────────────────────────
    m = results["all"]
    print(f"\n{'=' * 60}")
    print("  BACKTEST RESULTS — ALL DATA")
    print(f"{'=' * 60}")
    print(f"  Trades:         {m.get('total_trades', 0)}")
    print(f"  Win Rate:       {m.get('win_rate', 0):.2%}")
    print(f"  Profit Factor:  {m.get('profit_factor', 0):.2f}")
    print(f"  CAGR:           {m.get('cagr', 0):.2%}")
    print(f"  Max Drawdown:   {m.get('mdd', 0):.2%}")
    print(f"  Sharpe Ratio:   {m.get('sharpe', 0):.2f}")
    print(f"  Total PnL:      {m.get('total_pnl_usdt', 0):+,.2f} USDT")
    print(f"  Final Equity:   {m.get('final_equity', 0):,.2f} USDT")

    oos = results.get("out_of_sample", {})
    if oos:
        print(f"\n  OUT-OF-SAMPLE (last 20%)")
        print(f"    Win Rate:     {oos.get('win_rate', 0):.2%}")
        print(f"    Profit Factor:{oos.get('profit_factor', 0):.2f}")
        print(f"    Sharpe:       {oos.get('sharpe', 0):.2f}")
        print(f"    MDD:          {oos.get('mdd', 0):.2%}")

    # ── Quality gate result ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  QUALITY GATE: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"{'=' * 60}")
    for k, v in gate_results.items():
        status = "  ✅" if v["pass"] else "  ❌"
        print(f"{status} {k:25s} {v['value']:.3f}  (required {v['required']})")

    # ── Deep analysis (optional) ──────────────────────────────────────────────
    if not args.quiet and results.get("trades"):
        analyzer = BacktestAnalyzer()
        report = analyzer.analyze(
            trades=results["trades"],
            equity_curve=results["equity_curve"],
            initial_capital=args.capital,
        )
        analyzer.print_report(report)

        # Save markdown report
        os.makedirs("reports", exist_ok=True)
        md = analyzer.to_markdown(report)
        report_path = "reports/backtest_btc_synthetic.md"
        with open(report_path, "w") as f:
            f.write(md)
        print(f"\n  Markdown report saved to: {report_path}")

    if passed:
        print("\n  Quality gate PASSED.")
        print("  Set QUALITY_GATE_PASSED=true in .env to enable LIVE trading.")
        return 0
    else:
        print("\n  Quality gate FAILED. Adjust strategy parameters and re-run.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
