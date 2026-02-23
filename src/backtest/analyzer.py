"""
BacktestAnalyzer — deep performance analysis for backtest results.

Produces breakdowns by:
  - Trade mode (SCALP vs TREND_RIDE)
  - Exit reason distribution
  - Best/worst UTC hours
  - Cost analysis (fees + funding as % of gross)
  - Win/loss streak statistics
  - Filter effectiveness (if filter stats are available)

Usage:
  from src.backtest.analyzer import BacktestAnalyzer
  analyzer = BacktestAnalyzer()
  report = analyzer.analyze(trades, equity_curve, initial_capital=10_000)
  analyzer.print_report(report)
  md = analyzer.to_markdown(report)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AnalysisReport:
    # Core metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    mdd: float = 0.0
    cagr: float = 0.0
    total_pnl_usdt: float = 0.0
    final_equity: float = 0.0

    # Cost analysis
    total_fees_usdt: float = 0.0
    total_funding_usdt: float = 0.0
    fee_pct_of_gross: float = 0.0
    funding_pct_of_gross: float = 0.0

    # Streak analysis
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_usdt: float = 0.0
    avg_loss_usdt: float = 0.0

    # Breakdowns
    mode_breakdown: Dict = field(default_factory=dict)
    exit_reason_dist: Dict = field(default_factory=dict)
    hour_breakdown: Dict = field(default_factory=dict)

    # Filter stats
    trades_filtered: int = 0
    trades_passed: int = 0


class BacktestAnalyzer:
    def analyze(
        self,
        trades: list,
        equity_curve: pd.DataFrame,
        initial_capital: float = 10_000.0,
    ) -> AnalysisReport:
        """
        Compute deep analysis from a list of BacktestTrade objects and
        an equity curve DataFrame {ts, equity}.
        """
        report = AnalysisReport()

        if not trades:
            return report

        pnls = [t.pnl_net for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        report.total_trades = len(pnls)
        report.total_pnl_usdt = sum(pnls)
        report.win_rate = len(wins) / len(pnls)
        report.avg_win_usdt = float(np.mean(wins)) if wins else 0.0
        report.avg_loss_usdt = float(abs(np.mean(losses))) if losses else 0.0
        report.profit_factor = (
            float(sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
        )

        # ── Cost analysis ─────────────────────────────────────────────────────
        report.total_fees_usdt = sum(getattr(t, "fees", 0) for t in trades)
        report.total_funding_usdt = sum(getattr(t, "funding_cost", 0) for t in trades)
        gross_pnl = sum(getattr(t, "pnl_gross", t.pnl_net) for t in trades)
        if abs(gross_pnl) > 0:
            report.fee_pct_of_gross = report.total_fees_usdt / abs(gross_pnl)
            report.funding_pct_of_gross = report.total_funding_usdt / abs(gross_pnl)

        # ── Mode breakdown ────────────────────────────────────────────────────
        for mode in ("SCALP", "TREND_RIDE"):
            mt = [t for t in trades if getattr(t, "trade_mode", "SCALP") == mode]
            if mt:
                mt_pnls = [t.pnl_net for t in mt]
                mt_wins = [p for p in mt_pnls if p > 0]
                report.mode_breakdown[mode] = {
                    "count":     len(mt_pnls),
                    "win_rate":  len(mt_wins) / len(mt_pnls),
                    "total_pnl": float(np.sum(mt_pnls)),
                    "avg_pnl":   float(np.mean(mt_pnls)),
                }

        # ── Exit reason distribution ──────────────────────────────────────────
        reason_counts: Dict[str, int] = {}
        for t in trades:
            r = getattr(t, "exit_reason", "UNKNOWN")
            reason_counts[r] = reason_counts.get(r, 0) + 1
        total = sum(reason_counts.values())
        report.exit_reason_dist = {k: v / total for k, v in reason_counts.items()}

        # ── Hour breakdown (UTC entry hour) ───────────────────────────────────
        hour_pnl: Dict[int, List[float]] = {}
        for t in trades:
            entry_ts = getattr(t, "entry_ts", 0)
            if entry_ts and entry_ts > 0:
                h = pd.Timestamp(int(entry_ts), unit="ms", tz="UTC").hour
                hour_pnl.setdefault(h, []).append(t.pnl_net)
        for h, hp in hour_pnl.items():
            hw = [p for p in hp if p > 0]
            report.hour_breakdown[h] = {
                "count":    len(hp),
                "win_rate": len(hw) / len(hp),
                "avg_pnl":  float(np.mean(hp)),
            }

        # ── Streak analysis ───────────────────────────────────────────────────
        cur_win = cur_loss = 0
        for p in pnls:
            if p > 0:
                cur_win += 1
                cur_loss = 0
            else:
                cur_loss += 1
                cur_win = 0
            report.max_win_streak = max(report.max_win_streak, cur_win)
            report.max_loss_streak = max(report.max_loss_streak, cur_loss)

        # ── Equity curve metrics ──────────────────────────────────────────────
        if not equity_curve.empty and len(equity_curve) >= 2:
            eq = equity_curve["equity"].values
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / np.where(peak > 0, peak, 1)
            report.mdd = float(dd.max())
            report.final_equity = float(eq[-1])

            start_ts = float(equity_curve["ts"].iloc[0])
            end_ts = float(equity_curve["ts"].iloc[-1])
            years = (end_ts - start_ts) / (1000 * 365.25 * 86400)
            if years > 0.1 and initial_capital > 0:
                report.cagr = (report.final_equity / initial_capital) ** (1 / years) - 1

            # Sharpe (trade-level, annualised)
            if len(pnls) >= 2:
                pnl_arr = np.array(pnls)
                std = pnl_arr.std()
                if std > 0 and years > 0:
                    trades_per_year = len(pnls) / years
                    report.sharpe = (pnl_arr.mean() / std) * np.sqrt(trades_per_year)

        return report

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, report: AnalysisReport) -> None:
        W = 70
        print("\n" + "=" * W)
        print("  BACKTEST DEEP ANALYSIS")
        print("=" * W)
        print(f"  Total Trades:      {report.total_trades}")
        print(f"  Win Rate:          {report.win_rate:.2%}")
        print(f"  Profit Factor:     {report.profit_factor:.2f}")
        print(f"  Sharpe Ratio:      {report.sharpe:.2f}")
        print(f"  Max Drawdown:      {report.mdd:.2%}")
        print(f"  CAGR:              {report.cagr:.2%}")
        print(f"  Total PnL:         {report.total_pnl_usdt:+.2f} USDT")
        print(f"  Final Equity:      {report.final_equity:,.2f} USDT")
        print(f"  Avg Win:           +{report.avg_win_usdt:.2f} USDT")
        print(f"  Avg Loss:          -{report.avg_loss_usdt:.2f} USDT")

        if report.mode_breakdown:
            print()
            print("── Trade Mode Breakdown " + "─" * (W - 23))
            for mode, s in report.mode_breakdown.items():
                print(
                    f"  {mode:12s} | trades={s['count']:5d} | "
                    f"win_rate={s['win_rate']:.2%} | "
                    f"total_pnl={s['total_pnl']:+.2f} | "
                    f"avg_pnl={s['avg_pnl']:+.2f}"
                )

        if report.exit_reason_dist:
            print()
            print("── Exit Reason Distribution " + "─" * (W - 27))
            for reason, pct in sorted(report.exit_reason_dist.items(), key=lambda x: -x[1]):
                bar = "█" * int(pct * 30)
                print(f"  {reason:16s} {pct:5.1%}  {bar}")

        print()
        print("── Cost Analysis " + "─" * (W - 17))
        print(f"  Total Fees:        {report.total_fees_usdt:+.2f} USDT  "
              f"({report.fee_pct_of_gross:.2%} of gross)")
        if report.total_funding_usdt != 0:
            print(f"  Total Funding:     {report.total_funding_usdt:+.2f} USDT  "
                  f"({report.funding_pct_of_gross:.2%} of gross)")

        print()
        print("── Streak Analysis " + "─" * (W - 19))
        print(f"  Max Win Streak:    {report.max_win_streak}")
        print(f"  Max Loss Streak:   {report.max_loss_streak}")

        if report.hour_breakdown:
            print()
            print("── Best / Worst Hours (UTC entry) " + "─" * (W - 33))
            sorted_h = sorted(
                report.hour_breakdown.items(),
                key=lambda x: x[1]["avg_pnl"],
                reverse=True,
            )
            print("  Best 3 hours:")
            for h, s in sorted_h[:3]:
                print(f"    {h:02d}:00  trades={s['count']:4d}  "
                      f"win={s['win_rate']:.2%}  avg={s['avg_pnl']:+.2f}")
            print("  Worst 3 hours:")
            for h, s in sorted_h[-3:]:
                print(f"    {h:02d}:00  trades={s['count']:4d}  "
                      f"win={s['win_rate']:.2%}  avg={s['avg_pnl']:+.2f}")

        print("=" * W)

    def to_markdown(self, report: AnalysisReport) -> str:
        lines = [
            "# Backtest Analysis Report",
            "",
            "## Core Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Trades | {report.total_trades} |",
            f"| Win Rate | {report.win_rate:.2%} |",
            f"| Profit Factor | {report.profit_factor:.2f} |",
            f"| Sharpe Ratio | {report.sharpe:.2f} |",
            f"| Max Drawdown | {report.mdd:.2%} |",
            f"| CAGR | {report.cagr:.2%} |",
            f"| Total PnL | {report.total_pnl_usdt:+.2f} USDT |",
            f"| Final Equity | {report.final_equity:,.2f} USDT |",
            f"| Avg Win | +{report.avg_win_usdt:.2f} USDT |",
            f"| Avg Loss | -{report.avg_loss_usdt:.2f} USDT |",
            f"| Total Fees | {report.total_fees_usdt:.2f} USDT ({report.fee_pct_of_gross:.2%} of gross) |",
            "",
        ]

        if report.mode_breakdown:
            lines += [
                "## Trade Mode Breakdown",
                "",
                "| Mode | Trades | Win Rate | Total PnL | Avg PnL |",
                "|------|--------|----------|-----------|---------|",
            ]
            for mode, s in report.mode_breakdown.items():
                lines.append(
                    f"| {mode} | {s['count']} | {s['win_rate']:.2%} | "
                    f"{s['total_pnl']:+.2f} | {s['avg_pnl']:+.2f} |"
                )
            lines.append("")

        if report.exit_reason_dist:
            lines += [
                "## Exit Reason Distribution",
                "",
                "| Reason | Percentage |",
                "|--------|------------|",
            ]
            for reason, pct in sorted(report.exit_reason_dist.items(), key=lambda x: -x[1]):
                lines.append(f"| {reason} | {pct:.1%} |")
            lines.append("")

        lines += [
            "## Streak Analysis",
            "",
            f"- Max Win Streak: **{report.max_win_streak}**",
            f"- Max Loss Streak: **{report.max_loss_streak}**",
        ]

        return "\n".join(lines)
