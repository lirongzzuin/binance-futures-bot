"""
Backtest Report Generator.
Writes HTML + CSV reports to reports/ directory.
Evaluates quality gate.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


class QualityGate:
    def __init__(self, settings: Settings):
        self._s = settings

    def evaluate(self, metrics: Dict) -> Tuple[bool, Dict]:
        """
        Returns (passed, detail_dict)
        detail_dict: {metric: {"value": float, "required": str, "pass": bool}}
        """
        checks = {
            "sharpe": {
                "value": metrics.get("sharpe", 0),
                "required": f">= {self._s.QUALITY_GATE_SHARPE}",
                "pass": metrics.get("sharpe", 0) >= self._s.QUALITY_GATE_SHARPE,
            },
            "max_drawdown": {
                "value": metrics.get("mdd", 1),
                "required": f"<= {self._s.QUALITY_GATE_MDD:.0%}",
                "pass": metrics.get("mdd", 1) <= self._s.QUALITY_GATE_MDD,
            },
            "win_rate": {
                "value": metrics.get("win_rate", 0),
                "required": f">= {self._s.QUALITY_GATE_WIN_RATE:.0%}",
                "pass": metrics.get("win_rate", 0) >= self._s.QUALITY_GATE_WIN_RATE,
            },
            "profit_factor": {
                "value": metrics.get("profit_factor", 0),
                "required": f">= {self._s.QUALITY_GATE_PROFIT_FACTOR}",
                "pass": metrics.get("profit_factor", 0) >= self._s.QUALITY_GATE_PROFIT_FACTOR,
            },
            "min_trades": {
                "value": metrics.get("total_trades", 0),
                "required": f">= {self._s.QUALITY_GATE_MIN_TRADES}",
                "pass": metrics.get("total_trades", 0) >= self._s.QUALITY_GATE_MIN_TRADES,
            },
        }
        passed = all(c["pass"] for c in checks.values())
        return passed, checks


def generate_report(
    results: Dict,
    symbols: List[str],
    interval: str,
    years: int,
) -> str:
    """Generate HTML and CSV reports. Returns path to HTML report."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_name = f"backtest_{ts}"

    # ── CSV: all trades ───────────────────────────────────────────────────────
    trades = results.get("trades", [])
    if trades:
        trade_rows = []
        for t in trades:
            trade_rows.append({
                "symbol": t.symbol,
                "side": t.side,
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "leverage": t.leverage,
                "pnl_gross": t.pnl_gross,
                "fees": t.fees,
                "pnl_net": t.pnl_net,
                "exit_reason": t.exit_reason,
            })
        trades_df = pd.DataFrame(trade_rows)
        csv_path = REPORTS_DIR / f"{report_name}_trades.csv"
        trades_df.to_csv(csv_path, index=False)

    # ── CSV: equity curve ─────────────────────────────────────────────────────
    ec = results.get("equity_curve")
    if ec is not None and not ec.empty:
        ec.to_csv(REPORTS_DIR / f"{report_name}_equity.csv", index=False)

    # ── JSON: metrics ─────────────────────────────────────────────────────────
    metrics_data = {
        k: v for k, v in results.items()
        if k not in ("trades", "equity_curve")
    }
    json_path = REPORTS_DIR / f"{report_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=2, default=str)

    # ── HTML report ───────────────────────────────────────────────────────────
    html_path = REPORTS_DIR / f"{report_name}.html"
    _write_html(results, html_path, symbols, interval, years)

    log.info("Reports written: %s", REPORTS_DIR / report_name)
    return str(html_path)


def _write_html(
    results: Dict,
    path: Path,
    symbols: List[str],
    interval: str,
    years: int,
) -> None:
    m_all = results.get("all", {})
    m_is = results.get("in_sample", {})
    m_oos = results.get("out_of_sample", {})

    def pct(v): return f"{v:.2%}" if isinstance(v, float) else str(v)
    def num(v, d=2): return f"{v:,.{d}f}" if isinstance(v, (int, float)) else str(v)

    monthly = m_all.get("monthly_returns", {})
    monthly_rows = ""
    for month, ret in sorted(monthly.items()):
        color = "#2ecc71" if ret >= 0 else "#e74c3c"
        monthly_rows += f'<tr><td>{month}</td><td style="color:{color}">{pct(ret)}</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #0f0a07; color: #e0d5c8; margin: 0; padding: 24px; }}
  h1 {{ color: #ff892e; }} h2 {{ color: #ff892e; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0; }}
  .card {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; }}
  .card .label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
  .card .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
  .pass {{ color: #2ecc71; }} .fail {{ color: #e74c3c; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #222; }}
  th {{ background: rgba(255,137,46,0.2); }}
  .section {{ margin: 32px 0; }}
</style>
</head>
<body>
<h1>📊 Backtest Report</h1>
<p>Symbols: <strong>{", ".join(symbols)}</strong> | Interval: {interval}m | Period: {years} years</p>
<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="section">
<h2>Full Period Performance</h2>
<div class="grid">
  <div class="card"><div class="label">CAGR</div><div class="value">{pct(m_all.get('cagr', 0))}</div></div>
  <div class="card"><div class="label">Max Drawdown</div><div class="value" style="color:#e74c3c">{pct(m_all.get('mdd', 0))}</div></div>
  <div class="card"><div class="label">Sharpe Ratio</div><div class="value">{num(m_all.get('sharpe', 0))}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value">{pct(m_all.get('win_rate', 0))}</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value">{num(m_all.get('profit_factor', 0))}</div></div>
  <div class="card"><div class="label">Total Trades</div><div class="value">{m_all.get('total_trades', 0)}</div></div>
  <div class="card"><div class="label">Total PnL</div><div class="value">{num(m_all.get('total_pnl_usdt', 0))} USDT</div></div>
  <div class="card"><div class="label">Final Equity</div><div class="value">{num(m_all.get('final_equity', 0))} USDT</div></div>
</div>
</div>

<div class="section">
<h2>Walk-Forward Validation</h2>
<table>
<tr><th>Metric</th><th>In-Sample (80%)</th><th>Out-of-Sample (20%)</th></tr>
<tr><td>Sharpe</td><td>{num(m_is.get('sharpe', 0))}</td><td>{num(m_oos.get('sharpe', 0))}</td></tr>
<tr><td>CAGR</td><td>{pct(m_is.get('cagr', 0))}</td><td>{pct(m_oos.get('cagr', 0))}</td></tr>
<tr><td>Max Drawdown</td><td>{pct(m_is.get('mdd', 0))}</td><td>{pct(m_oos.get('mdd', 0))}</td></tr>
<tr><td>Win Rate</td><td>{pct(m_is.get('win_rate', 0))}</td><td>{pct(m_oos.get('win_rate', 0))}</td></tr>
<tr><td>Profit Factor</td><td>{num(m_is.get('profit_factor', 0))}</td><td>{num(m_oos.get('profit_factor', 0))}</td></tr>
<tr><td>Trades</td><td>{m_is.get('total_trades', 0)}</td><td>{m_oos.get('total_trades', 0)}</td></tr>
</table>
</div>

<div class="section">
<h2>Monthly Returns</h2>
<table>
<tr><th>Month</th><th>Return</th></tr>
{monthly_rows}
</table>
</div>

<div class="section">
<h2>Strategy Notes</h2>
<ul>
  <li>Fee model: Taker 0.055%, Maker 0.020%</li>
  <li>Slippage: 0.10% per trade</li>
  <li>Max leverage cap (risk engine): {results.get('all', {}).get('max_leverage_used', 'N/A')}</li>
  <li>Walk-forward split: 80% in-sample / 20% out-of-sample</li>
</ul>
</div>
</body>
</html>"""

    path.write_text(html)
