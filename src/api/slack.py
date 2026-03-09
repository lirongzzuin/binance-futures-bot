"""
Slack Incoming Webhook notifier.
All bot events, trade alerts, and daily briefings are sent here.
"""
from __future__ import annotations

import json
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

# Korean Standard Time = UTC+9
_KST = timezone(timedelta(hours=9))

from src.logger import get_logger

log = get_logger(__name__)


class SlackNotifier:
    def __init__(self, webhook_url: str):
        self._url = webhook_url
        self._enabled = bool(webhook_url)

    def _post(self, payload: Dict) -> None:
        if not self._enabled:
            log.debug("Slack disabled (no webhook URL): %s", payload.get("text", "")[:80])
            return
        try:
            resp = httpx.post(self._url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            log.error("Slack notification failed: %s", exc)

    def _send(self, text: str, blocks: Optional[List] = None) -> None:
        payload: Dict[str, Any] = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        self._post(payload)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def send_startup(self, mode: str) -> None:
        emoji = {"dry-run": "🤖", "paper": "📄", "live": "🔴"}.get(mode, "⚪")
        self._send(
            f"{emoji} *Bot Started* | Mode: `{mode.upper()}` | {_now()}"
        )

    def send_shutdown(self, reason: str = "manual") -> None:
        self._send(f"⏹ *Bot Stopped* | Reason: `{reason}` | {_now()}")

    def send_error(self, error: str, context: str = "") -> None:
        ctx = f" | context: `{context}`" if context else ""
        self._send(f"❌ *Error*{ctx}\n```{error[:500]}```")

    # ── Universe ──────────────────────────────────────────────────────────────

    def send_universe_update(self, symbols: List[str]) -> None:
        self._send(
            f"🔄 *Universe Updated* | Top {len(symbols)} coins: "
            + ", ".join(f"`{s}`" for s in symbols)
        )

    # ── Trade Events ──────────────────────────────────────────────────────────

    def send_signal(
        self, symbol: str, side: str, strength: float, regime: str
    ) -> None:
        arrow = "📈" if side == "Buy" else "📉"
        self._send(
            f"{arrow} *Signal* `{symbol}` | {side} | "
            f"strength={strength:.2f} | regime={regime}"
        )

    def send_order_placed(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        price: Optional[str],
        order_id: str,
        leverage: int,
    ) -> None:
        price_str = f"@ {price}" if price else "@ MARKET"
        self._send(
            f"📋 *Order Placed* `{symbol}` {side} {order_type} "
            f"qty={qty} {price_str} lev={leverage}× | id=`{order_id[:12]}`"
        )

    def send_order_filled(
        self,
        symbol: str,
        side: str,
        qty: str,
        avg_price: str,
        pnl: Optional[float] = None,
    ) -> None:
        pnl_str = f" | PnL={pnl:+.2f} USDT" if pnl is not None else ""
        self._send(
            f"✅ *Filled* `{symbol}` {side} qty={qty} @ {avg_price}{pnl_str}"
        )

    def send_order_partial(
        self, symbol: str, side: str, filled: str, total: str
    ) -> None:
        self._send(
            f"⚡ *Partial Fill* `{symbol}` {side} {filled}/{total}"
        )

    def send_order_cancelled(self, symbol: str, order_id: str, reason: str) -> None:
        self._send(
            f"🚫 *Cancelled* `{symbol}` id=`{order_id[:12]}` | {reason}"
        )

    def send_order_failed(self, symbol: str, side: str, error: str) -> None:
        self._send(f"💥 *Order Failed* `{symbol}` {side} | `{error[:200]}`")

    def send_position_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        leverage: int,
        stop_price: float,
        take_profit_price: float,
        signal_strength: float,
        order_type: str = "Limit",
    ) -> None:
        arrow = "📈" if side == "Buy" else "📉"
        direction = "LONG" if side == "Buy" else "SHORT"
        self._send(
            f"{arrow} *Opened* `{symbol}` {direction} ×{leverage} | {order_type}\n"
            f"진입가={entry_price:,.4f} | qty={qty} | "
            f"SL={stop_price:,.4f} | TP={take_profit_price:,.4f}\n"
            f"signal={signal_strength:.2f} | {_now()}"
        )

    def send_position_closed(
        self,
        symbol: str,
        side: str,          # "Buy" or "Sell" (entry side = position type)
        reason: str,
        pnl: float,
        pnl_pct: float,     # PnL as % of margin (return on invested capital)
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        leverage: int = 1,
    ) -> None:
        emoji = "🟢" if pnl >= 0 else "🔴"
        direction = "LONG" if side == "Buy" else "SHORT"
        price_str = (
            f" | {entry_price:,.4f}→{exit_price:,.4f}"
            if entry_price > 0 and exit_price > 0 else ""
        )
        lev_str = f" ×{leverage}" if leverage > 1 else ""
        self._send(
            f"{emoji} *Closed* `{symbol}` {direction}{lev_str}{price_str}\n"
            f"사유: {reason} | PnL={pnl:+.2f} USDT ({pnl_pct:+.2%})"
        )

    # ── Risk Alerts ───────────────────────────────────────────────────────────

    def send_risk_alert(self, alert: str) -> None:
        self._send(f"⚠️ *Risk Alert* | {alert}")

    def send_market_mode_changed(
        self,
        from_mode: str,
        to_mode: str,
        reason: str,
        symbols: List[str],
        leverage_cap: int,
    ) -> None:
        """매크로 시장 레짐 전환 알림."""
        mode_emoji = {
            "BULL": "🟢",
            "BEAR": "🔴",
            "SIDEWAYS": "🟡",
        }
        to_emoji = mode_emoji.get(to_mode, "⚪")
        sym_str = ", ".join(f"`{s}`" for s in symbols)
        self._send(
            f"{to_emoji} *Market Mode Changed*\n"
            f"  {from_mode} → *{to_mode}*\n"
            f"  거래 심볼: {sym_str}\n"
            f"  최대 레버리지: {leverage_cap}×\n"
            f"  근거: {reason}\n"
            f"  {_now()}"
        )

    def send_daily_loss_limit(self, loss_pct: float) -> None:
        self._send(
            f"⚠️ *Daily Loss Alert* | 오늘 손실 {loss_pct:.2%} — "
            f"거래는 계속 진행됩니다 (복리 모드)."
        )

    def send_drawdown_alert(self, dd_pct: float) -> None:
        self._send(
            f"🚨 *Max Drawdown Breach* | Current DD={dd_pct:.2%}. "
            f"Switching to conservative mode."
        )

    def send_circuit_breaker(self, reason: str) -> None:
        self._send(f"⚡ *Circuit Breaker Opened* | {reason}")

    # ── Profit Protection ─────────────────────────────────────────────────────

    def send_profit_transfer(
        self,
        amount: float,
        equity_before: float,
        equity_after: float,
        total_transferred: float,
    ) -> None:
        self._send(
            f"💰 *Profit Protected* | {amount:,.2f} USDT → Funding Account\n"
            f"Trading Account: {equity_before:,.2f} → {equity_after:,.2f} USDT\n"
            f"Total moved to safety: {total_transferred:,.2f} USDT"
        )

    # ── Real-time Dashboard ───────────────────────────────────────────────────

    def send_dashboard(
        self,
        equity: float,
        initial_capital: float,
        daily_pnl: float,
        total_pnl: float,
        positions: List[Dict],
        mode: str,
    ) -> None:
        """
        Real-time portfolio dashboard sent on a configurable interval.

        positions: list of dicts with keys:
          symbol, side ("Buy"/"Sell"), qty, entry_price,
          mark_price, unrealised_pnl, leverage
        """
        daily_pct = daily_pnl / initial_capital if initial_capital > 0 else 0.0
        total_pct = total_pnl / initial_capital if initial_capital > 0 else 0.0

        d_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        t_emoji = "🟢" if total_pnl >= 0 else "🔴"
        mode_emoji = {"live": "🔴", "paper": "📄", "dry-run": "🤖"}.get(mode, "⚪")

        # ── Build position lines ──────────────────────────────────────────────
        pos_lines = ""
        for p in positions:
            symbol = p.get("symbol", "?")
            side_raw = p.get("side", "Buy")
            side_label = "LONG" if side_raw == "Buy" else "SHORT"
            side_emoji = "📈" if side_raw == "Buy" else "📉"
            qty = p.get("qty", 0.0)
            entry = p.get("entry_price", 0.0)
            mark = p.get("mark_price", entry)
            upnl = p.get("unrealised_pnl", 0.0)
            lev = p.get("leverage", 1)
            notional = entry * qty
            upnl_pct = upnl / notional if notional > 0 else 0.0
            upnl_emoji = "🟢" if upnl >= 0 else "🔴"
            pos_lines += (
                f"  {side_emoji} `{symbol}` {side_label} ×{lev}\n"
                f"       qty={qty} | entry={entry:,.2f} → {mark:,.2f}\n"
                f"       {upnl_emoji} uPnL: {upnl:+.2f} USDT ({upnl_pct:+.2%})\n"
            )

        if not pos_lines:
            pos_lines = "  _포지션 없음_\n"

        sep = "─" * 36
        text = (
            f"📊 *Portfolio Dashboard* {mode_emoji} `{mode.upper()}`\n"
            f"{sep}\n"
            f"💰 Equity   : `{equity:>12,.2f} USDT`\n"
            f"{d_emoji} Today PnL : `{daily_pnl:>+12.2f} USDT`  ({daily_pct:+.2%})\n"
            f"{t_emoji} Total PnL : `{total_pnl:>+12.2f} USDT`  ({total_pct:+.2%})\n"
            f"{sep}\n"
            f"*Open Positions ({len(positions)})*\n"
            f"{pos_lines}"
            f"{sep}\n"
            f"_Updated: {_now()}_"
        )
        self._send(text)

    # ── Daily Briefing ────────────────────────────────────────────────────────

    def send_daily_briefing(self, report: Dict[str, Any]) -> None:
        """
        report keys: session, date, time, equity, period_pnl, period_pct,
                     total_pnl, total_pct, drawdown, positions, trades_today,
                     win_rate, leverage_avg, exposure_pct, mode
        """
        session = report.get("session", "브리핑")
        date = report.get("date", _now()[:10])
        time_str = report.get("time", _now()[11:])
        eq = report.get("equity", 0)
        p_pnl = report.get("period_pnl", 0)     # 마지막 브리핑 이후 PnL
        p_pct = report.get("period_pct", 0)
        t_pnl = report.get("total_pnl", 0)
        t_pct = report.get("total_pct", 0)
        dd = report.get("drawdown", 0)
        positions = report.get("positions", [])
        trades = report.get("trades_today", 0)
        wr = report.get("win_rate", 0)
        lev = report.get("leverage_avg", 0)
        exp = report.get("exposure_pct", 0)
        mode = report.get("mode", "?")

        pos_lines = ""
        for p in positions:
            side = p.get("side", "?")
            sym = p.get("symbol", "?")
            sz = p.get("size", 0)
            upnl = p.get("unrealised_pnl", 0)
            upnl_emoji = "🟢" if upnl >= 0 else "🔴"
            pos_lines += f"  • {upnl_emoji} `{sym}` {side} qty={sz} uPnL={upnl:+.2f}\n"
        if not pos_lines:
            pos_lines = "  _포지션 없음_\n"

        dd_emoji = "🟢" if dd < 0.05 else ("🟡" if dd < 0.15 else "🔴")
        p_emoji = "🟢" if p_pnl >= 0 else "🔴"
        t_emoji = "🟢" if t_pnl >= 0 else "🔴"
        mode_emoji = {"live": "🔴", "paper": "📄", "dry-run": "🤖"}.get(mode, "⚪")

        sep = "─" * 32
        text = (
            f"📊 *{session}* {mode_emoji} `{mode.upper()}`\n"
            f"{date}  {time_str}\n"
            f"{sep}\n"
            f"{p_emoji} 기간 수익 : `{p_pnl:>+10.2f} USDT`  ({p_pct:+.2%})\n"
            f"{t_emoji} 누적 수익 : `{t_pnl:>+10.2f} USDT`  ({t_pct:+.2%})\n"
            f"💰 잔고     : `{eq:>10,.2f} USDT`\n"
            f"{sep}\n"
            f"{dd_emoji} MDD: {dd:.2%} | 📦 {exp:.1%} 노출 | ×{lev:.1f} 평균레버\n"
            f"🎯 오늘 거래: {trades}회 | 최근 승률: {wr:.1%}\n"
            f"\n*Open Positions ({len(positions)}):*\n{pos_lines}"
        )
        self._send(text)


def _now() -> str:
    """Return current time formatted in Korean Standard Time (KST, UTC+9)."""
    return datetime.now(tz=_KST).strftime("%Y-%m-%d %H:%M KST")
