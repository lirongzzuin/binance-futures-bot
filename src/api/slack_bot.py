"""
Slack Bot — Socket Mode slash-command handler.

Runs as a daemon thread alongside the main trading loop.
Users can query live portfolio data at any time without waiting for
the next scheduled push.

Slash commands (register all four in the Slack App dashboard):
  /dashboard  — Full snapshot: equity + positions + PnL
  /positions  — Open positions with entry / mark / uPnL detail
  /pnl        — PnL summary (today / total / win-rate)
  /status     — Bot health: uptime, last cycle, circuit breaker

Setup (one-time):
  1. https://api.slack.com/apps → Create New App → From scratch
  2. Socket Mode → Enable → Generate App-Level Token (scope: connections:write)
     → copy as SLACK_APP_TOKEN (xapp-...)
  3. OAuth & Permissions → Bot Token Scopes: chat:write, commands
     → Install to workspace → copy Bot User OAuth Token as SLACK_BOT_TOKEN (xoxb-...)
  4. Slash Commands → Create each of the four commands above
     (Request URL can be anything — Socket Mode ignores it)
  5. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in fly.io secrets
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.logger import get_logger

log = get_logger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

class SlackBotHandler:
    """
    Wraps a slack-bolt App in Socket Mode, running in a background thread.

    data_provider: zero-argument callable that returns a DashboardData dict.
                   Called on every slash command — should be fast (< 2 s).
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        data_provider: Callable[[], Dict[str, Any]],
    ) -> None:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        self._data_provider = data_provider
        self._app = App(token=bot_token, logger=log)
        self._socket_handler = SocketModeHandler(self._app, app_token)
        self._thread: Optional[threading.Thread] = None

        self._register()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch Socket Mode listener in a non-blocking daemon thread."""
        self._thread = threading.Thread(
            target=self._run, name="slack-socket", daemon=True
        )
        self._thread.start()
        log.info("Slack Socket Mode listener started")

    def _run(self) -> None:
        try:
            self._socket_handler.start()
        except Exception as exc:
            log.error("Slack Socket Mode thread crashed: %s", exc)

    # ── Command Registration ───────────────────────────────────────────────────

    def _register(self) -> None:
        self._app.command("/dashboard")(self._handle_dashboard)
        self._app.command("/positions")(self._handle_positions)
        self._app.command("/pnl")(self._handle_pnl)
        self._app.command("/status")(self._handle_status)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_dashboard(self, ack, respond) -> None:
        ack()
        try:
            data = self._data_provider()
            respond(_fmt_dashboard(data))
        except Exception as exc:
            respond({"text": f"❌ 오류: `{exc}`"})

    def _handle_positions(self, ack, respond) -> None:
        ack()
        try:
            data = self._data_provider()
            respond(_fmt_positions(data))
        except Exception as exc:
            respond({"text": f"❌ 오류: `{exc}`"})

    def _handle_pnl(self, ack, respond) -> None:
        ack()
        try:
            data = self._data_provider()
            respond(_fmt_pnl(data))
        except Exception as exc:
            respond({"text": f"❌ 오류: `{exc}`"})

    def _handle_status(self, ack, respond) -> None:
        ack()
        try:
            data = self._data_provider()
            respond(_fmt_status(data))
        except Exception as exc:
            respond({"text": f"❌ 오류: `{exc}`"})


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_dashboard(data: Dict[str, Any]) -> Dict:
    equity = data.get("equity", 0.0)
    initial = data.get("initial_capital", equity or 1.0)
    daily_pnl = data.get("daily_pnl", 0.0)
    total_pnl = data.get("total_pnl", 0.0)
    positions: List[Dict] = data.get("positions", [])
    mode = data.get("mode", "?")

    daily_pct = daily_pnl / initial if initial else 0.0
    total_pct = total_pnl / initial if initial else 0.0

    mode_emoji = {"live": "🔴", "paper": "📄", "dry-run": "🤖"}.get(mode, "⚪")
    d_emoji = "🟢" if daily_pnl >= 0 else "🔴"
    t_emoji = "🟢" if total_pnl >= 0 else "🔴"

    # ── Header block ──────────────────────────────────────────────────────────
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📊 Portfolio Dashboard  {mode_emoji} {mode.upper()}",
            },
        },
        {"type": "divider"},
        # ── Equity + PnL ──────────────────────────────────────────────────────
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*💰 Equity*\n`{equity:,.2f} USDT`"},
                {"type": "mrkdwn", "text": f"*{d_emoji} Today PnL*\n`{daily_pnl:+.2f} USDT` ({daily_pct:+.2%})"},
                {"type": "mrkdwn", "text": f"*{t_emoji} Total PnL*\n`{total_pnl:+.2f} USDT` ({total_pct:+.2%})"},
                {"type": "mrkdwn", "text": f"*📍 Open Positions*\n`{len(positions)}`"},
            ],
        },
        {"type": "divider"},
    ]

    # ── Position blocks ───────────────────────────────────────────────────────
    if positions:
        for p in positions:
            blocks.append(_position_block(p))
    else:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "_현재 오픈 포지션 없음_"},
        })

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"_Updated: {_now()}_"}],
    })

    return {"blocks": blocks, "text": "Portfolio Dashboard"}


def _fmt_positions(data: Dict[str, Any]) -> Dict:
    positions: List[Dict] = data.get("positions", [])
    mode = data.get("mode", "?")

    if not positions:
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "📭 *현재 오픈 포지션 없음*"},
                }
            ],
            "text": "포지션 없음",
        }

    blocks: List[Dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📍 Open Positions ({len(positions)})  [{mode.upper()}]",
            },
        },
        {"type": "divider"},
    ]

    for p in positions:
        blocks.append(_position_block(p))

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"_Updated: {_now()}_"}],
    })

    return {"blocks": blocks, "text": f"Open positions: {len(positions)}"}


def _fmt_pnl(data: Dict[str, Any]) -> Dict:
    equity = data.get("equity", 0.0)
    initial = data.get("initial_capital", equity or 1.0)
    daily_pnl = data.get("daily_pnl", 0.0)
    weekly_pnl = data.get("weekly_pnl", 0.0)
    total_pnl = data.get("total_pnl", 0.0)
    trades_today = data.get("trades_today", 0)
    win_rate = data.get("win_rate", 0.0)

    daily_pct = daily_pnl / initial if initial else 0.0
    weekly_pct = weekly_pnl / initial if initial else 0.0
    total_pct = total_pnl / initial if initial else 0.0

    def _row(label: str, pnl: float, pct: float) -> Dict:
        emoji = "🟢" if pnl >= 0 else "🔴"
        return {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*{emoji} {label}*"},
                {"type": "mrkdwn", "text": f"`{pnl:+.2f} USDT`  ({pct:+.2%})"},
            ],
        }

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "💹 PnL Summary"},
        },
        {"type": "divider"},
        _row("Today", daily_pnl, daily_pct),
        _row("This Week", weekly_pnl, weekly_pct),
        _row("All Time", total_pnl, total_pct),
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*🎯 Trades Today*\n`{trades_today}`"},
                {"type": "mrkdwn", "text": f"*🏆 Win Rate (30d)*\n`{win_rate:.1%}`"},
                {"type": "mrkdwn", "text": f"*💰 Equity*\n`{equity:,.2f} USDT`"},
                {"type": "mrkdwn", "text": f"*📦 Initial Capital*\n`{initial:,.2f} USDT`"},
            ],
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"_Updated: {_now()}_"}],
        },
    ]

    return {"blocks": blocks, "text": "PnL Summary"}


def _fmt_status(data: Dict[str, Any]) -> Dict:
    mode = data.get("mode", "?")
    uptime_sec = data.get("uptime_seconds", 0.0)
    last_cycle_ts = data.get("last_cycle_ts", 0.0)
    circuit_open = data.get("circuit_breaker_open", False)
    open_positions = len(data.get("positions", []))

    uptime_str = _fmt_duration(uptime_sec)
    last_cycle_str = (
        _fmt_ago(last_cycle_ts) if last_cycle_ts > 0 else "_아직 실행 안 됨_"
    )
    cb_str = "🔴 OPEN (API 차단 중)" if circuit_open else "🟢 CLOSED (정상)"
    mode_emoji = {"live": "🔴", "paper": "📄", "dry-run": "🤖"}.get(mode, "⚪")

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🤖 Bot Status"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*{mode_emoji} Mode*\n`{mode.upper()}`"},
                {"type": "mrkdwn", "text": f"*⏱ Uptime*\n`{uptime_str}`"},
                {"type": "mrkdwn", "text": f"*🔄 Last Cycle*\n{last_cycle_str}"},
                {"type": "mrkdwn", "text": f"*⚡ Circuit Breaker*\n{cb_str}"},
                {"type": "mrkdwn", "text": f"*📍 Open Positions*\n`{open_positions}`"},
            ],
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"_Updated: {_now()}_"}],
        },
    ]

    return {"blocks": blocks, "text": "Bot Status"}


# ── Shared block builder ──────────────────────────────────────────────────────

def _position_block(p: Dict[str, Any]) -> Dict:
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

    # Price change direction arrow
    price_diff = mark - entry
    price_arrow = "↑" if price_diff >= 0 else "↓"

    return {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": (
                    f"*{side_emoji} `{symbol}`  {side_label}  ×{lev}*\n"
                    f"qty: `{qty}`"
                ),
            },
            {
                "type": "mrkdwn",
                "text": (
                    f"Entry: `{entry:,.2f}` {price_arrow} Mark: `{mark:,.2f}`\n"
                    f"{upnl_emoji} uPnL: `{upnl:+.2f} USDT` ({upnl_pct:+.2%})"
                ),
            },
        ],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_ago(ts: float) -> str:
    elapsed = time.time() - ts
    if elapsed < 60:
        return f"`{int(elapsed)}초 전`"
    if elapsed < 3600:
        return f"`{int(elapsed / 60)}분 전`"
    return f"`{int(elapsed / 3600)}시간 전`"
