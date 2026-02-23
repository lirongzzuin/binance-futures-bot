"""
Daily Briefing Generator.
Called by the scheduler at 9:00 AM KST every day.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.api.binance_client import BinanceClient
from src.api.slack import SlackNotifier
from src.config import Settings
from src.db.database import Database
from src.execution.state_machine import StateMachine
from src.logger import get_logger

log = get_logger(__name__)


class DailyBriefing:
    def __init__(
        self,
        settings: Settings,
        client: BinanceClient,
        db: Database,
        slack: SlackNotifier,
        state_machine: StateMachine,
    ):
        self._s = settings
        self._client = client
        self._db = db
        self._slack = slack
        self._sm = state_machine

    def send(self) -> None:
        """Compile and send daily briefing to Slack."""
        try:
            report = self._build_report()
            self._slack.send_daily_briefing(report)
            log.info("Daily briefing sent")
        except Exception as e:
            log.error("Daily briefing failed: %s", e)
            self._slack.send_error(str(e), context="daily_briefing")

    def _build_report(self) -> Dict[str, Any]:
        from datetime import date
        today = date.today().isoformat()

        # Equity
        equity = self._get_equity()
        total_pnl = self._db.get_total_pnl(mode=self._s.TRADING_MODE)
        initial = self._s.PAPER_CAPITAL_USDT
        total_pct = total_pnl / initial if initial > 0 else 0

        # Daily PnL
        daily_pnl = self._db.get_today_pnl()
        yesterday_equity = equity - daily_pnl
        daily_pct = daily_pnl / yesterday_equity if yesterday_equity > 0 else 0

        # Drawdown
        peak = self._db.get_peak_equity(mode=self._s.TRADING_MODE)
        drawdown = (peak - equity) / peak if peak > 0 else 0

        # Daily stats
        daily_stats = self._db.get_daily_stats(days=1)
        trades_today = daily_stats[0].get("trades", 0) if daily_stats else 0

        # 30-day win rate
        recent_trades = self._db.get_trades(limit=100, mode=self._s.TRADING_MODE)
        wins_30d = sum(1 for t in recent_trades if float(t.get("pnl", 0)) > 0)
        win_rate_30d = wins_30d / len(recent_trades) if recent_trades else 0

        # Open positions
        open_positions = self._sm.all_positions()
        pos_list = []
        total_lev = 0
        total_exposure = 0
        for p in open_positions:
            pos_list.append({
                "symbol": p.symbol,
                "side": p.state.value,
                "size": round(p.qty, 4),
                "unrealised_pnl": round(p.unrealised_pnl, 2),
            })
            total_lev += p.leverage
            total_exposure += p.qty * p.entry_price

        avg_lev = total_lev / len(open_positions) if open_positions else 0
        exposure_pct = total_exposure / equity if equity > 0 else 0

        return {
            "date": today,
            "equity": equity,
            "daily_pnl": daily_pnl,
            "daily_pct": daily_pct,
            "total_pnl": total_pnl,
            "total_pct": total_pct,
            "drawdown": drawdown,
            "positions": pos_list,
            "trades_today": trades_today,
            "win_rate_30d": win_rate_30d,
            "leverage_avg": avg_lev,
            "exposure_pct": exposure_pct,
            "mode": self._s.TRADING_MODE,
        }

    def _get_equity(self) -> float:
        if self._s.is_dry_run() or self._s.is_paper():
            # Paper: use initial capital + total PnL
            total_pnl = self._db.get_total_pnl(mode=self._s.TRADING_MODE)
            return self._s.PAPER_CAPITAL_USDT + total_pnl
        try:
            return self._client.get_usdt_equity()
        except Exception as e:
            log.error("Failed to fetch equity: %s", e)
            return 0.0
