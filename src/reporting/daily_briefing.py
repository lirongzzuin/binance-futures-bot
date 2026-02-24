"""
Daily Briefing Generator.
Called by the scheduler at 9:00 AM and 9:00 PM KST.

PnL 기준점 시스템:
  _PNL_BASELINE_KEY  : 누적 수익 기준 equity (reset_pnl_baseline() 로 재설정)
  _SNAPSHOT_KEY      : 마지막 브리핑 equity (브리핑마다 자동 갱신 → 기간 PnL)

reset_pnl_baseline() 호출 방법 (Binance 서버 배포 시):
  ssh <server>
  → python3 -c "
      from src.config import settings; from src.db.database import Database
      from src.api.binance_client import BinanceClient
      db=Database(settings); client=BinanceClient(settings)
      eq=client.get_usdt_equity()
      db.set_state('pnl_total_baseline', str(eq))
      db.set_state('briefing_equity_snapshot', str(eq))
      print(f'Baseline reset to {eq:.2f} USDT')
    "
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from src.api.binance_client import BinanceClient
from src.api.slack import SlackNotifier
from src.config import Settings
from src.db.database import Database
from src.execution.state_machine import StateMachine
from src.logger import get_logger

log = get_logger(__name__)

_KST = timezone(timedelta(hours=9))
_PNL_BASELINE_KEY = "pnl_total_baseline"    # 누적 PnL 기준점 (수동 재설정)
_SNAPSHOT_KEY = "briefing_equity_snapshot"  # 기간 PnL 기준점 (브리핑마다 갱신)


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
        self._last_equity: float = 0.0

    def reset_pnl_baseline(self) -> float:
        """
        누적 PnL + 기간 PnL 기준점을 현재 equity로 초기화.
        main.py 에서 또는 SSH 콘솔에서 직접 호출 가능.
        """
        current_equity = self._get_equity()
        self._db.set_state(_PNL_BASELINE_KEY, str(current_equity))
        self._db.set_state(_SNAPSHOT_KEY, str(current_equity))
        self._last_equity = current_equity
        log.info("PnL baseline reset to %.2f USDT", current_equity)
        return current_equity

    def send(self) -> None:
        """Compile and send briefing to Slack."""
        try:
            report = self._build_report()
            self._slack.send_daily_briefing(report)
            # 이번 브리핑 equity를 스냅샷으로 저장 (다음 브리핑 기간 PnL 계산용)
            self._last_equity = report["equity"]
            self._db.set_state(_SNAPSHOT_KEY, str(self._last_equity))
            log.info("Daily briefing sent | snapshot=%.2f", self._last_equity)
        except Exception as e:
            log.error("Daily briefing failed: %s", e)
            self._slack.send_error(str(e), context="daily_briefing")

    def _build_report(self) -> Dict[str, Any]:
        now_kst = datetime.now(tz=_KST)
        session = "오전 브리핑" if now_kst.hour < 15 else "오후 브리핑"
        equity = self._get_equity()

        # ── 누적 PnL (baseline 기준) ──────────────────────────────────────────
        baseline_raw = self._db.get_state(_PNL_BASELINE_KEY)
        if baseline_raw:
            baseline = float(baseline_raw)
        else:
            # 최초 실행: PAPER_CAPITAL_USDT를 baseline으로 사용
            baseline = self._s.PAPER_CAPITAL_USDT
        total_pnl = equity - baseline
        total_pct = total_pnl / baseline if baseline > 0 else 0.0

        # ── 기간 PnL (마지막 브리핑 이후) ─────────────────────────────────────
        snap_raw = self._db.get_state(_SNAPSHOT_KEY)
        if self._last_equity > 0:
            last_snap = self._last_equity
        elif snap_raw:
            last_snap = float(snap_raw)
            self._last_equity = last_snap
        else:
            last_snap = 0.0

        if last_snap > 0:
            period_pnl = equity - last_snap
            period_pct = period_pnl / last_snap
        else:
            period_pnl = self._db.get_today_pnl()
            period_pct = period_pnl / max(equity - period_pnl, 1.0)

        # ── Drawdown ──────────────────────────────────────────────────────────
        peak = self._db.get_peak_equity(mode=self._s.TRADING_MODE)
        if not peak or peak < equity:
            peak = equity  # 새 고점
        drawdown = (peak - equity) / peak if peak > 0 else 0.0

        # ── Daily stats ───────────────────────────────────────────────────────
        daily_stats = self._db.get_daily_stats(days=1)
        trades_today = daily_stats[0].get("trades", 0) if daily_stats else 0

        recent_trades = self._db.get_trades(limit=100, mode=self._s.TRADING_MODE)
        wins = sum(1 for t in recent_trades if float(t.get("pnl", 0)) > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0.0

        # ── Open positions ────────────────────────────────────────────────────
        open_positions = self._sm.all_positions()
        pos_list: List[Dict] = []
        total_lev = 0
        total_exposure = 0.0
        for p in open_positions:
            pos_list.append({
                "symbol": p.symbol,
                "side": p.state.value,
                "size": round(p.qty, 4),
                "unrealised_pnl": round(p.unrealised_pnl, 2),
                "leverage": p.leverage,
            })
            total_lev += p.leverage
            total_exposure += p.qty * p.entry_price

        avg_lev = total_lev / len(open_positions) if open_positions else 0.0
        exposure_pct = total_exposure / equity if equity > 0 else 0.0

        return {
            "session": session,
            "date": now_kst.strftime("%Y-%m-%d"),
            "time": now_kst.strftime("%H:%M KST"),
            "equity": equity,
            "baseline": baseline,
            "period_pnl": period_pnl,
            "period_pct": period_pct,
            "total_pnl": total_pnl,
            "total_pct": total_pct,
            "drawdown": drawdown,
            "positions": pos_list,
            "trades_today": trades_today,
            "win_rate": win_rate,
            "leverage_avg": avg_lev,
            "exposure_pct": exposure_pct,
            "mode": self._s.TRADING_MODE,
        }

    def _get_equity(self) -> float:
        if self._s.is_dry_run() or self._s.is_paper():
            total_pnl = self._db.get_total_pnl(mode=self._s.TRADING_MODE)
            return self._s.PAPER_CAPITAL_USDT + total_pnl
        try:
            return self._client.get_usdt_equity()
        except Exception as e:
            log.error("Failed to fetch equity: %s", e)
            return 0.0
