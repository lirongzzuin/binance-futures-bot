"""
Hard Guard Rails — enforces hard limits and halts trading in critical scenarios.

Guards checked in order:
  1. Daily loss monitoring (알림만, 거래 중지 없음 — 복리 극대화)
  2. Weekly loss monitoring (알림만, 거래 중지 없음)
  3. Max drawdown (30% 초과 시 HALT — 파산 방어)
  4. Spike / extreme volatility (스프레드 > 1% 시 진입 일시 중지)

When a guard trips, it either:
  - Logs a warning + Slack alert (daily/weekly loss → 거래 계속)
  - Forces position reduction (REDUCE mode)
  - Emergency-exits all positions (HALT mode)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)


class GuardAction(str, Enum):
    ALLOW = "ALLOW"          # proceed normally
    PAUSE = "PAUSE"          # no new entries, hold existing
    REDUCE = "REDUCE"        # reduce leverage / scale out
    HALT = "HALT"            # close all, stop trading


@dataclass
class GuardResult:
    action: GuardAction
    reason: str = ""
    reduce_pct: float = 0.0   # for REDUCE: how much to scale down (0-1)


class RiskGuards:
    def __init__(self, settings: Settings):
        self._s = settings

    def check(
        self,
        equity: float,
        peak_equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        initial_capital: float,
        spread_pct: float,
        vol_percentile: float,
    ) -> GuardResult:
        """
        Run all guards and return the most restrictive action.
        """
        results = [
            self._check_daily_loss(equity, daily_pnl, initial_capital),
            self._check_weekly_loss(equity, weekly_pnl, initial_capital),
            self._check_max_drawdown(equity, peak_equity),
            self._check_spike(spread_pct, vol_percentile),
        ]

        # Return the most restrictive result
        priority = {GuardAction.HALT: 4, GuardAction.REDUCE: 3,
                    GuardAction.PAUSE: 2, GuardAction.ALLOW: 1}
        results.sort(key=lambda r: priority[r.action], reverse=True)
        worst = results[0]

        if worst.action not in (GuardAction.ALLOW,):
            log.warning("Guard tripped: %s — %s", worst.action.value, worst.reason)

        return worst

    def _check_daily_loss(
        self, equity: float, daily_pnl: float, initial_capital: float
    ) -> GuardResult:
        """일일 손실 모니터링 — 거래 중지 없이 경고만 (복리 극대화 전략).

        경고는 caller(main.py)에서 Slack으로 전송하고 거래는 계속 진행.
        단, equity <= 0이면 즉시 HALT.
        """
        if equity <= 0:
            return GuardResult(GuardAction.HALT, "Equity <= 0")

        daily_loss_pct = -daily_pnl / max(equity - daily_pnl, 1)
        if daily_loss_pct >= self._s.MAX_DAILY_LOSS_PCT:
            # 거래 중지하지 않음 — main.py에서 Slack 알림 한 번만 전송 (KST 날짜 디바운스)
            return GuardResult(
                GuardAction.ALLOW,
                f"Daily loss {daily_loss_pct:.2%} ≥ limit {self._s.MAX_DAILY_LOSS_PCT:.2%}",
            )
        return GuardResult(GuardAction.ALLOW)

    def _check_weekly_loss(
        self, equity: float, weekly_pnl: float, initial_capital: float
    ) -> GuardResult:
        """주간 손실 모니터링 — 거래 중지 없이 경고만 (복리 극대화 전략)."""
        weekly_loss_pct = -weekly_pnl / max(equity - weekly_pnl, 1)
        if weekly_loss_pct >= self._s.MAX_WEEKLY_LOSS_PCT:
            log.debug(
                "Weekly loss: %.2f%% ≥ limit %.2f%% — 거래 계속 진행 (복리 모드)",
                weekly_loss_pct * 100, self._s.MAX_WEEKLY_LOSS_PCT * 100,
            )
        return GuardResult(GuardAction.ALLOW)

    def _check_max_drawdown(self, equity: float, peak_equity: float) -> GuardResult:
        """Reduce or halt if drawdown exceeds thresholds."""
        if peak_equity <= 0:
            return GuardResult(GuardAction.ALLOW)

        dd = (peak_equity - equity) / peak_equity
        hard_limit = self._s.MAX_DRAWDOWN_PCT
        reduce_limit = hard_limit * 0.70  # 70% of max → REDUCE

        if dd >= hard_limit:
            return GuardResult(
                GuardAction.HALT,
                f"Max drawdown {dd:.2%} ≥ hard limit {hard_limit:.2%}",
            )
        elif dd >= reduce_limit:
            reduce_pct = (dd - reduce_limit) / (hard_limit - reduce_limit)
            return GuardResult(
                GuardAction.REDUCE,
                f"Drawdown {dd:.2%} — reducing leverage",
                reduce_pct=min(0.75, reduce_pct),
            )
        return GuardResult(GuardAction.ALLOW)

    def _check_spike(self, spread_pct: float, vol_percentile: float) -> GuardResult:
        """Pause entries during extreme spread or volatility."""
        if spread_pct > 0.01:  # 1% spread = dangerous
            return GuardResult(
                GuardAction.PAUSE,
                f"Spread {spread_pct:.3%} is extreme — pausing entries",
            )
        if vol_percentile > 0.97:
            return GuardResult(
                GuardAction.REDUCE,
                f"Volatility percentile {vol_percentile:.2%} — reducing",
                reduce_pct=0.5,
            )
        return GuardResult(GuardAction.ALLOW)
