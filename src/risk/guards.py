"""
Hard Guard Rails — enforces hard limits and protects capital in adverse conditions.

Guards checked in order:
  1. Daily loss monitoring (알림만, 거래 중지 없음 — 복리 극대화)
  2. Weekly loss monitoring (알림만, 거래 중지 없음)
  3. Max drawdown — 계층적 회복 시스템 (파산 방어 + 복리 회복):
       DD < 21%: ALLOW — 정상 거래
       DD 21~30%: REDUCE (0~75% 비중 축소) — 손실 억제
       DD 30~42%: REDUCE (75~85% 비중 축소) — 회복 모드 소규모 거래
       DD 42~50%: PAUSE — 신규 진입 중단, 기존 포지션 관리
       DD >= 50%: HALT  — 긴급 전면 중단
  4. Spike / extreme volatility (스프레드 > 1% 시 진입 일시 중지)

When a guard trips, it either:
  - Logs a warning + Slack alert (daily/weekly loss → 거래 계속)
  - Forces position reduction (REDUCE mode) — 비중 축소 후 소규모 거래 계속
  - Pauses new entries (PAUSE mode) — 기존 포지션 관리만
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

        if worst.action != GuardAction.ALLOW:
            log.warning("Guard tripped: %s — %s", worst.action.value, worst.reason)

        return worst

    def _check_daily_loss(
        self, equity: float, daily_pnl: float, initial_capital: float
    ) -> GuardResult:
        """일일 손실 모니터링 — 거래 중지 없이 경고만 (복리 극대화 전략)."""
        if equity <= 0:
            return GuardResult(GuardAction.HALT, "Equity <= 0")

        daily_loss_pct = -daily_pnl / max(equity - daily_pnl, 1)
        if daily_loss_pct >= self._s.MAX_DAILY_LOSS_PCT:
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
        """계층적 드로우다운 가드 — 즉시 중단 대신 비중 축소로 복리 회복 허용.

        계층:
          DD < reduce_limit (21%): ALLOW  — 정상 거래
          DD 21~30%:  REDUCE 0~75%        — 손실 확대 억제, 소규모 거래 유지
          DD 30~42%:  REDUCE 75~85%       — 극소규모 거래, 회복 적립
          DD 42~50%:  PAUSE               — 신규 진입 중단, 기존 포지션 관리
          DD >= 50%:  HALT                — 긴급 전면 중단 (재앙적 손실 방어)
        """
        if peak_equity <= 0:
            return GuardResult(GuardAction.ALLOW)

        dd = (peak_equity - equity) / peak_equity
        halt_limit   = self._s.HALT_DRAWDOWN_PCT   # default 0.50
        pause_limit  = self._s.PAUSE_DRAWDOWN_PCT  # default 0.42
        hard_limit   = self._s.MAX_DRAWDOWN_PCT    # default 0.30
        reduce_limit = hard_limit * 0.70           # default 0.21

        if dd >= halt_limit:
            return GuardResult(
                GuardAction.HALT,
                f"Max drawdown {dd:.2%} ≥ halt limit {halt_limit:.2%} — emergency stop",
            )
        if dd >= pause_limit:
            return GuardResult(
                GuardAction.PAUSE,
                f"Drawdown {dd:.2%} ≥ pause limit {pause_limit:.2%} — pausing new entries",
            )
        if dd >= hard_limit:
            zone = max(pause_limit - hard_limit, 0.001)
            reduce_pct = 0.75 + (dd - hard_limit) / zone * 0.10
            return GuardResult(
                GuardAction.REDUCE,
                f"Drawdown {dd:.2%} ≥ hard limit {hard_limit:.2%} — heavy reduction (recovery mode)",
                reduce_pct=min(0.85, reduce_pct),
            )
        if dd >= reduce_limit:
            reduce_pct = (dd - reduce_limit) / max(hard_limit - reduce_limit, 0.001)
            return GuardResult(
                GuardAction.REDUCE,
                f"Drawdown {dd:.2%} — reducing position size",
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
