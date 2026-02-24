"""
ProfitProtector — automatic profit transfer to Funding Account.

Logic:
  When current equity exceeds initial_capital by PROFIT_PROTECT_THRESHOLD_PCT:
    1. Calculate profit = equity - initial_capital
    2. Calculate transfer = profit × PROFIT_PROTECT_TRANSFER_PCT
    3. Ensure trading account keeps ≥ initial_capital × PROFIT_PROTECT_MIN_CAPITAL_PCT
    4. Transfer remainder to Funding Account (safe, not in futures margin)
    5. Record transfer in DB and send Slack alert

This protects against the risk of holding too much capital in a cross-futures
account, which could all be at risk if positions move adversely.

Example (20k equity, 10k initial, threshold=20%, transfer=50%):
  profit           = 20000 - 10000 = 10000
  candidate amount = 10000 × 0.50 = 5000
  min keep         = 10000 × 1.10 = 11000
  remaining        = 20000 - 5000 = 15000 ≥ 11000 ✓
  → Transfer 5000 USDT to Funding Account
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from src.api.binance_client import BinanceClient
    from src.api.slack import SlackNotifier
    from src.config import Settings

log = get_logger(__name__)


@dataclass
class TransferRecord:
    timestamp: float
    amount: float
    equity_before: float
    equity_after: float
    total_transferred: float


class ProfitProtector:
    """
    Monitors trading account equity and automatically transfers
    a portion of profits to the safer Funding Account.
    """

    def __init__(
        self,
        client: "BinanceClient",
        settings: "Settings",
        slack: "SlackNotifier",
        initial_capital: float,
    ) -> None:
        self._client = client
        self._s = settings
        self._slack = slack
        self._initial_capital = initial_capital
        self._total_transferred: float = 0.0
        self._last_check_ts: float = 0.0
        self._transfer_history: list[TransferRecord] = []

    # ── Main check ────────────────────────────────────────────────────────────

    def check_and_protect(self, current_equity: float) -> Optional[float]:
        """
        Called periodically. Returns transfer amount if a transfer was made, else None.
        In paper/dry-run mode: simulates the transfer (logs + Slack, no real API call).
        In live mode: executes the real internal transfer.
        """
        if not self._s.PROFIT_PROTECT_ENABLED:
            return None

        # Rate limit: only check every N hours
        check_interval = self._s.PROFIT_PROTECT_CHECK_HOURS * 3600
        if time.time() - self._last_check_ts < check_interval:
            return None
        self._last_check_ts = time.time()

        transfer_amount = self._calculate_transfer(current_equity)
        if transfer_amount is None:
            return None

        log.info(
            "ProfitProtector: equity=%.2f initial=%.2f profit=%.2f → transfer=%.2f USDT",
            current_equity,
            self._initial_capital,
            current_equity - self._initial_capital,
            transfer_amount,
        )

        # Execute (or simulate)
        success = self._execute_transfer(transfer_amount, current_equity)
        if not success:
            return None

        self._total_transferred += transfer_amount
        record = TransferRecord(
            timestamp=time.time(),
            amount=transfer_amount,
            equity_before=current_equity,
            equity_after=current_equity - transfer_amount,
            total_transferred=self._total_transferred,
        )
        self._transfer_history.append(record)

        # Slack notification
        self._slack.send_profit_transfer(
            amount=transfer_amount,
            equity_before=current_equity,
            equity_after=current_equity - transfer_amount,
            total_transferred=self._total_transferred,
        )

        return transfer_amount

    # ── Calculation ───────────────────────────────────────────────────────────

    def _calculate_transfer(self, equity: float) -> Optional[float]:
        """
        Returns how much to transfer, or None if conditions not met.
        """
        # 복리 성장 단계: equity가 최소 임계값 미만이면 이체 건너뜀
        min_equity = self._s.PROFIT_PROTECT_MIN_EQUITY_USDT
        if equity < min_equity:
            log.debug(
                "ProfitProtector: equity=%.2f < min_threshold=%.2f — 복리 성장 단계, 이체 건너뜀",
                equity, min_equity,
            )
            return None

        profit = equity - self._initial_capital
        profit_pct = profit / self._initial_capital if self._initial_capital > 0 else 0

        # Must exceed profit threshold
        if profit_pct < self._s.PROFIT_PROTECT_THRESHOLD_PCT:
            log.debug(
                "ProfitProtector: profit=%.1f%% < threshold=%.1f%% — skip",
                profit_pct * 100,
                self._s.PROFIT_PROTECT_THRESHOLD_PCT * 100,
            )
            return None

        # Candidate: fraction of profit
        candidate = profit * self._s.PROFIT_PROTECT_TRANSFER_PCT

        # Safety: trading account must keep ≥ min_capital after transfer
        min_keep = self._initial_capital * self._s.PROFIT_PROTECT_MIN_CAPITAL_PCT
        max_transferable = equity - min_keep

        if max_transferable <= 0:
            log.debug("ProfitProtector: not enough margin to transfer safely")
            return None

        transfer_amount = min(candidate, max_transferable)

        # Minimum meaningful transfer: 10 USDT
        if transfer_amount < 10.0:
            return None

        return round(transfer_amount, 2)

    # ── Execution ─────────────────────────────────────────────────────────────

    def _execute_transfer(self, amount: float, equity: float) -> bool:
        """Execute the transfer. Paper/dry-run: simulated. Live: real API call."""
        if self._s.is_dry_run() or self._s.is_paper():
            log.info("[%s] Simulated transfer: %.2f USDT → Funding Account",
                     self._s.TRADING_MODE.upper(), amount)
            return True

        try:
            self._client.transfer_to_spot(amount)  # Binance: Futures → Spot Account
            return True
        except Exception as e:
            log.error("ProfitProtector: transfer failed: %s", e)
            return False

    # ── Status ────────────────────────────────────────────────────────────────

    def total_transferred(self) -> float:
        return self._total_transferred

    def transfer_count(self) -> int:
        return len(self._transfer_history)

    def status_dict(self) -> dict:
        return {
            "enabled": self._s.PROFIT_PROTECT_ENABLED,
            "initial_capital": self._initial_capital,
            "threshold_pct": self._s.PROFIT_PROTECT_THRESHOLD_PCT,
            "transfer_pct": self._s.PROFIT_PROTECT_TRANSFER_PCT,
            "total_transferred": self._total_transferred,
            "transfer_count": len(self._transfer_history),
        }
