"""
Trading State Machine.

Each symbol has exactly one state: FLAT | LONG | SHORT
Transitions are validated to prevent invalid state changes.

State machine governs:
  - Entry: FLAT → LONG | SHORT (requires signal + risk OK)
  - Exit:  LONG/SHORT → FLAT  (stop, TP, trailing, time, regime change)
  - Re-entry cooldown enforcement
  - Partial exit transitions (LONG → LONG with reduced size)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)


class PositionState(str, Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    PARTIAL_TP1 = "PARTIAL_TP1"
    TRAILING_STOP = "TRAILING_STOP"
    REGIME_CHANGE = "REGIME_CHANGE"
    SIGNAL_FLIP = "SIGNAL_FLIP"
    TIME_EXIT = "TIME_EXIT"
    RISK_GUARD = "RISK_GUARD"
    MANUAL = "MANUAL"
    LIQUIDATION_RISK = "LIQUIDATION_RISK"


@dataclass
class PositionInfo:
    symbol: str
    state: PositionState = PositionState.FLAT
    entry_price: float = 0.0
    qty: float = 0.0
    leverage: int = 1
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    tp1_price: float = 0.0        # first partial TP target
    tp2_price: float = 0.0        # second (final) TP target
    tp1_hit: bool = False          # True after TP1 partial exit executed
    trade_mode: str = "SCALP"     # SCALP or TREND_RIDE
    mode_max_bars: int = 15        # max bars for this mode
    partial_exit_pct: float = 0.60 # fraction to close at TP1
    trailing_stop_distance: float = 0.0
    highest_price: float = 0.0   # for long trailing stop
    lowest_price: float = 0.0    # for short trailing stop
    entry_ts: float = 0.0
    last_exit_ts: float = 0.0    # for cooldown
    last_exit_was_loss: bool = False   # 직전 청산이 손절이었는지
    last_exit_direction: int = 0       # 직전 포지션 방향 (+1 long, -1 short)
    consecutive_losses: int = 0
    unrealised_pnl: float = 0.0
    order_id: Optional[str] = None
    entry_reason: str = ""
    entry_fee: float = 0.0       # entry fee paid (USDT); pro-rated on partial exits

    @property
    def is_flat(self) -> bool:
        return self.state == PositionState.FLAT

    @property
    def is_long(self) -> bool:
        return self.state == PositionState.LONG

    @property
    def is_short(self) -> bool:
        return self.state == PositionState.SHORT


@dataclass
class ExitSignal:
    should_exit: bool
    reason: ExitReason = ExitReason.MANUAL
    partial: bool = False
    partial_pct: float = 0.0   # what fraction to close (0-1)
    message: str = ""


class StateMachine:
    def __init__(self, settings: Settings):
        self._s = settings
        self._positions: Dict[str, PositionInfo] = {}

    def get_position(self, symbol: str) -> PositionInfo:
        if symbol not in self._positions:
            self._positions[symbol] = PositionInfo(symbol=symbol)
        return self._positions[symbol]

    def all_positions(self) -> List[PositionInfo]:
        return [p for p in self._positions.values() if not p.is_flat]

    def open_count(self) -> int:
        return sum(1 for p in self._positions.values() if not p.is_flat)

    # ── Entry ────────────────────────────────────────────────────────────────

    def can_enter(self, symbol: str, direction: int = 0) -> bool:
        """
        Check cooldown and basic entry feasibility.

        Direction-aware cooldown (손절 후):
          - 동일 방향 재진입: REENTRY_SAME_DIR_COOLDOWN_SECONDS (45분)
          - 반대 방향 진입: REENTRY_COOLDOWN_SECONDS (30분)
          - 수익 청산 후: 기본 쿨다운 (30분)
        """
        pos = self.get_position(symbol)
        if not pos.is_flat:
            return False

        if pos.last_exit_ts > 0:
            elapsed = time.time() - pos.last_exit_ts
            # 손절 후 동일 방향: 더 긴 쿨다운 (같은 실수 반복 방지)
            if (pos.last_exit_was_loss
                    and direction != 0
                    and direction == pos.last_exit_direction):
                cooldown = self._s.REENTRY_SAME_DIR_COOLDOWN_SECONDS
            else:
                cooldown = self._s.REENTRY_COOLDOWN_SECONDS

            if elapsed < cooldown:
                remaining = cooldown - elapsed
                log.debug("Cooldown active for %s — %.0fs remaining", symbol, remaining)
                return False
        return True

    def enter(
        self,
        symbol: str,
        direction: int,   # +1 long, -1 short
        entry_price: float,
        qty: float,
        leverage: int,
        stop_price: float,
        take_profit_price: float,
        trailing_distance: float,
        order_id: Optional[str] = None,
        reason: str = "",
        tp1_price: float = 0.0,
        tp2_price: float = 0.0,
        trade_mode: str = "SCALP",
        mode_max_bars: int = 15,
        partial_exit_pct: float = 0.60,
        entry_fee: float = 0.0,
    ) -> bool:
        pos = self.get_position(symbol)
        if not pos.is_flat:
            log.warning("Cannot enter %s — already in %s", symbol, pos.state)
            return False

        new_state = PositionState.LONG if direction == 1 else PositionState.SHORT
        pos.state = new_state
        pos.entry_price = entry_price
        pos.qty = qty
        pos.leverage = leverage
        pos.stop_price = stop_price
        pos.take_profit_price = take_profit_price
        pos.tp1_price = tp1_price
        pos.tp2_price = tp2_price if tp2_price > 0 else take_profit_price
        pos.tp1_hit = False
        pos.trade_mode = trade_mode
        pos.mode_max_bars = mode_max_bars
        pos.partial_exit_pct = partial_exit_pct
        pos.trailing_stop_distance = trailing_distance
        pos.highest_price = entry_price
        pos.lowest_price = entry_price
        pos.entry_ts = time.time()
        pos.order_id = order_id
        pos.entry_reason = reason
        pos.unrealised_pnl = 0.0
        pos.entry_fee = entry_fee

        log.info(
            "Entered %s | state=%s mode=%s price=%.4f qty=%.4f lev=%d× "
            "stop=%.4f tp1=%.4f tp2=%.4f",
            symbol, new_state.value, trade_mode, entry_price, qty, leverage,
            stop_price, tp1_price, pos.tp2_price,
        )
        return True

    # ── Exit Check ───────────────────────────────────────────────────────────

    def check_exit(
        self,
        symbol: str,
        current_price: float,
        current_atr: float,
        signal_direction: int,    # current combined signal direction
        regime_flip: bool = False, # True if regime changed adversely
        bars_in_trade: int = 0,
        max_bars: int = 200,
        signal_strength: float = 1.0,  # current combined signal strength
        exit_threshold: float = 0.0,   # SIGNAL_EXIT_THRESHOLD from settings
    ) -> ExitSignal:
        pos = self.get_position(symbol)

        if pos.is_flat:
            return ExitSignal(should_exit=False)

        # Update trailing stop reference prices
        if pos.is_long:
            pos.highest_price = max(pos.highest_price, current_price)
            pos.unrealised_pnl = (current_price - pos.entry_price) * pos.qty
        else:
            pos.lowest_price = min(pos.lowest_price, current_price)
            pos.unrealised_pnl = (pos.entry_price - current_price) * pos.qty

        # ── 0. Partial TP1 check (before stop) ────────────────────────────────
        if not pos.tp1_hit and pos.tp1_price > 0:
            tp1_triggered = (
                (pos.is_long and current_price >= pos.tp1_price) or
                (pos.is_short and current_price <= pos.tp1_price)
            )
            if tp1_triggered:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.PARTIAL_TP1,
                    partial=True,
                    partial_pct=pos.partial_exit_pct,
                    message=f"TP1 hit at {current_price:.4f} (target {pos.tp1_price:.4f})",
                )

        # ── 1. Fixed stop loss ────────────────────────────────────────────────
        if pos.is_long and current_price <= pos.stop_price:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                message=f"Price {current_price:.4f} ≤ stop {pos.stop_price:.4f}",
            )
        if pos.is_short and current_price >= pos.stop_price:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                message=f"Price {current_price:.4f} ≥ stop {pos.stop_price:.4f}",
            )

        # ── 2. Take profit ────────────────────────────────────────────────────
        if pos.is_long and pos.take_profit_price > 0 and current_price >= pos.take_profit_price:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                message=f"Price {current_price:.4f} ≥ TP {pos.take_profit_price:.4f}",
            )
        if pos.is_short and pos.take_profit_price > 0 and current_price <= pos.take_profit_price:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                message=f"Price {current_price:.4f} ≤ TP {pos.take_profit_price:.4f}",
            )

        # ── 3. Trailing stop (only after TP1 partial exit) ────────────────────
        # Before TP1, the hard stop handles risk. Activating trailing before TP1
        # would create a tighter-than-intended stop that fires on normal noise.
        if pos.trailing_stop_distance > 0 and pos.tp1_hit:
            if pos.is_long:
                trailing_stop = pos.highest_price - pos.trailing_stop_distance
                if current_price <= trailing_stop and current_price > pos.stop_price:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.TRAILING_STOP,
                        message=f"Trailing stop hit: {current_price:.4f} ≤ {trailing_stop:.4f}",
                    )
            else:
                trailing_stop = pos.lowest_price + pos.trailing_stop_distance
                if current_price >= trailing_stop and current_price < pos.stop_price:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.TRAILING_STOP,
                        message=f"Trailing stop hit: {current_price:.4f} ≥ {trailing_stop:.4f}",
                    )

        # ── 4. Regime change / signal flip ────────────────────────────────────
        if regime_flip:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.REGIME_CHANGE,
                message="Regime changed adversely",
            )

        # Strong counter-signal → exit
        if pos.is_long and signal_direction == -1:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.SIGNAL_FLIP,
                message="Strong counter-signal against long",
            )
        if pos.is_short and signal_direction == 1:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.SIGNAL_FLIP,
                message="Strong counter-signal against short",
            )

        # ── 4b. Signal weakness exit (SIGNAL_EXIT_THRESHOLD) ─────────────────
        # If same direction but signal strength fell below exit threshold, exit
        if exit_threshold > 0 and signal_strength < exit_threshold:
            if (pos.is_long and signal_direction != -1) or (pos.is_short and signal_direction != 1):
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.SIGNAL_FLIP,
                    message=f"Signal strength {signal_strength:.2f} fell below exit threshold {exit_threshold:.2f}",
                )

        # ── 5. Time exit ──────────────────────────────────────────────────────
        effective_max_bars = pos.mode_max_bars if max_bars == 200 else max_bars
        if bars_in_trade >= effective_max_bars:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TIME_EXIT,
                message=f"Max bars ({effective_max_bars}) reached",
            )

        return ExitSignal(should_exit=False)

    # ── Partial Exit ──────────────────────────────────────────────────────────

    def partial_exit(
        self,
        symbol: str,
        qty_to_close: float,
        exit_price: float,
        partial_pct: float = 0.0,
    ) -> float:
        """
        Close `qty_to_close` units of the position at `exit_price`.
        Moves stop to breakeven, sets take_profit to tp2.
        Returns realised PnL for the closed portion.
        partial_pct: fraction of original position being closed (for entry_fee pro-rata).
        """
        pos = self.get_position(symbol)
        if pos.is_flat or qty_to_close <= 0:
            return 0.0

        if pos.is_long:
            pnl = (exit_price - pos.entry_price) * qty_to_close
        else:
            pnl = (pos.entry_price - exit_price) * qty_to_close

        # Reduce position size
        pos.qty = max(0.0, pos.qty - qty_to_close)

        # Pro-rate remaining entry fee (portion closed is no longer part of the position)
        if partial_pct > 0:
            pos.entry_fee = pos.entry_fee * (1.0 - partial_pct)

        # Move stop to breakeven, target to TP2
        pos.stop_price = pos.entry_price
        pos.take_profit_price = pos.tp2_price
        pos.tp1_hit = True

        log.info(
            "Partial exit %s | closed=%.4f remaining=%.4f pnl=%.4f "
            "stop_moved_to=%.4f tp2=%.4f",
            symbol, qty_to_close, pos.qty, pnl, pos.stop_price, pos.tp2_price,
        )
        return pnl

    # ── Exit ─────────────────────────────────────────────────────────────────

    def exit(self, symbol: str, exit_price: float) -> float:
        """Close position, return realised PnL."""
        pos = self.get_position(symbol)
        if pos.is_flat:
            return 0.0

        if pos.is_long:
            pnl = (exit_price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - exit_price) * pos.qty

        if pnl < 0:
            pos.consecutive_losses += 1
        else:
            pos.consecutive_losses = 0

        log.info(
            "Exited %s | state=%s entry=%.4f exit=%.4f qty=%.4f pnl=%.4f",
            symbol, pos.state.value, pos.entry_price, exit_price, pos.qty, pnl,
        )

        pos.last_exit_ts = time.time()
        pos.last_exit_was_loss = (pnl < 0)
        pos.last_exit_direction = 1 if pos.is_long else -1
        pos.state = PositionState.FLAT
        pos.entry_price = 0.0
        pos.qty = 0.0
        pos.leverage = 1
        pos.stop_price = 0.0
        pos.take_profit_price = 0.0
        pos.tp1_price = 0.0
        pos.tp2_price = 0.0
        pos.tp1_hit = False
        pos.trailing_stop_distance = 0.0
        pos.order_id = None
        pos.entry_fee = 0.0

        return pnl

    # ── Force Flat (Liquidation Recovery) ────────────────────────────────────

    def force_flat(self, symbol: str) -> None:
        """Force position to FLAT state — used when liquidation detected on exchange."""
        pos = self.get_position(symbol)
        if pos.is_flat:
            return
        log.warning("force_flat: %s state=%s — resetting to FLAT", symbol, pos.state.value)
        pos.last_exit_ts = time.time()
        pos.state = PositionState.FLAT
        pos.entry_price = 0.0
        pos.qty = 0.0
        pos.leverage = 1
        pos.stop_price = 0.0
        pos.take_profit_price = 0.0
        pos.tp1_price = 0.0
        pos.tp2_price = 0.0
        pos.tp1_hit = False
        pos.trailing_stop_distance = 0.0
        pos.order_id = None
        pos.entry_fee = 0.0

    # ── Sync with exchange ────────────────────────────────────────────────────

    def sync_from_exchange(self, symbol: str, exchange_positions: list) -> None:
        """
        Update local state from exchange data (called on restart).
        exchange_positions: list of position dicts from BinanceClient.get_positions()
        """
        pos = self.get_position(symbol)
        for ep in exchange_positions:
            if ep.get("symbol") != symbol:
                continue
            size = float(ep.get("size", 0))
            if size <= 0:
                continue
            side = ep.get("side", "")
            new_state = PositionState.LONG if side == "Buy" else PositionState.SHORT
            if pos.is_flat:
                pos.state = new_state
                # Exchange may return empty string "" for unset fields — use `or 0`
                pos.entry_price = float(ep.get("avgPrice", 0) or 0)
                pos.qty = size
                pos.leverage = int(float(ep.get("leverage", 1) or 1))
                pos.stop_price = float(ep.get("stopLoss", 0) or 0)
                pos.take_profit_price = float(ep.get("takeProfit", 0) or 0)
                log.info("State restored from exchange: %s %s qty=%.4f", symbol, new_state.value, size)
