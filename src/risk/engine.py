"""
Risk Engine — dynamic leverage and position sizing.

Design principles:
  1. 1-trade max loss is always bounded: equity × MAX_RISK_PER_TRADE_PCT
  2. Leverage is a derived quantity, not an input
  3. Multiple dampening factors prevent over-leveraging in adverse conditions
  4. Hard caps prevent edge-case blow-ups

Additional sizing methods (both used as soft caps):
  - AQR Volatility Targeting: scale position so daily P&L vol ≈ target
  - Quarter-Kelly: f* × 0.25 fraction of equity as stop-distance sizing cap

References:
  - "Volatility Targeting" (Moreira & Muir 2017, JF)
  - AQR "Managed Futures" research (publicly available)
  - Van K. Tharp "Trade Your Way to Financial Freedom" (Quarter-Kelly)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskParameters:
    """Output of the risk engine for a single trade opportunity."""
    leverage: int              # final leverage to set on exchange
    position_size_usdt: float  # notional value of the position
    margin_usdt: float         # required margin = notional / leverage
    qty_base: float            # quantity in base coin (position_size_usdt / price)
    stop_price: float          # recommended stop-loss price
    take_profit_price: float   # recommended take-profit price (TP2 after TP1 hit)
    tp1_price: float = 0.0     # first partial take-profit target
    tp2_price: float = 0.0     # second take-profit target (full close)
    trailing_stop_atr: float = 0.0  # ATR for trailing stop
    max_loss_usdt: float = 0.0      # max expected loss on this trade
    risk_ok: bool = True            # False if risk checks failed (no trade)
    reason: str = ""           # why risk_ok is False
    trade_mode: str = "SCALP"  # SCALP or TREND_RIDE
    partial_exit_pct: float = 0.60
    mode_max_bars: int = 15
    vol_target_size: float = 0.0   # AQR vol-targeting position size (informational)
    kelly_size: float = 0.0        # Quarter-Kelly position size cap (informational)


class RiskEngine:
    def __init__(self, settings: Settings):
        self._s = settings

    def calculate(
        self,
        equity: float,
        entry_price: float,
        side: str,              # "Buy" | "Sell"
        atr: float,
        signal_strength: float,
        spread_pct: float,
        funding_rate: float,
        vol_percentile: float,  # 0–1
        drawdown_pct: float,    # current drawdown as fraction
        consecutive_losses: int,
        symbol_max_leverage: int,
        symbol_info: dict,
        current_positions: int,
        trade_mode: Optional[str] = None,  # "SCALP" or "TREND_RIDE"
        recent_close: Optional[pd.Series] = None,  # for AQR vol-targeting
        win_stats: Optional[dict] = None,           # for quarter-Kelly cap
        bear_mode: bool = False,            # True = BEAR/SIDEWAYS macro regime
    ) -> RiskParameters:

        # ── Basic feasibility checks ─────────────────────────────────────────
        if equity <= 0:
            return self._no_trade("Equity <= 0")

        if current_positions >= self._s.MAX_CONCURRENT_POSITIONS:
            return self._no_trade(f"Max concurrent positions reached ({current_positions})")

        if atr <= 0 or entry_price <= 0:
            return self._no_trade("Invalid ATR or entry price")

        # ── Resolve trade mode and ATR multipliers ────────────────────────────
        resolved_mode = trade_mode or "SCALP"
        if resolved_mode == "TREND_RIDE":
            stop_atr_mult = self._s.TREND_STOP_ATR
            tp1_atr_mult = self._s.TREND_TP1_ATR
            tp2_atr_mult = self._s.TREND_TP2_ATR
            partial_exit_pct = self._s.TREND_PARTIAL_EXIT_PCT
            mode_max_bars = self._s.TREND_MAX_BARS
        else:
            stop_atr_mult = self._s.SCALP_STOP_ATR
            tp1_atr_mult = self._s.SCALP_TP1_ATR
            tp2_atr_mult = self._s.SCALP_TP2_ATR
            partial_exit_pct = self._s.SCALP_PARTIAL_EXIT_PCT
            mode_max_bars = self._s.SCALP_MAX_BARS

        # ── Step 1: Stop distance ────────────────────────────────────────────
        stop_distance = atr * stop_atr_mult
        stop_distance_pct = stop_distance / entry_price

        if stop_distance_pct <= 0:
            return self._no_trade("Stop distance zero")

        # ── Step 2: Base leverage from volatility targeting ───────────────────
        # Target: equity × risk_pct = loss if stop is hit
        # qty = equity × risk_pct / stop_distance
        # notional = qty × price
        # leverage = notional / margin = qty × price / (qty × price / leverage) → solve:
        # leverage = notional / (equity × risk_pct / stop_distance_pct)
        # → leverage = 1 / stop_distance_pct × (we scale later)
        #
        # Simpler: base_leverage = target_risk_pct / stop_distance_pct
        # This ensures 1 stop-hit = target_risk_pct of equity lost (at 1× leverage)
        # Then position_size = equity × risk_pct / (price × stop_distance_pct)
        target_risk = self._s.MAX_RISK_PER_TRADE_PCT
        base_leverage = target_risk / stop_distance_pct
        base_leverage = max(1.0, base_leverage)

        # ── Step 3: Dampening factors ─────────────────────────────────────────

        # (a) Volatility regime: reduce leverage when vol is high
        if vol_percentile > 0.80:
            vol_mod = 0.55
        elif vol_percentile > 0.65:
            vol_mod = 0.78
        elif vol_percentile > 0.50:
            vol_mod = 0.90
        else:
            vol_mod = 1.00

        # (b) Spread / liquidity: wide spread = poor execution
        if spread_pct > 0.005:
            liq_mod = 0.30
        elif spread_pct > 0.003:
            liq_mod = 0.60
        elif spread_pct > 0.001:
            liq_mod = 0.85
        else:
            liq_mod = 1.00

        # (c) Funding cost: high absolute funding → reduce leverage
        abs_funding = abs(funding_rate)
        if abs_funding > 0.001:
            fund_mod = 0.60
        elif abs_funding > 0.0005:
            fund_mod = 0.80
        else:
            fund_mod = 1.00

        # (d) Signal strength
        sig_mod = 0.50 + 0.50 * signal_strength  # 0.55 → 0.775, 1.0 → 1.0

        # (e) Account health (drawdown + consecutive losses)
        dd_mod = max(0.25, 1.0 - drawdown_pct * 3.0)
        consec_mod = max(0.25, 0.90 ** consecutive_losses)
        health_mod = min(dd_mod, consec_mod)

        # ── Step 4: Combined leverage ─────────────────────────────────────────
        raw_leverage = base_leverage * vol_mod * liq_mod * fund_mod * sig_mod * health_mod

        # Hard caps
        risk_engine_cap = self._s.BEAR_MAX_LEVERAGE_CAP if bear_mode else self._s.MAX_LEVERAGE_CAP
        exchange_cap = min(symbol_max_leverage, 100)

        # Additional safety: if drawdown > 15%, cap at 5×
        if drawdown_pct > 0.15:
            risk_engine_cap = min(risk_engine_cap, 5)
        # If drawdown > 10%, cap at 10×
        elif drawdown_pct > 0.10:
            risk_engine_cap = min(risk_engine_cap, 10)

        final_leverage = max(1, min(
            math.floor(raw_leverage),
            risk_engine_cap,
            exchange_cap,
        ))

        # ── Step 5: Position sizing ───────────────────────────────────────────
        # We want: loss if stop hit = equity × risk_pct
        # qty × stop_distance = equity × risk_pct
        # qty = equity × risk_pct / stop_distance (in base units)
        qty_base = equity * target_risk / stop_distance

        # ── AQR Volatility Targeting (Moreira & Muir 2017) ────────────────────
        # Scale position so daily P&L vol ≈ VOLATILITY_TARGET_PCT / sqrt(252)
        vol_target_qty: float = 0.0
        if recent_close is not None and len(recent_close) >= 20:
            log_rets = np.log(recent_close / recent_close.shift(1)).dropna()
            if len(log_rets) >= 20:
                daily_vol = float(log_rets.rolling(20).std().iloc[-1])
                if not pd.isna(daily_vol) and daily_vol > 0:
                    daily_target = self._s.VOLATILITY_TARGET_PCT / math.sqrt(252)
                    # qty × price × daily_vol ≈ equity × daily_target
                    vol_target_qty = equity * daily_target / (entry_price * daily_vol)
                    if vol_target_qty > 0:
                        qty_base = min(qty_base, vol_target_qty)

        # ── Quarter-Kelly position size cap (Van K. Tharp) ───────────────────
        kelly_qty: float = 0.0
        if win_stats is not None and win_stats.get("n_trades", 0) >= 30:
            wr = float(win_stats.get("win_rate", 0.5))
            avg_win = float(win_stats.get("avg_win", 1.0))
            avg_loss = float(win_stats.get("avg_loss", 1.0))
            if avg_loss > 0 and avg_win > 0:
                r_ratio = avg_win / avg_loss
                kelly_f = wr - (1.0 - wr) / r_ratio
                kelly_f = max(0.0, kelly_f)
                # Quarter-Kelly risk amount, converted to qty via stop distance
                quarter_kelly_risk = equity * kelly_f * self._s.USE_KELLY_FRACTION
                kelly_qty = quarter_kelly_risk / stop_distance
                if kelly_qty > 0:
                    qty_base = min(qty_base, kelly_qty)

        # Clamp to total exposure limit
        max_notional_per_trade = equity * self._s.MAX_SINGLE_EXPOSURE_PCT * final_leverage
        notional = qty_base * entry_price
        if notional > max_notional_per_trade:
            qty_base = max_notional_per_trade / entry_price
            notional = max_notional_per_trade

        margin = notional / final_leverage
        max_loss = qty_base * stop_distance  # = equity × risk_pct (by design)

        # ── Step 6: Stop / TP prices ──────────────────────────────────────────
        if side == "Buy":
            stop_price = entry_price - stop_distance
            tp1_price = entry_price + atr * tp1_atr_mult
            tp_price = entry_price + atr * tp2_atr_mult   # TP2 = full close target
        else:
            stop_price = entry_price + stop_distance
            tp1_price = entry_price - atr * tp1_atr_mult
            tp_price = entry_price - atr * tp2_atr_mult

        # ── Step 7: Minimum notional check ────────────────────────────────────
        min_notional = symbol_info.get("min_notional", 5.0)
        if notional < min_notional:
            return self._no_trade(
                f"Position notional {notional:.2f} < minimum {min_notional:.2f}"
            )

        # ── Step 7b: 수수료 손익분기점 체크 ──────────────────────────────────
        # TP1 도달 시 예상 수익이 수수료(진입 + 청산)의 1.5배 이상이어야 진입
        # round-trip: 진입 MAKER(0.020%) + TP1 청산 MAKER(0.020%) + 슬리피지×2
        round_trip_fee_pct = (
            self._s.BACKTEST_FEE_MAKER * 2 + self._s.BACKTEST_SLIPPAGE_PCT * 2
        )
        tp1_move_pct = (atr * tp1_atr_mult) / entry_price
        fee_safety_multiplier = 2.0  # TP1 수익이 수수료의 최소 2.0배
        # SCALP: ATR=0.107% → TP1=0.160% > fee×2.0=0.160% ✓ (최소 기준)
        # TREND: ATR=0.053% → TP1=0.160% > fee×2.0=0.160% ✓
        if tp1_move_pct < round_trip_fee_pct * fee_safety_multiplier:
            return self._no_trade(
                f"TP1 예상수익 {tp1_move_pct:.4%} < 수수료×{fee_safety_multiplier:.1f} "
                f"{round_trip_fee_pct * fee_safety_multiplier:.4%} — 수수료 손익분기 미달"
            )

        # ── Step 8: Liquidation distance check ────────────────────────────────
        # Require liquidation price to be at least 3× the stop distance away.
        # Binance USDT-perpetual maintenance margin rate ≈ 0.5% for most symbols.
        maintenance_margin_rate = 0.005
        if side == "Buy":
            liq_price = entry_price * (1.0 - 1.0 / final_leverage + maintenance_margin_rate)
            liq_distance = entry_price - liq_price
        else:
            liq_price = entry_price * (1.0 + 1.0 / final_leverage - maintenance_margin_rate)
            liq_distance = liq_price - entry_price

        min_liq_buffer = stop_distance * 3.0
        if liq_distance < min_liq_buffer:
            return self._no_trade(
                f"Liquidation too close: liq_dist={liq_distance:.4f} "
                f"< 3×stop={min_liq_buffer:.4f} (lev={final_leverage}× liq={liq_price:.4f})"
            )

        log.debug(
            "Risk calc | lev=%d× qty_base=%.4f notional=%.2f margin=%.2f "
            "stop=%.4f tp1=%.4f tp2=%.4f max_loss=%.2f mode=%s",
            final_leverage, qty_base, notional, margin,
            stop_price, tp1_price, tp_price, max_loss, resolved_mode,
        )

        return RiskParameters(
            leverage=final_leverage,
            position_size_usdt=notional,
            margin_usdt=margin,
            qty_base=qty_base,
            stop_price=stop_price,
            take_profit_price=tp_price,
            tp1_price=tp1_price,
            tp2_price=tp_price,
            trailing_stop_atr=atr * self._s.TRAILING_STOP_ATR_MULTIPLIER,
            max_loss_usdt=max_loss,
            risk_ok=True,
            trade_mode=resolved_mode,
            partial_exit_pct=partial_exit_pct,
            mode_max_bars=mode_max_bars,
            vol_target_size=vol_target_qty * entry_price if vol_target_qty > 0 else 0.0,
            kelly_size=kelly_qty * entry_price if kelly_qty > 0 else 0.0,
        )

    @staticmethod
    def _no_trade(reason: str) -> RiskParameters:
        return RiskParameters(
            leverage=1,
            position_size_usdt=0,
            margin_usdt=0,
            qty_base=0,
            stop_price=0,
            take_profit_price=0,
            tp1_price=0,
            tp2_price=0,
            trailing_stop_atr=0,
            max_loss_usdt=0,
            risk_ok=False,
            reason=reason,
            vol_target_size=0.0,
            kelly_size=0.0,
        )
