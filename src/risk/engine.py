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
        bear_mode: bool = False,  # True = BEAR/SIDEWAYS → BEAR_MAX_LEVERAGE_CAP 사용
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
        elif resolved_mode == "PURE_SCALP":
            stop_atr_mult = self._s.SCALP_PURE_STOP_ATR
            tp1_atr_mult = self._s.SCALP_PURE_TP1_ATR
            tp2_atr_mult = self._s.SCALP_PURE_TP2_ATR
            partial_exit_pct = self._s.SCALP_PURE_PARTIAL_EXIT_PCT
            mode_max_bars = self._s.SCALP_PURE_MAX_BARS
        else:
            stop_atr_mult = self._s.SCALP_STOP_ATR
            tp1_atr_mult = self._s.SCALP_TP1_ATR
            tp2_atr_mult = self._s.SCALP_TP2_ATR
            partial_exit_pct = self._s.SCALP_PARTIAL_EXIT_PCT
            mode_max_bars = self._s.SCALP_MAX_BARS

        # ── Signal-driven risk escalation ─────────────────────────────────────
        # 드로우다운이 높을수록 에스컬레이션 자동 억제 (안전 장치)
        dd_escalation_allowed = drawdown_pct <= 0.15

        target_risk = self._s.MAX_RISK_PER_TRADE_PCT
        if signal_strength >= 0.85 and dd_escalation_allowed:
            stop_atr_scale = 0.68
            trade_risk_pct = min(0.12, target_risk * 6.0)
        elif signal_strength >= 0.80 and dd_escalation_allowed:
            stop_atr_scale = 0.73
            trade_risk_pct = min(0.10, target_risk * 5.0)
        elif signal_strength >= 0.75 and dd_escalation_allowed:
            stop_atr_scale = 0.80
            trade_risk_pct = min(0.07, target_risk * 3.5)
        elif signal_strength >= 0.65:
            stop_atr_scale = 0.88
            trade_risk_pct = min(0.05, target_risk * 2.5)
        else:
            stop_atr_scale = 1.00
            trade_risk_pct = min(0.03, target_risk * 1.5)

        # ── Step 1: Stop distance (signal-scaled + vol-adjusted + minimum floor) ──
        # 변동성 적응형 스탑: ATR이 작을수록 배수 확대 (노이즈 대비)
        atr_pct = atr / entry_price if entry_price > 0 else 0.01
        if atr_pct < 0.003:
            vol_stop_mult = 2.0
        elif atr_pct < 0.006:
            vol_stop_mult = 1.5
        elif atr_pct < 0.012:
            vol_stop_mult = 1.0
        else:
            vol_stop_mult = 0.85

        stop_distance = atr * stop_atr_mult * stop_atr_scale * vol_stop_mult
        # 수수료 체크용 원본 TP1 저장 (vol 스케일 미적용)
        raw_tp1_atr_mult = tp1_atr_mult
        # TP도 동일한 vol_stop_mult 적용 → R:R 비율 유지
        tp1_atr_mult *= vol_stop_mult
        tp2_atr_mult *= vol_stop_mult

        # 모드별 최소 스탑 거리 플로어
        _min_stop_pct_map = {
            "PURE_SCALP": self._s.MIN_STOP_PCT_PURE_SCALP,
            "SCALP":      self._s.MIN_STOP_PCT_SCALP,
            "TREND_RIDE": self._s.MIN_STOP_PCT_TREND_RIDE,
        }
        min_stop_pct = _min_stop_pct_map.get(resolved_mode, self._s.MIN_STOP_PCT_SCALP)
        min_stop_dist = entry_price * min_stop_pct
        if stop_distance < min_stop_dist:
            tp_scale = min_stop_dist / stop_distance
            stop_distance = min_stop_dist
        else:
            tp_scale = 1.0

        stop_distance_pct = stop_distance / entry_price

        if stop_distance_pct <= 0:
            return self._no_trade("Stop distance zero")

        # ── Step 2: Base leverage ─────────────────────────────────────────────
        base_leverage = trade_risk_pct / stop_distance_pct
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

        # (d) Signal quality gate
        sig_mod = 0.65 + 0.35 * min(signal_strength, 1.0)

        # (e) Account health (drawdown + consecutive losses)
        dd_mod = max(0.25, 1.0 - drawdown_pct * 3.0)
        consec_mod = max(0.25, 0.90 ** consecutive_losses)
        health_mod = min(dd_mod, consec_mod)

        # ── Step 4: Combined leverage ─────────────────────────────────────────
        raw_leverage = base_leverage * vol_mod * liq_mod * fund_mod * sig_mod * health_mod

        # Hard caps
        risk_engine_cap = self._s.BEAR_MAX_LEVERAGE_CAP if bear_mode else self._s.MAX_LEVERAGE_CAP
        exchange_cap = min(symbol_max_leverage, 100)

        # Additional safety: 드로우다운 비례 캡
        if drawdown_pct > 0.20:
            risk_engine_cap = min(risk_engine_cap, max(5, risk_engine_cap // 4))
        elif drawdown_pct > 0.12:
            risk_engine_cap = min(risk_engine_cap, max(10, risk_engine_cap // 2))

        final_leverage = max(1, min(
            math.floor(raw_leverage),
            risk_engine_cap,
            exchange_cap,
        ))

        # ── Step 5: Position sizing ───────────────────────────────────────────
        qty_base = equity * trade_risk_pct / stop_distance

        # ── AQR Volatility Targeting (Moreira & Muir 2017) ────────────────────
        vol_target_qty: float = 0.0
        if recent_close is not None and len(recent_close) >= 20:
            log_rets = np.log(recent_close / recent_close.shift(1)).dropna()
            if len(log_rets) >= 20:
                daily_vol = float(log_rets.rolling(20).std().iloc[-1])
                if not pd.isna(daily_vol) and daily_vol > 0:
                    daily_target = self._s.VOLATILITY_TARGET_PCT / math.sqrt(252)
                    vol_target_qty = equity * daily_target / (entry_price * daily_vol)
                    if vol_target_qty > 0:
                        qty_base = min(qty_base, vol_target_qty)

        # ── Adaptive Half-Kelly position sizing ────────────────────────────────
        kelly_qty: float = 0.0
        kelly_min_trades = self._s.KELLY_MIN_TRADES
        if win_stats is not None and win_stats.get("n_trades", 0) >= kelly_min_trades:
            wr = float(win_stats.get("win_rate", 0.5))
            avg_win = float(win_stats.get("avg_win", 1.0))
            avg_loss = float(win_stats.get("avg_loss", 1.0))
            if avg_loss > 0 and avg_win > 0:
                r_ratio = avg_win / avg_loss
                kelly_f = wr - (1.0 - wr) / r_ratio
                kelly_f = max(0.0, kelly_f)
                effective_kelly = kelly_f * self._s.USE_KELLY_FRACTION * signal_strength
                kelly_risk = equity * min(effective_kelly, 0.10)
                kelly_qty = kelly_risk / stop_distance
                if kelly_qty > 0:
                    qty_base = min(qty_base, kelly_qty)
                    log.debug(
                        "Kelly sizing | wr=%.2f r_ratio=%.2f kelly_f=%.3f "
                        "effective=%.3f risk=%.2f qty=%.4f",
                        wr, r_ratio, kelly_f, effective_kelly, kelly_risk, kelly_qty,
                    )

        # Clamp to total exposure limit
        max_notional_per_trade = equity * self._s.MAX_SINGLE_EXPOSURE_PCT * final_leverage
        notional = qty_base * entry_price
        if notional > max_notional_per_trade:
            qty_base = max_notional_per_trade / entry_price
            notional = max_notional_per_trade

        margin = notional / final_leverage
        max_loss = qty_base * stop_distance

        # ── Step 6: Stop / TP prices ──────────────────────────────────────────
        if side == "Buy":
            stop_price = entry_price - stop_distance
            tp1_price = entry_price + atr * tp1_atr_mult * tp_scale
            tp_price = entry_price + atr * tp2_atr_mult * tp_scale
        else:
            stop_price = entry_price + stop_distance
            tp1_price = entry_price - atr * tp1_atr_mult * tp_scale
            tp_price = entry_price - atr * tp2_atr_mult * tp_scale

        # ── Step 7: Minimum notional check ────────────────────────────────────
        min_notional = symbol_info.get("min_notional", 5.0)
        if notional < min_notional:
            return self._no_trade(
                f"Position notional {notional:.2f} < minimum {min_notional:.2f}"
            )

        # ── Step 7b: 수수료 손익분기점 체크 ──────────────────────────────────
        entry_is_market = (
            spread_pct > self._s.SPREAD_TAKER_THRESHOLD
            or signal_strength > self._s.SIGNAL_URGENT_THRESHOLD
        )
        entry_fee_rate = (
            self._s.BACKTEST_FEE_TAKER if entry_is_market else self._s.BACKTEST_FEE_MAKER
        )
        round_trip_fee_pct = (
            entry_fee_rate + self._s.BACKTEST_FEE_MAKER + self._s.BACKTEST_SLIPPAGE_PCT * 2
        )
        tp1_move_pct = (atr * raw_tp1_atr_mult) / entry_price

        # 모드별 수수료 안전 배수
        if resolved_mode == "PURE_SCALP":
            fee_safety_multiplier = 1.5
        else:
            fee_safety_multiplier = 2.0

        if tp1_move_pct < round_trip_fee_pct * fee_safety_multiplier:
            entry_type = "Taker(시장가)" if entry_is_market else "Maker(PostOnly)"
            return self._no_trade(
                f"TP1 예상수익 {tp1_move_pct:.4%} < 수수료×{fee_safety_multiplier:.1f} "
                f"{round_trip_fee_pct * fee_safety_multiplier:.4%} "
                f"[진입={entry_type}, 모드={resolved_mode}] — 수수료 손익분기 미달"
            )

        # ── Step 8: Liquidation distance check ────────────────────────────────
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
