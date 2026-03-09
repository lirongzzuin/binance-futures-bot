"""
Multi-signal generator.

Produces a SignalBundle for each bar containing individual signals
from 5 distinct categories:
  1. Trend / Momentum  (EMA cross, MACD, RSI trend)
  2. Mean Reversion    (RSI extremes, Bollinger)
  3. Volatility        (ATR trend, BB squeeze)
  4. Volume / OI / Funding  (volume spike, OI divergence, funding bias)
  5. ICT               (BOS, CHoCH, FVG, sweep, premium/discount)

Each signal returns: -1 (bearish), 0 (neutral), +1 (bullish)
and a confidence weight (0.0–1.0).

Design principle: signals are independent — the combiner applies
regime-based weights and produces the final directional score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Settings
from src.strategy.indicators import (
    adx,
    atr,
    bollinger_bands,
    cmf,
    ema,
    hull_ma,
    keltner_channel,
    macd,
    obv,
    rsi,
    rsi_divergence,
    squeeze_momentum,
    stoch_rsi,
    volume_zscore,
    volatility_percentile,
    volume_ratio,
    williams_r,
    vwap as compute_vwap,
)
from src.strategy.ict import ICTResult, StructureBias


# ── Signal definition ─────────────────────────────────────────────────────────

@dataclass
class Signal:
    name: str
    direction: int   # -1, 0, +1
    confidence: float  # 0.0 – 1.0
    category: str    # trend|reversion|volatility|volume|ict


@dataclass
class SignalBundle:
    signals: List[Signal] = field(default_factory=list)
    last_price: float = 0.0
    atr_val: float = 0.0
    stop_distance: float = 0.0   # recommended stop distance (price units)

    def by_category(self, cat: str) -> List[Signal]:
        return [s for s in self.signals if s.category == cat]

    def net_direction(self) -> float:
        """Weighted average direction across all signals."""
        if not self.signals:
            return 0.0
        total_w = sum(s.confidence for s in self.signals)
        if total_w == 0:
            return 0.0
        return sum(s.direction * s.confidence for s in self.signals) / total_w


# ── Signal Generator ──────────────────────────────────────────────────────────

class SignalGenerator:
    def __init__(self, settings: Settings):
        self._s = settings

    def generate(
        self,
        df: pd.DataFrame,
        ict_result: ICTResult,
        funding_rate: float = 0.0,
        oi_change_pct: float = 0.0,
        spread_pct: float = 0.0,
        vwap: float = 0.0,
        vwap_distance_pct: float = 0.0,
        orderbook_imbalance: float = 0.0,
        htf_df: Optional[pd.DataFrame] = None,  # 일봉 데이터 (HTF 추세 필터)
    ) -> SignalBundle:
        bundle = SignalBundle()

        if len(df) < self._s.EMA_SLOW + 5:
            return bundle

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ATR
        atr_series = atr(high, low, close, self._s.ATR_PERIOD)
        atr_val = float(atr_series.iloc[-1])
        bundle.last_price = float(close.iloc[-1])
        bundle.atr_val = atr_val
        bundle.stop_distance = atr_val * self._s.STOP_ATR_MULTIPLIER

        # ── Category 1: Trend / Momentum ─────────────────────────────────────
        bundle.signals += self._trend_signals(close, high, low)

        # ── Category 2: Mean Reversion ────────────────────────────────────────
        bundle.signals += self._reversion_signals(close)

        # ── Category 3: Volatility ────────────────────────────────────────────
        bundle.signals += self._volatility_signals(close, high, low, atr_series)

        # ── Category 4: Volume / OI / Funding ────────────────────────────────
        bundle.signals += self._volume_signals(close, volume, oi_change_pct, funding_rate)

        # ── Category 5: ICT ───────────────────────────────────────────────────
        bundle.signals += self._ict_signals(ict_result, bundle.last_price)

        # ── Category 6: VWAP ─────────────────────────────────────────────────
        bundle.signals += self._vwap_signal(close, vwap, vwap_distance_pct)

        # ── Category 7: Order Book ────────────────────────────────────────────
        bundle.signals += self._orderbook_signal(orderbook_imbalance)

        # ── Category 8: Keltner Channel ───────────────────────────────────────
        bundle.signals += self._keltner_signal(close, high, low)

        # ── Category 9: Hull MA Direction ─────────────────────────────────────
        bundle.signals += self._hma_signal(close)

        # ── Category 10: Stochastic RSI ───────────────────────────────────────
        bundle.signals += self._stochrsi_signal(close)

        # ── Category 11: Williams %R ──────────────────────────────────────────
        bundle.signals += self._williams_r_signal(close, high, low)

        # ── Category 12: Chaikin Money Flow ──────────────────────────────────
        bundle.signals += self._cmf_signal(close, high, low, volume)

        # ── Category 13: RSI Divergence ───────────────────────────────────────
        bundle.signals += self._rsi_divergence_signal(close)

        # ── Category 14: OBV Trend ────────────────────────────────────────────
        bundle.signals += self._obv_signal(close, volume)

        # ── Category 15: Squeeze Momentum ─────────────────────────────────────
        bundle.signals += self._squeeze_signal(close, high, low)

        # ── Category 16: HTF Trend Filter (일봉 추세 방향) ────────────────────
        # 가장 중요한 필터 — 일봉 추세 반대 방향 진입 차단
        if htf_df is not None and len(htf_df) >= 50:
            bundle.signals += self._htf_trend_signals(htf_df)

        return bundle

    # ── Trend / Momentum ──────────────────────────────────────────────────────

    def _trend_signals(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[Signal]:
        signals = []

        # EMA cross (fast vs mid)
        ef = ema(close, self._s.EMA_FAST)
        em = ema(close, self._s.EMA_MID)
        es = ema(close, self._s.EMA_SLOW)

        ef_now, ef_prev = float(ef.iloc[-1]), float(ef.iloc[-2])
        em_now, em_prev = float(em.iloc[-1]), float(em.iloc[-2])
        es_now = float(es.iloc[-1])
        price_now = float(close.iloc[-1])

        # Fast crosses mid
        cross_dir = 0
        if ef_prev <= em_prev and ef_now > em_now:
            cross_dir = 1
        elif ef_prev >= em_prev and ef_now < em_now:
            cross_dir = -1

        # Full EMA alignment
        if ef_now > em_now > es_now and price_now > ef_now:
            trend_dir = 1
        elif ef_now < em_now < es_now and price_now < ef_now:
            trend_dir = -1
        else:
            trend_dir = 0

        signals.append(Signal(
            name="ema_cross",
            direction=cross_dir,
            confidence=0.8 if cross_dir != 0 else 0.4,
            category="trend",
        ))
        signals.append(Signal(
            name="ema_alignment",
            direction=trend_dir,
            confidence=0.9 if trend_dir != 0 else 0.3,
            category="trend",
        ))

        # MACD
        macd_df = macd(close)
        hist_now = float(macd_df["histogram"].iloc[-1])
        hist_prev = float(macd_df["histogram"].iloc[-2])
        macd_dir = 0
        if hist_prev <= 0 < hist_now:
            macd_dir = 1
        elif hist_prev >= 0 > hist_now:
            macd_dir = -1
        elif hist_now > 0:
            macd_dir = 1
        else:
            macd_dir = -1

        signals.append(Signal(
            name="macd",
            direction=macd_dir,
            confidence=0.6 if abs(hist_now) > abs(hist_prev) else 0.35,
            category="trend",
        ))

        # RSI momentum (trend mode: >50 bullish, <50 bearish)
        rsi_s = rsi(close, self._s.RSI_PERIOD)
        rsi_val = float(rsi_s.iloc[-1])
        if pd.isna(rsi_val):
            rsi_val = 50.0
        rsi_dir = 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
        signals.append(Signal(
            name="rsi_momentum",
            direction=rsi_dir,
            confidence=min(1.0, abs(rsi_val - 50) / 50),
            category="trend",
        ))

        return signals

    # ── Mean Reversion ────────────────────────────────────────────────────────

    def _reversion_signals(self, close: pd.Series) -> List[Signal]:
        signals = []

        rsi_s = rsi(close, self._s.RSI_PERIOD)
        rsi_val = float(rsi_s.iloc[-1])

        # RSI extreme (only fire at real extremes)
        if rsi_val < 30:
            dir_, conf = 1, (30 - rsi_val) / 30
        elif rsi_val > 70:
            dir_, conf = -1, (rsi_val - 70) / 30
        else:
            dir_, conf = 0, 0.0
        signals.append(Signal(name="rsi_extreme", direction=dir_, confidence=conf, category="reversion"))

        # Bollinger band
        bb = bollinger_bands(close, self._s.BB_PERIOD, self._s.BB_STD)
        pct_b = float(bb["pct_b"].iloc[-1]) if not pd.isna(bb["pct_b"].iloc[-1]) else 0.5

        if pct_b < 0.05:
            bb_dir, bb_conf = 1, min(1.0, (0.05 - pct_b) * 20)
        elif pct_b > 0.95:
            bb_dir, bb_conf = -1, min(1.0, (pct_b - 0.95) * 20)
        else:
            bb_dir, bb_conf = 0, 0.0
        signals.append(Signal(name="bb_extreme", direction=bb_dir, confidence=bb_conf, category="reversion"))

        return signals

    # ── Volatility ────────────────────────────────────────────────────────────

    def _volatility_signals(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        atr_series: pd.Series,
    ) -> List[Signal]:
        signals = []

        # ATR expansion (trend momentum confirmation)
        atr_now = float(atr_series.iloc[-1])
        atr_avg = float(atr_series.rolling(20).mean().iloc[-1]) if len(atr_series) >= 20 else atr_now
        atr_expanding = atr_now > atr_avg * 1.2

        # Direction comes from price vs EMA slow
        es = ema(close, self._s.EMA_SLOW)
        price_vs_ema = 1 if float(close.iloc[-1]) > float(es.iloc[-1]) else -1

        if atr_expanding:
            signals.append(Signal(
                name="atr_expansion",
                direction=price_vs_ema,
                confidence=min(1.0, atr_now / atr_avg - 1.0),
                category="volatility",
            ))
        else:
            signals.append(Signal(name="atr_expansion", direction=0, confidence=0.2, category="volatility"))

        # Volatility percentile (high vol = reduce confidence, not direction)
        vol_pct = volatility_percentile(close, short=20, long=100)
        vp = float(vol_pct.iloc[-1]) if not pd.isna(vol_pct.iloc[-1]) else 0.5
        # High vol → neutral (preserve direction but lower confidence)
        signals.append(Signal(
            name="vol_regime",
            direction=0,
            confidence=max(0.0, 1.0 - vp),   # low vol = higher confidence in other signals
            category="volatility",
        ))

        return signals

    # ── Volume / OI / Funding ─────────────────────────────────────────────────

    def _volume_signals(
        self,
        close: pd.Series,
        volume: pd.Series,
        oi_change_pct: float,
        funding_rate: float,
    ) -> List[Signal]:
        signals = []

        # Volume Z-score (>2 = spike)
        vol_z = volume_zscore(volume, 20)
        vz = float(vol_z.iloc[-1]) if not pd.isna(vol_z.iloc[-1]) else 0.0
        price_change = float(close.pct_change().iloc[-1]) if len(close) >= 2 else 0.0

        vol_dir = 0
        vol_conf = 0.0
        if vz > 2.0:
            vol_dir = 1 if price_change > 0 else -1
            vol_conf = min(1.0, vz / 4.0)
        signals.append(Signal(name="volume_spike", direction=vol_dir, confidence=vol_conf, category="volume"))

        # OI change
        # OI rising + price rising → trend continuation (bullish)
        # OI rising + price falling → trend continuation (bearish)
        # OI falling + price rising → weakening rally (slight bearish)
        if abs(oi_change_pct) > 0.01:
            if oi_change_pct > 0 and price_change > 0:
                oi_dir, oi_conf = 1, min(1.0, oi_change_pct * 10)
            elif oi_change_pct > 0 and price_change < 0:
                oi_dir, oi_conf = -1, min(1.0, oi_change_pct * 10)
            elif oi_change_pct < 0 and price_change > 0:
                oi_dir, oi_conf = -1, min(1.0, abs(oi_change_pct) * 5)  # divergence = weak
            else:
                oi_dir, oi_conf = 1, min(1.0, abs(oi_change_pct) * 5)
        else:
            oi_dir, oi_conf = 0, 0.0
        signals.append(Signal(name="oi_flow", direction=oi_dir, confidence=oi_conf, category="volume"))

        # Funding rate bias
        # High positive funding → market crowded long → slight bearish bias
        # High negative funding → market crowded short → slight bullish bias
        fr_threshold = 0.0003  # 0.03% per 8h is significant
        if funding_rate > fr_threshold:
            fr_dir = -1
            fr_conf = min(1.0, funding_rate / 0.001)
        elif funding_rate < -fr_threshold:
            fr_dir = 1
            fr_conf = min(1.0, abs(funding_rate) / 0.001)
        else:
            fr_dir = 0
            fr_conf = 0.0
        signals.append(Signal(name="funding_bias", direction=fr_dir, confidence=fr_conf, category="volume"))

        return signals

    # ── VWAP Signals ──────────────────────────────────────────────────────────

    def _vwap_signal(
        self,
        close: pd.Series,
        vwap_val: float,
        vwap_distance_pct: float,
    ) -> List[Signal]:
        signals = []

        if vwap_val <= 0:
            return signals

        current_price = float(close.iloc[-1])

        # vwap_distance: bullish if price above VWAP, bearish below
        dist = abs(vwap_distance_pct)
        if vwap_distance_pct > 0:
            dir_ = 1
        elif vwap_distance_pct < 0:
            dir_ = -1
        else:
            dir_ = 0

        conf = min(1.0, dist * 20)  # 5% distance → full confidence
        signals.append(Signal(
            name="vwap_distance",
            direction=dir_,
            confidence=conf,
            category="vwap",
        ))

        # vwap_cross: price crossed VWAP this bar
        if len(close) >= 2:
            prev_price = float(close.iloc[-2])
            cross_dir = 0
            if prev_price < vwap_val <= current_price:
                cross_dir = 1    # bullish cross above VWAP
            elif prev_price > vwap_val >= current_price:
                cross_dir = -1   # bearish cross below VWAP

            if cross_dir != 0:
                signals.append(Signal(
                    name="vwap_cross",
                    direction=cross_dir,
                    confidence=0.75,
                    category="vwap",
                ))

        # short_momentum: 3-bar EMA vs VWAP alignment
        if len(close) >= 6:
            ema3 = float(ema(close, 3).iloc[-1])
            if ema3 > vwap_val and current_price > vwap_val:
                mom_dir = 1
                mom_conf = min(1.0, (ema3 - vwap_val) / vwap_val * 50)
            elif ema3 < vwap_val and current_price < vwap_val:
                mom_dir = -1
                mom_conf = min(1.0, (vwap_val - ema3) / vwap_val * 50)
            else:
                mom_dir = 0
                mom_conf = 0.1

            signals.append(Signal(
                name="vwap_momentum",
                direction=mom_dir,
                confidence=mom_conf,
                category="vwap",
            ))

        return signals

    # ── Order Book Signal ─────────────────────────────────────────────────────

    def _orderbook_signal(self, imbalance: float) -> List[Signal]:
        signals = []

        if imbalance > 0.15:
            direction = 1
            conf = min(1.0, (imbalance - 0.15) * 4 + 0.3)
        elif imbalance < -0.15:
            direction = -1
            conf = min(1.0, (abs(imbalance) - 0.15) * 4 + 0.3)
        else:
            direction = 0
            conf = 0.0

        signals.append(Signal(
            name="orderbook_pressure",
            direction=direction,
            confidence=conf,
            category="orderbook",
        ))

        return signals

    # ── Keltner Channel Signal ────────────────────────────────────────────────

    def _keltner_signal(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
    ) -> List[Signal]:
        signals = []
        try:
            kc = keltner_channel(
                close, high, low,
                self._s.KELTNER_PERIOD,
                self._s.KELTNER_ATR_MULT,
            )
            upper = float(kc["upper"].iloc[-1])
            lower = float(kc["lower"].iloc[-1])
            mid = float(kc["mid"].iloc[-1])
            price = float(close.iloc[-1])
            pct_k_raw = kc["pct_k"].iloc[-1]
            pct_k = float(pct_k_raw) if not pd.isna(pct_k_raw) else 0.5

            channel_width = upper - lower
            if channel_width <= 0 or pd.isna(upper):
                return signals

            # Breakout above/below channel → strong momentum
            if price > upper:
                dir_ = 1
                conf = min(1.0, 0.6 + (price - upper) / channel_width)
            elif price < lower:
                dir_ = -1
                conf = min(1.0, 0.6 + (lower - price) / channel_width)
            elif pct_k > 0.70:
                dir_ = 1
                conf = (pct_k - 0.70) * 2.0
            elif pct_k < 0.30:
                dir_ = -1
                conf = (0.30 - pct_k) * 2.0
            else:
                dir_ = 0
                conf = 0.1

            signals.append(Signal(
                name="keltner_breakout",
                direction=dir_,
                confidence=min(1.0, max(0.0, conf)),
                category="trend",
            ))
        except Exception:
            pass
        return signals

    # ── Hull MA Direction Signal ──────────────────────────────────────────────

    def _hma_signal(self, close: pd.Series) -> List[Signal]:
        signals = []
        try:
            hma_series = hull_ma(close, self._s.HULL_MA_PERIOD)
            valid = hma_series.dropna()
            if len(valid) < 3:
                return signals

            hma_now = float(valid.iloc[-1])
            hma_prev = float(valid.iloc[-2])
            price = float(close.iloc[-1])

            slope = (hma_now - hma_prev) / (abs(hma_prev) + 1e-9)

            # Direction from slope
            slope_threshold = 0.0003  # 0.03% per bar
            if slope > slope_threshold:
                dir_ = 1
                conf = min(1.0, slope / 0.002)
            elif slope < -slope_threshold:
                dir_ = -1
                conf = min(1.0, abs(slope) / 0.002)
            else:
                dir_ = 0
                conf = 0.1

            # Strengthen when price is on the right side of HMA
            if dir_ == 1 and price > hma_now:
                conf = min(1.0, conf + 0.15)
            elif dir_ == -1 and price < hma_now:
                conf = min(1.0, conf + 0.15)

            signals.append(Signal(
                name="hma_direction",
                direction=dir_,
                confidence=min(1.0, max(0.0, conf)),
                category="trend",
            ))
        except Exception:
            pass
        return signals

    # ── Stochastic RSI Signal ─────────────────────────────────────────────────

    def _stochrsi_signal(self, close: pd.Series) -> List[Signal]:
        """
        Stochastic RSI: Enter on oversold/overbought extremes with K/D crossover.
        K crosses D from below in oversold zone (K<20) → bullish.
        K crosses D from above in overbought zone (K>80) → bearish.
        """
        signals = []
        try:
            srsi = stoch_rsi(
                close,
                rsi_period=self._s.STOCHRSI_RSI_PERIOD,
                stoch_period=self._s.STOCHRSI_STOCH_PERIOD,
                smooth_k=self._s.STOCHRSI_SMOOTH_K,
                smooth_d=self._s.STOCHRSI_SMOOTH_D,
            )
            if len(srsi.dropna()) < 3:
                return signals

            k_now = float(srsi["k"].iloc[-1])
            k_prev = float(srsi["k"].iloc[-2])
            d_now = float(srsi["d"].iloc[-1])
            d_prev = float(srsi["d"].iloc[-2])

            if pd.isna(k_now) or pd.isna(d_now):
                return signals

            # Bullish: K crosses above D in oversold zone (< 30)
            if k_prev <= d_prev and k_now > d_now and k_now < 30:
                conf = min(1.0, (30 - k_now) / 30 + 0.3)
                signals.append(Signal(
                    name="stochrsi_bullish",
                    direction=1,
                    confidence=conf,
                    category="reversion",
                ))
            # Bearish: K crosses below D in overbought zone (> 70)
            elif k_prev >= d_prev and k_now < d_now and k_now > 70:
                conf = min(1.0, (k_now - 70) / 30 + 0.3)
                signals.append(Signal(
                    name="stochrsi_bearish",
                    direction=-1,
                    confidence=conf,
                    category="reversion",
                ))
            # Strong momentum continuation: K and D both > 70 (trending up)
            elif k_now > 70 and d_now > 70:
                signals.append(Signal(
                    name="stochrsi_momentum_up",
                    direction=1,
                    confidence=min(1.0, (k_now - 70) / 30),
                    category="momentum",
                ))
            # Strong momentum continuation: K and D both < 30 (trending down)
            elif k_now < 30 and d_now < 30:
                signals.append(Signal(
                    name="stochrsi_momentum_down",
                    direction=-1,
                    confidence=min(1.0, (30 - k_now) / 30),
                    category="momentum",
                ))
            else:
                signals.append(Signal(
                    name="stochrsi_neutral",
                    direction=0,
                    confidence=0.1,
                    category="reversion",
                ))
        except Exception:
            pass
        return signals

    # ── Williams %R Signal ────────────────────────────────────────────────────

    def _williams_r_signal(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> List[Signal]:
        """
        Williams %R: Detect overbought/oversold conditions.
        Below -80 = oversold → bullish reversal signal.
        Above -20 = overbought → bearish reversal signal.
        """
        signals = []
        try:
            wr = williams_r(high, low, close, self._s.WILLIAMS_R_PERIOD)
            if wr.empty or pd.isna(wr.iloc[-1]):
                return signals

            wr_now = float(wr.iloc[-1])
            wr_prev = float(wr.iloc[-2]) if len(wr) >= 2 else wr_now

            if wr_now < -80:
                # Deeply oversold — bullish
                conf = min(1.0, (-80 - wr_now) / 20 + 0.4)
                signals.append(Signal(
                    name="williams_r_oversold",
                    direction=1,
                    confidence=conf,
                    category="reversion",
                ))
            elif wr_now > -20:
                # Deeply overbought — bearish
                conf = min(1.0, (wr_now + 20) / 20 + 0.4)
                signals.append(Signal(
                    name="williams_r_overbought",
                    direction=-1,
                    confidence=conf,
                    category="reversion",
                ))
            elif wr_prev < -80 and wr_now >= -80:
                # Exiting oversold — bullish momentum
                signals.append(Signal(
                    name="williams_r_exit_oversold",
                    direction=1,
                    confidence=0.65,
                    category="trend",
                ))
            elif wr_prev > -20 and wr_now <= -20:
                # Exiting overbought — bearish momentum
                signals.append(Signal(
                    name="williams_r_exit_overbought",
                    direction=-1,
                    confidence=0.65,
                    category="trend",
                ))
        except Exception:
            pass
        return signals

    # ── Chaikin Money Flow Signal ─────────────────────────────────────────────

    def _cmf_signal(
        self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series
    ) -> List[Signal]:
        """
        CMF: Positive = buying pressure, Negative = selling pressure.
        CMF > 0.1 and rising → bullish confirmation.
        CMF < -0.1 and falling → bearish confirmation.
        """
        signals = []
        try:
            cmf_series = cmf(high, low, close, volume, self._s.CMF_PERIOD)
            if cmf_series.empty or pd.isna(cmf_series.iloc[-1]):
                return signals

            cmf_now = float(cmf_series.iloc[-1])
            cmf_prev = float(cmf_series.iloc[-2]) if len(cmf_series) >= 2 else cmf_now

            if cmf_now > 0.1:
                # Buying pressure
                rising = cmf_now > cmf_prev
                conf = min(1.0, cmf_now * 5 + (0.15 if rising else 0))
                signals.append(Signal(
                    name="cmf_bullish",
                    direction=1,
                    confidence=conf,
                    category="volume",
                ))
            elif cmf_now < -0.1:
                # Selling pressure
                falling = cmf_now < cmf_prev
                conf = min(1.0, abs(cmf_now) * 5 + (0.15 if falling else 0))
                signals.append(Signal(
                    name="cmf_bearish",
                    direction=-1,
                    confidence=conf,
                    category="volume",
                ))
            else:
                signals.append(Signal(
                    name="cmf_neutral",
                    direction=0,
                    confidence=0.1,
                    category="volume",
                ))
        except Exception:
            pass
        return signals

    # ── RSI Divergence Signal ─────────────────────────────────────────────────

    def _rsi_divergence_signal(self, close: pd.Series) -> List[Signal]:
        """
        Detect RSI divergences — powerful reversal signals.
        Bullish divergence (price lower low, RSI higher low) → buy.
        Bearish divergence (price higher high, RSI lower high) → sell.
        """
        signals = []
        try:
            rsi_s = rsi(close, self._s.RSI_PERIOD)
            div = rsi_divergence(close, rsi_s, lookback=self._s.RSI_DIVERGENCE_LOOKBACK)

            if div.empty:
                return signals

            div_val = int(div.iloc[-1])
            if div_val == 1:
                signals.append(Signal(
                    name="rsi_divergence_bullish",
                    direction=1,
                    confidence=0.80,
                    category="reversion",
                ))
            elif div_val == -1:
                signals.append(Signal(
                    name="rsi_divergence_bearish",
                    direction=-1,
                    confidence=0.80,
                    category="reversion",
                ))
        except Exception:
            pass
        return signals

    # ── OBV Trend Signal ──────────────────────────────────────────────────────

    def _obv_signal(self, close: pd.Series, volume: pd.Series) -> List[Signal]:
        """
        OBV trend confirmation:
        OBV EMA trending up with price up → bullish confirmation.
        OBV EMA trending down with price up → bearish divergence.
        """
        signals = []
        try:
            obv_series = obv(close, volume)
            obv_ema = obv_series.ewm(span=self._s.OBV_EMA_PERIOD, adjust=False).mean()

            if len(obv_ema.dropna()) < 3:
                return signals

            obv_now = float(obv_ema.iloc[-1])
            obv_prev = float(obv_ema.iloc[-3])
            price_now = float(close.iloc[-1])
            price_prev = float(close.iloc[-3])

            obv_rising = obv_now > obv_prev
            price_rising = price_now > price_prev

            if obv_rising and price_rising:
                # OBV confirms price up move — bullish
                conf = min(1.0, abs(obv_now - obv_prev) / (abs(obv_prev) + 1e-9) * 10)
                signals.append(Signal(
                    name="obv_bullish_confirm",
                    direction=1,
                    confidence=min(0.75, conf + 0.3),
                    category="volume",
                ))
            elif not obv_rising and not price_rising:
                # OBV confirms price down move — bearish
                conf = min(1.0, abs(obv_now - obv_prev) / (abs(obv_prev) + 1e-9) * 10)
                signals.append(Signal(
                    name="obv_bearish_confirm",
                    direction=-1,
                    confidence=min(0.75, conf + 0.3),
                    category="volume",
                ))
            elif not obv_rising and price_rising:
                # OBV divergence — price up but OBV falling → bearish divergence
                signals.append(Signal(
                    name="obv_bearish_divergence",
                    direction=-1,
                    confidence=0.60,
                    category="volume",
                ))
            elif obv_rising and not price_rising:
                # OBV divergence — price down but OBV rising → bullish divergence
                signals.append(Signal(
                    name="obv_bullish_divergence",
                    direction=1,
                    confidence=0.60,
                    category="volume",
                ))
        except Exception:
            pass
        return signals

    # ── Squeeze Momentum Signal ───────────────────────────────────────────────

    def _squeeze_signal(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> List[Signal]:
        """
        Squeeze Momentum (John Carter / LazyBear):
        After squeeze fires (squeeze_off after squeeze_on), follow momentum direction.
        Rising positive momentum → bullish.
        Falling negative momentum → bearish.
        Extremely high confidence when squeeze just released.
        """
        signals = []
        try:
            sq = squeeze_momentum(
                close, high, low,
                bb_period=self._s.SQUEEZE_BB_PERIOD,
                kc_period=self._s.SQUEEZE_KC_PERIOD,
            )
            if len(sq.dropna(subset=["momentum"])) < 3:
                return signals

            mom_now = float(sq["momentum"].iloc[-1])
            mom_prev = float(sq["momentum"].iloc[-2]) if len(sq) >= 2 else mom_now
            was_squeezing = bool(sq["squeeze_on"].iloc[-2]) if len(sq) >= 2 else False
            currently_squeezing = bool(sq["squeeze_on"].iloc[-1])

            # Squeeze just released (high-conviction breakout setup)
            squeeze_released = was_squeezing and not currently_squeezing

            if pd.isna(mom_now):
                return signals

            if mom_now > 0 and mom_now > mom_prev:
                # Rising positive momentum → bullish
                conf = 0.80 if squeeze_released else 0.55
                signals.append(Signal(
                    name="squeeze_bullish",
                    direction=1,
                    confidence=conf,
                    category="squeeze",
                ))
            elif mom_now < 0 and mom_now < mom_prev:
                # Falling negative momentum → bearish
                conf = 0.80 if squeeze_released else 0.55
                signals.append(Signal(
                    name="squeeze_bearish",
                    direction=-1,
                    confidence=conf,
                    category="squeeze",
                ))
            elif mom_now > 0 and mom_now < mom_prev:
                # Momentum fading — weakening bull
                signals.append(Signal(
                    name="squeeze_bull_fade",
                    direction=0,
                    confidence=0.2,
                    category="squeeze",
                ))
            elif mom_now < 0 and mom_now > mom_prev:
                # Momentum fading — weakening bear
                signals.append(Signal(
                    name="squeeze_bear_fade",
                    direction=0,
                    confidence=0.2,
                    category="squeeze",
                ))
        except Exception:
            pass
        return signals

    # ── HTF Trend Filter (일봉 기반 추세 방향) ────────────────────────────────

    def _htf_trend_signals(self, htf_df: pd.DataFrame) -> list:
        """
        일봉 기반 추세 방향 신호 — 가장 높은 가중치 카테고리.

        규칙:
        - EMA9 > EMA21 > EMA50 > EMA200 (완전 정렬) → 강한 상승 추세
        - EMA50 > EMA200 (골든크로스 구간) → 불마켓 (long bias)
        - EMA50 < EMA200 (데스크로스 구간) → 베어마켓 (short bias)
        - RSI가 40-70 구간에서 롱, 30-60 구간에서 숏 진입 선호

        이 필터가 발동되면 역방향 신호는 combiner에서 대부분 차단됨.
        """
        signals = []
        try:
            close = htf_df["close"]
            high = htf_df["high"]
            low = htf_df["low"]

            # EMA200 기준: 215봉 이상 필요 (EMA 수렴용 워밍업 포함)
            min_htf_bars = self._s.HTF_EMA_TREND + 15
            if len(close) < min_htf_bars:
                return signals

            # ── HTF EMA 계산 ──────────────────────────────────────────────────
            ef = ema(close, self._s.HTF_EMA_FAST)
            em = ema(close, self._s.HTF_EMA_MID)
            es = ema(close, self._s.HTF_EMA_SLOW)
            et = ema(close, self._s.HTF_EMA_TREND)

            ef_now = float(ef.iloc[-1])
            em_now = float(em.iloc[-1])
            es_now = float(es.iloc[-1])
            et_now = float(et.iloc[-1])
            price_now = float(close.iloc[-1])

            # ── EMA 정렬 점수 ─────────────────────────────────────────────────
            # full_bull: 모든 EMA 완전 정렬 (가장 강한 추세)
            full_bull = ef_now > em_now > es_now > et_now and price_now > ef_now
            # mild_bull: EMA50 > EMA200 = 불마켓
            mild_bull = es_now > et_now
            full_bear = ef_now < em_now < es_now < et_now and price_now < ef_now
            mild_bear = es_now < et_now  # EMA50 < EMA200 = 베어마켓

            # ── HTF ADX ───────────────────────────────────────────────────────
            adx_df_htf = adx(high, low, close, self._s.ADX_PERIOD)
            htf_adx = float(adx_df_htf["adx"].iloc[-1])
            htf_plus_di = float(adx_df_htf["plus_di"].iloc[-1])
            htf_minus_di = float(adx_df_htf["minus_di"].iloc[-1])
            adx_trending = htf_adx >= self._s.HTF_ADX_THRESHOLD
            adx_bullish = htf_plus_di > htf_minus_di

            # ── HTF RSI ───────────────────────────────────────────────────────
            rsi_s = rsi(close, self._s.RSI_PERIOD)
            htf_rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

            # ── HTF 방향 결정 ─────────────────────────────────────────────────
            if full_bull and adx_trending:
                conf = min(1.0, 0.85 + min(0.15, (htf_adx - 20) / 100))
                signals.append(Signal(
                    name="htf_strong_bull",
                    direction=1,
                    confidence=conf,
                    category="htf_trend",
                ))
            elif mild_bull:
                if adx_trending and adx_bullish:
                    conf = 0.80
                elif adx_bullish:
                    conf = 0.65
                else:
                    conf = 0.55  # 조정 중이지만 불마켓 유지
                signals.append(Signal(
                    name="htf_mild_bull",
                    direction=1,
                    confidence=conf,
                    category="htf_trend",
                ))
            elif full_bear and adx_trending:
                conf = min(1.0, 0.85 + min(0.15, (htf_adx - 20) / 100))
                signals.append(Signal(
                    name="htf_strong_bear",
                    direction=-1,
                    confidence=conf,
                    category="htf_trend",
                ))
            elif mild_bear:
                if adx_trending and not adx_bullish:
                    conf = 0.80
                elif not adx_bullish:
                    conf = 0.65
                else:
                    conf = 0.55
                signals.append(Signal(
                    name="htf_mild_bear",
                    direction=-1,
                    confidence=conf,
                    category="htf_trend",
                ))
            else:
                signals.append(Signal(
                    name="htf_neutral",
                    direction=0,
                    confidence=0.15,
                    category="htf_trend",
                ))

            # ── RSI 진입 품질 보조 신호 (동일 방향만) ────────────────────────
            rsi_ok_for_long = self._s.HTF_RSI_BULL_MIN <= htf_rsi <= self._s.HTF_RSI_BULL_MAX
            rsi_ok_for_short = self._s.HTF_RSI_BEAR_MIN <= htf_rsi <= self._s.HTF_RSI_BEAR_MAX

            if rsi_ok_for_long and mild_bull:
                signals.append(Signal(
                    name="htf_rsi_bull_zone",
                    direction=1,
                    confidence=0.65,
                    category="htf_trend",
                ))
            elif rsi_ok_for_short and mild_bear:
                signals.append(Signal(
                    name="htf_rsi_bear_zone",
                    direction=-1,
                    confidence=0.65,
                    category="htf_trend",
                ))

        except Exception:
            pass

        return signals

    # ── ICT Signals ───────────────────────────────────────────────────────────

    def _ict_signals(self, ict: ICTResult, current_price: float) -> List[Signal]:
        signals = []

        # Structure bias
        if ict.bias == StructureBias.BULLISH:
            signals.append(Signal(name="ict_structure", direction=1, confidence=0.7, category="ict"))
        elif ict.bias == StructureBias.BEARISH:
            signals.append(Signal(name="ict_structure", direction=-1, confidence=0.7, category="ict"))
        else:
            signals.append(Signal(name="ict_structure", direction=0, confidence=0.2, category="ict"))

        # BOS
        if ict.bos_bullish:
            signals.append(Signal(name="ict_bos", direction=1, confidence=0.8, category="ict"))
        elif ict.bos_bearish:
            signals.append(Signal(name="ict_bos", direction=-1, confidence=0.8, category="ict"))

        # CHoCH (higher weight — reversal signal)
        if ict.choch_bullish:
            signals.append(Signal(name="ict_choch", direction=1, confidence=0.85, category="ict"))
        elif ict.choch_bearish:
            signals.append(Signal(name="ict_choch", direction=-1, confidence=0.85, category="ict"))

        # Liquidity sweeps
        if ict.recent_sweep_bullish:
            signals.append(Signal(name="ict_sweep", direction=1, confidence=0.9, category="ict"))
        elif ict.recent_sweep_bearish:
            signals.append(Signal(name="ict_sweep", direction=-1, confidence=0.9, category="ict"))

        # FVG proximity
        if ict.nearest_fvg_long and ict.nearest_fvg_long.price_near_zone(current_price):
            signals.append(Signal(name="ict_fvg_demand", direction=1, confidence=0.8, category="ict"))
        if ict.nearest_fvg_short and ict.nearest_fvg_short.price_near_zone(current_price):
            signals.append(Signal(name="ict_fvg_supply", direction=-1, confidence=0.8, category="ict"))

        # Premium / Discount
        if ict.price_in_discount:
            signals.append(Signal(name="ict_discount", direction=1, confidence=0.6, category="ict"))
        elif ict.price_in_premium:
            signals.append(Signal(name="ict_premium", direction=-1, confidence=0.6, category="ict"))

        return signals
