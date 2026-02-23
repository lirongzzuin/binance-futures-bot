"""
Market Regime Detector.

Classifies each bar into one of:
  TRENDING_UP | TRENDING_DOWN | RANGING | HIGH_VOLATILITY

Uses a multi-factor scoring approach:
  1. ADX + Directional Index → trend strength and direction
  2. EMA structure (fast/mid/slow alignment) → trend direction
  3. Realized volatility percentile → volatility regime
  4. Volume ratio → conviction filter
  5. OI change → trend continuation / divergence hint

Reference: ADX methodology (Wilder 1978), systematic trend-following
literature (Covel 2004, AQR trend research).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from src.config import Settings
from src.strategy.indicators import (
    adx,
    efficiency_ratio,
    ema,
    hurst_exponent,
    realized_volatility,
    volatility_percentile,
    volume_ratio,
)


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class RegimeResult:
    regime: Regime
    adx_val: float
    plus_di: float
    minus_di: float
    vol_percentile: float
    ema_aligned_up: bool
    ema_aligned_down: bool
    volume_ratio: float
    trend_score: float    # -1 (strong down) to +1 (strong up)
    confidence: float     # 0 to 1
    hurst_val: float = 0.5   # Hurst exponent (0–1); >0.55 trending, <0.45 mean-reverting
    er_val: float = 0.5      # Kaufman Efficiency Ratio (0–1); >0.40 = directional


class RegimeDetector:
    def __init__(self, settings: Settings):
        self._s = settings

    def detect(
        self,
        df: pd.DataFrame,
        oi_change_pct: float = 0.0,
        hurst_max_lags: int = 20,
    ) -> RegimeResult:
        """
        hurst_max_lags: set to 0 to skip Hurst computation (backtest fast mode).
        Hurst is O(n × lags) in Python loops — expensive for 100k+ evaluations.
        """
        """
        df must have columns: open, high, low, close, volume
        Returns RegimeResult for the latest bar.
        """
        if len(df) < self._s.EMA_SLOW + 10:
            return RegimeResult(
                regime=Regime.RANGING,
                adx_val=0,
                plus_di=0,
                minus_di=0,
                vol_percentile=0.5,
                ema_aligned_up=False,
                ema_aligned_down=False,
                volume_ratio=1.0,
                trend_score=0.0,
                confidence=0.0,
                hurst_val=0.5,
                er_val=0.5,
            )

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── 1. ADX ────────────────────────────────────────────────────────────
        adx_df = adx(high, low, close, self._s.ADX_PERIOD)
        adx_val = float(adx_df["adx"].iloc[-1])
        plus_di = float(adx_df["plus_di"].iloc[-1])
        minus_di = float(adx_df["minus_di"].iloc[-1])

        adx_trend = adx_val >= self._s.ADX_TREND_THRESHOLD
        adx_range = adx_val < self._s.ADX_RANGE_THRESHOLD

        # ── 2. EMA structure ──────────────────────────────────────────────────
        ema_f = ema(close, self._s.EMA_FAST).iloc[-1]
        ema_m = ema(close, self._s.EMA_MID).iloc[-1]
        ema_s = ema(close, self._s.EMA_SLOW).iloc[-1]
        last_close = float(close.iloc[-1])

        ema_aligned_up = ema_f > ema_m > ema_s and last_close > ema_f
        ema_aligned_down = ema_f < ema_m < ema_s and last_close < ema_f

        # EMA slope (5-bar)
        ema_slow_series = ema(close, self._s.EMA_SLOW)
        if len(ema_slow_series) >= 6:
            ema_slope = (float(ema_slow_series.iloc[-1]) - float(ema_slow_series.iloc[-6])) / float(ema_slow_series.iloc[-6])
        else:
            ema_slope = 0.0

        # ── 3. Volatility percentile ──────────────────────────────────────────
        rv_pct = volatility_percentile(close, short=self._s.VOLATILITY_LOOKBACK, long=100)
        vol_pct = float(rv_pct.iloc[-1]) if not pd.isna(rv_pct.iloc[-1]) else 0.5
        high_vol = vol_pct > 0.85

        # ── 4. Volume ratio ───────────────────────────────────────────────────
        vol_r = volume_ratio(volume, short=5, long=20)
        vol_ratio_val = float(vol_r.iloc[-1]) if not pd.isna(vol_r.iloc[-1]) else 1.0

        # ── 5. OI signal ──────────────────────────────────────────────────────
        # OI rising with trend → continuation; OI falling with price move → caution
        oi_confirms_trend = (
            (oi_change_pct > 0.01 and ema_aligned_up) or
            (oi_change_pct > 0.01 and ema_aligned_down)
        )

        # ── 6. Hurst Exponent (Peters 1991) ───────────────────────────────────
        # Skipped when hurst_max_lags=0 (backtest fast mode)
        hurst_val = 0.5
        if hurst_max_lags > 0:
            hurst_val = hurst_exponent(close, max_lags=hurst_max_lags)

        # ── 7. Efficiency Ratio (Kaufman) ─────────────────────────────────────
        er_series = efficiency_ratio(close, period=10)
        er_val = 0.5
        valid_er = er_series.dropna()
        if len(valid_er) > 0:
            er_val = float(valid_er.iloc[-1])
        if pd.isna(er_val):
            er_val = 0.5

        # ── Composite trend score ─────────────────────────────────────────────
        # Range: -1 (strong down) to +1 (strong up)
        di_score = 0.0
        if adx_val > 0:
            di_score = (plus_di - minus_di) / (plus_di + minus_di + 1e-9)

        ema_score = 1.0 if ema_aligned_up else (-1.0 if ema_aligned_down else ema_slope * 10)
        ema_score = max(-1.0, min(1.0, ema_score))

        trend_score = 0.50 * di_score + 0.40 * ema_score + 0.10 * (1.0 if oi_confirms_trend else 0.0)
        trend_score = max(-1.0, min(1.0, trend_score))

        # ── Regime classification ─────────────────────────────────────────────
        # Trending requires ADX confirmation AND either Hurst > 0.50 OR ER > 0.40
        # This filters out low-quality "trend" signals in choppy markets
        trending_confirmed = adx_trend and (hurst_val > 0.50 or er_val > 0.40)

        if high_vol and not adx_trend:
            regime = Regime.HIGH_VOLATILITY
            confidence = vol_pct
        elif trending_confirmed and trend_score > 0.2:
            regime = Regime.TRENDING_UP
            confidence = adx_val / 50.0 * (0.5 + 0.5 * trend_score)
            # Boost confidence when Hurst aligns with momentum
            if hurst_val > 0.55:
                confidence = min(1.0, confidence * 1.15)
        elif trending_confirmed and trend_score < -0.2:
            regime = Regime.TRENDING_DOWN
            confidence = adx_val / 50.0 * (0.5 - 0.5 * trend_score)
            if hurst_val > 0.55:
                confidence = min(1.0, confidence * 1.15)
        else:
            regime = Regime.RANGING
            confidence = max(0.0, 1.0 - adx_val / self._s.ADX_TREND_THRESHOLD)

        confidence = max(0.0, min(1.0, confidence))

        return RegimeResult(
            regime=regime,
            adx_val=adx_val,
            plus_di=plus_di,
            minus_di=minus_di,
            vol_percentile=vol_pct,
            ema_aligned_up=ema_aligned_up,
            ema_aligned_down=ema_aligned_down,
            volume_ratio=vol_ratio_val,
            trend_score=trend_score,
            confidence=confidence,
            hurst_val=hurst_val,
            er_val=er_val,
        )
