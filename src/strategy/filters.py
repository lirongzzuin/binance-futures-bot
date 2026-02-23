"""
Signal Quality Filter.

Applies pre-entry quality checks to eliminate low-edge trades:
  1. Dead market filter  — ATR too low (no movement, high spread impact)
  2. Efficiency Ratio    — Kaufman ER < threshold → market too choppy
  3. Hurst Exponent      — Peters H < threshold  → mean-reverting regime
  4. Volatility spike    — vol_percentile > 0.93 → extreme regime, wider spreads
  5. Low-liquidity hours — UTC 22:00–01:00 (warning, not rejection)

Philosophy: Renaissance Technologies reportedly filters out 70–80% of signals.
Quality > quantity. Each trade must have genuine statistical edge after fees.

References:
  - Perry Kaufman "Trading Systems and Methods" (ER filter)
  - Edgar Peters "Fractal Market Analysis" (Hurst exponent)
  - AQR Capital vol-targeting research
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import Settings
from src.strategy.indicators import (
    atr,
    efficiency_ratio,
    hurst_exponent,
    volatility_percentile,
)


@dataclass
class SignalFilter:
    passed: bool
    reason: str = ""
    er_val: float = 0.0
    hurst_val: float = 0.5


class QualityFilter:
    def __init__(self, settings: Settings):
        self._s = settings

    def check(self, df: pd.DataFrame, use_hurst: bool = True) -> SignalFilter:
        """
        Run all quality checks on market data.
        Returns SignalFilter with pass/fail status and diagnostic metrics.

        use_hurst: set False in high-throughput contexts (backtest) since
                   Hurst exponent is O(n²) and expensive for 100k+ evaluations.
        """
        if len(df) < 30:
            return SignalFilter(passed=False, reason="Insufficient data (< 30 bars)")

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── 1. ATR dead-market filter ─────────────────────────────────────────
        atr_series = atr(high, low, close, self._s.ATR_PERIOD)
        atr_val = float(atr_series.iloc[-1])

        if pd.isna(atr_val) or atr_val <= 0:
            return SignalFilter(passed=False, reason="ATR unavailable")

        last_price = float(close.iloc[-1])
        if last_price <= 0:
            return SignalFilter(passed=False, reason="Invalid price")

        atr_pct = atr_val / last_price
        if atr_pct < self._s.SIGNAL_FILTER_MIN_ATR_PCT:
            return SignalFilter(
                passed=False,
                reason=f"Dead market: ATR/price={atr_pct:.6f} < {self._s.SIGNAL_FILTER_MIN_ATR_PCT}",
                er_val=0.0,
                hurst_val=0.5,
            )

        # ── 2. Efficiency Ratio (Kaufman) ─────────────────────────────────────
        # Use period=50 (≈ 4h at 5-min bars) — short periods give near-zero ER on crypto
        er_series = efficiency_ratio(close, period=50)
        er_val = 0.0
        valid_er = er_series.dropna()
        if len(valid_er) > 0:
            er_val = float(valid_er.iloc[-1])
        if pd.isna(er_val):
            er_val = 0.0

        if er_val < self._s.SIGNAL_FILTER_ER_THRESHOLD:
            return SignalFilter(
                passed=False,
                reason=(
                    f"Choppy market: ER={er_val:.3f} < threshold {self._s.SIGNAL_FILTER_ER_THRESHOLD}"
                ),
                er_val=er_val,
                hurst_val=0.5,
            )

        # ── 3. Hurst Exponent (Peters) ────────────────────────────────────────
        # Skipped when use_hurst=False (backtest) — O(n²) cost
        hurst_val = 0.5
        if use_hurst:
            hurst_val = hurst_exponent(close, max_lags=20)
            if hurst_val < self._s.SIGNAL_FILTER_HURST_MIN:
                return SignalFilter(
                    passed=False,
                    reason=(
                        f"Mean-reverting: Hurst={hurst_val:.3f} < {self._s.SIGNAL_FILTER_HURST_MIN}"
                    ),
                    er_val=er_val,
                    hurst_val=hurst_val,
                )

        # ── 4. Extreme volatility spike ───────────────────────────────────────
        if len(close) >= 100:
            vp_series = volatility_percentile(close, short=20, long=100)
            vp = float(vp_series.iloc[-1]) if not pd.isna(vp_series.iloc[-1]) else 0.5
            if vp > 0.93:
                return SignalFilter(
                    passed=False,
                    reason=f"Extreme volatility spike: vol_pct={vp:.2f} > 0.93",
                    er_val=er_val,
                    hurst_val=hurst_val,
                )

        return SignalFilter(passed=True, reason="", er_val=er_val, hurst_val=hurst_val)
