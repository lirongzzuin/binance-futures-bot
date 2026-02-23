"""
Trade Mode Detector — decides between SCALP and TREND_RIDE for each signal.

SCALP mode:   tight exits, quick TP, 5-min bars, 15-bar max hold
TREND_RIDE:   wider stops, partial TP, longer hold, rides momentum

Scoring: 8 conditions evaluated; score ≥ TREND_RIDE_SCORE_THRESHOLD → TREND_RIDE
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)


class TradeMode(str, Enum):
    SCALP = "SCALP"
    TREND_RIDE = "TREND_RIDE"


@dataclass
class TradeModeResult:
    mode: TradeMode
    score: int
    score_details: Dict[str, int]


@dataclass
class ScalpModeParams:
    stop_atr: float
    tp1_atr: float
    tp2_atr: float
    max_bars: int
    partial_exit_pct: float


class TradeModeDetector:
    def __init__(self, settings: Settings):
        self._s = settings

    def detect(
        self,
        regime,             # RegimeResult
        ict_result,         # ICTResult
        bundle,             # SignalBundle
        funding_rate: float,
        spread_pct: float,
        direction: int,     # +1 long, -1 short
    ) -> TradeModeResult:
        """
        Score each condition; sum ≥ TREND_RIDE_SCORE_THRESHOLD → TREND_RIDE.

        Positive signals (+1 each):
          - ADX ≥ 30
          - |OI change| ≥ 1.5%  (via bundle; approximated from volume signals)
          - BOS aligned with direction
          - |net_direction| ≥ 0.60
          - vol_percentile < 0.70   (not choppy)
          - EMA aligned with direction

        Negative signals (-1 each):
          - |funding| ≥ 0.0008  (crowd risk)
          - spread_pct ≥ 0.002  (poor execution)
        """
        details: Dict[str, int] = {}

        # +1: ADX ≥ 30 — strong trend
        adx_val = getattr(regime, "adx_val", 0.0)
        details["adx_strong"] = 1 if adx_val >= 30.0 else 0

        # +1: |OI change| ≥ 1.5% — check via oi_flow signal confidence
        oi_score = 0
        for sig in bundle.signals:
            if sig.name == "oi_flow" and sig.direction == direction and sig.confidence >= 0.15:
                oi_score = 1
                break
        details["oi_aligned"] = oi_score

        # +1: BOS aligned with direction
        bos_bullish = getattr(ict_result, "bos_bullish", False)
        bos_bearish = getattr(ict_result, "bos_bearish", False)
        bos_aligned = (direction == 1 and bos_bullish) or (direction == -1 and bos_bearish)
        details["bos_aligned"] = 1 if bos_aligned else 0

        # +1: |net_direction| ≥ 0.60
        net_dir = abs(bundle.net_direction())
        details["strong_net_dir"] = 1 if net_dir >= 0.60 else 0

        # +1: vol_percentile < 0.70 — not in choppy/high-vol regime
        vol_pct = getattr(regime, "vol_percentile", 0.5)
        details["low_vol"] = 1 if vol_pct < 0.70 else 0

        # +1: EMA aligned with direction
        ema_up = getattr(regime, "ema_aligned_up", False)
        ema_down = getattr(regime, "ema_aligned_down", False)
        ema_aligned = (direction == 1 and ema_up) or (direction == -1 and ema_down)
        details["ema_aligned"] = 1 if ema_aligned else 0

        # -1: |funding| ≥ 0.0008 — crowd risk
        details["funding_risk"] = -1 if abs(funding_rate) >= 0.0008 else 0

        # -1: spread_pct ≥ 0.002 — poor execution
        details["wide_spread"] = -1 if spread_pct >= 0.002 else 0

        score = sum(details.values())
        mode = (
            TradeMode.TREND_RIDE
            if score >= self._s.TREND_RIDE_SCORE_THRESHOLD
            else TradeMode.SCALP
        )

        log.debug("TradeModeDetector | score=%d mode=%s details=%s", score, mode.value, details)

        return TradeModeResult(mode=mode, score=score, score_details=details)

    def get_params(self, mode: TradeMode) -> ScalpModeParams:
        """Return ATR multipliers and hold limits for a given trade mode."""
        if mode == TradeMode.TREND_RIDE:
            return ScalpModeParams(
                stop_atr=self._s.TREND_STOP_ATR,
                tp1_atr=self._s.TREND_TP1_ATR,
                tp2_atr=self._s.TREND_TP2_ATR,
                max_bars=self._s.TREND_MAX_BARS,
                partial_exit_pct=self._s.TREND_PARTIAL_EXIT_PCT,
            )
        return ScalpModeParams(
            stop_atr=self._s.SCALP_STOP_ATR,
            tp1_atr=self._s.SCALP_TP1_ATR,
            tp2_atr=self._s.SCALP_TP2_ATR,
            max_bars=self._s.SCALP_MAX_BARS,
            partial_exit_pct=self._s.SCALP_PARTIAL_EXIT_PCT,
        )
