"""
ICT (Inner Circle Trader) concept implementation.

All rules are purely mechanical / rule-based, derived from publicly
available ICT educational content. No ambiguous discretionary judgement.

Implemented concepts:
  1. Swing Highs / Lows  (N-bar pivot detection)
  2. BOS  — Break of Structure
  3. CHoCH — Change of Character
  4. FVG  — Fair Value Gap (Imbalance)
  5. Liquidity Sweep (Stop Hunt)
  6. Premium / Discount filter

References:
  - ICT public YouTube / podcast content (freely available)
  - Urban Forex, various open-source ICT implementations on GitHub
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Settings


# ── Data structures ───────────────────────────────────────────────────────────

class StructureBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class SwingPoint:
    idx: int
    price: float
    is_high: bool   # True = SwingHigh, False = SwingLow
    bar_time: Optional[int] = None


@dataclass
class FVG:
    """Fair Value Gap / Imbalance zone."""
    top: float
    bottom: float
    is_bullish: bool    # True = gap above (demand), False = gap below (supply)
    bar_idx: int
    filled: bool = False

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    def price_in_zone(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def price_near_zone(self, price: float, tolerance: float = 0.001) -> bool:
        buffer = (self.top - self.bottom) * (1 + tolerance)
        return (self.bottom - buffer) <= price <= (self.top + buffer)


@dataclass
class LiquiditySweep:
    """A wick that pierces a prior swing extreme then reverses."""
    bar_idx: int
    level: float
    is_upward: bool     # True = swept above swing high (bearish reversal)
    confirmed: bool = False


@dataclass
class ICTResult:
    bias: StructureBias = StructureBias.NEUTRAL
    bos_bullish: bool = False          # Recent Break of Structure up
    bos_bearish: bool = False
    choch_bullish: bool = False        # CHoCH = potential reversal signal
    choch_bearish: bool = False
    fvgs_bullish: List[FVG] = field(default_factory=list)  # unfilled demand FVGs
    fvgs_bearish: List[FVG] = field(default_factory=list)  # unfilled supply FVGs
    price_in_discount: bool = False    # below 50% of range → bullish bias
    price_in_premium: bool = False     # above 50% of range → bearish bias
    recent_sweep_bullish: bool = False # sweep of lows → potential long
    recent_sweep_bearish: bool = False # sweep of highs → potential short
    nearest_fvg_long: Optional[FVG] = None   # closest demand FVG to current price
    nearest_fvg_short: Optional[FVG] = None  # closest supply FVG
    ict_long_score: float = 0.0    # composite 0-1 for long
    ict_short_score: float = 0.0   # composite 0-1 for short
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)


# ── ICT Analyzer ─────────────────────────────────────────────────────────────

class ICTAnalyzer:
    def __init__(self, settings: Settings):
        self._s = settings

    def analyze(self, df: pd.DataFrame) -> ICTResult:
        """
        df must have: open, high, low, close (indexed 0..N-1)
        Returns ICTResult for the current (latest) bar.
        """
        result = ICTResult()
        n = len(df)
        period = self._s.ICT_SWING_PERIOD
        lookback = self._s.ICT_FVG_LOOKBACK
        range_period = self._s.ICT_RANGE_PERIOD

        if n < period * 2 + 5:
            return result

        # ── 1. Swing Highs / Lows ────────────────────────────────────────────
        highs, lows = self._find_swings(df, period)
        result.swing_highs = highs
        result.swing_lows = lows

        # ── 2. Market Structure (BOS / CHoCH) ────────────────────────────────
        self._analyze_structure(df, highs, lows, result)

        # ── 3. Fair Value Gaps ────────────────────────────────────────────────
        fvgs = self._find_fvgs(df, lookback)
        current_price = float(df["close"].iloc[-1])
        active_fvgs_bull = [f for f in fvgs if f.is_bullish and not f.filled
                            and f.top < current_price * 1.05]
        active_fvgs_bear = [f for f in fvgs if not f.is_bullish and not f.filled
                            and f.bottom > current_price * 0.95]

        result.fvgs_bullish = active_fvgs_bull
        result.fvgs_bearish = active_fvgs_bear

        # Nearest FVGs
        if active_fvgs_bull:
            result.nearest_fvg_long = min(
                active_fvgs_bull, key=lambda f: abs(f.mid - current_price)
            )
        if active_fvgs_bear:
            result.nearest_fvg_short = min(
                active_fvgs_bear, key=lambda f: abs(f.mid - current_price)
            )

        # ── 4. Premium / Discount ─────────────────────────────────────────────
        recent_high = float(df["high"].iloc[-range_period:].max())
        recent_low = float(df["low"].iloc[-range_period:].min())
        if recent_high > recent_low:
            range_pct = (current_price - recent_low) / (recent_high - recent_low)
            result.price_in_discount = range_pct < 0.40  # below 40%
            result.price_in_premium = range_pct > 0.60   # above 60%

        # ── 5. Liquidity Sweeps ───────────────────────────────────────────────
        self._analyze_sweeps(df, highs, lows, result)

        # ── 6. Composite ICT scores ───────────────────────────────────────────
        result.ict_long_score = self._compute_long_score(result, current_price)
        result.ict_short_score = self._compute_short_score(result, current_price)

        return result

    # ── Swing Detection ───────────────────────────────────────────────────────

    def _find_swings(
        self, df: pd.DataFrame, period: int
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        highs: List[SwingPoint] = []
        lows: List[SwingPoint] = []
        n = len(df)

        for i in range(period, n - period):
            window_h = df["high"].iloc[i - period: i + period + 1]
            window_l = df["low"].iloc[i - period: i + period + 1]

            if float(df["high"].iloc[i]) == float(window_h.max()):
                highs.append(SwingPoint(
                    idx=i,
                    price=float(df["high"].iloc[i]),
                    is_high=True,
                ))
            if float(df["low"].iloc[i]) == float(window_l.min()):
                lows.append(SwingPoint(
                    idx=i,
                    price=float(df["low"].iloc[i]),
                    is_high=False,
                ))

        # Keep last N swing points for relevance
        return highs[-20:], lows[-20:]

    # ── BOS / CHoCH ───────────────────────────────────────────────────────────

    def _analyze_structure(
        self,
        df: pd.DataFrame,
        highs: List[SwingPoint],
        lows: List[SwingPoint],
        result: ICTResult,
    ) -> None:
        n = len(df)
        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if n >= 2 else last_close

        if len(highs) >= 1:
            last_sh = highs[-1]
            # BOS bullish: close above last swing high
            if prev_close <= last_sh.price < last_close:
                result.bos_bullish = True

        if len(lows) >= 1:
            last_sl = lows[-1]
            # BOS bearish: close below last swing low
            if prev_close >= last_sl.price > last_close:
                result.bos_bearish = True

        # CHoCH: reversal after established BOS
        # (simplified: BOS in one direction while bias was opposite)
        if result.bos_bullish and len(lows) >= 2:
            # was trending down (recent swing low < previous swing low)
            if lows[-1].price < lows[-2].price:
                result.choch_bullish = True  # potential reversal up
        if result.bos_bearish and len(highs) >= 2:
            if highs[-1].price > highs[-2].price:
                result.choch_bearish = True  # potential reversal down

        # Overall bias
        if result.bos_bullish and not result.bos_bearish:
            result.bias = StructureBias.BULLISH
        elif result.bos_bearish and not result.bos_bullish:
            result.bias = StructureBias.BEARISH
        elif len(highs) >= 2 and len(lows) >= 2:
            # Higher highs and higher lows → bullish
            if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
                result.bias = StructureBias.BULLISH
            elif highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
                result.bias = StructureBias.BEARISH

    # ── Fair Value Gaps ───────────────────────────────────────────────────────

    def _find_fvgs(self, df: pd.DataFrame, lookback: int) -> List[FVG]:
        """
        Bullish FVG:  df[i-1].high < df[i+1].low  →  demand zone
        Bearish FVG:  df[i-1].low  > df[i+1].high →  supply zone
        """
        fvgs: List[FVG] = []
        start = max(1, len(df) - lookback)

        for i in range(start, len(df) - 1):
            h_prev = float(df["high"].iloc[i - 1])
            l_prev = float(df["low"].iloc[i - 1])
            h_next = float(df["high"].iloc[i + 1])
            l_next = float(df["low"].iloc[i + 1])
            close_cur = float(df["close"].iloc[-1])

            # Bullish FVG (gap = demand zone)
            if h_prev < l_next:
                fvg = FVG(top=l_next, bottom=h_prev, is_bullish=True, bar_idx=i)
                # Check if price has since filled the gap
                subsequent_lows = df["low"].iloc[i + 1:]
                if float(subsequent_lows.min()) <= h_prev:
                    fvg.filled = True
                fvgs.append(fvg)

            # Bearish FVG (gap = supply zone)
            elif l_prev > h_next:
                fvg = FVG(top=l_prev, bottom=h_next, is_bullish=False, bar_idx=i)
                subsequent_highs = df["high"].iloc[i + 1:]
                if float(subsequent_highs.max()) >= l_prev:
                    fvg.filled = True
                fvgs.append(fvg)

        return fvgs

    # ── Liquidity Sweeps ──────────────────────────────────────────────────────

    def _analyze_sweeps(
        self,
        df: pd.DataFrame,
        highs: List[SwingPoint],
        lows: List[SwingPoint],
        result: ICTResult,
    ) -> None:
        """
        A sweep occurs when the current bar's wick pierces a prior swing
        but the bar closes back on the opposite side.
        """
        if len(df) < 2:
            return

        cur_high = float(df["high"].iloc[-1])
        cur_low = float(df["low"].iloc[-1])
        cur_close = float(df["close"].iloc[-1])
        cur_open = float(df["open"].iloc[-1])

        # Sweep of a swing high → bearish reversal signal
        if highs:
            last_sh_price = highs[-1].price
            if cur_high > last_sh_price and cur_close < last_sh_price:
                result.recent_sweep_bearish = True  # wick above, close below → sell

        # Sweep of a swing low → bullish reversal signal
        if lows:
            last_sl_price = lows[-1].price
            if cur_low < last_sl_price and cur_close > last_sl_price:
                result.recent_sweep_bullish = True  # wick below, close above → buy

    # ── Composite Scores ──────────────────────────────────────────────────────

    def _compute_long_score(self, result: ICTResult, current_price: float) -> float:
        score = 0.0
        factors = 0

        if result.bos_bullish:
            score += 1.0; factors += 1
        if result.choch_bullish:
            score += 1.0; factors += 1
        if result.recent_sweep_bullish:
            score += 1.0; factors += 1
        if result.price_in_discount:
            score += 0.8; factors += 1
        if result.bias == StructureBias.BULLISH:
            score += 0.6; factors += 1
        if result.nearest_fvg_long and result.nearest_fvg_long.price_near_zone(current_price):
            score += 1.0; factors += 1

        return score / max(factors, 1)

    def _compute_short_score(self, result: ICTResult, current_price: float) -> float:
        score = 0.0
        factors = 0

        if result.bos_bearish:
            score += 1.0; factors += 1
        if result.choch_bearish:
            score += 1.0; factors += 1
        if result.recent_sweep_bearish:
            score += 1.0; factors += 1
        if result.price_in_premium:
            score += 0.8; factors += 1
        if result.bias == StructureBias.BEARISH:
            score += 0.6; factors += 1
        if result.nearest_fvg_short and result.nearest_fvg_short.price_near_zone(current_price):
            score += 1.0; factors += 1

        return score / max(factors, 1)
