"""
Macro Regime Detector — BTC 일봉/주봉 기반 시장 상황 판단.

Bull market  : EMA50 > EMA200, ADX > 20, 20일 ROC > +5%
              → 멀티코인 스캘핑 (최대 10× 레버리지)
Bear/Sideways: EMA50 < EMA200 OR ROC_20 < −5% OR ADX < 15 (횡보)
              → BTC + ETH만 거래 (최대 50× 레버리지, 단기 스캘핑)

전환 확정 룰:
  - BULL 전환: N일 연속 BULL 신호 필요 (MACRO_BULL_CONFIRM_DAYS)
  - BEAR/SIDEWAYS 전환: N일 연속 비-BULL 신호 (MACRO_BEAR_CONFIRM_DAYS)
  → 이로써 휩소(잦은 전환) 방지
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

from src.logger import get_logger

if TYPE_CHECKING:
    from src.api.binance_client import BinanceClient
    from src.config import Settings

log = get_logger(__name__)


class MacroRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class MacroRegimeResult:
    regime: MacroRegime
    confidence: float          # 0~1; 높을수록 확신
    ema50: float = 0.0
    ema200: float = 0.0
    adx: float = 0.0
    roc_20: float = 0.0        # 20일 ROC (%)
    reason: str = ""
    symbols: List[str] = field(default_factory=list)  # 허용 거래 심볼


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up = high - high.shift(1)
    dn = low.shift(1) - low
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)

    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0)


# ── Detector ──────────────────────────────────────────────────────────────────

class MacroRegimeDetector:
    """BTC 일봉 데이터를 기반으로 거시 시장 상황을 판단한다."""

    def __init__(self, client: "BinanceClient", settings: "Settings"):
        self._client = client
        self._s = settings
        self._btc_symbol = "BTCUSDT"

    def detect(self) -> MacroRegimeResult:
        """
        BTC 일봉 200개를 가져와 매크로 레짐을 계산한다.
        API 실패 시 SIDEWAYS를 기본값으로 반환 (보수적).
        """
        try:
            df = self._fetch_daily("BTCUSDT", limit=250)
            if df is None or len(df) < 210:
                log.warning("MacroRegime: BTC 일봉 데이터 부족 — SIDEWAYS 기본값 사용")
                return self._default()

            result = self._compute(df)
            log.info(
                "MacroRegime: %s (confidence=%.2f) ema50=%.0f ema200=%.0f "
                "adx=%.1f roc20=%.1f%% | %s",
                result.regime.value,
                result.confidence,
                result.ema50,
                result.ema200,
                result.adx,
                result.roc_20,
                result.reason,
            )
            return result
        except Exception as e:
            log.error("MacroRegime detect error: %s — SIDEWAYS fallback", e)
            return self._default()

    def _fetch_daily(self, symbol: str, limit: int = 250) -> Optional[pd.DataFrame]:
        """일봉 OHLCV를 DataFrame으로 반환."""
        # Binance interval: "1d" for daily
        raw = self._client.get_kline(symbol=symbol, interval="1d", limit=limit)
        if not raw:
            return None
        # Binance: [open_time, open, high, low, close, volume, close_time,
        #           quote_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
        df = pd.DataFrame(raw, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        for col in ["ts", "open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def _compute(self, df: pd.DataFrame) -> MacroRegimeResult:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── EMA50 / EMA200 ────────────────────────────────────────────────────
        ema50 = float(_ema(close, 50).iloc[-1])
        ema200 = float(_ema(close, 200).iloc[-1])
        price = float(close.iloc[-1])

        # ── ADX (14일) ────────────────────────────────────────────────────────
        adx_series = _adx(high, low, close, period=14)
        adx = float(adx_series.iloc[-1])

        # ── 20일 ROC (Rate of Change) ─────────────────────────────────────────
        roc_20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else 0.0

        # ── 스코어링 ──────────────────────────────────────────────────────────
        bull_score = 0
        reasons = []

        # 1) EMA50 > EMA200 (golden cross zone) — 가장 중요한 지표
        ema_cross_pct = (ema50 - ema200) / ema200 * 100 if ema200 > 0 else 0
        if ema50 > ema200:
            bull_score += 3
            reasons.append(f"EMA50>EMA200(+{ema_cross_pct:.1f}%)")
        elif ema_cross_pct < -3:
            bull_score -= 2
            reasons.append(f"EMA50<<EMA200({ema_cross_pct:.1f}%)")
        else:
            reasons.append(f"EMA50≈EMA200({ema_cross_pct:.1f}%)")

        # 2) 가격이 EMA200 위에 있는가
        price_above_ema200 = price > ema200
        if price_above_ema200:
            bull_score += 1
            reasons.append("price>EMA200")
        else:
            bull_score -= 1
            reasons.append("price<EMA200")

        # 3) ADX — 추세 강도
        if adx >= 25:
            # 강한 추세 — 방향에 따라 +/-
            if ema50 > ema200:
                bull_score += 2
                reasons.append(f"ADX={adx:.0f}(strong bull trend)")
            else:
                bull_score -= 2
                reasons.append(f"ADX={adx:.0f}(strong bear trend)")
        elif adx < 15:
            # 약한 추세 — 횡보 신호
            reasons.append(f"ADX={adx:.0f}(ranging)")

        # 4) 20일 모멘텀
        if roc_20 >= 8:
            bull_score += 2
            reasons.append(f"ROC20=+{roc_20:.1f}%(strong)")
        elif roc_20 >= 3:
            bull_score += 1
            reasons.append(f"ROC20=+{roc_20:.1f}%")
        elif roc_20 <= -10:
            bull_score -= 2
            reasons.append(f"ROC20={roc_20:.1f}%(crash)")
        elif roc_20 <= -3:
            bull_score -= 1
            reasons.append(f"ROC20={roc_20:.1f}%(weak)")
        else:
            reasons.append(f"ROC20={roc_20:.1f}%(neutral)")

        # ── 레짐 결정 (bull_score -8 ~ +8) ──────────────────────────────────
        # BULL: score >= 4 (EMA 골든크로스 + 강한 모멘텀)
        # SIDEWAYS: -1 <= score < 4
        # BEAR: score < -1

        max_possible = 8
        if bull_score >= 4:
            regime = MacroRegime.BULL
            confidence = min(1.0, bull_score / max_possible)
        elif bull_score < -1:
            regime = MacroRegime.BEAR
            confidence = min(1.0, abs(bull_score) / max_possible)
        else:
            regime = MacroRegime.SIDEWAYS
            confidence = 0.5

        # ── 허용 심볼 설정 ────────────────────────────────────────────────────
        if regime == MacroRegime.BULL:
            symbols = [s.strip() for s in self._s.BULL_MODE_SYMBOLS.split(",") if s.strip()]
        else:
            symbols = [s.strip() for s in self._s.BEAR_MODE_SYMBOLS.split(",") if s.strip()]

        return MacroRegimeResult(
            regime=regime,
            confidence=confidence,
            ema50=ema50,
            ema200=ema200,
            adx=adx,
            roc_20=roc_20,
            reason=" | ".join(reasons),
            symbols=symbols,
        )

    def _default(self) -> MacroRegimeResult:
        """API 실패 또는 데이터 부족 시 보수적 기본값 (SIDEWAYS → BTC/ETH만)."""
        symbols = [s.strip() for s in self._s.BEAR_MODE_SYMBOLS.split(",") if s.strip()]
        return MacroRegimeResult(
            regime=MacroRegime.SIDEWAYS,
            confidence=0.5,
            reason="fallback: insufficient data",
            symbols=symbols,
        )
