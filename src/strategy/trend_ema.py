"""
Trend EMA Strategy — Phase-locked Direction System

핵심 설계 철학:
  시장 장세(Phase)가 진입 방향을 결정한다.
  EMA 정렬이 1차 조건, 나머지는 확인 지표.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
장세 판단 (HTF 15분봉 기반):
  BULL_TREND  : EMA21 > EMA50 > EMA200 + ADX ≥ threshold → LONG만 허용
  BEAR_TREND  : EMA21 < EMA50 < EMA200 + ADX ≥ threshold → SHORT만 허용
  SIDEWAYS    : 위 조건 미충족 → 볼린저 밴드 극단값 양방향 (높은 임계치)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
진입 스코어링 (5점 만점):

[BULL_TREND / BEAR_TREND] — 최소 3점
  (+2) EMA9/EMA21 정렬이 Phase 방향과 일치 (핵심 조건)
  (+1) 가격이 EMA21 ±ZONE_ATR×ATR 내 (EMA 풀백/반등 구간)
  (+1) RSI가 방향에 맞는 범위 내 (과매수/과매도 회피)
  (+1) 거래량 ≥ 20봉 평균 × VOLUME_MULT
  (-1) |funding rate| > 0.001 (과도한 포지션 쏠림)
  (-1) spread > 0.2% (체결 비용 과다)

[SIDEWAYS] — 최소 4점
  (+2) 가격이 볼린저 밴드 하단/상단 극단값
  (+1) RSI 35 이하(롱) / 65 이상(숏)
  (+1) 거래량 스파이크 (평균 × 1.5)
  (+1) 방향 일치 캔들 바디 > 0.15×ATR
  (-1) |funding rate| > 0.001
  (-1) spread > 0.2%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
신호 강도 → 레버리지 티어 (engine.py 경계 정렬):
  추세 score 3 → 0.65 (4-5×)
  추세 score 4 → 0.77 (8-10×)
  추세 score 5 → 0.85 (20×)
  횡보  score 4 → 0.65 (4-5×) — 낮은 레버리지 유지
  횡보  score 5 → 0.69 (4-5×)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Settings
from src.logger import get_logger
from src.strategy.combiner import CombinedSignal
from src.strategy.indicators import adx, atr, bollinger_bands, ema, rsi
from src.strategy.regime import Regime

log = get_logger(__name__)

# 레버리지 티어 경계 정렬 강도 매핑
_TREND_SCORE_STRENGTH: dict[int, float] = {
    3: 0.65,   # ≥0.65 → 4-5× tier (최소 진입)
    4: 0.77,   # ≥0.75 → 8-10× tier
    5: 0.85,   # ≥0.85 → 20× tier
}
_SIDEWAYS_SCORE_STRENGTH: dict[int, float] = {
    4: 0.65,   # ≥0.65 → 4-5× tier (횡보는 보수적)
    5: 0.69,   # ≥0.65 → 4-5× tier (약간 높은 신뢰)
}


class MarketPhase(str, Enum):
    BULL_TREND = "bull_trend"    # EMA21>50>200 + ADX 강세 → LONG만
    BEAR_TREND = "bear_trend"    # EMA21<50<200 + ADX 강세 → SHORT만
    SIDEWAYS = "sideways"        # 조건 미충족 → BB 극단 양방향


@dataclass
class PhaseResult:
    phase: MarketPhase
    htf_ema21: float
    htf_ema50: float
    htf_ema200: float
    htf_adx: float
    htf_plus_di: float
    htf_minus_di: float


class TrendEMAStrategy:
    """
    EMA 정렬 기반 장세 추종 전략.

    장세를 먼저 판단하고, 장세 방향에 맞는 신호만 생성.
    역추세 진입을 구조적으로 차단.
    """

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    # ── Public Interface ───────────────────────────────────────────────────────

    def generate(
        self,
        ltf_df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        vwap_val: float = 0.0,
        funding_rate: float = 0.0,
        spread_pct: float = 0.0,
        oi_change_pct: float = 0.0,
    ) -> Tuple[CombinedSignal, float]:
        """
        LTF(3m) + HTF(15m) 데이터를 사용해 진입 신호 생성.

        Returns:
            (CombinedSignal, ltf_atr)
        """
        if len(ltf_df) < 50:
            return self._neutral(), 0.0

        ltf_close = ltf_df["close"]
        ltf_high = ltf_df["high"]
        ltf_low = ltf_df["low"]
        ltf_volume = ltf_df["volume"]
        ltf_open = ltf_df["open"]

        ltf_atr_val = float(atr(ltf_high, ltf_low, ltf_close, 14).iloc[-1])
        if ltf_atr_val <= 0 or np.isnan(ltf_atr_val):
            return self._neutral(), 0.0

        # HTF 장세 판단
        has_htf = htf_df is not None and len(htf_df) >= 200
        if has_htf:
            phase_r = self._detect_phase(htf_df)
        else:
            # HTF 없으면 LTF로 대체 (정밀도 낮음)
            phase_r = self._detect_phase(ltf_df) if len(ltf_df) >= 200 else PhaseResult(
                MarketPhase.SIDEWAYS, 0, 0, 0, 0, 0, 0
            )

        log.debug(
            "TrendEMA phase=%s ema21=%.2f ema50=%.2f ema200=%.2f adx=%.1f",
            phase_r.phase, phase_r.htf_ema21, phase_r.htf_ema50,
            phase_r.htf_ema200, phase_r.htf_adx,
        )

        # 장세별 신호 생성
        if phase_r.phase == MarketPhase.BULL_TREND:
            signal = self._bull_entry(ltf_close, ltf_high, ltf_low, ltf_volume, ltf_atr_val, funding_rate, spread_pct)
        elif phase_r.phase == MarketPhase.BEAR_TREND:
            signal = self._bear_entry(ltf_close, ltf_high, ltf_low, ltf_volume, ltf_atr_val, funding_rate, spread_pct)
        else:
            signal = self._sideways_entry(ltf_close, ltf_high, ltf_low, ltf_volume, ltf_open, ltf_atr_val, funding_rate, spread_pct)

        if signal.direction != 0:
            log.info(
                "TrendEMA signal | phase=%s direction=%+d strength=%.2f score=%.0f",
                phase_r.phase, signal.direction, signal.strength,
                signal.long_score if signal.direction == 1 else signal.short_score,
            )

        return signal, ltf_atr_val

    def detect_phase(self, htf_df: pd.DataFrame) -> MarketPhase:
        """외부에서 장세 조회용 (로깅/테스트)."""
        return self._detect_phase(htf_df).phase

    # ── Phase Detection ────────────────────────────────────────────────────────

    def _detect_phase(self, df: pd.DataFrame) -> PhaseResult:
        """
        HTF 데이터로 시장 장세 판단.

        BULL_TREND : EMA21 > EMA50 > EMA200 AND ADX ≥ threshold
        BEAR_TREND : EMA21 < EMA50 < EMA200 AND ADX ≥ threshold
        SIDEWAYS   : 그 외 (ADX 약하거나 EMA 혼재)
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        htf_ema21 = float(ema(close, 21).iloc[-1])
        htf_ema50 = float(ema(close, 50).iloc[-1])
        htf_ema200 = float(ema(close, 200).iloc[-1])

        adx_df = adx(high, low, close, 14)
        htf_adx = float(adx_df["adx"].iloc[-1])
        htf_plus_di = float(adx_df["plus_di"].iloc[-1])
        htf_minus_di = float(adx_df["minus_di"].iloc[-1])

        adx_threshold = self._s.TREND_EMA_ADX_THRESHOLD  # 기본 20.0
        is_trending = htf_adx >= adx_threshold

        # 완전 정렬 여부 확인
        bull_aligned = (htf_ema21 > htf_ema50) and (htf_ema50 > htf_ema200)
        bear_aligned = (htf_ema21 < htf_ema50) and (htf_ema50 < htf_ema200)

        # DI 방향성 확인: ADX가 강해도 방향이 맞아야 함
        # BULL_TREND: +DI > -DI (매수 압력 우위)
        # BEAR_TREND: -DI > +DI (매도 압력 우위)
        di_bull = (htf_plus_di > htf_minus_di)
        di_bear = (htf_minus_di > htf_plus_di)

        if bull_aligned and is_trending and di_bull:
            phase = MarketPhase.BULL_TREND
        elif bear_aligned and is_trending and di_bear:
            phase = MarketPhase.BEAR_TREND
        else:
            phase = MarketPhase.SIDEWAYS

        return PhaseResult(phase, htf_ema21, htf_ema50, htf_ema200, htf_adx, htf_plus_di, htf_minus_di)

    # ── Bull Trend Entry ───────────────────────────────────────────────────────

    def _bull_entry(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        ltf_atr_val: float,
        funding_rate: float,
        spread_pct: float,
    ) -> CombinedSignal:
        """
        BULL_TREND: EMA21 풀백 구간 LONG 진입.
        모멘텀 재개 하드필터 적용 — 풀백이 끝나고 실제 반등 확인 후 진입.
        """
        current_price = float(close.iloc[-1])
        ltf_ema9 = float(ema(close, 9).iloc[-1])
        ltf_ema21 = float(ema(close, 21).iloc[-1])
        ltf_rsi = float(rsi(close, 14).iloc[-1])
        current_vol = float(volume.iloc[-1])
        avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else current_vol

        # ── 하드 필터: 모멘텀 재개 확인 (3-bar pattern) ──────────────────
        # 풀백 후 모멘텀 재개를 확인하지 않으면 떨어지는 칼날을 잡을 위험
        if len(close) >= 4:
            closes = close.iloc[-4:-1].values  # 직전 3봉
            lows = low.iloc[-3:-1].values      # 직전 2봉 low
            momentum_ok = (
                (closes[2] > closes[1] > closes[0]) or  # 3봉 연속 상승
                (lows[1] > lows[0])                       # higher low 형성
            )
            if not momentum_ok:
                log.debug("TrendEMA BULL: momentum not resuming — skipping")
                return self._neutral()

        score = 0

        # 조건 1: LTF EMA 정렬 (핵심 +2점)
        if ltf_ema9 > ltf_ema21:
            score += 2

        # 조건 2: 가격이 EMA21 풀백 존 내 (+1점)
        zone_atr = self._s.TREND_EMA_ZONE_ATR_MULT
        zone_low = ltf_ema21 - ltf_atr_val * zone_atr
        zone_high = ltf_ema21 + ltf_atr_val * zone_atr
        if zone_low <= current_price <= zone_high:
            score += 1

        # 조건 3: RSI 매수 범위 (+1점)
        rsi_min = self._s.TREND_EMA_RSI_BULL_MIN
        rsi_max = self._s.TREND_EMA_RSI_BULL_MAX
        if rsi_min <= ltf_rsi <= rsi_max:
            score += 1

        # 조건 4: 거래량 확인 (+1점)
        vol_mult = self._s.TREND_EMA_VOLUME_MULT
        if avg_vol > 0 and current_vol >= avg_vol * vol_mult:
            score += 1

        # 패널티
        if abs(funding_rate) > 0.001:
            score -= 1
        if spread_pct > 0.002:
            score -= 1

        min_score = self._s.TREND_EMA_MIN_SCORE
        if score >= min_score:
            strength = _TREND_SCORE_STRENGTH.get(min(score, 5), 0.65)
            return CombinedSignal(
                direction=1,
                strength=strength,
                regime=Regime.TRENDING_UP,
                long_score=float(score),
                short_score=0.0,
                category_scores={"trend_ema": float(score)},
            )
        return self._neutral()

    # ── Bear Trend Entry ───────────────────────────────────────────────────────

    def _bear_entry(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        ltf_atr_val: float,
        funding_rate: float,
        spread_pct: float,
    ) -> CombinedSignal:
        """
        BEAR_TREND: EMA21 반등 구간 SHORT 진입.
        모멘텀 재개 하드필터 적용 — 반등이 끝나고 실제 재하락 확인 후 진입.
        """
        current_price = float(close.iloc[-1])
        ltf_ema9 = float(ema(close, 9).iloc[-1])
        ltf_ema21 = float(ema(close, 21).iloc[-1])
        ltf_rsi = float(rsi(close, 14).iloc[-1])
        current_vol = float(volume.iloc[-1])
        avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else current_vol

        # ── 하드 필터: 하락 모멘텀 재개 확인 ────────────────────────────
        if len(close) >= 4:
            closes = close.iloc[-4:-1].values
            highs = high.iloc[-3:-1].values
            momentum_ok = (
                (closes[2] < closes[1] < closes[0]) or  # 3봉 연속 하락
                (highs[1] < highs[0])                     # lower high 형성
            )
            if not momentum_ok:
                log.debug("TrendEMA BEAR: momentum not resuming — skipping")
                return self._neutral()

        score = 0

        # 조건 1: LTF EMA 정렬 (하락 방향, 핵심 +2점)
        if ltf_ema9 < ltf_ema21:
            score += 2

        # 조건 2: 가격이 EMA21 반등 존 내 (+1점)
        zone_atr = self._s.TREND_EMA_ZONE_ATR_MULT
        zone_low = ltf_ema21 - ltf_atr_val * zone_atr
        zone_high = ltf_ema21 + ltf_atr_val * zone_atr
        if zone_low <= current_price <= zone_high:
            score += 1

        # 조건 3: RSI 숏 범위 (+1점)
        rsi_min = self._s.TREND_EMA_RSI_BEAR_MIN
        rsi_max = self._s.TREND_EMA_RSI_BEAR_MAX
        if rsi_min <= ltf_rsi <= rsi_max:
            score += 1

        # 조건 4: 거래량 확인 (+1점)
        vol_mult = self._s.TREND_EMA_VOLUME_MULT
        if avg_vol > 0 and current_vol >= avg_vol * vol_mult:
            score += 1

        # 패널티
        if abs(funding_rate) > 0.001:
            score -= 1
        if spread_pct > 0.002:
            score -= 1

        min_score = self._s.TREND_EMA_MIN_SCORE
        if score >= min_score:
            strength = _TREND_SCORE_STRENGTH.get(min(score, 5), 0.65)
            return CombinedSignal(
                direction=-1,
                strength=strength,
                regime=Regime.TRENDING_DOWN,
                long_score=0.0,
                short_score=float(score),
                category_scores={"trend_ema": float(score)},
            )
        return self._neutral()

    # ── Sideways Entry ─────────────────────────────────────────────────────────

    def _sideways_entry(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        open_: pd.Series,
        ltf_atr_val: float,
        funding_rate: float,
        spread_pct: float,
    ) -> CombinedSignal:
        """
        SIDEWAYS: 볼린저 밴드 극단값에서 평균 회귀 진입.

        전략: 하단 밴드 과매도 LONG / 상단 밴드 과매수 SHORT.
        추세 진입보다 높은 점수 요구 (신뢰도 강화).
        """
        current_price = float(close.iloc[-1])
        current_open = float(open_.iloc[-1])
        current_vol = float(volume.iloc[-1])
        avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else current_vol

        ltf_rsi = float(rsi(close, 14).iloc[-1])
        bb = bollinger_bands(close, 20, 2.0)
        bb_upper = float(bb["upper"].iloc[-1])
        bb_lower = float(bb["lower"].iloc[-1])

        penalty = 0
        if abs(funding_rate) > 0.001:
            penalty += 1
        if spread_pct > 0.002:
            penalty += 1

        # 롱 스코어 (볼린저 하단 과매도 반등)
        long_score = 0
        if current_price <= bb_lower:
            long_score += 2  # 하단 터치
        elif current_price <= bb_lower + ltf_atr_val * 0.5:
            long_score += 1  # 하단 근처

        rsi_range_long_max = self._s.TREND_EMA_RSI_RANGE_LONG_MAX   # 35.0
        if ltf_rsi < rsi_range_long_max:
            long_score += 1

        vol_mult_range = 1.5  # 횡보장은 더 강한 거래량 요구
        if avg_vol > 0 and current_vol >= avg_vol * vol_mult_range:
            long_score += 1

        candle_body = close.iloc[-1] - open_.iloc[-1]
        if candle_body > 0 and candle_body > ltf_atr_val * 0.15:
            long_score += 1  # 양봉 확인

        # 숏 스코어 (볼린저 상단 과매수 되돌림)
        short_score = 0
        if current_price >= bb_upper:
            short_score += 2  # 상단 터치
        elif current_price >= bb_upper - ltf_atr_val * 0.5:
            short_score += 1  # 상단 근처

        rsi_range_short_min = self._s.TREND_EMA_RSI_RANGE_SHORT_MIN  # 65.0
        if ltf_rsi > rsi_range_short_min:
            short_score += 1

        if avg_vol > 0 and current_vol >= avg_vol * vol_mult_range:
            short_score += 1

        candle_body_s = open_.iloc[-1] - close.iloc[-1]
        if candle_body_s > 0 and candle_body_s > ltf_atr_val * 0.15:
            short_score += 1  # 음봉 확인

        long_score -= penalty
        short_score -= penalty

        min_score = self._s.TREND_EMA_RANGE_MIN_SCORE  # 4 (추세보다 높음)

        if long_score >= min_score and long_score > short_score:
            strength = _SIDEWAYS_SCORE_STRENGTH.get(min(long_score, 5), 0.65)
            return CombinedSignal(
                direction=1,
                strength=strength,
                regime=Regime.RANGING,
                long_score=float(long_score),
                short_score=0.0,
                category_scores={"trend_ema_range": float(long_score)},
            )
        elif short_score >= min_score and short_score > long_score:
            strength = _SIDEWAYS_SCORE_STRENGTH.get(min(short_score, 5), 0.65)
            return CombinedSignal(
                direction=-1,
                strength=strength,
                regime=Regime.RANGING,
                long_score=0.0,
                short_score=float(short_score),
                category_scores={"trend_ema_range": float(short_score)},
            )
        return self._neutral()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _neutral(self) -> CombinedSignal:
        return CombinedSignal(
            direction=0,
            strength=0.0,
            regime=Regime.RANGING,
            long_score=0.0,
            short_score=0.0,
            category_scores={},
        )
