"""
Pure Scalping Strategy — EMA Ribbon + VWAP + RSI Momentum Scalping

알고리즘 개요:
  - HTF (기본 15m): 추세 바이어스 + VWAP 기준선 제공
  - LTF (기본 3m):  진입 신호 생성 (EMA 존 풀백 + RSI 모멘텀 + 볼륨 확인)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
롱 진입 조건 (9점 만점):
  htf_ema_bull  (+2): HTF EMA9 > EMA21  → HTF 단기 상승 추세
  htf_vwap_bull (+1): HTF 가격 > VWAP  → 세션 내 매수 우위
  ltf_ema_align (+2): LTF EMA9 > EMA21 → LTF 단기 상승 추세 확인
  price_in_zone (+1): 가격이 EMA21 ± ATR×EMA_ZONE 내 → 풀백 진입 구간
  rsi_in_zone   (+1): RSI 40–65         → 모멘텀 형성 중, 과매수 아님
  volume_spike  (+1): 현재 볼륨 ≥ 20봉 평균 × VOLUME_MULT → 참여도 확인
  bull_candle   (+1): 마지막 봉이 양봉  → 즉각적 방향성 확인
  adverse_fund  (-1): |funding| > 0.001 → 한쪽으로 쏠린 포지션 위험
  wide_spread   (-1): 스프레드 > 0.2%  → 체결 비용 과다

숏 진입: 롱의 정반대 조건

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스코어링:
  score ≥ ENTRY_MIN_SCORE (5): 진입 허용 (지정가/PostOnly 선호)
  score = MAX_POSITIVE_SCORE (9): 긴급 진입 → 시장가 선호
  score < ENTRY_MIN_SCORE: 진입 없음 (direction=0)

신호 강도 매핑 (engine.py 티어 경계 정렬):
  score=5: 0.60 (2-3× 레버리지)
  score=6: 0.69 (4-5× 레버리지)
  score=7: 0.77 (8-10× 레버리지)
  score=8: 0.83 (15-18× 레버리지)
  score=9: 0.91 (20×, 시장가 트리거)

수수료 최적화 (Binance):
  score < 9 AND spread < 0.2%: PostOnly 지정가 (Maker 0.02%)
  score = 9 OR spread > 0.2%: 시장가 (Taker 0.04%)
  → signal_strength가 SIGNAL_URGENT_THRESHOLD(0.90) 이상일 때 자동 시장가 전환
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Settings
from src.logger import get_logger
from src.strategy.combiner import CombinedSignal
from src.strategy.indicators import atr, ema, rsi, vwap as compute_vwap
from src.strategy.regime import Regime

log = get_logger(__name__)

# 점수 계산 상수
_MAX_POSITIVE_SCORE = 9   # htf_ema(2) + htf_vwap(1) + ltf_ema(2) + zone(1) + rsi(1) + vol(1) + candle(1)
_MAX_PENALTY_SCORE = -2   # adverse_funding(-1) + wide_spread(-1)

# 레버리지 티어 경계 정렬 강도 매핑 (engine.py 티어 기준)
# engine.py 티어: <0.65→2-3×, ≥0.65→4-5×, ≥0.75→8-10×, ≥0.80→15-18×, ≥0.85→20×
# score 9만 SIGNAL_URGENT_THRESHOLD(0.90) 초과 → 시장가(Taker) 트리거
_SCORE_STRENGTH_MAP: dict[int, float] = {
    5: 0.60,   # < 0.65 → default tier (2-3×)
    6: 0.69,   # ≥ 0.65 → 4-5× tier
    7: 0.77,   # ≥ 0.75 → 8-10× tier
    8: 0.83,   # ≥ 0.80 → 15-18× tier
    # 9 → urgent_threshold + 0.01 = 0.91 → ≥0.85(20×) + Market 트리거
}


@dataclass
class ScalpSignalDetail:
    """스캘핑 신호 상세 정보 (디버깅/로깅용)."""
    score: int
    max_score: int
    direction: int
    atr_val: float
    long_details: Dict[str, int]
    short_details: Dict[str, int]
    htf_ema9: float
    htf_ema21: float
    htf_vwap: float
    ltf_ema9: float
    ltf_ema21: float
    ltf_rsi: float


class ScalpingStrategy:
    """
    순수 스캘핑 전략: EMA Ribbon + VWAP + RSI 모멘텀 스캘핑.

    주요 특징:
    - LTF (3m) 봉으로 정밀 진입 타이밍
    - HTF (15m) 봉으로 추세 방향 필터
    - 9점 스코어링 시스템으로 진입 품질 제어
    - 기존 RiskEngine / StateMachine과 완전 호환 (PURE_SCALP 모드 사용)
    - 수수료 최적화: 낮은 점수/좁은 스프레드 → PostOnly, score=9 → Market
    """

    def __init__(self, settings: Settings):
        self._s = settings

    def generate(
        self,
        ltf_df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        *,
        vwap_val: float = 0.0,
        funding_rate: float = 0.0,
        spread_pct: float = 0.0,
        oi_change_pct: float = 0.0,
    ) -> Tuple[CombinedSignal, float]:
        """
        스캘핑 신호 생성.

        Args:
            ltf_df: LTF OHLCV 데이터프레임 (3m 봉)
            htf_df: HTF OHLCV 데이터프레임 (15m 봉, None 가능)
            vwap_val: 사전 계산된 VWAP 값 (collector에서 제공)
            funding_rate: 현재 펀딩 비율
            spread_pct: 오더북 스프레드 비율
            oi_change_pct: OI 변화율 (미사용, 확장 예정)

        Returns:
            (CombinedSignal, atr_value)
              CombinedSignal: direction(±1,0), strength(0-1), regime
              atr_value: LTF ATR 값 → RiskEngine 포지션 사이징에 사용
        """
        if ltf_df.empty or len(ltf_df) < 50:
            log.debug("ScalpStrategy: LTF 데이터 부족 (bars=%d)", len(ltf_df) if not ltf_df.empty else 0)
            return self._neutral(), 0.0

        # ── LTF 지표 계산 ────────────────────────────────────────────────────
        close_ltf = ltf_df["close"]
        high_ltf = ltf_df["high"]
        low_ltf = ltf_df["low"]
        vol_ltf = ltf_df["volume"]

        ema9_ltf_s = ema(close_ltf, self._s.SCALP_PURE_EMA_FAST)
        ema21_ltf_s = ema(close_ltf, self._s.SCALP_PURE_EMA_MID)
        rsi14_ltf_s = rsi(close_ltf, self._s.SCALP_PURE_RSI_PERIOD)
        atr14_ltf_s = atr(high_ltf, low_ltf, close_ltf, 14)
        vol_ma20_s = vol_ltf.rolling(20).mean()

        # 현재 봉 값
        current_price = float(close_ltf.iloc[-1])
        current_open = float(ltf_df["open"].iloc[-1])
        current_vol = float(vol_ltf.iloc[-1])
        ltf_ema9 = float(ema9_ltf_s.iloc[-1])
        ltf_ema21 = float(ema21_ltf_s.iloc[-1])
        ltf_rsi = float(rsi14_ltf_s.iloc[-1])
        ltf_atr = float(atr14_ltf_s.iloc[-1])
        avg_vol = float(vol_ma20_s.iloc[-1]) if not pd.isna(vol_ma20_s.iloc[-1]) else 0.0

        if ltf_atr <= 0 or np.isnan(ltf_atr):
            log.debug("ScalpStrategy: ATR 이상값 (atr=%.6f)", ltf_atr)
            return self._neutral(), 0.0

        # NaN 값 검증
        for val, name in [
            (ltf_ema9, "ema9"), (ltf_ema21, "ema21"), (ltf_rsi, "rsi")
        ]:
            if np.isnan(val):
                return self._neutral(), ltf_atr

        # ── HTF 지표 계산 ────────────────────────────────────────────────────
        htf_ema9 = 0.0
        htf_ema21 = 0.0
        htf_vwap = vwap_val   # collector에서 사전 계산된 VWAP
        htf_price = 0.0
        has_htf = htf_df is not None and len(htf_df) >= 30

        if has_htf:
            close_htf = htf_df["close"]
            high_htf = htf_df["high"]
            low_htf = htf_df["low"]
            vol_htf = htf_df["volume"]

            ema9_htf_s = ema(close_htf, self._s.SCALP_PURE_EMA_FAST)
            ema21_htf_s = ema(close_htf, self._s.SCALP_PURE_EMA_MID)
            htf_ema9 = float(ema9_htf_s.iloc[-1])
            htf_ema21 = float(ema21_htf_s.iloc[-1])
            htf_price = float(close_htf.iloc[-1])

            # VWAP이 사전 계산 안 된 경우 HTF 데이터로 직접 계산
            if htf_vwap <= 0:
                vwap_period = min(96, len(htf_df))
                vwap_series = compute_vwap(high_htf, low_htf, close_htf, vol_htf, vwap_period)
                if not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]):
                    htf_vwap = float(vwap_series.iloc[-1])

        # ── 롱/숏 스코어 계산 ────────────────────────────────────────────────
        long_score, long_details = self._score_long(
            current_price=current_price,
            current_open=current_open,
            current_vol=current_vol,
            avg_vol=avg_vol,
            ltf_ema9=ltf_ema9,
            ltf_ema21=ltf_ema21,
            ltf_atr=ltf_atr,
            ltf_rsi=ltf_rsi,
            htf_ema9=htf_ema9,
            htf_ema21=htf_ema21,
            htf_vwap=htf_vwap,
            htf_price=htf_price,
            has_htf=has_htf,
            funding_rate=funding_rate,
            spread_pct=spread_pct,
        )

        short_score, short_details = self._score_short(
            current_price=current_price,
            current_open=current_open,
            current_vol=current_vol,
            avg_vol=avg_vol,
            ltf_ema9=ltf_ema9,
            ltf_ema21=ltf_ema21,
            ltf_atr=ltf_atr,
            ltf_rsi=ltf_rsi,
            htf_ema9=htf_ema9,
            htf_ema21=htf_ema21,
            htf_vwap=htf_vwap,
            htf_price=htf_price,
            has_htf=has_htf,
            funding_rate=funding_rate,
            spread_pct=spread_pct,
        )

        min_score = self._s.SCALP_PURE_ENTRY_MIN_SCORE

        # ── 방향 결정 ────────────────────────────────────────────────────────
        if long_score >= min_score and long_score > short_score:
            direction = 1
            score = long_score
            details = long_details
            # 레짐: HTF EMA 정렬이 있으면 TRENDING_UP, 아니면 RANGING
            regime = Regime.TRENDING_UP if (has_htf and htf_ema9 > htf_ema21 and htf_ema21 > 0) else Regime.RANGING

        elif short_score >= min_score and short_score > long_score:
            direction = -1
            score = short_score
            details = short_details
            regime = Regime.TRENDING_DOWN if (has_htf and htf_ema9 < htf_ema21 and htf_ema21 > 0) else Regime.RANGING

        else:
            log.debug(
                "ScalpStrategy: 진입 조건 미충족 (long=%d short=%d min=%d)",
                long_score, short_score, min_score,
            )
            return self._neutral(), ltf_atr

        # ── 신호 강도 매핑 (엔진 티어 경계 정렬) ─────────────────────────────
        # score 5→0.60(2-3×), 6→0.69(4-5×), 7→0.77(8-10×), 8→0.83(15-18×)
        # score 9→0.91: ≥0.85(20×) + SIGNAL_URGENT_THRESHOLD 초과 → 시장가 트리거
        urgent_threshold = self._s.SIGNAL_URGENT_THRESHOLD  # 0.90

        if score >= _MAX_POSITIVE_SCORE:
            # 최고 점수(9)만 시장가 트리거 → 즉시 체결 보장
            strength = urgent_threshold + 0.01  # 0.91
        else:
            # score 5-8: 티어 경계 정렬 강도 → PostOnly (수수료 절감)
            strength = _SCORE_STRENGTH_MAP.get(score, 0.60)

        strength = min(0.95, strength)

        log.debug(
            "ScalpStrategy | dir=%+d score=%d/%d strength=%.2f "
            "long_score=%d short_score=%d ltf_atr=%.4f",
            direction, score, _MAX_POSITIVE_SCORE, strength,
            long_score, short_score, ltf_atr,
        )

        signal = CombinedSignal(
            direction=direction,
            strength=strength,
            regime=regime,
            long_score=long_score / _MAX_POSITIVE_SCORE,
            short_score=short_score / _MAX_POSITIVE_SCORE,
            category_scores=dict(details),
        )
        return signal, ltf_atr

    def get_signal_detail(
        self,
        ltf_df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        *,
        vwap_val: float = 0.0,
        funding_rate: float = 0.0,
        spread_pct: float = 0.0,
    ) -> Optional[ScalpSignalDetail]:
        """
        디버깅/로깅용 상세 신호 정보 반환.
        generate()와 동일한 로직이지만 ScalpSignalDetail 반환.
        """
        if ltf_df.empty or len(ltf_df) < 50:
            return None

        close_ltf = ltf_df["close"]
        high_ltf = ltf_df["high"]
        low_ltf = ltf_df["low"]
        vol_ltf = ltf_df["volume"]

        ema9_ltf_s = ema(close_ltf, self._s.SCALP_PURE_EMA_FAST)
        ema21_ltf_s = ema(close_ltf, self._s.SCALP_PURE_EMA_MID)
        rsi14_ltf_s = rsi(close_ltf, self._s.SCALP_PURE_RSI_PERIOD)
        atr14_ltf_s = atr(high_ltf, low_ltf, close_ltf, 14)
        vol_ma20_s = vol_ltf.rolling(20).mean()

        current_price = float(close_ltf.iloc[-1])
        current_open = float(ltf_df["open"].iloc[-1])
        current_vol = float(vol_ltf.iloc[-1])
        ltf_ema9 = float(ema9_ltf_s.iloc[-1])
        ltf_ema21 = float(ema21_ltf_s.iloc[-1])
        ltf_rsi = float(rsi14_ltf_s.iloc[-1])
        ltf_atr = float(atr14_ltf_s.iloc[-1])
        avg_vol = float(vol_ma20_s.iloc[-1]) if not pd.isna(vol_ma20_s.iloc[-1]) else 0.0

        htf_ema9 = 0.0
        htf_ema21 = 0.0
        htf_vwap = vwap_val
        htf_price = 0.0
        has_htf = htf_df is not None and len(htf_df) >= 30

        if has_htf:
            close_htf = htf_df["close"]
            high_htf = htf_df["high"]
            low_htf = htf_df["low"]
            vol_htf = htf_df["volume"]
            htf_ema9 = float(ema(close_htf, self._s.SCALP_PURE_EMA_FAST).iloc[-1])
            htf_ema21 = float(ema(close_htf, self._s.SCALP_PURE_EMA_MID).iloc[-1])
            htf_price = float(close_htf.iloc[-1])
            if htf_vwap <= 0:
                vwap_series = compute_vwap(high_htf, low_htf, close_htf, vol_htf, min(96, len(htf_df)))
                if not vwap_series.empty:
                    htf_vwap = float(vwap_series.iloc[-1])

        long_score, long_details = self._score_long(
            current_price, current_open, current_vol, avg_vol,
            ltf_ema9, ltf_ema21, ltf_atr, ltf_rsi,
            htf_ema9, htf_ema21, htf_vwap, htf_price, has_htf,
            funding_rate, spread_pct,
        )
        short_score, short_details = self._score_short(
            current_price, current_open, current_vol, avg_vol,
            ltf_ema9, ltf_ema21, ltf_atr, ltf_rsi,
            htf_ema9, htf_ema21, htf_vwap, htf_price, has_htf,
            funding_rate, spread_pct,
        )

        best_score = max(long_score, short_score)
        direction = 1 if long_score > short_score else (-1 if short_score > long_score else 0)

        return ScalpSignalDetail(
            score=best_score,
            max_score=_MAX_POSITIVE_SCORE,
            direction=direction,
            atr_val=ltf_atr,
            long_details=long_details,
            short_details=short_details,
            htf_ema9=htf_ema9,
            htf_ema21=htf_ema21,
            htf_vwap=htf_vwap,
            ltf_ema9=ltf_ema9,
            ltf_ema21=ltf_ema21,
            ltf_rsi=ltf_rsi,
        )

    # ── 롱 스코어 계산 ────────────────────────────────────────────────────────

    def _score_long(
        self,
        current_price: float,
        current_open: float,
        current_vol: float,
        avg_vol: float,
        ltf_ema9: float,
        ltf_ema21: float,
        ltf_atr: float,
        ltf_rsi: float,
        htf_ema9: float,
        htf_ema21: float,
        htf_vwap: float,
        htf_price: float,
        has_htf: bool,
        funding_rate: float,
        spread_pct: float,
    ) -> Tuple[int, Dict[str, int]]:
        """롱 진입 스코어 계산 (9점 만점)."""
        d: Dict[str, int] = {}

        # ── HTF 바이어스 (최대 +3) ─────────────────────────────────────────
        if has_htf and htf_ema9 > 0 and htf_ema21 > 0:
            # +2: HTF EMA9 > EMA21 (단기 상승 추세)
            d["htf_ema_bull"] = 2 if htf_ema9 > htf_ema21 else 0
            # +1: 가격이 VWAP 위 (세션 내 수요 우위)
            d["htf_vwap_bull"] = 1 if (htf_vwap > 0 and htf_price > htf_vwap) else 0
        else:
            # HTF 데이터 없음: 바이어스 점수 0 (진입 가능하지만 낮은 점수)
            d["htf_ema_bull"] = 0
            d["htf_vwap_bull"] = 0

        # ── LTF EMA 정렬 (최대 +2) ────────────────────────────────────────
        # +2: EMA9 > EMA21 (단기 상승 추세 확인)
        d["ltf_ema_aligned"] = 2 if (ltf_ema9 > ltf_ema21 and ltf_ema21 > 0) else 0

        # ── 가격 EMA 존 (최대 +1) ──────────────────────────────────────────
        # +1: 현재 가격이 EMA21 ± (ATR × EMA_ZONE_ATR) 내 (풀백 진입 구간)
        zone_threshold = ltf_atr * self._s.SCALP_PURE_EMA_ZONE_ATR
        d["price_in_zone"] = 1 if (ltf_ema21 > 0 and abs(current_price - ltf_ema21) <= zone_threshold) else 0

        # ── RSI 모멘텀 구간 (최대 +1) ─────────────────────────────────────
        # +1: RSI가 40–65 범위 (모멘텀 형성 중, 과매수 아님)
        rsi_min = self._s.SCALP_PURE_RSI_LONG_MIN
        rsi_max = self._s.SCALP_PURE_RSI_LONG_MAX
        d["rsi_in_zone"] = 1 if (rsi_min <= ltf_rsi <= rsi_max) else 0

        # ── 볼륨 확인 (최대 +1) ────────────────────────────────────────────
        # +1: 현재 봉 볼륨 ≥ 20봉 평균 × VOLUME_MULT
        vol_mult = self._s.SCALP_PURE_VOLUME_MULT
        d["volume_spike"] = 1 if (avg_vol > 0 and current_vol >= avg_vol * vol_mult) else 0

        # ── 캔들 방향 (최대 +1) ────────────────────────────────────────────
        # +1: 마지막 봉이 양봉 (close > open)
        d["bull_candle"] = 1 if current_price > current_open else 0

        # ── 패널티 ────────────────────────────────────────────────────────
        # -1: 펀딩 비율 과다 (한쪽 포지션 쏠림 위험)
        d["adverse_funding"] = -1 if abs(funding_rate) > 0.001 else 0
        # -1: 스프레드 과다 (체결 비용 증가) — Binance 기준 0.2%
        d["wide_spread"] = -1 if spread_pct > 0.002 else 0

        score = sum(d.values())
        return score, d

    # ── 숏 스코어 계산 ────────────────────────────────────────────────────────

    def _score_short(
        self,
        current_price: float,
        current_open: float,
        current_vol: float,
        avg_vol: float,
        ltf_ema9: float,
        ltf_ema21: float,
        ltf_atr: float,
        ltf_rsi: float,
        htf_ema9: float,
        htf_ema21: float,
        htf_vwap: float,
        htf_price: float,
        has_htf: bool,
        funding_rate: float,
        spread_pct: float,
    ) -> Tuple[int, Dict[str, int]]:
        """숏 진입 스코어 계산 (9점 만점, 롱의 정반대 조건)."""
        d: Dict[str, int] = {}

        # ── HTF 바이어스 (하락 방향) ──────────────────────────────────────
        if has_htf and htf_ema9 > 0 and htf_ema21 > 0:
            d["htf_ema_bear"] = 2 if htf_ema9 < htf_ema21 else 0
            d["htf_vwap_bear"] = 1 if (htf_vwap > 0 and htf_price < htf_vwap) else 0
        else:
            d["htf_ema_bear"] = 0
            d["htf_vwap_bear"] = 0

        # ── LTF EMA 정렬 (하락) ────────────────────────────────────────────
        d["ltf_ema_aligned"] = 2 if (ltf_ema9 < ltf_ema21 and ltf_ema21 > 0) else 0

        # ── 가격 EMA 존 ────────────────────────────────────────────────────
        zone_threshold = ltf_atr * self._s.SCALP_PURE_EMA_ZONE_ATR
        d["price_in_zone"] = 1 if (ltf_ema21 > 0 and abs(current_price - ltf_ema21) <= zone_threshold) else 0

        # ── RSI 모멘텀 구간 (숏: 35–60) ───────────────────────────────────
        rsi_min = self._s.SCALP_PURE_RSI_SHORT_MIN
        rsi_max = self._s.SCALP_PURE_RSI_SHORT_MAX
        d["rsi_in_zone"] = 1 if (rsi_min <= ltf_rsi <= rsi_max) else 0

        # ── 볼륨 확인 ──────────────────────────────────────────────────────
        vol_mult = self._s.SCALP_PURE_VOLUME_MULT
        d["volume_spike"] = 1 if (avg_vol > 0 and current_vol >= avg_vol * vol_mult) else 0

        # ── 캔들 방향 (음봉) ───────────────────────────────────────────────
        d["bear_candle"] = 1 if current_price < current_open else 0

        # ── 패널티 ────────────────────────────────────────────────────────
        d["adverse_funding"] = -1 if abs(funding_rate) > 0.001 else 0
        d["wide_spread"] = -1 if spread_pct > 0.002 else 0

        score = sum(d.values())
        return score, d

    # ── 중립 신호 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _neutral() -> CombinedSignal:
        return CombinedSignal(
            direction=0,
            strength=0.0,
            regime=Regime.RANGING,
            long_score=0.0,
            short_score=0.0,
            category_scores={},
        )
