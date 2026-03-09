"""
Signal Combiner — applies regime-based weights to the signal bundle
and produces a final CombinedSignal with direction and strength.

MTF 전략 (15분봉+일봉) 가중치 구조:
  htf_trend가 최우선 필터 역할 — 일봉 추세와 역방향 신호 차단

  TRENDING_UP/DOWN : htf_trend(0.35) + trend(0.25) + ict(0.20) + volume(0.15)
                   + reversion(0.05)
  RANGING          : htf_trend(0.35) + ict(0.25) + volume(0.15) + reversion(0.15)
                   + trend(0.10)
  HIGH_VOLATILITY  : htf_trend(0.40) + ict(0.25) + volume(0.20) + trend(0.10)
                   + reversion(0.05)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.config import Settings
from src.strategy.regime import Regime, RegimeResult
from src.strategy.signals import SignalBundle, Signal


@dataclass
class CombinedSignal:
    direction: int       # +1 long, -1 short, 0 neutral
    strength: float      # 0.0 – 1.0 (how confident we are)
    regime: Regime
    long_score: float    # raw long score
    short_score: float   # raw short score
    category_scores: Dict[str, float]

    @property
    def is_long(self) -> bool:
        return self.direction == 1 and self.strength > 0

    @property
    def is_short(self) -> bool:
        return self.direction == -1 and self.strength > 0


# ── Category weights by regime ────────────────────────────────────────────────

_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    Regime.TRENDING_UP: {
        # 불마켓 추세 추종 전략: HTF 방향 + 모멘텀 + ICT 구조
        # 15분봉에서 reversion(dip-buying)은 false signal이 많음
        # → 추세 방향 모멘텀 진입이 더 높은 승률 달성
        "htf_trend": 0.35, "trend": 0.25, "ict": 0.20, "volume": 0.15,
        "reversion": 0.05, "vwap": 0.00, "squeeze": 0.00, "volatility": 0.00,
        "orderbook": 0.00, "momentum": 0.00,
    },
    Regime.TRENDING_DOWN: {
        # 베어마켓 추세 추종 숏 전략: HTF 방향 + 모멘텀 + ICT 구조
        "htf_trend": 0.35, "trend": 0.25, "ict": 0.20, "volume": 0.15,
        "reversion": 0.05, "vwap": 0.00, "squeeze": 0.00, "volatility": 0.00,
        "orderbook": 0.00, "momentum": 0.00,
    },
    Regime.RANGING: {
        # 횡보장: HTF 방향 + ICT 지지/저항 + 소량 reversion(범위 경계 매매)
        "htf_trend": 0.35, "ict": 0.25, "volume": 0.15, "reversion": 0.15,
        "trend": 0.10, "vwap": 0.00, "squeeze": 0.00, "volatility": 0.00,
        "orderbook": 0.00, "momentum": 0.00,
    },
    Regime.HIGH_VOLATILITY: {
        # 고변동성: HTF 방향 + ICT 브레이크아웃 + volume 폭발 확인
        "htf_trend": 0.40, "ict": 0.25, "volume": 0.20, "trend": 0.10,
        "reversion": 0.05, "vwap": 0.00, "volatility": 0.00, "squeeze": 0.00,
        "orderbook": 0.00, "momentum": 0.00,
    },
}


class SignalCombiner:
    def __init__(self, settings: Settings):
        self._s = settings

    def combine(
        self,
        bundle: SignalBundle,
        regime_result: RegimeResult,
    ) -> CombinedSignal:
        regime = regime_result.regime
        weights = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS[Regime.RANGING])

        # ── Per-category weighted scores ──────────────────────────────────────
        category_long: Dict[str, float] = {}
        category_short: Dict[str, float] = {}
        category_scores: Dict[str, float] = {}

        for cat in ("trend", "reversion", "volatility", "volume", "ict", "vwap", "orderbook",
                    "momentum", "squeeze", "htf_trend"):
            cat_signals = bundle.by_category(cat)
            if not cat_signals:
                category_long[cat] = 0.0
                category_short[cat] = 0.0
                category_scores[cat] = 0.0
                continue

            total_w = sum(s.confidence for s in cat_signals)
            if total_w == 0:
                category_scores[cat] = 0.0
                continue

            # Separate long and short components
            long_score = sum(
                s.confidence for s in cat_signals if s.direction == 1
            ) / total_w
            short_score = sum(
                s.confidence for s in cat_signals if s.direction == -1
            ) / total_w

            category_long[cat] = long_score
            category_short[cat] = short_score
            category_scores[cat] = (long_score - short_score)  # net: -1 to +1

        # ── Aggregate with regime weights (renormalized over active categories) ─
        # Categories with no signal data (vwap=0, oi=0, orderbook=0 in backtest)
        # are excluded so inactive data doesn't dilute the scores.
        active_weight_sum = sum(
            weights.get(c, 0) for c in weights
            if category_long.get(c, 0) > 0 or category_short.get(c, 0) > 0
        )
        if active_weight_sum > 0:
            norm = 1.0 / active_weight_sum
            total_long = sum(weights.get(c, 0) * norm * category_long.get(c, 0) for c in weights)
            total_short = sum(weights.get(c, 0) * norm * category_short.get(c, 0) for c in weights)
        else:
            total_long = 0.0
            total_short = 0.0

        # ── Regime directional filter ─────────────────────────────────────────
        # 15M 레짐 방향 조정: 완전 차단 대신 역방향 신호 감쇠
        # 완전 차단(0.0)은 HTF 필터와 충돌 → 유효한 페이드 트레이드 차단
        # HTF가 "베어"이고 15M이 일시적 TRENDING_UP(반등)이면 숏 허용 필요
        if regime == Regime.TRENDING_UP:
            total_short *= 0.25  # 15M 상승장에서 숏 대폭 감쇠
        elif regime == Regime.TRENDING_DOWN:
            total_long *= 0.25   # 15M 하락장에서 롱 대폭 감쇠

        # High volatility: slight reduction (still allow breakout scalps)
        if regime == Regime.HIGH_VOLATILITY:
            total_long *= 0.75
            total_short *= 0.75

        # ── HTF 방향성 정렬 필터 ─────────────────────────────────────────────────
        # htf_trend 카테고리의 net score (-1 to +1)
        htf_net = category_scores.get("htf_trend", 0.0)

        # 규칙:
        # htf_net >= +0.50 (일봉 강한 상승장): 롱만 허용, 숏 완전 차단
        # htf_net <= -0.50 (일봉 강한 하락장): 숏만 허용, 롱 완전 차단
        # |htf_net| < 0.50 (일봉 중립/약한 추세): 양방향 모두 차단 (노이즈 거래 방지)
        HTF_BIAS_MIN = 0.50    # 최소 방향성 요구 (강한 HTF 추세만)
        HTF_MILD_BULL = 0.70   # 중간~강한 상승: 롱 허용하지만 일부 감쇠
        HTF_MILD_BEAR = 0.70   # 중간~강한 하락: 숏 허용하지만 일부 감쇠

        if htf_net >= HTF_BIAS_MIN:
            # 일봉 상승 → 숏 완전 차단, 롱 허용
            total_short = 0.0
            if htf_net < HTF_MILD_BULL:
                total_long *= 0.50
        elif htf_net <= -HTF_BIAS_MIN:
            # 일봉 하락 → 롱 완전 차단, 숏 허용
            total_long = 0.0
            if htf_net > -HTF_MILD_BEAR:
                total_short *= 0.50
        else:
            # 일봉 중립 구간: 양방향 완전 차단
            total_long = 0.0
            total_short = 0.0

        # ── Determine direction and strength ──────────────────────────────────
        threshold = self._s.SIGNAL_ENTRY_THRESHOLD

        if total_long >= threshold and total_long > total_short:
            direction = 1
            strength = total_long
        elif total_short >= threshold and total_short > total_long:
            direction = -1
            strength = total_short
        else:
            direction = 0
            strength = max(total_long, total_short)

        # Regime confidence amplifies strength (1.0–1.3x).
        strength = min(1.0, strength * (1.0 + 0.3 * regime_result.confidence))

        return CombinedSignal(
            direction=direction,
            strength=strength,
            regime=regime,
            long_score=total_long,
            short_score=total_short,
            category_scores=category_scores,
        )
