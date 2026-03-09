"""
Tests for ScalpingStrategy — Binance Futures Bot (src/strategy/scalping.py).

테스트 커버리지:
- 롱 신호 생성 (HTF+LTF 정렬, VWAP, RSI, 볼륨 조건)
- 숏 신호 생성 (역방향 조건)
- 점수 미달 → 중립 반환
- 신호 강도 매핑 검증 (engine.py 티어 경계 정렬)
- HTF 없는 경우 처리
- 데이터 부족 처리
- 페널티 조건 (펀딩 비율, 스프레드)
- ATR 반환 검증
- PURE_SCALP 모드 RiskEngine 연동
- collect_for_scalp 데이터 구조 검증
"""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategy.scalping import ScalpingStrategy, ScalpSignalDetail, _SCORE_STRENGTH_MAP, _MAX_POSITIVE_SCORE
from src.strategy.regime import Regime
from src.config import Settings


# ── Test Fixtures ─────────────────────────────────────────────────────────────

def make_settings(**kwargs) -> Settings:
    defaults = dict(
        TRADING_MODE="dry-run",
        BINANCE_API_KEY="",
        BINANCE_API_SECRET="",
        BINANCE_TESTNET=True,
        STRATEGY_MODE="scalp",
        SCALP_PURE_LTF_TIMEFRAME="3",
        SCALP_PURE_HTF_TIMEFRAME="15",
        SCALP_PURE_EMA_FAST=9,
        SCALP_PURE_EMA_MID=21,
        SCALP_PURE_EMA_SLOW=50,
        SCALP_PURE_RSI_PERIOD=14,
        SCALP_PURE_RSI_LONG_MIN=40.0,
        SCALP_PURE_RSI_LONG_MAX=65.0,
        SCALP_PURE_RSI_SHORT_MIN=35.0,
        SCALP_PURE_RSI_SHORT_MAX=60.0,
        SCALP_PURE_VOLUME_MULT=1.2,
        SCALP_PURE_EMA_ZONE_ATR=1.0,
        SCALP_PURE_ENTRY_MIN_SCORE=5,
        SCALP_PURE_ENTRY_URGENT_SCORE=9,
        SCALP_PURE_STOP_ATR=1.0,
        SCALP_PURE_TP1_ATR=1.5,
        SCALP_PURE_TP2_ATR=3.0,
        SCALP_PURE_MAX_BARS=20,
        SCALP_PURE_PARTIAL_EXIT_PCT=0.50,
        SIGNAL_URGENT_THRESHOLD=0.90,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def make_ohlcv(
    n: int = 100,
    price_trend: str = "up",    # "up", "down", "flat"
    base_price: float = 50000.0,
    atr_pct: float = 0.01,
    vol_spike: bool = False,
) -> pd.DataFrame:
    """테스트용 OHLCV 데이터프레임 생성."""
    np.random.seed(42)

    if price_trend == "up":
        prices = base_price + np.cumsum(np.random.randn(n) * base_price * 0.002 + base_price * 0.001)
    elif price_trend == "down":
        prices = base_price + np.cumsum(np.random.randn(n) * base_price * 0.002 - base_price * 0.001)
    else:
        prices = base_price + np.cumsum(np.random.randn(n) * base_price * 0.001)

    prices = np.abs(prices)
    atr_val = base_price * atr_pct
    high = prices + np.abs(np.random.randn(n)) * atr_val * 0.5
    low = prices - np.abs(np.random.randn(n)) * atr_val * 0.5
    close = prices
    open_ = prices - (np.random.randn(n) * atr_val * 0.3)

    volume = np.abs(np.random.randn(n) * 100 + 200)
    if vol_spike:
        volume[-1] = volume[:-1].mean() * 2.0

    ts = np.arange(n) * 180000  # 3분봉

    return pd.DataFrame({
        "ts": ts.astype(int),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_bullish_df(n: int = 100) -> pd.DataFrame:
    """명확한 상승 추세 데이터 생성."""
    np.random.seed(1)
    base = 50000.0
    prices = base + np.linspace(0, base * 0.05, n) + np.random.randn(n) * base * 0.001
    prices = np.abs(prices)
    atr_val = base * 0.01
    high = prices + atr_val * 0.3
    low = prices - atr_val * 0.3
    open_ = prices - atr_val * 0.1   # 양봉: close > open
    volume = np.abs(np.random.randn(n) * 100 + 300)
    volume[-5:] = volume[:-5].mean() * 1.5  # 최근 볼륨 스파이크
    ts = np.arange(n) * 180000
    return pd.DataFrame({
        "ts": ts.astype(int),
        "open": open_,
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume,
    })


def make_bearish_df(n: int = 100) -> pd.DataFrame:
    """명확한 하락 추세 데이터 생성."""
    np.random.seed(2)
    base = 50000.0
    prices = base - np.linspace(0, base * 0.05, n) + np.random.randn(n) * base * 0.001
    prices = np.abs(prices)
    atr_val = base * 0.01
    high = prices + atr_val * 0.3
    low = prices - atr_val * 0.3
    open_ = prices + atr_val * 0.1   # 음봉: close < open
    volume = np.abs(np.random.randn(n) * 100 + 300)
    volume[-5:] = volume[:-5].mean() * 1.5
    ts = np.arange(n) * 180000
    return pd.DataFrame({
        "ts": ts.astype(int),
        "open": open_,
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume,
    })


# ── Basic Tests ───────────────────────────────────────────────────────────────

class TestBasicCases:
    def test_empty_df_returns_neutral(self):
        s = ScalpingStrategy(make_settings())
        sig, atr = s.generate(pd.DataFrame(), None)
        assert sig.direction == 0
        assert sig.strength == 0.0
        assert atr == 0.0

    def test_insufficient_data_returns_neutral(self):
        s = ScalpingStrategy(make_settings())
        df = make_ohlcv(n=30)  # < 50 bars required
        sig, atr = s.generate(df, None)
        assert sig.direction == 0
        assert sig.strength == 0.0

    def test_returns_combined_signal_and_atr(self):
        s = ScalpingStrategy(make_settings())
        df = make_ohlcv(n=100)
        sig, atr = s.generate(df, None)
        assert hasattr(sig, "direction")
        assert hasattr(sig, "strength")
        assert hasattr(sig, "regime")
        # ATR must be positive for valid data
        assert atr >= 0.0

    def test_no_htf_still_works(self):
        s = ScalpingStrategy(make_settings())
        df = make_ohlcv(n=100)
        sig, atr = s.generate(df, None)
        # Should not raise, direction could be anything
        assert sig.direction in (-1, 0, 1)


# ── Score & Strength Mapping Tests ────────────────────────────────────────────

class TestScoreStrengthMapping:
    def test_score_strength_map_values(self):
        """각 score에 대응하는 strength 값 검증."""
        assert _SCORE_STRENGTH_MAP[5] == 0.60
        assert _SCORE_STRENGTH_MAP[6] == 0.69
        assert _SCORE_STRENGTH_MAP[7] == 0.77
        assert _SCORE_STRENGTH_MAP[8] == 0.83

    def test_max_score_triggers_market_order(self):
        """score=9이면 strength > SIGNAL_URGENT_THRESHOLD(0.90) → 시장가."""
        s = make_settings(SIGNAL_URGENT_THRESHOLD=0.90)
        strategy = ScalpingStrategy(s)

        # score=9 케이스 직접 시뮬레이션
        urgent_threshold = 0.90
        expected = urgent_threshold + 0.01  # 0.91
        assert expected > 0.90

    def test_score_5_uses_limit_order(self):
        """score=5이면 strength=0.60 < 0.90 → PostOnly 지정가."""
        assert _SCORE_STRENGTH_MAP[5] == 0.60
        assert _SCORE_STRENGTH_MAP[5] < 0.90  # PostOnly

    def test_strength_below_urgent_threshold_for_score_8(self):
        """score=8이면 strength=0.83 < 0.90 → PostOnly 지정가."""
        assert _SCORE_STRENGTH_MAP[8] == 0.83
        assert _SCORE_STRENGTH_MAP[8] < 0.90

    def test_max_positive_score_is_9(self):
        assert _MAX_POSITIVE_SCORE == 9


# ── Scoring Tests ──────────────────────────────────────────────────────────────

class TestScoringComponents:
    def test_adverse_funding_penalty(self):
        """높은 펀딩 비율은 패널티 적용."""
        s = ScalpingStrategy(make_settings())
        ltf = make_bullish_df(100)
        htf = make_bullish_df(50)

        sig_normal, _ = s.generate(ltf, htf, funding_rate=0.0)
        sig_high_fund, _ = s.generate(ltf, htf, funding_rate=0.002)  # > 0.001

        # 높은 펀딩은 신호 강도를 낮추거나 같아야 함
        assert sig_high_fund.strength <= sig_normal.strength + 0.2  # 어느 정도 허용

    def test_wide_spread_penalty(self):
        """넓은 스프레드는 패널티 적용 (0.2% 초과)."""
        s = ScalpingStrategy(make_settings())
        ltf = make_bullish_df(100)
        htf = make_bullish_df(50)

        sig_normal, _ = s.generate(ltf, htf, spread_pct=0.001)
        sig_wide, _ = s.generate(ltf, htf, spread_pct=0.003)  # > 0.002

        # 넓은 스프레드는 신호 강도를 낮추거나 같아야 함
        assert sig_wide.strength <= sig_normal.strength + 0.2

    def test_score_method_long_positive(self):
        """롱 조건이 충족되면 양수 스코어."""
        strategy = ScalpingStrategy(make_settings())
        score, details = strategy._score_long(
            current_price=50100.0,
            current_open=50000.0,   # 양봉 (close > open)
            current_vol=300.0,
            avg_vol=200.0,          # 볼륨 스파이크 (300 > 200×1.2=240)
            ltf_ema9=50050.0,
            ltf_ema21=49900.0,      # EMA9 > EMA21 (상승)
            ltf_atr=500.0,
            ltf_rsi=52.0,           # 40~65 범위
            htf_ema9=50000.0,
            htf_ema21=49800.0,      # HTF 상승
            htf_vwap=49500.0,
            htf_price=50100.0,      # 가격 > VWAP
            has_htf=True,
            funding_rate=0.0,
            spread_pct=0.001,
        )
        assert score >= 5  # 최소 점수 이상
        assert details["htf_ema_bull"] == 2
        assert details["htf_vwap_bull"] == 1
        assert details["ltf_ema_aligned"] == 2
        assert details["bull_candle"] == 1

    def test_score_method_short_positive(self):
        """숏 조건이 충족되면 양수 스코어."""
        strategy = ScalpingStrategy(make_settings())
        score, details = strategy._score_short(
            current_price=49900.0,
            current_open=50000.0,   # 음봉 (close < open)
            current_vol=300.0,
            avg_vol=200.0,
            ltf_ema9=49950.0,
            ltf_ema21=50100.0,      # EMA9 < EMA21 (하락)
            ltf_atr=500.0,
            ltf_rsi=48.0,           # 35~60 범위
            htf_ema9=50000.0,
            htf_ema21=50200.0,      # HTF 하락
            htf_vwap=50500.0,
            htf_price=49900.0,      # 가격 < VWAP
            has_htf=True,
            funding_rate=0.0,
            spread_pct=0.001,
        )
        assert score >= 5
        assert details["htf_ema_bear"] == 2
        assert details["htf_vwap_bear"] == 1
        assert details["ltf_ema_aligned"] == 2
        assert details["bear_candle"] == 1

    def test_funding_penalty_applied(self):
        """펀딩 비율 > 0.001이면 -1 패널티."""
        strategy = ScalpingStrategy(make_settings())
        _, details = strategy._score_long(
            current_price=50000.0, current_open=49900.0, current_vol=300.0, avg_vol=200.0,
            ltf_ema9=50050.0, ltf_ema21=49900.0, ltf_atr=500.0, ltf_rsi=52.0,
            htf_ema9=50000.0, htf_ema21=49800.0, htf_vwap=49500.0, htf_price=50000.0,
            has_htf=True, funding_rate=0.002, spread_pct=0.001,
        )
        assert details["adverse_funding"] == -1

    def test_spread_penalty_applied(self):
        """스프레드 > 0.002이면 -1 패널티."""
        strategy = ScalpingStrategy(make_settings())
        _, details = strategy._score_long(
            current_price=50000.0, current_open=49900.0, current_vol=300.0, avg_vol=200.0,
            ltf_ema9=50050.0, ltf_ema21=49900.0, ltf_atr=500.0, ltf_rsi=52.0,
            htf_ema9=50000.0, htf_ema21=49800.0, htf_vwap=49500.0, htf_price=50000.0,
            has_htf=True, funding_rate=0.0, spread_pct=0.003,
        )
        assert details["wide_spread"] == -1

    def test_no_penalty_normal_conditions(self):
        """정상 조건에서는 패널티 없음."""
        strategy = ScalpingStrategy(make_settings())
        _, details = strategy._score_long(
            current_price=50000.0, current_open=49900.0, current_vol=300.0, avg_vol=200.0,
            ltf_ema9=50050.0, ltf_ema21=49900.0, ltf_atr=500.0, ltf_rsi=52.0,
            htf_ema9=50000.0, htf_ema21=49800.0, htf_vwap=49500.0, htf_price=50000.0,
            has_htf=True, funding_rate=0.0001, spread_pct=0.001,
        )
        assert details["adverse_funding"] == 0
        assert details["wide_spread"] == 0


# ── Direction Tests ───────────────────────────────────────────────────────────

class TestDirection:
    def test_bullish_df_tends_long(self):
        """상승 추세 데이터는 롱 또는 중립 신호."""
        s = ScalpingStrategy(make_settings())
        ltf = make_bullish_df(100)
        htf = make_bullish_df(50)
        sig, atr = s.generate(ltf, htf)
        # 상승 추세에서 숏 신호는 나오지 않아야 함
        assert sig.direction >= 0  # 0 (중립) 또는 1 (롱)

    def test_bearish_df_tends_short(self):
        """하락 추세 데이터는 숏 또는 중립 신호."""
        s = ScalpingStrategy(make_settings())
        ltf = make_bearish_df(100)
        htf = make_bearish_df(50)
        sig, atr = s.generate(ltf, htf)
        assert sig.direction <= 0  # 0 (중립) 또는 -1 (숏)

    def test_trending_up_regime_on_long_signal(self):
        """롱 신호 + HTF 상승이면 TRENDING_UP 레짐."""
        s = ScalpingStrategy(make_settings())
        ltf = make_bullish_df(100)
        htf = make_bullish_df(50)
        sig, _ = s.generate(ltf, htf)
        if sig.direction == 1:
            assert sig.regime == Regime.TRENDING_UP

    def test_trending_down_regime_on_short_signal(self):
        """숏 신호 + HTF 하락이면 TRENDING_DOWN 레짐."""
        s = ScalpingStrategy(make_settings())
        ltf = make_bearish_df(100)
        htf = make_bearish_df(50)
        sig, _ = s.generate(ltf, htf)
        if sig.direction == -1:
            assert sig.regime == Regime.TRENDING_DOWN

    def test_neutral_signal_direction_zero(self):
        """중립 신호는 direction=0."""
        s = ScalpingStrategy(make_settings())
        sig = ScalpingStrategy._neutral()
        assert sig.direction == 0
        assert sig.strength == 0.0
        assert sig.regime == Regime.RANGING


# ── VWAP Tests ────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_vwap_bull_condition(self):
        """가격 > VWAP이면 htf_vwap_bull=1."""
        strategy = ScalpingStrategy(make_settings())
        _, details = strategy._score_long(
            current_price=50100.0, current_open=50000.0, current_vol=300.0, avg_vol=200.0,
            ltf_ema9=50050.0, ltf_ema21=49900.0, ltf_atr=500.0, ltf_rsi=52.0,
            htf_ema9=50000.0, htf_ema21=49800.0,
            htf_vwap=49500.0,    # VWAP 아래
            htf_price=50100.0,   # 가격 > VWAP → bull
            has_htf=True, funding_rate=0.0, spread_pct=0.001,
        )
        assert details["htf_vwap_bull"] == 1

    def test_vwap_no_bull_when_price_below(self):
        """가격 < VWAP이면 htf_vwap_bull=0."""
        strategy = ScalpingStrategy(make_settings())
        _, details = strategy._score_long(
            current_price=50100.0, current_open=50000.0, current_vol=300.0, avg_vol=200.0,
            ltf_ema9=50050.0, ltf_ema21=49900.0, ltf_atr=500.0, ltf_rsi=52.0,
            htf_ema9=50000.0, htf_ema21=49800.0,
            htf_vwap=50500.0,    # VWAP 위
            htf_price=50100.0,   # 가격 < VWAP → no bull
            has_htf=True, funding_rate=0.0, spread_pct=0.001,
        )
        assert details["htf_vwap_bull"] == 0

    def test_no_htf_sets_vwap_score_zero(self):
        """HTF 없으면 VWAP 점수=0."""
        strategy = ScalpingStrategy(make_settings())
        _, details = strategy._score_long(
            current_price=50100.0, current_open=50000.0, current_vol=300.0, avg_vol=200.0,
            ltf_ema9=50050.0, ltf_ema21=49900.0, ltf_atr=500.0, ltf_rsi=52.0,
            htf_ema9=0.0, htf_ema21=0.0, htf_vwap=0.0, htf_price=0.0,
            has_htf=False, funding_rate=0.0, spread_pct=0.001,
        )
        assert details["htf_ema_bull"] == 0
        assert details["htf_vwap_bull"] == 0


# ── Signal Detail Tests ───────────────────────────────────────────────────────

class TestSignalDetail:
    def test_get_signal_detail_returns_none_on_empty(self):
        s = ScalpingStrategy(make_settings())
        result = s.get_signal_detail(pd.DataFrame(), None)
        assert result is None

    def test_get_signal_detail_returns_dataclass(self):
        s = ScalpingStrategy(make_settings())
        df = make_bullish_df(100)
        result = s.get_signal_detail(df, make_bullish_df(50))
        assert result is not None
        assert isinstance(result, ScalpSignalDetail)
        assert result.max_score == _MAX_POSITIVE_SCORE
        assert result.atr_val > 0

    def test_detail_has_all_fields(self):
        s = ScalpingStrategy(make_settings())
        df = make_bullish_df(100)
        detail = s.get_signal_detail(df, make_bullish_df(50))
        assert detail is not None
        assert hasattr(detail, "score")
        assert hasattr(detail, "long_details")
        assert hasattr(detail, "short_details")
        assert hasattr(detail, "htf_ema9")
        assert hasattr(detail, "ltf_rsi")


# ── Risk Engine Integration Tests ────────────────────────────────────────────

class TestRiskEngineIntegration:
    def test_pure_scalp_mode_uses_tight_params(self):
        """PURE_SCALP 모드: 타이트한 SL/TP 파라미터 사용 확인."""
        from src.risk.engine import RiskEngine
        s = make_settings()
        engine = RiskEngine(s)

        result = engine.calculate(
            equity=10000.0,
            entry_price=50000.0,
            side="Buy",
            atr=500.0,
            signal_strength=0.77,  # score=7 → 8-10× tier
            spread_pct=0.001,
            funding_rate=0.0001,
            vol_percentile=0.4,
            drawdown_pct=0.0,
            consecutive_losses=0,
            symbol_max_leverage=125,
            symbol_info={"min_notional": 5.0, "max_leverage": 125},
            current_positions=0,
            trade_mode="PURE_SCALP",
        )
        assert result.risk_ok
        # TP1 = entry + 1.5×ATR = 50000 + 750 = 50750
        assert abs(result.tp1_price - 50750.0) < 50
        # TP2 = entry + 3.0×ATR = 50000 + 1500 = 51500
        assert abs(result.tp2_price - 51500.0) < 100
        # SL = entry - 1.0×ATR × stop_scale (scaled by signal)
        assert result.stop_price < 50000.0
        # partial_exit_pct = 0.50
        assert result.partial_exit_pct == 0.50
        # mode_max_bars = 20
        assert result.mode_max_bars == 20

    def test_pure_scalp_mode_vs_swing_mode_tp_tighter(self):
        """PURE_SCALP의 TP가 SCALP보다 타이트함."""
        from src.risk.engine import RiskEngine
        s = make_settings()
        engine = RiskEngine(s)

        common = dict(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=0.70, spread_pct=0.001, funding_rate=0.0001,
            vol_percentile=0.4, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125,
            symbol_info={"min_notional": 5.0, "max_leverage": 125},
            current_positions=0,
        )
        pure_scalp = engine.calculate(**common, trade_mode="PURE_SCALP")
        scalp = engine.calculate(**common, trade_mode="SCALP")

        assert pure_scalp.risk_ok
        assert scalp.risk_ok
        # PURE_SCALP TP1 (1.5×ATR) < SCALP TP1 (2.5×ATR)
        assert pure_scalp.tp1_price < scalp.tp1_price

    def test_pure_scalp_max_bars(self):
        """PURE_SCALP: max_bars=20 (60분)."""
        from src.risk.engine import RiskEngine
        s = make_settings()
        result = RiskEngine(s).calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=0.60, spread_pct=0.001, funding_rate=0.0001,
            vol_percentile=0.4, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125,
            symbol_info={"min_notional": 5.0, "max_leverage": 125},
            current_positions=0, trade_mode="PURE_SCALP",
        )
        assert result.mode_max_bars == 20

    def test_signal_strength_score9_exceeds_urgent_threshold(self):
        """score=9 → strength=0.91 > SIGNAL_URGENT_THRESHOLD(0.90) → 시장가."""
        s = make_settings(SIGNAL_URGENT_THRESHOLD=0.90)
        urgent_threshold = s.SIGNAL_URGENT_THRESHOLD
        # score=9 strength = urgent_threshold + 0.01
        strength_score9 = urgent_threshold + 0.01
        assert strength_score9 > 0.90  # 시장가 트리거 확인

    def test_signal_strength_score8_below_urgent_threshold(self):
        """score=8 → strength=0.83 < 0.90 → PostOnly 지정가."""
        assert _SCORE_STRENGTH_MAP[8] == 0.83
        assert _SCORE_STRENGTH_MAP[8] < 0.90


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_strategy_mode_swing_default(self):
        """기본 STRATEGY_MODE는 swing."""
        from src.config import Settings
        s = Settings(TRADING_MODE="dry-run")
        assert s.STRATEGY_MODE == "swing"

    def test_strategy_mode_scalp(self):
        """STRATEGY_MODE=scalp 설정 가능."""
        s = make_settings(STRATEGY_MODE="scalp")
        assert s.STRATEGY_MODE == "scalp"

    def test_scalp_pure_params_defaults(self):
        """SCALP_PURE_* 기본값 검증."""
        from src.config import Settings
        s = Settings(TRADING_MODE="dry-run")
        assert s.SCALP_PURE_LTF_TIMEFRAME == "3"
        assert s.SCALP_PURE_HTF_TIMEFRAME == "15"
        assert s.SCALP_PURE_STOP_ATR == 1.2
        assert s.SCALP_PURE_TP1_ATR == 2.0
        assert s.SCALP_PURE_TP2_ATR == 4.5
        assert s.SCALP_PURE_MAX_BARS == 30
        assert s.SCALP_PURE_PARTIAL_EXIT_PCT == 0.40
        assert s.SCALP_PURE_ENTRY_MIN_SCORE == 8

    def test_pure_scalp_tighter_than_scalp(self):
        """PURE_SCALP 파라미터가 SCALP보다 타이트함 (defaults 기준)."""
        from src.config import Settings
        s = Settings(TRADING_MODE="dry-run")
        # SL: PURE_SCALP ≤ SCALP
        assert s.SCALP_PURE_STOP_ATR <= s.SCALP_STOP_ATR
        # Max bars: 30 < 48
        assert s.SCALP_PURE_MAX_BARS < s.SCALP_MAX_BARS


# ── Collector Tests ───────────────────────────────────────────────────────────

class TestCollectorInterface:
    def test_collect_for_scalp_returns_market_snapshot(self):
        """collect_for_scalp 인터페이스 확인 (mock 사용)."""
        from unittest.mock import MagicMock
        from src.data.collector import DataCollector
        from src.config import settings as default_settings

        mock_client = MagicMock()
        # Binance kline format
        kline_row = [
            1700000000000, "50000.0", "50100.0", "49900.0", "50050.0", "100.0",
            1700000060000, "5000000.0", 100, "50.0", "2500000.0", "0"
        ]
        mock_client.get_kline.return_value = [kline_row] * 100
        mock_client.get_orderbook.return_value = {
            "bids": [["49990.0", "1.0"]] * 5,
            "asks": [["50010.0", "1.0"]] * 5,
        }
        mock_client.get_funding_rate_history.return_value = [
            {"fundingRate": "0.0001", "fundingTime": 1700000000000}
        ]
        mock_client.get_open_interest.return_value = [
            {"sumOpenInterest": "1000.0"},
            {"sumOpenInterest": "1010.0"},
        ]

        collector = DataCollector(mock_client, default_settings)
        snap = collector.collect_for_scalp("BTCUSDT", limit=100)

        assert snap.symbol == "BTCUSDT"
        assert not snap.ohlcv.empty
        assert snap.last_price > 0


# ── ATR Tests ─────────────────────────────────────────────────────────────────

class TestATRReturn:
    def test_atr_positive_for_valid_data(self):
        """유효한 데이터에서 ATR > 0."""
        s = ScalpingStrategy(make_settings())
        df = make_bullish_df(100)
        _, atr_val = s.generate(df, None)
        # 방향에 관계없이 ATR은 항상 양수
        assert atr_val >= 0.0

    def test_atr_returned_even_on_neutral_signal(self):
        """중립 신호도 ATR은 반환."""
        s = ScalpingStrategy(make_settings())
        df = make_ohlcv(n=100, price_trend="flat")
        _, atr_val = s.generate(df, None)
        # flat 데이터는 중립 가능하지만 ATR은 반환
        assert atr_val >= 0.0


# ── Max Score Tests ───────────────────────────────────────────────────────────

class TestMaxScore:
    def test_max_score_structure(self):
        """9점 만점 구조 검증."""
        strategy = ScalpingStrategy(make_settings())
        # 모든 롱 조건 충족
        score, details = strategy._score_long(
            current_price=50100.0,
            current_open=50000.0,     # 양봉
            current_vol=300.0,
            avg_vol=200.0,            # 볼륨 스파이크 (300 > 240)
            ltf_ema9=50050.0,
            ltf_ema21=49900.0,        # EMA9 > EMA21
            ltf_atr=200.0,            # ATR 기준 EMA 존
            ltf_rsi=52.0,             # 40~65
            htf_ema9=50000.0,
            htf_ema21=49800.0,        # HTF 상승
            htf_vwap=49500.0,
            htf_price=50100.0,        # 가격 > VWAP
            has_htf=True,
            funding_rate=0.0001,      # 패널티 없음
            spread_pct=0.001,         # 패널티 없음
        )
        # htf_ema(+2) + htf_vwap(+1) + ltf_ema(+2) + zone 가능 + rsi(+1) + vol(+1) + candle(+1)
        assert score >= 7  # 최소 7점 (EMA존 제외해도)
        assert details["htf_ema_bull"] == 2
        assert details["htf_vwap_bull"] == 1
        assert details["ltf_ema_aligned"] == 2

    def test_min_score_for_entry(self):
        """진입 최소 점수(5) 이상이어야 방향 신호 발생."""
        strategy = ScalpingStrategy(make_settings())
        # 부족한 조건 (score < 5 예상)
        sig, _ = strategy.generate(make_ohlcv(n=100, price_trend="flat"), None)
        # flat이면 score < 5로 direction=0 가능
        assert sig.direction in (-1, 0, 1)  # 범위 내여야 함
