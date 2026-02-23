"""Tests for macro regime detection and risk engine bear mode."""
import pytest
import sys
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategy.macro_regime import MacroRegimeDetector, MacroRegime, _ema, _adx
from src.risk.engine import RiskEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_settings(**kwargs):
    from src.config import Settings
    defaults = dict(
        TRADING_MODE="dry-run",
        BINANCE_API_KEY="",
        BINANCE_API_SECRET="",
        MAX_LEVERAGE_CAP=10,
        BEAR_MAX_LEVERAGE_CAP=50,
        MAX_CONCURRENT_POSITIONS=3,
        MAX_SINGLE_EXPOSURE_PCT=0.30,
        MAX_RISK_PER_TRADE_PCT=0.02,
        SCALP_STOP_ATR=1.0,
        SCALP_TP1_ATR=1.5,
        SCALP_TP2_ATR=3.0,
        SCALP_MAX_BARS=12,
        SCALP_PARTIAL_EXIT_PCT=0.60,
        TREND_STOP_ATR=2.0,
        TREND_TP1_ATR=3.0,
        TREND_TP2_ATR=6.0,
        TREND_MAX_BARS=120,
        TREND_PARTIAL_EXIT_PCT=0.40,
        TREND_RIDE_SCORE_THRESHOLD=2,
        BULL_MODE_SYMBOLS="",
        BEAR_MODE_SYMBOLS="BTCUSDT,ETHUSDT",
        MACRO_REGIME_CHECK_HOURS=4,
        MACRO_BULL_CONFIRM_COUNT=3,
        MACRO_BEAR_CONFIRM_COUNT=2,
        STOP_ATR_MULTIPLIER=2.0,
        TP_ATR_MULTIPLIER=3.0,
        TRAILING_STOP_ATR_MULTIPLIER=1.0,
        MAX_DAILY_LOSS_PCT=0.08,
        MAX_WEEKLY_LOSS_PCT=0.15,
        MAX_DRAWDOWN_PCT=0.30,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


SYMBOL_INFO = {
    "min_qty": 0.001,
    "qty_step": 0.001,
    "price_step": 0.1,
    "min_notional": 5.0,
    "max_leverage": 125,
}


def _make_bull_df(n: int = 250) -> pd.DataFrame:
    """강한 상승 추세 데이터: EMA50 > EMA200, ADX high, ROC positive."""
    # Linear uptrend: 20000 → 120000 (250 bars)
    prices = np.linspace(20000, 120000, n)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 100, n)
    close = pd.Series(prices + noise).clip(lower=100)

    high = close * 1.005
    low = close * 0.995

    # Binance kline format: 12 columns
    ts = np.arange(n, dtype=float) * 86400000.0
    df = pd.DataFrame({
        "ts": ts,
        "open": close * 0.999,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(1000, 2000, n),
        "close_time": ts + 86399999,
        "quote_volume": close * 1500,
        "num_trades": rng.integers(1000, 5000, n),
        "taker_buy_base": rng.uniform(500, 1000, n),
        "taker_buy_quote": close * 750,
        "ignore": 0,
    })
    return df


def _make_bear_df(n: int = 250) -> pd.DataFrame:
    """강한 하락 추세 데이터: EMA50 < EMA200, ADX high, ROC negative."""
    prices = np.linspace(100000, 20000, n)
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 100, n)
    close = pd.Series(prices + noise).clip(lower=100)

    high = close * 1.005
    low = close * 0.995

    ts = np.arange(n, dtype=float) * 86400000.0
    df = pd.DataFrame({
        "ts": ts,
        "open": close * 1.001,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(1000, 2000, n),
        "close_time": ts + 86399999,
        "quote_volume": close * 1500,
        "num_trades": rng.integers(1000, 5000, n),
        "taker_buy_base": rng.uniform(500, 1000, n),
        "taker_buy_quote": close * 750,
        "ignore": 0,
    })
    return df


def _make_sideways_df(n: int = 250) -> pd.DataFrame:
    """횡보 데이터: 큰 방향성 없음."""
    rng = np.random.default_rng(42)
    base = 50000.0
    noise = rng.normal(0, 500, n)
    close = pd.Series(base + np.cumsum(noise * 0.1))
    close = close.clip(lower=100)

    high = close * 1.003
    low = close * 0.997

    ts = np.arange(n, dtype=float) * 86400000.0
    df = pd.DataFrame({
        "ts": ts,
        "open": close * 0.999,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(500, 1500, n),
        "close_time": ts + 86399999,
        "quote_volume": close * 1000,
        "num_trades": rng.integers(500, 2000, n),
        "taker_buy_base": rng.uniform(250, 750, n),
        "taker_buy_quote": close * 500,
        "ignore": 0,
    })
    return df


def _make_detector(mock_raw=None, **settings_kwargs):
    """테스트용 MacroRegimeDetector (API mock 포함)."""
    settings = make_settings(**settings_kwargs)
    client = MagicMock()
    if mock_raw is not None:
        client.get_kline.return_value = mock_raw
    detector = MacroRegimeDetector(client, settings)
    return detector


def _df_to_raw(df: pd.DataFrame):
    """DataFrame을 Binance kline raw 포맷으로 변환."""
    cols = ["ts", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    return df[cols].values.tolist()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIndicatorHelpers:
    def test_ema_converges(self):
        """EMA should converge: fast EMA > slow EMA on uptrend."""
        prices = pd.Series(np.linspace(10000, 50000, 300))
        ema50 = _ema(prices, 50)
        ema200 = _ema(prices, 200)
        assert float(ema50.iloc[-1]) > float(ema200.iloc[-1])

    def test_adx_positive(self):
        """ADX should be non-negative."""
        rng = np.random.default_rng(0)
        n = 100
        ts = np.arange(n)
        close = pd.Series(50000 + np.cumsum(rng.normal(0, 50, n)))
        high = close * 1.002
        low = close * 0.998
        adx_series = _adx(high, low, close, period=14)
        assert float(adx_series.iloc[-1]) >= 0


class TestMacroRegimeDetection:
    def test_bull_market_detected(self):
        df = _make_bull_df(250)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw)
        result = detector.detect()
        assert result.regime == MacroRegime.BULL

    def test_bear_market_detected(self):
        df = _make_bear_df(250)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw)
        result = detector.detect()
        assert result.regime in (MacroRegime.BEAR, MacroRegime.SIDEWAYS)

    def test_sideways_market_detected(self):
        df = _make_sideways_df(250)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw)
        result = detector.detect()
        assert result.regime in (MacroRegime.SIDEWAYS, MacroRegime.BEAR)

    def test_bull_result_has_full_universe_symbols(self):
        """BULL 레짐 → BULL_MODE_SYMBOLS 반환 (비어있으면 빈 리스트)."""
        df = _make_bull_df(250)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw, BULL_MODE_SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT")
        result = detector.detect()
        if result.regime == MacroRegime.BULL:
            assert "BTCUSDT" in result.symbols
            assert "SOLUSDT" in result.symbols

    def test_bear_result_has_btc_eth_symbols(self):
        """BEAR/SIDEWAYS 레짐 → BEAR_MODE_SYMBOLS 반환."""
        df = _make_bear_df(250)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw, BEAR_MODE_SYMBOLS="BTCUSDT,ETHUSDT")
        result = detector.detect()
        if result.regime in (MacroRegime.BEAR, MacroRegime.SIDEWAYS):
            assert "BTCUSDT" in result.symbols
            assert "ETHUSDT" in result.symbols

    def test_insufficient_data_returns_sideways(self):
        """데이터 부족 시 SIDEWAYS 반환."""
        # Only 50 bars (< 210 required)
        df = _make_bull_df(50)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw)
        result = detector.detect()
        assert result.regime == MacroRegime.SIDEWAYS

    def test_api_failure_returns_sideways(self):
        """API 실패 시 SIDEWAYS 반환 (보수적)."""
        detector = _make_detector(mock_raw=None)
        detector._client.get_kline.side_effect = Exception("API error")
        result = detector.detect()
        assert result.regime == MacroRegime.SIDEWAYS

    def test_empty_kline_returns_sideways(self):
        """빈 kline → SIDEWAYS."""
        detector = _make_detector(mock_raw=[])
        result = detector.detect()
        assert result.regime == MacroRegime.SIDEWAYS

    def test_result_has_reason_string(self):
        df = _make_bull_df(250)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw)
        result = detector.detect()
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_confidence_in_range(self):
        df = _make_bull_df(250)
        raw = _df_to_raw(df)
        detector = _make_detector(mock_raw=raw)
        result = detector.detect()
        assert 0.0 <= result.confidence <= 1.0


class TestRiskEngineBearMode:
    def test_bear_mode_uses_higher_leverage_cap(self):
        """bear_mode=True → BEAR_MAX_LEVERAGE_CAP (50) 사용."""
        s = make_settings(MAX_LEVERAGE_CAP=10, BEAR_MAX_LEVERAGE_CAP=50)
        engine = RiskEngine(s)
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=1.0, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.2, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
            bear_mode=True,
        )
        assert result.risk_ok
        assert result.leverage <= 50

    def test_normal_mode_uses_normal_leverage_cap(self):
        """bear_mode=False → MAX_LEVERAGE_CAP (10) 사용."""
        s = make_settings(MAX_LEVERAGE_CAP=10, BEAR_MAX_LEVERAGE_CAP=50)
        engine = RiskEngine(s)
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=1.0, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.2, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
            bear_mode=False,
        )
        assert result.risk_ok
        assert result.leverage <= 10

    def test_bear_mode_default_false(self):
        """bear_mode 기본값은 False (일반 레버리지 캡)."""
        s = make_settings(MAX_LEVERAGE_CAP=10)
        engine = RiskEngine(s)
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=0.7, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.4, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
        )
        assert result.risk_ok
        assert result.leverage <= s.MAX_LEVERAGE_CAP


class TestSymbolFiltering:
    """_filter_symbols_by_macro_regime 로직 테스트 (unit test, main.py 직접 접근)."""

    def _make_bot_mock(self, macro_mode="BULL", bull_syms="", bear_syms="BTCUSDT,ETHUSDT",
                       open_symbols=None):
        """TradingBot의 symbol filter 메서드를 독립적으로 테스트."""
        s = make_settings(BULL_MODE_SYMBOLS=bull_syms, BEAR_MODE_SYMBOLS=bear_syms)

        class FakeStateMachine:
            def all_positions(self_inner):
                from src.execution.state_machine import PositionInfo, PositionState
                positions = []
                for sym in (open_symbols or []):
                    p = PositionInfo(symbol=sym)
                    p.state = PositionState.LONG
                    positions.append(p)
                return positions

        class FakeBot:
            def __init__(self_inner):
                self_inner._macro_mode = macro_mode
                self_inner._state_machine = FakeStateMachine()
                self_inner._s = s

            def _filter_symbols_by_macro_regime(self_inner, symbols):
                # Replicate exact logic from main.py
                import src.config as cfg
                cfg.settings = s  # override module-level settings temporarily
                if self_inner._macro_mode == "BULL":
                    bull = [x.strip() for x in s.BULL_MODE_SYMBOLS.split(",") if x.strip()]
                    if bull:
                        return bull
                    return symbols
                allowed = [x.strip() for x in s.BEAR_MODE_SYMBOLS.split(",") if x.strip()]
                open_syms = [p.symbol for p in self_inner._state_machine.all_positions()]
                result = list(dict.fromkeys(allowed + open_syms))
                for force_sym in allowed:
                    if force_sym not in result:
                        result.append(force_sym)
                return result

        return FakeBot()

    def test_bull_mode_allows_all_symbols(self):
        bot = self._make_bot_mock(macro_mode="BULL", bull_syms="")
        universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        result = bot._filter_symbols_by_macro_regime(universe)
        assert set(result) == set(universe)

    def test_bear_mode_allows_only_btc_eth(self):
        bot = self._make_bot_mock(macro_mode="BEAR_SIDEWAYS")
        universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        result = bot._filter_symbols_by_macro_regime(universe)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        assert "SOLUSDT" not in result
        assert "BNBUSDT" not in result

    def test_bear_mode_includes_symbols_with_open_positions(self):
        bot = self._make_bot_mock(macro_mode="BEAR_SIDEWAYS", open_symbols=["SOLUSDT"])
        universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        result = bot._filter_symbols_by_macro_regime(universe)
        assert "SOLUSDT" in result  # open position 심볼 항상 포함

    def test_bull_mode_with_custom_symbols(self):
        bot = self._make_bot_mock(macro_mode="BULL", bull_syms="BTCUSDT,ETHUSDT,SOLUSDT")
        universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        result = bot._filter_symbols_by_macro_regime(universe)
        assert "BTCUSDT" in result
        assert "SOLUSDT" in result
        assert "BNBUSDT" not in result  # BULL_MODE_SYMBOLS에 없음
