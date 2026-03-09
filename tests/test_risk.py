"""Tests for risk engine and guard rails."""
import pytest
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.risk.engine import RiskEngine
from src.risk.guards import RiskGuards, GuardAction
from src.execution.state_machine import StateMachine


def make_settings(**kwargs):
    from src.config import Settings
    defaults = dict(
        TRADING_MODE="dry-run",
        BINANCE_API_KEY="",
        BINANCE_API_SECRET="",
        MAX_RISK_PER_TRADE_PCT=0.02,
        MAX_LEVERAGE_CAP=10,
        BEAR_MAX_LEVERAGE_CAP=50,
        MAX_CONCURRENT_POSITIONS=3,
        MAX_SINGLE_EXPOSURE_PCT=0.30,
        STOP_ATR_MULTIPLIER=2.0,
        TP_ATR_MULTIPLIER=3.0,
        TRAILING_STOP_ATR_MULTIPLIER=1.0,
        MAX_DAILY_LOSS_PCT=0.08,
        MAX_WEEKLY_LOSS_PCT=0.15,
        MAX_DRAWDOWN_PCT=0.30,
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


class TestRiskEngine:
    def setup_method(self):
        self.settings = make_settings()
        self.engine = RiskEngine(self.settings)

    def _calc(self, **kwargs):
        defaults = dict(
            equity=10000.0,
            entry_price=50000.0,
            side="Buy",
            atr=500.0,
            signal_strength=0.7,
            spread_pct=0.001,
            funding_rate=0.0001,
            vol_percentile=0.4,
            drawdown_pct=0.0,
            consecutive_losses=0,
            symbol_max_leverage=125,
            symbol_info=SYMBOL_INFO,
            current_positions=0,
        )
        defaults.update(kwargs)
        return self.engine.calculate(**defaults)

    def test_basic_calculation_returns_valid(self):
        result = self._calc()
        assert result.risk_ok
        assert result.leverage >= 1
        assert result.qty_base > 0
        assert result.position_size_usdt > 0

    def test_max_loss_bounded(self):
        """1-trade max loss should be ≤ risk_pct of equity."""
        result = self._calc(equity=10000.0)
        assert result.max_loss_usdt <= 10000.0 * 0.03  # 2% + buffer

    def test_leverage_within_caps(self):
        result = self._calc()
        assert result.leverage <= self.settings.MAX_LEVERAGE_CAP
        assert result.leverage <= SYMBOL_INFO["max_leverage"]
        assert result.leverage >= 1

    def test_high_volatility_reduces_leverage(self):
        normal = self._calc(vol_percentile=0.3)
        high_vol = self._calc(vol_percentile=0.90)
        assert high_vol.leverage <= normal.leverage

    def test_high_spread_reduces_leverage(self):
        normal = self._calc(spread_pct=0.001)
        wide = self._calc(spread_pct=0.006)
        assert wide.leverage <= normal.leverage

    def test_drawdown_reduces_leverage(self):
        no_dd = self._calc(drawdown_pct=0.0)
        high_dd = self._calc(drawdown_pct=0.15)
        assert high_dd.leverage <= no_dd.leverage

    def test_max_concurrent_positions(self):
        result = self._calc(current_positions=3)
        assert not result.risk_ok

    def test_zero_equity_no_trade(self):
        result = self._calc(equity=0)
        assert not result.risk_ok

    def test_stop_price_correct_long(self):
        result = self._calc(entry_price=50000, atr=500, side="Buy")
        assert result.stop_price < 50000
        assert result.take_profit_price > 50000

    def test_stop_price_correct_short(self):
        result = self._calc(entry_price=50000, atr=500, side="Sell")
        assert result.stop_price > 50000
        assert result.take_profit_price < 50000

    def test_min_notional_check(self):
        """Very small equity should fail min notional."""
        result = self._calc(equity=1.0)
        if result.risk_ok:
            assert result.position_size_usdt >= SYMBOL_INFO["min_notional"]

    def test_drawdown_hard_cap_at_15_pct(self):
        result = self._calc(drawdown_pct=0.16)
        assert result.leverage <= 5

    def test_consecutive_losses_reduce_leverage(self):
        no_loss = self._calc(consecutive_losses=0)
        many_loss = self._calc(consecutive_losses=5)
        assert many_loss.leverage <= no_loss.leverage

    def test_scalp_mode_tp1_tp2_present(self):
        result = self._calc(trade_mode="SCALP")
        assert result.risk_ok
        assert result.tp1_price > 0
        assert result.tp2_price > 0
        assert result.tp1_price < result.tp2_price  # TP1 closer than TP2

    def test_trend_ride_mode_wider_stops(self):
        scalp = self._calc(trade_mode="SCALP", side="Buy")
        trend = self._calc(trade_mode="TREND_RIDE", side="Buy")
        assert trend.stop_price < scalp.stop_price

    def test_trend_ride_mode_higher_tp(self):
        scalp = self._calc(trade_mode="SCALP", side="Buy")
        trend = self._calc(trade_mode="TREND_RIDE", side="Buy")
        assert trend.take_profit_price > scalp.take_profit_price

    def test_trade_mode_propagated(self):
        result = self._calc(trade_mode="TREND_RIDE")
        assert result.trade_mode == "TREND_RIDE"

    def test_default_mode_is_scalp(self):
        result = self._calc()
        assert result.trade_mode == "SCALP"

    def test_scalp_mode_max_bars(self):
        result = self._calc(trade_mode="SCALP")
        assert result.mode_max_bars == self.settings.SCALP_MAX_BARS

    def test_trend_ride_mode_max_bars(self):
        result = self._calc(trade_mode="TREND_RIDE")
        assert result.mode_max_bars == self.settings.TREND_MAX_BARS

    def test_tp1_short_position(self):
        result = self._calc(trade_mode="SCALP", side="Sell", entry_price=50000, atr=500)
        assert result.tp1_price < 50000    # for short, TP1 below entry
        assert result.tp2_price < result.tp1_price  # TP2 even lower

    # ── Fee Breakeven Check ────────────────────────────────────────────────────

    def test_fee_breakeven_rejects_tiny_atr(self):
        """ATR too small → TP1 profit below fee × 1.5 → no trade."""
        # With ATR=1 on a 50000 asset, move is 0.002%
        # SCALP TP1 = 1.5 ATR = 1.5 × 1 = 1.5 → 0.003% — still tiny
        # Binance round-trip fee = 0.02%×2 + 0.02%×2 = 0.08%, threshold = 0.12%
        result = self._calc(atr=1.0, entry_price=50000)
        assert not result.risk_ok

    def test_fee_breakeven_passes_normal_atr(self):
        """Normal ATR passes fee breakeven check."""
        # ATR=500 on 50000 → SCALP TP1=1.5×500/50000=1.5% >> 0.12% threshold
        result = self._calc(atr=500, entry_price=50000)
        assert result.risk_ok

    def test_wide_spread_uses_taker_fee_in_breakeven(self):
        """스프레드 > SPREAD_TAKER_THRESHOLD → Taker 수수료로 손익분기점 계산."""
        s = make_settings(
            SPREAD_TAKER_THRESHOLD=0.003,
            SIGNAL_URGENT_THRESHOLD=0.90,
            BACKTEST_FEE_MAKER=0.0002,
            BACKTEST_FEE_TAKER=0.0004,
            BACKTEST_SLIPPAGE_PCT=0.0002,
        )
        engine = RiskEngine(s)
        # 스프레드 넓음 → 시장가 진입 → Taker 수수료 0.04%
        # round_trip = 0.04%+0.02%+0.04% = 0.10% → min_tp1 = 0.20%
        # ATR=4, SCALP_TP1=1.5 → tp1_move = 1.5*4/50000 = 0.012% → 거부
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=4.0,
            signal_strength=0.7, spread_pct=0.005, funding_rate=0.0,
            vol_percentile=0.3, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
        )
        assert not result.risk_ok
        assert "Taker" in result.reason

    def test_urgent_signal_uses_taker_fee_in_breakeven(self):
        """신호 강도 > SIGNAL_URGENT_THRESHOLD → Taker 수수료로 계산."""
        s = make_settings(
            SPREAD_TAKER_THRESHOLD=0.003,
            SIGNAL_URGENT_THRESHOLD=0.90,
            BACKTEST_FEE_MAKER=0.0002,
            BACKTEST_FEE_TAKER=0.0004,
            BACKTEST_SLIPPAGE_PCT=0.0002,
        )
        engine = RiskEngine(s)
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=4.0,
            signal_strength=0.95, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.3, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
        )
        assert not result.risk_ok
        assert "Taker" in result.reason

    def test_normal_spread_uses_maker_fee_in_breakeven(self):
        """정상 스프레드 + 낮은 신호 → PostOnly(Maker) 수수료로 계산."""
        s = make_settings(
            SPREAD_TAKER_THRESHOLD=0.003,
            SIGNAL_URGENT_THRESHOLD=0.90,
            BACKTEST_FEE_MAKER=0.0002,
            BACKTEST_FEE_TAKER=0.0004,
            BACKTEST_SLIPPAGE_PCT=0.0002,
        )
        engine = RiskEngine(s)
        # Maker: round_trip = 0.02%+0.02%+0.04% = 0.08% → min_tp1 = 0.16%
        # ATR=60: tp1 = 1.5*60/50000 = 0.18% > 0.16% → 통과
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=60.0,
            signal_strength=0.70, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.3, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
        )
        assert result.risk_ok

    def test_maker_vs_taker_breakeven_difference(self):
        """Maker 기준은 통과하지만 Taker 기준은 거부되는 경계 케이스."""
        s = make_settings(
            SPREAD_TAKER_THRESHOLD=0.003,
            SIGNAL_URGENT_THRESHOLD=0.90,
            BACKTEST_FEE_MAKER=0.0002,
            BACKTEST_FEE_TAKER=0.0004,
            BACKTEST_SLIPPAGE_PCT=0.0002,
        )
        engine = RiskEngine(s)
        # ATR=60: tp1_move = 1.5*60/50000 = 0.18%
        # Maker min_tp1 = 0.16% → 통과
        # Taker min_tp1 = 0.20% → 거부
        maker_result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=60.0,
            signal_strength=0.70, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.3, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
        )
        taker_result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=60.0,
            signal_strength=0.95, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.3, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
        )
        assert maker_result.risk_ok, "Maker 경로는 통과해야 함"
        assert not taker_result.risk_ok, "Taker 경로는 거부돼야 함"

    # ── Volatility Targeting (AQR) ────────────────────────────────────────────

    def test_vol_targeting_with_recent_close(self):
        """With recent_close, vol_target_size should be computed."""
        rng = np.random.default_rng(42)
        prices = 50000.0 + np.cumsum(rng.normal(0, 50, 60))
        recent_close = pd.Series(prices)

        s = make_settings(VOLATILITY_TARGET_PCT=0.15)
        engine = RiskEngine(s)
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=0.7, spread_pct=0.001, funding_rate=0.0001,
            vol_percentile=0.4, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
            recent_close=recent_close,
        )
        assert result.risk_ok
        assert result.vol_target_size >= 0.0

    def test_no_vol_targeting_without_recent_close(self):
        """Without recent_close, vol_target_size should be 0."""
        result = self._calc()
        assert result.vol_target_size == 0.0

    # ── Quarter-Kelly ─────────────────────────────────────────────────────────

    def test_kelly_size_computed_with_win_stats(self):
        """With 30+ trades win_stats, kelly_size should be computed."""
        win_stats = {
            "n_trades": 100,
            "win_rate": 0.55,
            "avg_win": 120.0,
            "avg_loss": 80.0,
        }
        s = make_settings(USE_KELLY_FRACTION=0.25)
        engine = RiskEngine(s)
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=0.7, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.4, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
            win_stats=win_stats,
        )
        assert result.risk_ok
        assert result.kelly_size >= 0.0

    def test_no_kelly_size_without_win_stats(self):
        """Without win_stats, kelly_size should be 0."""
        result = self._calc()
        assert result.kelly_size == 0.0

    def test_kelly_not_applied_with_few_trades(self):
        """Kelly should not constrain with fewer than 30 trades."""
        win_stats = {
            "n_trades": 10,   # not enough trades
            "win_rate": 0.55,
            "avg_win": 120.0,
            "avg_loss": 80.0,
        }
        result_no_stats = self._calc()
        result_few_trades = self._calc(win_stats=win_stats)
        assert result_no_stats.qty_base == pytest.approx(result_few_trades.qty_base, rel=0.01)

    # ── Bear Mode ─────────────────────────────────────────────────────────────

    def test_bear_mode_uses_higher_leverage_cap(self):
        """bear_mode=True should use BEAR_MAX_LEVERAGE_CAP (50) not MAX_LEVERAGE_CAP (10)."""
        s = make_settings(MAX_LEVERAGE_CAP=10, BEAR_MAX_LEVERAGE_CAP=50)
        engine = RiskEngine(s)
        # High signal strength + low atr → leverage would be high, but capped
        result = engine.calculate(
            equity=10000.0, entry_price=50000.0, side="Buy", atr=500.0,
            signal_strength=1.0, spread_pct=0.001, funding_rate=0.0,
            vol_percentile=0.2, drawdown_pct=0.0, consecutive_losses=0,
            symbol_max_leverage=125, symbol_info=SYMBOL_INFO, current_positions=0,
            bear_mode=True,
        )
        assert result.risk_ok
        assert result.leverage <= 50  # bear mode allows up to 50×

    def test_normal_mode_uses_normal_leverage_cap(self):
        """bear_mode=False (default) should use MAX_LEVERAGE_CAP (10)."""
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
        assert result.leverage <= 10  # normal mode capped at 10×

    def test_bear_mode_default_false(self):
        """bear_mode should default to False."""
        result = self._calc()
        assert result.leverage <= self.settings.MAX_LEVERAGE_CAP


class TestStateMachine:
    def setup_method(self):
        self.settings = make_settings()
        self.sm = StateMachine(self.settings)

    def test_entry_fee_stored(self):
        """entry_fee should be stored on enter."""
        self.sm.enter(
            symbol="BTCUSDT", direction=1, entry_price=50000.0, qty=0.1,
            leverage=10, stop_price=49000.0, take_profit_price=53000.0,
            trailing_distance=500.0, entry_fee=10.0,
        )
        pos = self.sm.get_position("BTCUSDT")
        assert pos.entry_fee == pytest.approx(10.0)

    def test_partial_exit_prorates_entry_fee(self):
        """partial_exit with partial_pct=0.6 should reduce entry_fee by 60%."""
        self.sm.enter(
            symbol="BTCUSDT", direction=1, entry_price=50000.0, qty=0.1,
            leverage=10, stop_price=49000.0, take_profit_price=53000.0,
            trailing_distance=500.0, entry_fee=10.0,
            tp1_price=51000.0, tp2_price=53000.0, partial_exit_pct=0.60,
        )
        pos = self.sm.get_position("BTCUSDT")
        self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=51000.0, partial_pct=0.60)
        assert pos.entry_fee == pytest.approx(4.0, rel=1e-6)  # 10 × (1 - 0.6) = 4.0

    def test_exit_resets_entry_fee(self):
        """After full exit, entry_fee should be 0."""
        self.sm.enter(
            symbol="BTCUSDT", direction=1, entry_price=50000.0, qty=0.1,
            leverage=10, stop_price=49000.0, take_profit_price=53000.0,
            trailing_distance=500.0, entry_fee=10.0,
        )
        self.sm.exit("BTCUSDT", exit_price=51000.0)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.entry_fee == 0.0

    def test_force_flat_resets_entry_fee(self):
        """force_flat should reset entry_fee to 0."""
        self.sm.enter(
            symbol="BTCUSDT", direction=1, entry_price=50000.0, qty=0.1,
            leverage=10, stop_price=49000.0, take_profit_price=53000.0,
            trailing_distance=500.0, entry_fee=10.0,
        )
        self.sm.force_flat("BTCUSDT")
        pos = self.sm.get_position("BTCUSDT")
        assert pos.entry_fee == 0.0

    def test_entry_fee_default_zero(self):
        """entry_fee should default to 0 if not provided."""
        self.sm.enter(
            symbol="ETHUSDT", direction=-1, entry_price=3000.0, qty=1.0,
            leverage=5, stop_price=3100.0, take_profit_price=2700.0,
            trailing_distance=30.0,
        )
        pos = self.sm.get_position("ETHUSDT")
        assert pos.entry_fee == 0.0


class TestRiskGuards:
    def setup_method(self):
        self.settings = make_settings()
        self.guards = RiskGuards(self.settings)

    def _check(self, **kwargs):
        defaults = dict(
            equity=10000.0,
            peak_equity=10000.0,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            initial_capital=10000.0,
            spread_pct=0.001,
            vol_percentile=0.5,
        )
        defaults.update(kwargs)
        return self.guards.check(**defaults)

    def test_normal_conditions_allow(self):
        result = self._check()
        assert result.action == GuardAction.ALLOW

    def test_daily_loss_limit_allows_with_warning(self):
        # 복리 모드: 일일 손실 한도 초과 시 PAUSE 대신 ALLOW (거래 계속)
        result = self._check(equity=9100, daily_pnl=-900)  # ~9% loss
        assert result.action == GuardAction.ALLOW
        # reason에 "Daily loss" 포함 → main.py에서 Slack 알림 전송
        assert "Daily loss" in result.reason

    def test_weekly_loss_limit_allows_with_warning(self):
        # 복리 모드: 주간 손실 한도 초과 시 PAUSE 대신 ALLOW (거래 계속)
        result = self._check(equity=8400, weekly_pnl=-1600)  # ~16% weekly
        assert result.action == GuardAction.ALLOW

    def test_max_drawdown_heavy_reduce(self):
        # 31% DD → heavy REDUCE (30~42% zone)
        result = self._check(equity=6900, peak_equity=10000)  # 31% DD
        assert result.action == GuardAction.REDUCE

    def test_max_drawdown_pause(self):
        # 43% DD → PAUSE (42~50% zone)
        result = self._check(equity=5700, peak_equity=10000)  # 43% DD
        assert result.action == GuardAction.PAUSE

    def test_max_drawdown_halt(self):
        # 51% DD → HALT (>= 50%)
        result = self._check(equity=4900, peak_equity=10000)  # 51% DD
        assert result.action == GuardAction.HALT

    def test_moderate_drawdown_reduce(self):
        # ~22% drawdown → REDUCE (70% of 30% limit = 21%)
        result = self._check(equity=7800, peak_equity=10000)  # 22% DD
        assert result.action in (GuardAction.REDUCE, GuardAction.ALLOW)

    def test_extreme_spread_pauses(self):
        result = self._check(spread_pct=0.015)
        assert result.action in (GuardAction.PAUSE, GuardAction.REDUCE)

    def test_extreme_volatility_reduces(self):
        result = self._check(vol_percentile=0.98)
        assert result.action in (GuardAction.REDUCE, GuardAction.PAUSE)

    def test_zero_equity_halts(self):
        result = self._check(equity=0)
        assert result.action == GuardAction.HALT
