"""Tests for order manager and state machine."""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.execution.state_machine import StateMachine, PositionState, ExitReason, ExitSignal


def make_settings(**kwargs):
    from src.config import Settings
    defaults = dict(
        TRADING_MODE="dry-run",
        BINANCE_API_KEY="",
        BINANCE_API_SECRET="",
        REENTRY_COOLDOWN_SECONDS=0,  # no cooldown in tests
        ATR_PERIOD=14,
        STOP_ATR_MULTIPLIER=2.0,
        TP_ATR_MULTIPLIER=3.0,
        TRAILING_STOP_ATR_MULTIPLIER=1.5,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


class TestStateMachine:
    def setup_method(self):
        self.settings = make_settings()
        self.sm = StateMachine(self.settings)

    def _enter_long(self, symbol="BTCUSDT", entry=50000.0, qty=0.01, leverage=5,
                    tp1_price=0.0, tp2_price=0.0, trade_mode="SCALP",
                    mode_max_bars=15, partial_exit_pct=0.60):
        return self.sm.enter(
            symbol=symbol, direction=1,
            entry_price=entry, qty=qty, leverage=leverage,
            stop_price=entry - 1000,
            take_profit_price=entry + 1500,
            trailing_distance=750,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            trade_mode=trade_mode,
            mode_max_bars=mode_max_bars,
            partial_exit_pct=partial_exit_pct,
        )

    def _enter_short(self, symbol="BTCUSDT", entry=50000.0, qty=0.01, leverage=5,
                     tp1_price=0.0, tp2_price=0.0, trade_mode="SCALP",
                     mode_max_bars=15, partial_exit_pct=0.60):
        return self.sm.enter(
            symbol=symbol, direction=-1,
            entry_price=entry, qty=qty, leverage=leverage,
            stop_price=entry + 1000,
            take_profit_price=entry - 1500,
            trailing_distance=750,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            trade_mode=trade_mode,
            mode_max_bars=mode_max_bars,
            partial_exit_pct=partial_exit_pct,
        )

    def test_initial_state_flat(self):
        pos = self.sm.get_position("BTCUSDT")
        assert pos.state == PositionState.FLAT
        assert pos.is_flat

    def test_enter_long(self):
        result = self._enter_long()
        assert result is True
        pos = self.sm.get_position("BTCUSDT")
        assert pos.is_long
        assert pos.entry_price == 50000.0

    def test_enter_short(self):
        result = self._enter_short()
        assert result is True
        pos = self.sm.get_position("BTCUSDT")
        assert pos.is_short

    def test_cannot_enter_when_already_open(self):
        self._enter_long()
        result = self._enter_long()  # try again
        assert result is False
        # State should still be LONG, not double-entered
        pos = self.sm.get_position("BTCUSDT")
        assert pos.is_long

    def test_stop_loss_long(self):
        self._enter_long(entry=50000)
        # Price drops to stop level
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=48999,
                                       current_atr=500, signal_direction=0)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.STOP_LOSS

    def test_stop_loss_short(self):
        self._enter_short(entry=50000)
        # Price rises to stop level
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=51001,
                                       current_atr=500, signal_direction=0)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.STOP_LOSS

    def test_take_profit_long(self):
        self._enter_long(entry=50000)
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=51500,
                                       current_atr=500, signal_direction=0)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.TAKE_PROFIT

    def test_take_profit_short(self):
        self._enter_short(entry=50000)
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=48500,
                                       current_atr=500, signal_direction=0)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.TAKE_PROFIT

    def test_trailing_stop_long(self):
        # Trailing stop only fires AFTER tp1 partial exit.
        # Simulate: enter → trigger TP1 partial → price rises → trailing fires.
        self._enter_long(entry=50000)
        # Manually mark tp1 as hit (simulates partial_exit() being called)
        pos = self.sm.get_position("BTCUSDT")
        pos.tp1_hit = True
        pos.stop_price = 50000  # breakeven stop after TP1
        # Price rises to 52000 (new high), trailing dist=750
        self.sm.check_exit("BTCUSDT", current_price=52000, current_atr=500, signal_direction=0)
        # Price drops 750 from high (52000-750=51250 > 50000 breakeven) → trailing fires
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=51249,
                                       current_atr=500, signal_direction=0)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.TRAILING_STOP

    def test_regime_change_exit(self):
        self._enter_long()
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=50100,
                                       current_atr=500, signal_direction=0,
                                       regime_flip=True)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.REGIME_CHANGE

    def test_signal_flip_exits(self):
        self._enter_long()
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=50100,
                                       current_atr=500, signal_direction=-1)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.SIGNAL_FLIP

    def test_time_exit(self):
        self._enter_long()
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=50100,
                                       current_atr=500, signal_direction=0,
                                       bars_in_trade=201, max_bars=200)
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.TIME_EXIT

    def test_normal_conditions_no_exit(self):
        self._enter_long(entry=50000)
        # Price moves slightly in favour — no exit triggered
        exit_sig = self.sm.check_exit("BTCUSDT", current_price=50200,
                                       current_atr=500, signal_direction=1)
        assert not exit_sig.should_exit

    def test_exit_returns_pnl(self):
        self._enter_long(entry=50000, qty=0.01)
        pnl = self.sm.exit("BTCUSDT", exit_price=51000)
        assert abs(pnl - 10.0) < 0.001  # (51000-50000) * 0.01 = 10

    def test_exit_short_pnl(self):
        self._enter_short(entry=50000, qty=0.01)
        pnl = self.sm.exit("BTCUSDT", exit_price=49000)
        assert abs(pnl - 10.0) < 0.001  # (50000-49000) * 0.01 = 10

    def test_state_after_exit_is_flat(self):
        self._enter_long()
        self.sm.exit("BTCUSDT", exit_price=51000)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.is_flat

    def test_consecutive_losses_tracked(self):
        # Loss → increment
        self._enter_long(entry=50000, qty=0.01)
        pnl = self.sm.exit("BTCUSDT", exit_price=49000)  # loss
        pos = self.sm.get_position("BTCUSDT")
        assert pos.consecutive_losses == 1

    def test_open_count(self):
        self.sm.enter("BTCUSDT", 1, 50000, 0.01, 5, 49000, 51500, 750)
        self.sm.enter("ETHUSDT", 1, 3000, 0.1, 5, 2940, 3090, 45)
        assert self.sm.open_count() == 2

    def test_cooldown_prevents_reentry(self):
        settings = make_settings(REENTRY_COOLDOWN_SECONDS=300)
        sm = StateMachine(settings)
        sm.enter("BTCUSDT", 1, 50000, 0.01, 5, 49000, 51500, 750)
        sm.exit("BTCUSDT", 51000)
        # Should be blocked by cooldown
        assert not sm.can_enter("BTCUSDT")

    # ── Partial TP1 tests ─────────────────────────────────────────────────────

    def test_partial_tp1_triggered_for_long(self):
        """When price hits TP1, a partial exit signal should be returned."""
        self._enter_long(
            entry=50000, qty=0.10,
            tp1_price=50500,
            tp2_price=52000,
            partial_exit_pct=0.60,
        )
        exit_sig = self.sm.check_exit(
            "BTCUSDT", current_price=50500,
            current_atr=500, signal_direction=1
        )
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.PARTIAL_TP1
        assert exit_sig.partial is True
        assert exit_sig.partial_pct == 0.60

    def test_partial_tp1_triggered_for_short(self):
        """When price hits TP1 (below entry for short), partial exit returned."""
        self._enter_short(
            entry=50000, qty=0.10,
            tp1_price=49500,
            tp2_price=48000,
            partial_exit_pct=0.60,
        )
        exit_sig = self.sm.check_exit(
            "BTCUSDT", current_price=49500,
            current_atr=500, signal_direction=-1
        )
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.PARTIAL_TP1
        assert exit_sig.partial is True

    def test_partial_exit_reduces_qty(self):
        self._enter_long(
            entry=50000, qty=0.10,
            tp1_price=50500,
            tp2_price=52000,
            partial_exit_pct=0.60,
        )
        pnl = self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=50500)
        pos = self.sm.get_position("BTCUSDT")
        assert abs(pos.qty - 0.04) < 1e-9
        assert pnl > 0

    def test_partial_exit_moves_stop_to_breakeven(self):
        self._enter_long(
            entry=50000, qty=0.10,
            tp1_price=50500,
            tp2_price=52000,
        )
        self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=50500)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.stop_price == 50000.0   # moved to entry (breakeven)

    def test_tp1_hit_flag_set_after_partial_exit(self):
        self._enter_long(
            entry=50000, qty=0.10,
            tp1_price=50500,
            tp2_price=52000,
        )
        self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=50500)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.tp1_hit is True

    def test_no_second_tp1_after_already_hit(self):
        """After tp1_hit=True, TP1 price check should not trigger again."""
        self._enter_long(
            entry=50000, qty=0.10,
            tp1_price=50500,
            tp2_price=52000,
            partial_exit_pct=0.60,
        )
        # Trigger partial exit
        self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=50500)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.tp1_hit is True

        # Check again at same price — should NOT trigger PARTIAL_TP1 again
        exit_sig = self.sm.check_exit(
            "BTCUSDT", current_price=50500,
            current_atr=500, signal_direction=1
        )
        assert exit_sig.reason != ExitReason.PARTIAL_TP1

    def test_take_profit_price_updated_to_tp2(self):
        """After partial exit, take_profit_price should point to TP2."""
        self._enter_long(
            entry=50000, qty=0.10,
            tp1_price=50500,
            tp2_price=52000,
        )
        self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=50500)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.take_profit_price == 52000.0

    def test_mode_max_bars_used_in_time_exit(self):
        """SCALP mode should exit at SCALP_MAX_BARS, not 200."""
        self._enter_long(
            entry=50000, qty=0.01,
            trade_mode="SCALP", mode_max_bars=15,
        )
        # Should NOT exit before max_bars
        no_exit = self.sm.check_exit(
            "BTCUSDT", current_price=50100,
            current_atr=500, signal_direction=1,
            bars_in_trade=14, max_bars=200,  # pass 200 as default
        )
        assert not no_exit.should_exit

        # Should exit AT max_bars
        exit_sig = self.sm.check_exit(
            "BTCUSDT", current_price=50100,
            current_atr=500, signal_direction=1,
            bars_in_trade=15, max_bars=200,
        )
        assert exit_sig.should_exit
        assert exit_sig.reason == ExitReason.TIME_EXIT

    def test_trade_mode_stored_in_position(self):
        self._enter_long(trade_mode="TREND_RIDE", mode_max_bars=120)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.trade_mode == "TREND_RIDE"
        assert pos.mode_max_bars == 120

    def test_state_reset_after_full_exit(self):
        """After full exit, tp1/tp2/tp1_hit should all reset."""
        self._enter_long(
            entry=50000, qty=0.01,
            tp1_price=50500, tp2_price=52000,
        )
        self.sm.exit("BTCUSDT", exit_price=52000)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.is_flat
        assert pos.tp1_price == 0.0
        assert pos.tp2_price == 0.0
        assert pos.tp1_hit is False
