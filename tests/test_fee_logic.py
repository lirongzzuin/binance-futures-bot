"""
수수료 로직 테스트 — 지정가/시장가 주문 타입별 수수료 정확성 검증

검증 항목:
  1. GTX(PostOnly) 성공 → Maker fee(0.02%)
  2. GTX 거부 후 Market fallback → Taker fee(0.04%)
  3. spread > SPREAD_TAKER_THRESHOLD → Market 즉시 선택 → Taker fee
  4. signal_strength > 0.90 → Market 즉시 선택 → Taker fee
  5. STOP_LOSS 청산 → Market → Taker fee
  6. TAKE_PROFIT 청산 → Limit → Maker fee
  7. TRAILING_STOP 청산 → Market("STOP" 포함) → Taker fee
  8. SIGNAL_FLIP/REGIME_CHANGE/TIME_EXIT 청산 → Limit → Maker fee
  9. Partial TP1(PARTIAL_TP1) 청산 → Limit → Maker fee
 10. Partial TP1 후 full exit — entry_fee pro-rata 처리 정확성
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from src.execution.order_manager import OrderManager, ManagedOrder, OrderStatus
from src.execution.state_machine import StateMachine, ExitReason


FEE_MAKER = 0.0002   # 0.02%
FEE_TAKER = 0.0004   # 0.04%


# ── 공통 Fixtures ─────────────────────────────────────────────────────────────

def make_settings(**kwargs):
    from src.config import Settings
    defaults = dict(
        TRADING_MODE="dry-run",
        BINANCE_API_KEY="",
        BINANCE_API_SECRET="",
        BINANCE_HEDGE_MODE=True,
        SPREAD_TAKER_THRESHOLD=0.003,
        BACKTEST_FEE_MAKER=FEE_MAKER,
        BACKTEST_FEE_TAKER=FEE_TAKER,
        BACKTEST_SLIPPAGE_PCT=0.0002,
        LIMIT_ORDER_TIMEOUT_SECONDS=5,
        LIMIT_ORDER_MAX_RETRIES=3,
        REENTRY_COOLDOWN_SECONDS=0,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def make_order_manager(settings=None):
    s = settings or make_settings()
    mock_client = MagicMock()
    mock_slack = MagicMock()
    return OrderManager(mock_client, s, mock_slack)


def make_risk_params(qty=0.01, leverage=5, entry_price=50000.0, stop_price=49000.0,
                     tp=51500.0, tp1=50500.0, tp2=51500.0):
    from src.risk.engine import RiskParameters
    return RiskParameters(
        leverage=leverage,
        position_size_usdt=qty * entry_price,
        margin_usdt=qty * entry_price / leverage,
        qty_base=qty,
        stop_price=stop_price,
        take_profit_price=tp,
        tp1_price=tp1,
        tp2_price=tp2,
        trailing_stop_atr=500.0,
        risk_ok=True,
        trade_mode="SCALP",
    )


SYMBOL_INFO = {
    "qty_step": 0.001,
    "price_step": 0.1,
    "min_qty": 0.001,
    "max_leverage": 125,
}


# ── 1. 진입 주문 타입 테스트 ──────────────────────────────────────────────────

class TestEntryOrderType:
    """진입 주문이 상황에 따라 LIMIT 또는 MARKET으로 올바르게 선택되는지 검증."""

    def test_low_spread_limit_order_chosen(self):
        """spread < threshold, signal ≤ 0.90 → LIMIT(GTX) 선택."""
        om = make_order_manager()
        rp = make_risk_params()
        om._client.round_qty.return_value = "0.010"
        om._client.round_price.return_value = "49999.9"

        # dry-run이면 _place_limit_with_retry의 dry-run 분기 → LIMIT으로 처리
        order = om.open_position(
            symbol="BTCUSDT",
            direction=1,
            risk_params=rp,
            spread_pct=0.001,       # 0.1% < 0.3% threshold
            signal_strength=0.75,   # ≤ 0.90
            current_price=50000.0,
            symbol_info=SYMBOL_INFO,
        )
        assert order is not None
        assert order.order_type == "LIMIT"

    def test_high_spread_market_order_chosen(self):
        """spread > threshold → MARKET 즉시 선택."""
        om = make_order_manager()
        rp = make_risk_params()
        om._client.round_qty.return_value = "0.010"

        order = om.open_position(
            symbol="BTCUSDT",
            direction=1,
            risk_params=rp,
            spread_pct=0.005,       # 0.5% > 0.3% threshold
            signal_strength=0.75,
            current_price=50000.0,
            symbol_info=SYMBOL_INFO,
        )
        assert order is not None
        assert order.order_type == "MARKET"

    def test_high_signal_strength_market_order_chosen(self):
        """signal_strength > 0.90 → MARKET 즉시 선택."""
        om = make_order_manager()
        rp = make_risk_params()
        om._client.round_qty.return_value = "0.010"

        order = om.open_position(
            symbol="BTCUSDT",
            direction=1,
            risk_params=rp,
            spread_pct=0.001,
            signal_strength=0.95,   # > 0.90
            current_price=50000.0,
            symbol_info=SYMBOL_INFO,
        )
        assert order is not None
        assert order.order_type == "MARKET"

    def test_gtx_fallback_sets_market_type(self):
        """GTX 거부(-5022) 후 Market fallback → order_type이 MARKET으로 설정됨."""
        settings = make_settings(TRADING_MODE="paper")  # non-dry-run to trigger real path
        mock_client = MagicMock()
        mock_slack = MagicMock()
        om = OrderManager(mock_client, settings, mock_slack)

        # position mode / leverage 설정 성공
        mock_client.set_position_mode.return_value = None
        mock_client.set_leverage.return_value = None
        mock_client.round_qty.return_value = "0.010"
        mock_client.round_price.return_value = "49999.9"

        # GTX 주문 → -5022 에러 발생
        from src.api.binance_client import BinanceClientError
        mock_client.place_order.side_effect = [
            BinanceClientError(-5022, "GTX rejected"),   # limit attempt 1
            {"orderId": "123", "avgPrice": "50000", "executedQty": "0.01"},  # market success
        ]

        rp = make_risk_params()
        order = om.open_position(
            symbol="BTCUSDT",
            direction=1,
            risk_params=rp,
            spread_pct=0.001,
            signal_strength=0.75,
            current_price=50000.0,
            symbol_info=SYMBOL_INFO,
        )
        assert order is not None
        assert order.order_type == "MARKET", (
            "GTX 거부 후 Market fallback 시 order_type이 MARKET이어야 합니다"
        )


# ── 2. 청산 주문 타입 테스트 ──────────────────────────────────────────────────

class TestExitOrderType:
    """청산 사유별 주문 타입이 올바른지 검증."""

    def _make_close_order(self, reason: str, spread=0.001) -> ManagedOrder:
        om = make_order_manager()
        om._client.round_qty.return_value = "0.010"
        om._client.round_price.return_value = "50001.0"
        return om.close_position(
            symbol="BTCUSDT",
            direction=1,
            qty=0.01,
            current_price=50000.0,
            symbol_info=SYMBOL_INFO,
            reason=reason,
        )

    def test_stop_loss_uses_market(self):
        order = self._make_close_order("STOP_LOSS")
        assert order.order_type == "MARKET"

    def test_trailing_stop_uses_market(self):
        """TRAILING_STOP에 'STOP' 포함 → MARKET."""
        order = self._make_close_order("TRAILING_STOP")
        assert order.order_type == "MARKET"

    def test_risk_guard_uses_market(self):
        """RISK_GUARD에 'GUARD' 포함 → MARKET."""
        order = self._make_close_order("RISK_GUARD")
        assert order.order_type == "MARKET"

    def test_liquidation_risk_uses_market(self):
        """LIQUIDATION_RISK에 'LIQUIDATION' 포함 → MARKET."""
        order = self._make_close_order("LIQUIDATION_RISK")
        assert order.order_type == "MARKET"

    def test_take_profit_uses_limit(self):
        order = self._make_close_order("TAKE_PROFIT")
        assert order.order_type == "LIMIT"

    def test_signal_flip_uses_limit(self):
        order = self._make_close_order("SIGNAL_FLIP")
        assert order.order_type == "LIMIT"

    def test_regime_change_uses_limit(self):
        order = self._make_close_order("REGIME_CHANGE")
        assert order.order_type == "LIMIT"

    def test_time_exit_uses_limit(self):
        order = self._make_close_order("TIME_EXIT")
        assert order.order_type == "LIMIT"

    def test_partial_tp1_uses_limit(self):
        order = self._make_close_order("PARTIAL_TP1")
        assert order.order_type == "LIMIT"


# ── 3. 수수료 계산 단위 테스트 ────────────────────────────────────────────────

class TestFeeCalculation:
    """order_type 기반 수수료 계산이 올바른지 수동 계산으로 검증."""

    def _calc_entry_fee(self, order_type: str, qty: float, price: float) -> float:
        rate = FEE_TAKER if order_type == "MARKET" else FEE_MAKER
        return qty * price * rate

    def _calc_exit_fee(self, order_type: str, qty: float, price: float) -> float:
        rate = FEE_TAKER if order_type == "MARKET" else FEE_MAKER
        return qty * price * rate

    def test_limit_entry_fee_is_maker(self):
        qty, price = 0.1, 50000.0
        fee = self._calc_entry_fee("LIMIT", qty, price)
        assert fee == pytest.approx(qty * price * FEE_MAKER)   # 1.0 USDT

    def test_market_entry_fee_is_taker(self):
        qty, price = 0.1, 50000.0
        fee = self._calc_entry_fee("MARKET", qty, price)
        assert fee == pytest.approx(qty * price * FEE_TAKER)   # 2.0 USDT

    def test_market_entry_fee_double_maker(self):
        """Taker fee가 Maker fee의 2배여야 한다."""
        qty, price = 0.1, 50000.0
        maker_fee = self._calc_entry_fee("LIMIT", qty, price)
        taker_fee = self._calc_entry_fee("MARKET", qty, price)
        assert taker_fee == pytest.approx(maker_fee * 2.0)

    def test_stop_loss_exit_fee_is_taker(self):
        """STOP_LOSS → MARKET → Taker fee."""
        qty, price = 0.1, 49000.0
        fee = self._calc_exit_fee("MARKET", qty, price)
        assert fee == pytest.approx(qty * price * FEE_TAKER)

    def test_take_profit_exit_fee_is_maker(self):
        """TAKE_PROFIT → LIMIT → Maker fee."""
        qty, price = 0.1, 51500.0
        fee = self._calc_exit_fee("LIMIT", qty, price)
        assert fee == pytest.approx(qty * price * FEE_MAKER)

    def test_net_pnl_long_limit_entry_limit_tp(self):
        """Long 포지션 — GTX 진입 + TP 지정가 청산 — net PnL 정확성."""
        entry_price, tp_price = 50000.0, 51500.0
        qty, leverage = 0.1, 5

        gross_pnl = (tp_price - entry_price) * qty          # 150 USDT
        entry_fee = qty * entry_price * FEE_MAKER            # 1.0 USDT
        exit_fee  = qty * tp_price  * FEE_MAKER              # 1.03 USDT
        net_pnl   = gross_pnl - entry_fee - exit_fee        # 147.97 USDT

        assert gross_pnl == pytest.approx(150.0)
        assert entry_fee == pytest.approx(1.0)
        assert exit_fee  == pytest.approx(1.03)
        assert net_pnl   == pytest.approx(147.97)

    def test_net_pnl_long_market_entry_market_sl(self):
        """Long 포지션 — Market 진입 + STOP_LOSS Market 청산 — Taker×2."""
        entry_price, sl_price = 50000.0, 49000.0
        qty = 0.1

        gross_pnl = (sl_price - entry_price) * qty           # -100 USDT
        entry_fee = qty * entry_price * FEE_TAKER             # 2.0 USDT
        exit_fee  = qty * sl_price  * FEE_TAKER               # 1.96 USDT
        net_pnl   = gross_pnl - entry_fee - exit_fee         # -103.96 USDT

        assert gross_pnl == pytest.approx(-100.0)
        assert entry_fee == pytest.approx(2.0)
        assert exit_fee  == pytest.approx(1.96)
        assert net_pnl   == pytest.approx(-103.96)

    def test_partial_then_full_exit_entry_fee_prorata(self):
        """Partial TP1 → Full TP2: entry_fee pro-rata 처리 후 합계 정확성."""
        entry_price = 50000.0
        qty_total   = 0.10
        partial_pct = 0.60
        qty_partial = qty_total * partial_pct   # 0.06
        qty_remain  = qty_total - qty_partial   # 0.04
        tp1_price   = 50500.0
        tp2_price   = 51500.0

        entry_fee_total = qty_total * entry_price * FEE_MAKER   # 1.0 USDT

        # --- Partial TP1 ---
        entry_fee_partial = entry_fee_total * partial_pct       # 0.60 USDT
        exit_fee_tp1      = qty_partial * tp1_price * FEE_MAKER # 0.606 USDT
        total_fee_tp1     = entry_fee_partial + exit_fee_tp1
        gross_tp1         = (tp1_price - entry_price) * qty_partial  # 30 USDT
        net_tp1           = gross_tp1 - total_fee_tp1

        # 남은 entry_fee (state_machine.partial_exit 로직과 동일)
        remaining_entry_fee = entry_fee_total * (1.0 - partial_pct)  # 0.40 USDT

        # --- Full TP2 ---
        exit_fee_tp2  = qty_remain * tp2_price * FEE_MAKER       # 0.412 USDT
        total_fee_tp2 = remaining_entry_fee + exit_fee_tp2
        gross_tp2     = (tp2_price - entry_price) * qty_remain   # 60 USDT
        net_tp2       = gross_tp2 - total_fee_tp2

        # 합계
        total_gross = gross_tp1 + gross_tp2                      # 90 USDT
        total_fee   = total_fee_tp1 + total_fee_tp2
        total_net   = net_tp1 + net_tp2

        assert entry_fee_total  == pytest.approx(1.0,   rel=1e-6)
        assert total_gross      == pytest.approx(90.0,  rel=1e-6)
        # 총 수수료 = entry_fee 전액 + TP1 exit fee + TP2 exit fee
        expected_total_fee = entry_fee_total + exit_fee_tp1 + exit_fee_tp2
        assert total_fee        == pytest.approx(expected_total_fee, rel=1e-6)
        assert total_net        == pytest.approx(total_gross - expected_total_fee, rel=1e-6)
        # 수수료 감안해도 충분한 수익
        assert total_net > 0


# ── 4. 수수료 우위(Maker vs Taker) 인식 테스트 ───────────────────────────────

class TestFeeAdvantage:
    """Maker 수수료가 Taker보다 유리한 상황에서 올바른 주문 타입이 선택되는지."""

    def test_limit_order_saves_fee_vs_market(self):
        """동일 조건에서 Limit 주문이 Market보다 수수료가 절반."""
        qty, price = 1.0, 50000.0
        maker_fee = qty * price * FEE_MAKER
        taker_fee = qty * price * FEE_TAKER
        saving = taker_fee - maker_fee
        assert saving == pytest.approx(qty * price * (FEE_TAKER - FEE_MAKER))
        assert saving > 0   # Maker가 항상 유리

    def test_round_trip_maker_maker_cheaper_than_taker_taker(self):
        """Maker+Maker round-trip < Taker+Taker round-trip."""
        qty, entry, exit_ = 0.1, 50000.0, 51000.0
        round_trip_maker = qty * entry * FEE_MAKER + qty * exit_ * FEE_MAKER
        round_trip_taker = qty * entry * FEE_TAKER + qty * exit_ * FEE_TAKER
        assert round_trip_maker < round_trip_taker

    def test_breakeven_higher_for_taker_entry(self):
        """Taker로 진입하면 손익분기점이 더 높아진다 (Long 기준)."""
        entry, qty, leverage = 50000.0, 0.1, 5
        # Maker 진입: 손익분기점 = entry + entry_fee / qty
        maker_entry_fee = qty * entry * FEE_MAKER
        maker_breakeven = entry + maker_entry_fee / qty + entry * FEE_MAKER  # 청산 수수료도 고려

        # Taker 진입: 손익분기점
        taker_entry_fee = qty * entry * FEE_TAKER
        taker_breakeven = entry + taker_entry_fee / qty + entry * FEE_TAKER

        assert taker_breakeven > maker_breakeven


# ── 5. 상태머신 entry_fee 관리 통합 테스트 ────────────────────────────────────

class TestStateMachineEntryFee:
    """StateMachine이 entry_fee를 올바르게 저장·감소·리셋하는지 검증."""

    def setup_method(self):
        self.sm = StateMachine(make_settings())

    def _enter(self, entry_fee=10.0):
        return self.sm.enter(
            symbol="BTCUSDT", direction=1,
            entry_price=50000.0, qty=0.10, leverage=5,
            stop_price=49000.0, take_profit_price=52000.0,
            trailing_distance=500.0,
            tp1_price=50500.0, tp2_price=52000.0,
            partial_exit_pct=0.60,
            entry_fee=entry_fee,
        )

    def test_entry_fee_stored(self):
        self._enter(entry_fee=10.0)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.entry_fee == pytest.approx(10.0)

    def test_entry_fee_prorated_on_partial_exit(self):
        self._enter(entry_fee=10.0)
        self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=50500.0, partial_pct=0.60)
        pos = self.sm.get_position("BTCUSDT")
        # 60% 청산 → 40% 남음
        assert pos.entry_fee == pytest.approx(10.0 * 0.40, rel=1e-6)

    def test_entry_fee_zero_after_full_exit(self):
        self._enter(entry_fee=10.0)
        self.sm.exit("BTCUSDT", exit_price=51000.0)
        pos = self.sm.get_position("BTCUSDT")
        assert pos.entry_fee == 0.0

    def test_entry_fee_zero_after_force_flat(self):
        self._enter(entry_fee=10.0)
        self.sm.force_flat("BTCUSDT")
        pos = self.sm.get_position("BTCUSDT")
        assert pos.entry_fee == 0.0

    def test_partial_then_full_entry_fee_sum(self):
        """Partial + Full exit에서 차감된 entry_fee 합계 = 원래 entry_fee."""
        initial_fee = 10.0
        self._enter(entry_fee=initial_fee)

        partial_pct = 0.60
        # partial exit: 60% 차감
        fee_used_partial = initial_fee * partial_pct
        self.sm.partial_exit("BTCUSDT", qty_to_close=0.06, exit_price=50500.0,
                              partial_pct=partial_pct)
        pos = self.sm.get_position("BTCUSDT")
        remaining = pos.entry_fee

        # full exit: 나머지 40% 사용
        fee_used_full = remaining
        self.sm.exit("BTCUSDT", exit_price=52000.0)

        assert fee_used_partial + fee_used_full == pytest.approx(initial_fee, rel=1e-6)
