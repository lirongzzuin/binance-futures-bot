"""
미체결 지정가 주문 관리 테스트

검증 항목:
  1. cancel_stale_orders — dry-run 시 API 미호출
  2. cancel_stale_orders — 실제 모드: cancel_all_open_orders 호출 확인
  3. cancel_stale_orders — 내부 추적 상태(orders, sent_ids) 정리
  4. audit_stale_orders — dry-run 시 API 미호출
  5. audit_stale_orders — 포지션 없는 심볼의 오래된 SL/TP 주문 취소
  6. audit_stale_orders — 포지션 있는 심볼은 SL/TP 주문 유지
  7. audit_stale_orders — 포지션 없는 심볼의 LIMIT 주문 취소 (타임아웃 초과)
  8. audit_stale_orders — 포지션 없어도 LIMIT 주문이 타임아웃 이내면 유지
  9. cancel_stale_orders — API 오류 시 예외 발생 없이 경고만 로깅
 10. audit_stale_orders — 빈 주문 목록 시 정상 처리
 11. _set_sl_tp — SL/TP 주문 ID가 ManagedOrder에 저장됨
 12. _set_sl_tp — dry-run 시 SL/TP 설정 건너뜀
 13. open_position 성공 후 sl_order_id / tp_order_id 추적 가능
 14. 전체 청산 흐름: 진입 → SL/TP 설정 → 청산 → 잔여 주문 취소
"""
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch, call
from src.execution.order_manager import OrderManager, ManagedOrder, OrderStatus
from src.api.binance_client import BinanceClientError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_settings(mode="dry-run", stale_timeout=300, audit_interval=5, **kwargs):
    from src.config import Settings
    defaults = dict(
        TRADING_MODE=mode,
        BINANCE_API_KEY="",
        BINANCE_API_SECRET="",
        BINANCE_HEDGE_MODE=True,
        SPREAD_TAKER_THRESHOLD=0.003,
        BACKTEST_FEE_MAKER=0.0002,
        BACKTEST_FEE_TAKER=0.0004,
        BACKTEST_SLIPPAGE_PCT=0.0002,
        LIMIT_ORDER_TIMEOUT_SECONDS=5,
        LIMIT_ORDER_MAX_RETRIES=3,
        REENTRY_COOLDOWN_SECONDS=0,
        STALE_ORDER_TIMEOUT_SECONDS=stale_timeout,
        STALE_ORDER_AUDIT_INTERVAL_MINUTES=audit_interval,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def make_order_manager(mode="dry-run", **kwargs):
    s = make_settings(mode=mode, **kwargs)
    mock_client = MagicMock()
    mock_slack = MagicMock()
    mock_client.round_qty.return_value = "0.010"
    mock_client.round_price.return_value = "49999.9"
    return OrderManager(mock_client, s, mock_slack), mock_client, mock_slack


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


# ── 1. cancel_stale_orders — dry-run ─────────────────────────────────────────

class TestCancelStaleOrders:

    def test_dry_run_skips_api_call(self):
        """dry-run 모드에서 cancel_stale_orders는 API를 호출하지 않는다."""
        om, mock_client, _ = make_order_manager(mode="dry-run")
        om.cancel_stale_orders("BTCUSDT")
        mock_client.cancel_all_open_orders.assert_not_called()

    def test_paper_mode_calls_cancel_api(self):
        """paper/live 모드에서 cancel_all_open_orders가 호출된다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.cancel_all_open_orders.return_value = {}
        om.cancel_stale_orders("BTCUSDT")
        mock_client.cancel_all_open_orders.assert_called_once_with("BTCUSDT")

    def test_clears_internal_order_tracking(self):
        """cancel_stale_orders 후 해당 심볼의 주문이 내부 상태에서 제거된다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.cancel_all_open_orders.return_value = {}

        # 내부 상태에 가짜 주문 추가
        btc_order = ManagedOrder(client_id="bot-abc123", symbol="BTCUSDT", side="BUY")
        eth_order = ManagedOrder(client_id="bot-def456", symbol="ETHUSDT", side="BUY")
        om._orders["bot-abc123"] = btc_order
        om._orders["bot-def456"] = eth_order
        om._sent_ids.add("bot-abc123")
        om._sent_ids.add("bot-def456")

        om.cancel_stale_orders("BTCUSDT")

        # BTCUSDT 주문만 제거됨
        assert "bot-abc123" not in om._orders
        assert "bot-abc123" not in om._sent_ids
        # ETHUSDT 주문은 유지
        assert "bot-def456" in om._orders
        assert "bot-def456" in om._sent_ids

    def test_api_error_does_not_raise(self):
        """cancel_all_open_orders API 오류 시 예외 없이 경고만 로깅한다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.cancel_all_open_orders.side_effect = Exception("Network error")

        # 예외가 전파되어서는 안 됨
        try:
            om.cancel_stale_orders("BTCUSDT")
        except Exception:
            pytest.fail("cancel_stale_orders raised an exception")

    def test_no_orders_for_symbol_is_safe(self):
        """주문이 없는 심볼에 대해 cancel_stale_orders를 호출해도 안전하다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.cancel_all_open_orders.return_value = {}
        om.cancel_stale_orders("SOLUSDT")  # 주문 없음
        mock_client.cancel_all_open_orders.assert_called_once_with("SOLUSDT")


# ── 2. audit_stale_orders ─────────────────────────────────────────────────────

class TestAuditStaleOrders:

    def _make_order_entry(self, symbol, order_id, order_type, age_seconds):
        """테스트용 미체결 주문 dict 생성."""
        created_ms = int((time.time() - age_seconds) * 1000)
        return {
            "symbol": symbol,
            "orderId": order_id,
            "type": order_type,
            "time": created_ms,
            "status": "NEW",
        }

    def test_dry_run_skips_audit(self):
        """dry-run 모드에서 audit은 API를 호출하지 않는다."""
        om, mock_client, _ = make_order_manager(mode="dry-run")
        om.audit_stale_orders(active_symbols=set())
        mock_client.get_open_orders.assert_not_called()

    def test_empty_open_orders_is_safe(self):
        """미체결 주문이 없으면 아무것도 하지 않는다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = []
        om.audit_stale_orders(active_symbols=set())
        mock_client.cancel_order.assert_not_called()

    def test_cancels_sl_tp_for_symbol_without_position(self):
        """포지션 없는 심볼의 SL/TP 주문은 즉시 취소된다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 111, "STOP_MARKET", age_seconds=10),
            self._make_order_entry("BTCUSDT", 222, "TAKE_PROFIT_MARKET", age_seconds=10),
        ]
        mock_client.cancel_order.return_value = {}

        # BTCUSDT는 active_symbols에 없음 (포지션 없음)
        om.audit_stale_orders(active_symbols=set())

        assert mock_client.cancel_order.call_count == 2
        cancelled_ids = [c.args[1] for c in mock_client.cancel_order.call_args_list]
        assert "111" in cancelled_ids
        assert "222" in cancelled_ids

    def test_keeps_sl_tp_for_symbol_with_position(self):
        """포지션이 있는 심볼의 SL/TP 주문은 취소하지 않는다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 111, "STOP_MARKET", age_seconds=10),
            self._make_order_entry("BTCUSDT", 222, "TAKE_PROFIT_MARKET", age_seconds=10),
        ]

        # BTCUSDT에 활성 포지션 있음
        om.audit_stale_orders(active_symbols={"BTCUSDT"})

        mock_client.cancel_order.assert_not_called()

    def test_cancels_stale_limit_entry_without_position(self):
        """포지션 없는 심볼의 타임아웃 초과 LIMIT 진입 주문은 취소된다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 333, "LIMIT", age_seconds=400),  # 400s > 300s
        ]
        mock_client.cancel_order.return_value = {}

        om.audit_stale_orders(active_symbols=set())

        mock_client.cancel_order.assert_called_once_with("BTCUSDT", "333")

    def test_keeps_fresh_limit_entry_without_position(self):
        """포지션 없어도 LIMIT 주문이 타임아웃 이내면 유지한다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 444, "LIMIT", age_seconds=100),  # 100s < 300s
        ]

        om.audit_stale_orders(active_symbols=set())

        mock_client.cancel_order.assert_not_called()

    def test_keeps_limit_with_active_position(self):
        """포지션 있는 심볼의 LIMIT 주문은 시간 초과여도 유지한다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 555, "LIMIT", age_seconds=600),
        ]

        om.audit_stale_orders(active_symbols={"BTCUSDT"})

        mock_client.cancel_order.assert_not_called()

    def test_cancel_error_does_not_abort_audit(self):
        """한 주문 취소 오류가 나머지 주문 감사를 막지 않는다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 100, "STOP_MARKET", age_seconds=10),
            self._make_order_entry("ETHUSDT", 200, "STOP_MARKET", age_seconds=10),
        ]
        mock_client.cancel_order.side_effect = [
            Exception("First cancel failed"),
            {},  # Second succeeds
        ]

        # 예외가 전파되어서는 안 됨
        try:
            om.audit_stale_orders(active_symbols=set())
        except Exception:
            pytest.fail("audit_stale_orders raised an exception on cancel failure")

        # 두 번 호출 시도됨 (첫 번째 실패해도 두 번째 시도)
        assert mock_client.cancel_order.call_count == 2

    def test_get_open_orders_error_handled_gracefully(self):
        """get_open_orders API 오류 시 예외 없이 처리된다."""
        om, mock_client, _ = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.side_effect = Exception("API error")

        try:
            om.audit_stale_orders(active_symbols=set())
        except Exception:
            pytest.fail("audit_stale_orders raised on get_open_orders failure")

        mock_client.cancel_order.assert_not_called()

    def test_slack_alert_sent_when_orders_cancelled(self):
        """미체결 주문이 취소되면 Slack 알림이 발송된다."""
        om, mock_client, mock_slack = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 999, "STOP_MARKET", age_seconds=10),
        ]
        mock_client.cancel_order.return_value = {}

        om.audit_stale_orders(active_symbols=set())

        mock_slack.send_risk_alert.assert_called_once()

    def test_no_slack_alert_when_nothing_cancelled(self):
        """취소된 주문이 없으면 Slack 알림을 발송하지 않는다."""
        om, mock_client, mock_slack = make_order_manager(mode="paper", stale_timeout=300)
        mock_client.get_open_orders.return_value = [
            self._make_order_entry("BTCUSDT", 888, "STOP_MARKET", age_seconds=10),
        ]
        # BTCUSDT에 활성 포지션 → 취소 안 함

        om.audit_stale_orders(active_symbols={"BTCUSDT"})

        mock_slack.send_risk_alert.assert_not_called()


# ── 3. _set_sl_tp SL/TP 주문 ID 추적 ─────────────────────────────────────────

class TestSetSlTpTracking:

    def test_dry_run_skips_sl_tp(self):
        """dry-run 모드에서는 SL/TP 주문을 설정하지 않는다."""
        om, mock_client, _ = make_order_manager(mode="dry-run")
        managed = ManagedOrder(
            client_id="bot-test",
            symbol="BTCUSDT",
            side="BUY",
            qty="0.01",
            position_side="LONG",
            status=OrderStatus.FILLED,
        )
        rp = make_risk_params()
        om._set_sl_tp(managed, rp, SYMBOL_INFO)
        mock_client.set_trading_stop.assert_not_called()
        assert managed.sl_order_id is None
        assert managed.tp_order_id is None

    def test_sl_tp_order_ids_stored(self):
        """SL/TP 주문 설정 성공 시 order ID가 managed에 저장된다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.set_trading_stop.return_value = {
            "sl": {"orderId": 11111},
            "tp": {"orderId": 22222},
        }
        managed = ManagedOrder(
            client_id="bot-test",
            symbol="BTCUSDT",
            side="BUY",
            qty="0.01",
            position_side="LONG",
            status=OrderStatus.FILLED,
        )
        rp = make_risk_params()
        om._set_sl_tp(managed, rp, SYMBOL_INFO)

        assert managed.sl_order_id == "11111"
        assert managed.tp_order_id == "22222"

    def test_sl_tp_partial_result_handled(self):
        """SL만 설정되고 TP가 없어도 예외 없이 처리된다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.set_trading_stop.return_value = {
            "sl": {"orderId": 33333},
            # tp 없음
        }
        managed = ManagedOrder(
            client_id="bot-test",
            symbol="BTCUSDT",
            side="BUY",
            qty="0.01",
            position_side="LONG",
            status=OrderStatus.FILLED,
        )
        rp = make_risk_params()
        om._set_sl_tp(managed, rp, SYMBOL_INFO)

        assert managed.sl_order_id == "33333"
        assert managed.tp_order_id is None

    def test_invalid_stop_price_skips_sl_tp(self):
        """stop_price <= 0 이면 SL/TP 설정을 건너뛴다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        managed = ManagedOrder(
            client_id="bot-test",
            symbol="BTCUSDT",
            side="BUY",
            qty="0.01",
            position_side="LONG",
            status=OrderStatus.FILLED,
        )
        rp = make_risk_params(stop_price=0.0)  # invalid
        om._set_sl_tp(managed, rp, SYMBOL_INFO)
        mock_client.set_trading_stop.assert_not_called()


# ── 4. 전체 진입 → SL/TP → 청산 → 주문 정리 통합 흐름 ────────────────────────

class TestFullOrderLifecycle:

    def test_entry_order_stores_sl_tp_ids(self):
        """진입 성공 후 sl_order_id / tp_order_id가 ManagedOrder에 기록된다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.set_position_mode.return_value = None
        mock_client.set_leverage.return_value = None
        mock_client.round_qty.return_value = "0.010"
        mock_client.round_price.return_value = "49999.9"
        mock_client.place_order.return_value = {
            "orderId": "77777",
            "avgPrice": "50000",
            "executedQty": "0.01",
        }
        mock_client.get_order.return_value = {
            "status": "FILLED",
            "avgPrice": "50000",
            "executedQty": "0.01",
        }
        mock_client.set_trading_stop.return_value = {
            "sl": {"orderId": 88888},
            "tp": {"orderId": 99999},
        }

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
        assert order.sl_order_id == "88888"
        assert order.tp_order_id == "99999"

    def test_cancel_stale_called_after_position_close(self):
        """포지션 청산 후 cancel_stale_orders가 올바른 심볼로 호출된다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.cancel_all_open_orders.return_value = {}
        mock_client.round_qty.return_value = "0.010"
        mock_client.round_price.return_value = "50001.0"
        mock_client.place_order.return_value = {
            "orderId": "66666",
            "avgPrice": "50001",
            "executedQty": "0.01",
        }
        mock_client.get_order.return_value = {
            "status": "FILLED",
            "avgPrice": "50001",
            "executedQty": "0.01",
        }

        om.close_position(
            symbol="BTCUSDT",
            direction=1,
            qty=0.01,
            current_price=50001.0,
            symbol_info=SYMBOL_INFO,
            reason="TAKE_PROFIT",
        )
        # 청산 후 cancel_stale_orders 호출
        om.cancel_stale_orders("BTCUSDT")

        mock_client.cancel_all_open_orders.assert_called_once_with("BTCUSDT")

    def test_idempotent_cancel_multiple_calls(self):
        """cancel_stale_orders를 여러 번 호출해도 안전하다."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.cancel_all_open_orders.return_value = {}

        om.cancel_stale_orders("BTCUSDT")
        om.cancel_stale_orders("BTCUSDT")  # 두 번째 호출

        # 두 번 모두 호출됨 (멱등성 — 각각 독립적)
        assert mock_client.cancel_all_open_orders.call_count == 2

    def test_multiple_symbols_independent_cancellation(self):
        """여러 심볼에 대해 각각 독립적으로 cancel_stale_orders 동작."""
        om, mock_client, _ = make_order_manager(mode="paper")
        mock_client.cancel_all_open_orders.return_value = {}

        # BTC, ETH 각각의 주문 추가
        om._orders["bot-btc1"] = ManagedOrder(client_id="bot-btc1", symbol="BTCUSDT", side="BUY")
        om._orders["bot-eth1"] = ManagedOrder(client_id="bot-eth1", symbol="ETHUSDT", side="BUY")

        om.cancel_stale_orders("BTCUSDT")

        assert "bot-btc1" not in om._orders
        assert "bot-eth1" in om._orders  # ETH 주문은 유지
        mock_client.cancel_all_open_orders.assert_called_once_with("BTCUSDT")


# ── 5. 설정 값 검증 ────────────────────────────────────────────────────────────

class TestStaleOrderConfig:

    def test_default_stale_timeout(self):
        """STALE_ORDER_TIMEOUT_SECONDS 기본값 확인."""
        from src.config import Settings
        s = Settings(TRADING_MODE="dry-run", BINANCE_API_KEY="", BINANCE_API_SECRET="")
        assert s.STALE_ORDER_TIMEOUT_SECONDS == 300

    def test_default_audit_interval(self):
        """STALE_ORDER_AUDIT_INTERVAL_MINUTES 기본값 확인."""
        from src.config import Settings
        s = Settings(TRADING_MODE="dry-run", BINANCE_API_KEY="", BINANCE_API_SECRET="")
        assert s.STALE_ORDER_AUDIT_INTERVAL_MINUTES == 5

    def test_custom_stale_timeout(self):
        """STALE_ORDER_TIMEOUT_SECONDS 커스텀 설정."""
        s = make_settings(stale_timeout=120)
        assert s.STALE_ORDER_TIMEOUT_SECONDS == 120
