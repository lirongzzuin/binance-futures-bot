"""
Order Manager — idempotent order lifecycle management for Binance Futures.

Order type selection:
  - Default: Limit (GTX/PostOnly) for maker rebate
  - Switch to Market if:
    * Spread > threshold
    * Previous limit attempt timed out
    * Signal urgency is very high

Idempotency:
  - Each trade intent generates a deterministic client_order_id
  - Duplicate detection prevents double-sends on restart

Stale Order Management:
  - cancel_stale_orders(symbol): 포지션 청산 후 잔여 SL/TP 주문 취소
  - audit_stale_orders(active_symbols): 주기적으로 미체결 주문 스캔 및 정리

Binance Futures specifics:
  - Hedge mode: positionSide = "LONG" | "SHORT"
  - PostOnly: timeInForce = "GTX" (Good Till Crossing)
  - SL: type="STOP_MARKET", closePosition="true"
  - TP: type="TAKE_PROFIT" (limit), closePosition="true"
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set

from src.api.binance_client import BinanceClient, BinanceClientError, CircuitOpenError
from src.api.slack import SlackNotifier
from src.config import Settings
from src.logger import get_logger
from src.risk.engine import RiskParameters

log = get_logger(__name__)


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class ManagedOrder:
    client_id: str
    exchange_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    qty: str = ""
    price: Optional[str] = None
    order_type: str = "LIMIT"
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_price: float = 0.0
    attempts: int = 0
    created_ts: float = field(default_factory=time.time)
    position_side: str = "BOTH"   # "LONG" | "SHORT" | "BOTH"
    sl_order_id: Optional[str] = None  # 거래소 SL 주문 ID (취소 추적용)
    tp_order_id: Optional[str] = None  # 거래소 TP 주문 ID (취소 추적용)


class OrderManager:
    def __init__(
        self,
        client: BinanceClient,
        settings: Settings,
        slack: SlackNotifier,
    ):
        self._client = client
        self._s = settings
        self._slack = slack
        self._orders: Dict[str, ManagedOrder] = {}
        self._sent_ids: Set[str] = set()

    # ── Entry Order ───────────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        direction: int,         # +1 long, -1 short
        risk_params: RiskParameters,
        spread_pct: float,
        signal_strength: float,
        current_price: float,
        symbol_info: dict,
    ) -> Optional[ManagedOrder]:
        """Place an entry order. Returns ManagedOrder on success, None on failure."""
        # Binance: BUY to open long, SELL to open short
        side = "BUY" if direction == 1 else "SELL"
        qty_str = self._client.round_qty(risk_params.qty_base, symbol_info)
        client_id = self._make_client_id(symbol, side, "entry")

        if client_id in self._sent_ids:
            log.warning("Duplicate entry order prevented for %s %s", symbol, side)
            return self._orders.get(client_id)

        # ── Set position mode + leverage ──────────────────────────────────────
        if not self._s.is_dry_run():
            try:
                self._client.set_position_mode(
                    symbol=symbol,
                    hedge_mode=self._s.BINANCE_HEDGE_MODE,
                )
                self._client.set_leverage(symbol, risk_params.leverage)
            except Exception as e:
                log.error("Failed to init position mode/leverage for %s: %s", symbol, e)
                return None

        # ── positionSide for hedge mode ────────────────────────────────────────
        # Binance hedge mode: "LONG" for long position, "SHORT" for short
        if self._s.BINANCE_HEDGE_MODE:
            position_side = "LONG" if direction == 1 else "SHORT"
        else:
            position_side = "BOTH"

        # ── 진입 전 미체결 주문 정리 (주문 누적 방지) ─────────────────────────
        try:
            self._client.cancel_all_orders(symbol, side=side)
        except Exception as e:
            log.warning("pre-entry cancel_all_orders failed for %s %s: %s", symbol, side, e)

        # ── Choose order type ──────────────────────────────────────────────────
        use_market = (
            spread_pct > self._s.SPREAD_TAKER_THRESHOLD or
            signal_strength > self._s.SIGNAL_URGENT_THRESHOLD
        )

        managed = ManagedOrder(
            client_id=client_id,
            symbol=symbol,
            side=side,
            qty=qty_str,
            order_type="MARKET" if use_market else "LIMIT",
            position_side=position_side,
        )

        if use_market:
            success = self._place_market(managed, current_price, symbol_info)
        else:
            success = self._place_limit_with_retry(
                managed, current_price, symbol_info, direction
            )

        if success:
            self._set_sl_tp(managed, risk_params, symbol_info)
            self._sent_ids.add(client_id)
            self._orders[client_id] = managed
            return managed

        log.warning("Entry order failed for %s %s — all attempts exhausted", symbol, side)
        return None

    # ── Exit Order ────────────────────────────────────────────────────────────

    def close_position(
        self,
        symbol: str,
        direction: int,     # existing position direction
        qty: float,
        current_price: float,
        symbol_info: dict,
        reason: str = "",
    ) -> Optional[ManagedOrder]:
        """Close an existing position."""
        # To close a position, use opposite side
        # But in hedge mode, we identify by positionSide
        # To close long: SELL + positionSide=LONG
        # To close short: BUY + positionSide=SHORT
        side = "SELL" if direction == 1 else "BUY"
        qty_str = self._client.round_qty(qty, symbol_info)
        client_id = self._make_client_id(symbol, side, "exit")

        if client_id in self._sent_ids:
            log.warning("Duplicate exit order prevented for %s", symbol)
            return self._orders.get(client_id)

        use_market = (
            "STOP" in reason.upper() or
            "GUARD" in reason.upper() or
            "LIQUIDATION" in reason.upper()
        )

        if self._s.BINANCE_HEDGE_MODE:
            position_side = "LONG" if direction == 1 else "SHORT"
        else:
            position_side = "BOTH"

        # ── 청산 전 미체결 주문 정리 (주문 누적 방지) ──────────────────────────
        try:
            self._client.cancel_all_orders(symbol, side=side)
        except Exception as e:
            log.warning("pre-exit cancel_all_orders failed for %s %s: %s", symbol, side, e)

        managed = ManagedOrder(
            client_id=client_id,
            symbol=symbol,
            side=side,
            qty=qty_str,
            order_type="MARKET" if use_market else "LIMIT",
            position_side=position_side,
        )

        if use_market:
            success = self._place_market(managed, current_price, symbol_info, is_close=True)
        else:
            success = self._place_limit_with_retry(
                managed, current_price, symbol_info, -direction, is_close=True
            )

        if success:
            self._sent_ids.add(client_id)
            self._orders[client_id] = managed
            return managed

        log.warning("Close order failed for %s %s reason=%s", symbol, side, reason)
        return None

    # ── Stale Order Management ────────────────────────────────────────────────

    def cancel_stale_orders(self, symbol: str) -> None:
        """
        포지션 청산 후 거래소에 남아있는 SL/TP 및 미체결 진입 주문을 모두 취소.

        호출 시점:
          - _execute_exit (전체 청산 후)
          - _emergency_close_all (긴급 청산 후)

        주의: dry-run 모드에서는 실제 API 호출 없음.
        """
        if self._s.is_dry_run():
            return

        try:
            result = self._client.cancel_all_open_orders(symbol)
            log.info(
                "Stale order cleanup: cancelled all open orders for %s | result=%s",
                symbol, result,
            )
        except Exception as e:
            log.warning("cancel_stale_orders: failed for %s: %s", symbol, e)

        # 내부 추적 상태 정리 (해당 심볼의 주문 ID 제거)
        stale_ids = [cid for cid, o in self._orders.items() if o.symbol == symbol]
        for cid in stale_ids:
            self._sent_ids.discard(cid)
            del self._orders[cid]

        if stale_ids:
            log.debug("Removed %d tracked order(s) for %s from internal state", len(stale_ids), symbol)

    def audit_stale_orders(self, active_symbols: set) -> None:
        """
        주기적 미체결 주문 감사 (스케줄러에서 주기적으로 호출).

        로직:
          1. 거래소에서 전체 미체결 주문 목록 조회
          2. 활성 포지션이 없는 심볼의 SL/TP 주문 → 즉시 취소 (잔여 SL/TP)
          3. 활성 포지션이 없는 심볼의 LIMIT 주문 중 STALE_ORDER_TIMEOUT_SECONDS 초과 → 취소
          (포지션이 있는 심볼의 주문은 보호를 위해 유지)

        active_symbols: 현재 오픈 포지션이 있는 심볼 집합 (e.g. {"BTCUSDT", "ETHUSDT"})
        """
        if self._s.is_dry_run():
            return

        try:
            all_open = self._client.get_open_orders()
        except Exception as e:
            log.warning("audit_stale_orders: failed to fetch open orders: %s", e)
            return

        if not all_open:
            return

        now_ms = time.time() * 1000
        cancelled_count = 0
        for order in all_open:
            symbol = order.get("symbol", "")
            order_id = str(order.get("orderId", ""))
            order_type = order.get("type", "")
            created_ms = int(order.get("time", 0))
            age_s = (now_ms - created_ms) / 1000 if created_ms else 0

            # SL/TP 타입 주문: 포지션이 없으면 즉시 취소
            is_sl_tp_type = order_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT")
            no_position = symbol not in active_symbols

            # 취소 조건:
            # (a) SL/TP 주문이고 포지션이 없는 경우 → 잔여 SL/TP
            # (b) 지정가 진입 주문이 STALE_ORDER_TIMEOUT_SECONDS 초과 → 포지션 무관
            should_cancel = (
                (is_sl_tp_type and no_position) or
                (age_s > self._s.STALE_ORDER_TIMEOUT_SECONDS and
                 order_type in ("LIMIT",) and no_position)
            )

            if not should_cancel:
                if age_s > self._s.STALE_ORDER_TIMEOUT_SECONDS:
                    log.debug(
                        "audit: order %s (%s) for %s is %.0fs old — still has active position, keeping",
                        order_id, order_type, symbol, age_s,
                    )
                continue

            try:
                self._client.cancel_order(symbol, order_id)
                cancelled_count += 1
                log.info(
                    "audit: cancelled stale %s order %s for %s (age=%.0fs, active_pos=%s)",
                    order_type, order_id, symbol, age_s, symbol in active_symbols,
                )
            except Exception as e:
                log.warning(
                    "audit: failed to cancel order %s for %s: %s", order_id, symbol, e
                )

        if cancelled_count > 0:
            log.info("audit_stale_orders: cancelled %d stale order(s)", cancelled_count)
            self._slack.send_risk_alert(
                f"⚠️ 미체결 주문 정리: {cancelled_count}개 취소 완료"
            )

    # ── Internal placers ──────────────────────────────────────────────────────

    def _place_market(
        self, managed: ManagedOrder, current_price: float, symbol_info: dict,
        is_close: bool = False,
    ) -> bool:
        # 시장가로 실제 체결됨을 기록 (GTX 거부 후 폴백 포함) — 수수료 계산에 사용
        managed.order_type = "MARKET"
        if self._s.is_dry_run():
            managed.status = OrderStatus.FILLED
            managed.avg_price = current_price
            managed.filled_qty = float(managed.qty)
            managed.price = str(current_price)
            log.info("[DRY-RUN] Market %s %s qty=%s @ %.4f",
                     managed.side, managed.symbol, managed.qty, current_price)
            return True

        try:
            params = dict(
                symbol=managed.symbol,
                side=managed.side,
                qty=managed.qty,
                order_type="MARKET",
                client_order_id=managed.client_id,
                position_side=managed.position_side,
            )
            result = self._client.place_order(**params)
            managed.exchange_id = str(result.get("orderId", ""))
            managed.status = OrderStatus.FILLED
            managed.avg_price = float(result.get("avgPrice", 0) or current_price)
            managed.price = str(managed.avg_price or current_price)
            return True
        except BinanceClientError as e:
            # -2022: ReduceOnly order rejected — position already zero
            if e.code == -2022 and is_close:
                log.warning("-2022: ReduceOnly rejected for %s — position already closed", managed.symbol)
                managed.status = OrderStatus.FILLED
                managed.avg_price = current_price
                managed.price = str(current_price)
                return True
            log.error("Market order failed: %s", e)
            managed.status = OrderStatus.FAILED
            return False
        except CircuitOpenError:
            log.error("Market order skipped (circuit open): %s %s", managed.symbol, managed.side)
            raise
        except Exception as e:
            log.error("Market order unexpected error: %s", e)
            managed.status = OrderStatus.FAILED
            return False

    def _place_limit_with_retry(
        self,
        managed: ManagedOrder,
        current_price: float,
        symbol_info: dict,
        direction: int,
        is_close: bool = False,
    ) -> bool:
        timeout = self._s.LIMIT_ORDER_TIMEOUT_SECONDS
        max_retries = self._s.LIMIT_ORDER_MAX_RETRIES

        for attempt in range(1, max_retries + 1):
            managed.attempts = attempt
            offset = symbol_info.get("price_step", 0.01)
            if direction == 1:
                limit_price = current_price - offset
            else:
                limit_price = current_price + offset

            price_str = self._client.round_price(limit_price, symbol_info)
            managed.price = price_str

            if self._s.is_dry_run():
                managed.status = OrderStatus.FILLED
                managed.avg_price = float(price_str)
                managed.filled_qty = float(managed.qty)
                log.info("[DRY-RUN] Limit %s %s qty=%s @ %s",
                         managed.side, managed.symbol, managed.qty, price_str)
                return True

            try:
                result = self._client.place_order(
                    symbol=managed.symbol,
                    side=managed.side,
                    qty=managed.qty,
                    order_type="LIMIT",
                    price=price_str,
                    time_in_force="GTX",    # PostOnly — Maker fee
                    client_order_id=managed.client_id,
                    position_side=managed.position_side,
                )
                managed.exchange_id = str(result.get("orderId", ""))
                managed.status = OrderStatus.PENDING
            except BinanceClientError as e:
                # -5022: GTX order would be immediately taker — fall back to market
                if e.code == -5022:
                    log.info(
                        "GTX (PostOnly) rejected for %s attempt %d — falling back to market",
                        managed.symbol, attempt,
                    )
                    return self._place_market(managed, current_price, symbol_info, is_close=is_close)
                # -2022: ReduceOnly rejected — position already closed
                elif e.code == -2022 and is_close:
                    log.warning("-2022: ReduceOnly rejected for %s — position already closed", managed.symbol)
                    managed.status = OrderStatus.FILLED
                    managed.avg_price = current_price
                    managed.price = str(current_price)
                    return True
                else:
                    log.warning("Limit order attempt %d failed: %s", attempt, e)
                    if attempt == max_retries:
                        log.info("Falling back to market order for %s", managed.symbol)
                        return self._place_market(managed, current_price, symbol_info, is_close=is_close)
                    continue

            fill_result = self._wait_for_fill(managed, timeout)
            if fill_result:
                return True

            # Not filled — cancel and re-quote
            if managed.exchange_id:
                try:
                    self._client.cancel_order(managed.symbol, managed.exchange_id)
                    log.info(
                        "Limit order timed out, re-quoting %s (attempt %d/%d)",
                        managed.symbol, attempt, max_retries,
                    )
                except Exception:
                    pass

            if attempt == max_retries:
                log.info(
                    "All %d limit attempts exhausted for %s — falling back to market",
                    max_retries, managed.symbol,
                )
                return self._place_market(managed, current_price, symbol_info, is_close=is_close)

        return False

    def _wait_for_fill(self, managed: ManagedOrder, timeout: float) -> bool:
        if self._s.is_dry_run():
            return True

        start = time.time()
        while time.time() - start < timeout:
            try:
                order = self._client.get_order(managed.symbol, managed.exchange_id)
                if order:
                    status = order.get("status", "")
                    if status == "FILLED":
                        managed.status = OrderStatus.FILLED
                        managed.avg_price = float(order.get("avgPrice", 0))
                        managed.filled_qty = float(order.get("executedQty", 0))
                        return True
                    elif status == "PARTIALLY_FILLED":
                        managed.status = OrderStatus.PARTIAL
                        managed.filled_qty = float(order.get("executedQty", 0))
                    elif status in ("CANCELED", "REJECTED", "EXPIRED"):
                        managed.status = OrderStatus.CANCELLED
                        return False
            except Exception as e:
                log.warning("Order status check failed: %s", e)
            time.sleep(2)

        return False

    def _set_sl_tp(
        self,
        managed: ManagedOrder,
        risk_params: RiskParameters,
        symbol_info: dict,
    ) -> None:
        """
        Attach exchange-side SL/TP orders to the filled position.
        Binance uses separate STOP_MARKET and TAKE_PROFIT orders.

        Order type policy:
          SL → STOP_MARKET: 즉각 실행 (Taker fee)
          TP → TAKE_PROFIT_MARKET: closePosition=true 지원, 봇 TP1 로직과 충돌 방지

        체결된 SL/TP 주문 ID를 managed.sl_order_id / managed.tp_order_id에 저장 (로깅/추적용).
        포지션 청산 시 cancel_stale_orders()가 cancel_all_open_orders()로 일괄 정리함.
        """
        if self._s.is_dry_run():
            return

        if risk_params.stop_price <= 0:
            log.warning("_set_sl_tp: invalid stop_price for %s — skipping", managed.symbol)
            return

        sl_price = self._client.round_price(risk_params.stop_price, symbol_info)
        tp2 = risk_params.tp2_price if risk_params.tp2_price > 0 else risk_params.take_profit_price
        tp_price = self._client.round_price(tp2, symbol_info) if tp2 > 0 else None

        try:
            result = self._client.set_trading_stop(
                symbol=managed.symbol,
                stop_loss=sl_price,
                take_profit=tp_price,
                position_side=managed.position_side,
                sl_order_type="STOP_MARKET",
                tp_order_type="TAKE_PROFIT_MARKET",
                tp_limit_price=tp_price,
            )
            # SL/TP 주문 ID 추적 (취소 시 사용)
            sl_result = result.get("sl", {})
            tp_result = result.get("tp", {})
            if sl_result:
                managed.sl_order_id = str(sl_result.get("orderId", ""))
            if tp_result:
                managed.tp_order_id = str(tp_result.get("orderId", ""))
            log.info(
                "Exchange SL/TP set | %s positionSide=%s sl=%s(STOP_MARKET) tp=%s(TAKE_PROFIT@TP2) "
                "sl_id=%s tp_id=%s",
                managed.symbol, managed.position_side, sl_price, tp_price,
                managed.sl_order_id, managed.tp_order_id,
            )
        except Exception as e:
            log.error("Failed to set exchange SL/TP for %s: %s", managed.symbol, e)

    # ── Client Order ID ───────────────────────────────────────────────────────

    def _make_client_id(self, symbol: str, side: str, tag: str) -> str:
        """Deterministic client order ID for idempotency within the same minute."""
        minute = int(time.time()) // 60
        raw = f"{symbol}-{side}-{tag}-{minute}"
        h = hashlib.md5(raw.encode()).hexdigest()[:16]
        return f"bot-{h}"

    def get_order(self, client_id: str) -> Optional[ManagedOrder]:
        return self._orders.get(client_id)
