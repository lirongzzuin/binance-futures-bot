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

        # ── Choose order type ──────────────────────────────────────────────────
        use_market = (
            spread_pct > self._s.SPREAD_TAKER_THRESHOLD or
            signal_strength > 0.90
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

    # ── Internal placers ──────────────────────────────────────────────────────

    def _place_market(
        self, managed: ManagedOrder, current_price: float, symbol_info: dict,
        is_close: bool = False,
    ) -> bool:
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
          TP → TAKE_PROFIT (Limit at TP2): Maker fee + 봇 TP1 로직과 충돌 방지
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
            self._client.set_trading_stop(
                symbol=managed.symbol,
                stop_loss=sl_price,
                take_profit=tp_price,
                position_side=managed.position_side,
                sl_order_type="STOP_MARKET",
                tp_order_type="TAKE_PROFIT",
                tp_limit_price=tp_price,
            )
            log.info(
                "Exchange SL/TP set | %s positionSide=%s sl=%s(STOP_MARKET) tp=%s(TAKE_PROFIT@TP2)",
                managed.symbol, managed.position_side, sl_price, tp_price,
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
