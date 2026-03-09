"""
Binance Futures API client.
Wraps python-binance with:
 - Automatic rate-limit retry (exponential back-off, max 10 retries)
 - Circuit breaker (fail-fast after 5 consecutive errors, 60 s cool-down)
 - All required market-data + trading endpoints for USDT-margined perpetuals
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)

# ── Exceptions ────────────────────────────────────────────────────────────────

class BinanceClientError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(f"[{code}] {msg}")
        self.code = code


# Error codes that are benign — should NOT trip the circuit breaker
_BENIGN_ERROR_CODES = frozenset({
    -1135,  # Invalid JSON parameters (symbol format mismatch — not a real API failure)
    -2011,  # Unknown order sent (cancel on already-filled or non-existent order)
    -2013,  # No such order (cancel on already-closed order)
    -4046,  # No need to change position mode (already in hedge mode)
    -4061,  # Order's position side does not match user's setting (edge case on mode switch)
    -1021,  # Timestamp for this request is outside of the recvWindow (transient)
})


class CircuitOpenError(Exception):
    """Raised when circuit breaker is OPEN."""


# ── Circuit Breaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at > self._recovery_timeout:
                self._opened_at = None
                self._failures = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._opened_at = time.time()
                log.error("Circuit breaker OPENED after %d failures", self._failures)


# ── Binance Client ────────────────────────────────────────────────────────────

class BinanceClient:
    """Thin wrapper around python-binance with retry and circuit breaker."""

    # Binance Futures interval mapping (config stores as string "5", "60")
    _INTERVAL_MAP = {
        "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
        "60": "1h", "120": "2h", "240": "4h", "360": "6h",
        "480": "8h", "720": "12h", "1440": "1d",
    }

    def __init__(self, settings: Settings):
        self._settings = settings
        self._circuit = CircuitBreaker()

        if settings.BINANCE_TESTNET:
            self._client = Client(
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET,
                testnet=True,
            )
        else:
            self._client = Client(
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET,
            )

        # Cache for exchange info (symbol → constraints)
        self._symbol_info_cache: Dict[str, Dict] = {}

        log.info(
            "BinanceClient initialised | testnet=%s mode=%s",
            settings.BINANCE_TESTNET,
            settings.TRADING_MODE,
        )

    def _interval(self, tf: str) -> str:
        """Convert config timeframe string to Binance interval string."""
        return self._INTERVAL_MAP.get(str(tf), f"{tf}m")

    # ── Internal call wrapper ─────────────────────────────────────────────

    def _call(self, fn_name: str, **kwargs: Any) -> Any:
        if self._circuit.is_open:
            raise CircuitOpenError("Circuit breaker is OPEN — skipping API call")

        fn = getattr(self._client, fn_name)

        for attempt in range(1, 11):
            try:
                result = fn(**kwargs)
                self._circuit.record_success()
                return result
            except BinanceAPIException as e:
                code = e.code
                # Rate limit → back off
                if code == -1003 or e.status_code == 429:
                    wait = min(2 ** attempt, 60)
                    log.warning("Rate limit hit, waiting %ss (attempt %d)", wait, attempt)
                    time.sleep(wait)
                    continue
                if code not in _BENIGN_ERROR_CODES:
                    self._circuit.record_failure()
                raise BinanceClientError(code, e.message)
            except BinanceRequestException as e:
                self._circuit.record_failure()
                wait = min(2 ** attempt, 60)
                log.warning("Request error (attempt %d/10): %s — retry in %ss", attempt, e, wait)
                time.sleep(wait)
            except CircuitOpenError:
                raise
            except Exception as e:
                self._circuit.record_failure()
                wait = min(2 ** attempt, 60)
                log.warning("API error (attempt %d/10): %s — retry in %ss", attempt, e, wait)
                time.sleep(wait)

        raise BinanceClientError(-1, "Max retries exceeded")

    # ── Market Data ──────────────────────────────────────────────────────────

    def get_tickers(self) -> List[Dict]:
        """All USDT-margined perpetual ticker stats (24h)."""
        result = self._call("futures_ticker")
        # Filter to USDT pairs only
        return [t for t in result if str(t.get("symbol", "")).endswith("USDT")]

    def get_kline(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[List]:
        """
        OHLCV candles from Binance Futures.
        Returns raw list: [open_time, open, high, low, close, volume, close_time,
                           quote_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
        interval: Binance format string e.g. "5m", "1h"
        """
        params: Dict[str, Any] = dict(
            symbol=symbol, interval=interval, limit=limit
        )
        if start is not None:
            params["startTime"] = start
        if end is not None:
            params["endTime"] = end
        return self._call("futures_klines", **params)

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Order book snapshot. Returns dict with 'bids' and 'asks' lists.
        Valid Binance depth limits: 5, 10, 20, 50, 100, 500, 1000.
        """
        return self._call("futures_order_book", symbol=symbol, limit=limit)

    def get_funding_rate_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Historical funding rates. Returns list with 'fundingRate', 'fundingTime'.
        Testnet does not support this endpoint — returns [] on failure without
        tripping the circuit breaker (non-critical optional data).
        """
        try:
            fn = getattr(self._client, "futures_funding_rate")
            result = fn(symbol=symbol, limit=limit)
            self._circuit.record_success()
            return result
        except Exception:
            return []

    def get_open_interest(self, symbol: str, period: str = "5m", limit: int = 10) -> List[Dict]:
        """Open interest history. Returns list with 'sumOpenInterest', 'timestamp'.
        Testnet does not support openInterestHist — returns [] without tripping
        the circuit breaker (non-critical optional data).
        """
        try:
            fn = getattr(self._client, "futures_open_interest_hist")
            result = fn(symbol=symbol, period=period, limit=limit)
            self._circuit.record_success()
            return result
        except Exception:
            # Fallback: current OI snapshot (supported on testnet)
            try:
                fn2 = getattr(self._client, "futures_open_interest")
                current = fn2(symbol=symbol)
                return [current] if current else []
            except Exception:
                return []

    def get_exchange_info(self) -> Dict:
        """Full exchange info including symbol filters and precision."""
        return self._call("futures_exchange_info")

    def get_instruments_info(self, symbol: str) -> Dict:
        """Symbol metadata. Cached after first call."""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        info = self.get_exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                self._symbol_info_cache[symbol] = s
                return s
        return {}

    # ── Account & Position ───────────────────────────────────────────────────

    def get_usdt_equity(self) -> float:
        """Total USDT balance in futures account."""
        balances = self._call("futures_account_balance")
        for b in balances:
            if b.get("asset") == "USDT":
                return float(b.get("balance", 0))
        return 0.0

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        All open positions (or for a specific symbol).
        In hedge mode returns LONG and SHORT sides separately.
        Normalises to Bybit-compatible dict format for consistency with state machine.
        """
        if symbol:
            raw = self._call("futures_position_information", symbol=symbol)
        else:
            raw = self._call("futures_position_information")

        result = []
        for p in raw:
            amt = float(p.get("positionAmt", 0))
            if amt == 0:
                continue
            position_side = p.get("positionSide", "BOTH")
            # Normalise: LONG → side="Buy", SHORT → side="Sell"
            if position_side == "LONG":
                side = "Buy"
            elif position_side == "SHORT":
                side = "Sell"
            else:
                side = "Buy" if amt > 0 else "Sell"
            result.append({
                "symbol": p.get("symbol"),
                "side": side,
                "size": str(abs(amt)),
                "avgPrice": p.get("entryPrice", "0"),
                "markPrice": p.get("markPrice", "0"),
                "unrealisedPnl": p.get("unRealizedProfit", "0"),
                "leverage": p.get("leverage", "1"),
                "stopLoss": "0",
                "takeProfit": "0",
                "positionSide": position_side,
            })
        return result

    def get_position(self, symbol: str, side: Optional[str] = None) -> Optional[Dict]:
        """Single symbol position, or None if flat."""
        positions = self.get_positions(symbol)
        for p in positions:
            if float(p.get("size", 0)) <= 0:
                continue
            if side is None:
                return p
            if p.get("side", "") == side:
                return p
        return None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        try:
            self._call(
                "futures_change_leverage",
                symbol=symbol,
                leverage=leverage,
            )
            log.info("Leverage set to %s× for %s", leverage, symbol)
        except BinanceClientError as e:
            if e.code == -4028:
                log.debug("Leverage already set to %s for %s", leverage, symbol)
            else:
                raise

    def set_position_mode(self, symbol: str, hedge_mode: bool = True) -> None:
        """
        Set hedge mode (dualSidePosition) for the account.
        Note: Binance applies this at account level, not per symbol.
        """
        try:
            self._call(
                "futures_change_position_mode",
                dualSidePosition="true" if hedge_mode else "false",
            )
            mode_name = "Hedge" if hedge_mode else "One-Way"
            log.info("Position mode set to %s", mode_name)
        except BinanceClientError as e:
            if e.code in (-4046, -4059):
                log.debug("Position mode already set correctly")
            else:
                raise

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> None:
        """Set margin type (ISOLATED or CROSSED) for a symbol."""
        try:
            self._call(
                "futures_change_margin_type",
                symbol=symbol,
                marginType=margin_type,
            )
        except BinanceClientError as e:
            if e.code == -4046 or "No need to change" in str(e):
                log.debug("Margin type already set to %s for %s", margin_type, symbol)
            else:
                log.warning("Failed to set margin type for %s: %s", symbol, e)

    # ── Order Management ─────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,              # "BUY" | "SELL"
        qty: str,
        order_type: str = "LIMIT",
        price: Optional[str] = None,
        time_in_force: str = "GTX",  # GTX = PostOnly (Maker only)
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        position_side: str = "BOTH",  # "LONG" | "SHORT" | "BOTH"
    ) -> Dict:
        """
        Place a USDT-margined perpetual order.

        position_side:
          "BOTH"  = One-Way mode (default)
          "LONG"  = Hedge mode, Long position
          "SHORT" = Hedge mode, Short position

        time_in_force:
          "GTX" = PostOnly (Maker rebate) — rejected if would be taker
          "GTC" = Good Till Cancelled
          "IOC" = Immediate or Cancel
        """
        params: Dict[str, Any] = dict(
            symbol=symbol,
            side=side.upper(),
            type=order_type.upper(),
            quantity=qty,
            positionSide=position_side,
        )
        if client_order_id:
            params["newClientOrderId"] = client_order_id[:36]  # Binance max 36 chars

        if order_type.upper() == "LIMIT":
            params["price"] = price
            params["timeInForce"] = time_in_force
        else:
            # Market orders
            if reduce_only:
                params["reduceOnly"] = "true"

        result = self._call("futures_create_order", **params)
        log.info(
            "Order placed | %s %s %s qty=%s price=%s side=%s orderId=%s",
            symbol, side, order_type, qty, price, position_side, result.get("orderId"),
        )
        return result

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        try:
            return self._call(
                "futures_cancel_order",
                symbol=symbol,
                orderId=int(order_id),
            )
        except BinanceClientError as e:
            if e.code in (-2011, -2013):
                log.debug("cancel_order %s: order %s already filled/expired", symbol, order_id)
                return {}
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        if symbol:
            return self._call("futures_get_open_orders", symbol=symbol)
        return self._call("futures_get_open_orders")

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        try:
            return self._call(
                "futures_get_order",
                symbol=symbol,
                orderId=int(order_id),
            )
        except Exception:
            return None

    def get_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[Dict]:
        """Look up an order by its client order ID."""
        try:
            return self._call(
                "futures_get_order",
                symbol=symbol,
                origClientOrderId=client_order_id[:36],
            )
        except Exception:
            return None

    # ── Exchange-side SL/TP ───────────────────────────────────────────────────

    def set_trading_stop(
        self,
        symbol: str,
        stop_loss: Optional[str] = None,
        take_profit: Optional[str] = None,
        position_side: str = "BOTH",
        sl_order_type: str = "STOP_MARKET",      # SL: Market (urgent)
        tp_order_type: str = "TAKE_PROFIT_MARKET",  # TP: Market (closePosition=true 지원)
        tp_limit_price: Optional[str] = None,
    ) -> Dict:
        """
        Set exchange-side SL/TP orders for an open position.
        In Binance, these are separate orders (unlike Bybit's set_trading_stop).

        Order type policy:
          SL → STOP_MARKET: 손절 즉각 실행 (Taker fee)
          TP → TAKE_PROFIT (Limit): 익절 지정가 체결 (Maker fee)
        """
        results = {}

        # Determine close side based on position_side
        if position_side == "LONG":
            close_side = "SELL"
        elif position_side == "SHORT":
            close_side = "BUY"
        else:
            # One-way mode: need to infer from position
            close_side = "SELL"  # default, caller should use hedge mode

        # Set Stop Loss
        if stop_loss:
            try:
                sl_params = dict(
                    symbol=symbol,
                    side=close_side,
                    type=sl_order_type,         # STOP_MARKET
                    stopPrice=stop_loss,
                    closePosition="true",
                    positionSide=position_side,
                )
                sl_result = self._call("futures_create_order", **sl_params)
                results["sl"] = sl_result
                log.info(
                    "Exchange SL set | %s positionSide=%s stopPrice=%s",
                    symbol, position_side, stop_loss,
                )
            except BinanceClientError as e:
                log.error("Failed to set exchange SL for %s: %s", symbol, e)

        # Set Take Profit
        if take_profit:
            try:
                tp_params = dict(
                    symbol=symbol,
                    side=close_side,
                    type=tp_order_type,         # TAKE_PROFIT (limit)
                    stopPrice=take_profit,       # trigger price
                    price=tp_limit_price or take_profit,  # limit price
                    closePosition="true",
                    positionSide=position_side,
                    timeInForce="GTC",
                )
                tp_result = self._call("futures_create_order", **tp_params)
                results["tp"] = tp_result
                log.info(
                    "Exchange TP set | %s positionSide=%s stopPrice=%s limitPrice=%s",
                    symbol, position_side, take_profit, tp_limit_price or take_profit,
                )
            except BinanceClientError as e:
                log.error("Failed to set exchange TP for %s: %s", symbol, e)

        return results

    def cancel_all_open_orders(self, symbol: str) -> Dict:
        """Cancel all open orders for a symbol (useful for cleanup)."""
        try:
            return self._call("futures_cancel_all_open_orders", symbol=symbol)
        except Exception as e:
            log.warning("Failed to cancel all orders for %s: %s", symbol, e)
            return {}

    def cancel_all_orders(self, symbol: str, side: Optional[str] = None) -> int:
        """심볼의 모든 미체결 지정가 주문을 취소. side 필터 지원 (None이면 전체).
        Returns: 취소된 주문 수.
        """
        open_orders = self.get_open_orders(symbol)
        cancelled = 0
        for order in open_orders:
            if side:
                order_side = order.get("side", "").upper()
                if order_side != side.upper():
                    continue
            order_id = str(order.get("orderId", ""))
            if not order_id:
                continue
            try:
                self.cancel_order(symbol, order_id)
                cancelled += 1
            except Exception as e:
                log.warning("cancel_all_orders: failed to cancel %s for %s: %s", order_id, symbol, e)
        if cancelled:
            log.info("Cancelled %d stale order(s) for %s (side=%s)", cancelled, symbol, side or "all")
        return cancelled

    # ── Asset Transfers ───────────────────────────────────────────────────────

    def transfer_to_spot(self, amount: float, asset: str = "USDT") -> Dict:
        """
        Transfer funds from Futures Account → Spot Account.
        Used by ProfitProtector to lock in profits safely.
        type=2: FUTURES → SPOT
        """
        try:
            result = self._call(
                "futures_transfer",
                asset=asset,
                amount=str(round(amount, 2)),
                type=2,  # FUTURES → SPOT
            )
            log.info("Transferred %.2f %s to Spot Account | txId=%s", amount, asset, result.get("tranId"))
            return result
        except Exception as e:
            log.error("Transfer to Spot failed: %s", e)
            return {}

    def get_spot_balance(self, asset: str = "USDT") -> float:
        """Get USDT balance in Spot Account (safe storage)."""
        try:
            account = self._call("get_account")
            for b in account.get("balances", []):
                if b.get("asset") == asset:
                    return float(b.get("free", 0))
        except Exception as e:
            log.warning("Failed to get spot balance: %s", e)
        return 0.0

    # ── Execution Helpers ────────────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Returns convenient symbol constraints dict:
          min_qty, qty_step, price_step, min_notional, max_leverage, qty_precision, price_precision
        """
        info = self.get_instruments_info(symbol)
        if not info:
            return {}

        min_qty = 0.001
        qty_step = 0.001
        price_step = 0.01
        min_notional = 5.0
        qty_precision = info.get("quantityPrecision", 3)
        price_precision = info.get("pricePrecision", 2)

        for f in info.get("filters", []):
            ft = f.get("filterType", "")
            if ft == "LOT_SIZE":
                min_qty = float(f.get("minQty", 0.001))
                qty_step = float(f.get("stepSize", 0.001))
            elif ft == "PRICE_FILTER":
                price_step = float(f.get("tickSize", 0.01))
            elif ft == "MIN_NOTIONAL":
                min_notional = float(f.get("notional", 5))

        # Get max leverage from bracket info
        max_lev = 125
        try:
            brackets = self._call("futures_leverage_bracket", symbol=symbol)
            if brackets:
                max_lev = int(brackets[0].get("brackets", [{}])[0].get("initialLeverage", 125))
        except Exception:
            pass

        return {
            "min_qty": min_qty,
            "qty_step": qty_step,
            "price_step": price_step,
            "min_notional": min_notional,
            "max_leverage": max_lev,
            "qty_precision": qty_precision,
            "price_precision": price_precision,
        }

    def round_qty(self, qty: float, symbol_info: Dict) -> str:
        step = symbol_info.get("qty_step", 0.001)
        precision = symbol_info.get("qty_precision", 3)
        rounded = round(qty // step * step, precision + 2)
        rounded = max(rounded, symbol_info.get("min_qty", step))
        return f"{rounded:.{precision}f}"

    def round_price(self, price: float, symbol_info: Dict) -> str:
        step = symbol_info.get("price_step", 0.01)
        precision = symbol_info.get("price_precision", 2)
        rounded = round(price // step * step, precision + 2)
        return f"{rounded:.{precision}f}"
