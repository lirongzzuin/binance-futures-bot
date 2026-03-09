"""
MarketData collector — fetches and packages OHLCV, orderbook,
funding rate and open interest into a single MarketSnapshot.
Adapted for Binance Futures API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.api.binance_client import BinanceClient
from src.config import Settings
from src.logger import get_logger
from src.strategy.indicators import vwap as compute_vwap_series

log = get_logger(__name__)

# Binance interval mapping (config stores as plain number string)
_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h",
    "480": "8h", "720": "12h", "1440": "1d",
}

# Binance OI period mapping
_OI_PERIOD_MAP = {
    "1": "5m", "3": "5m", "5": "5m",
    "15": "15m", "30": "30m", "60": "1h",
    "120": "2h", "240": "4h",
}


@dataclass
class OrderBookSnapshot:
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0
    bid_depth_top5: float = 0.0
    ask_depth_top5: float = 0.0
    imbalance: float = 0.0


@dataclass
class MarketSnapshot:
    symbol: str
    ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)
    trend_ohlcv: Optional[pd.DataFrame] = None
    orderbook: OrderBookSnapshot = field(default_factory=OrderBookSnapshot)
    funding_rate: float = 0.0
    next_funding_ts: int = 0
    open_interest: float = 0.0
    oi_prev: float = 0.0
    oi_change_pct: float = 0.0
    last_price: float = 0.0
    vwap: float = 0.0
    vwap_distance_pct: float = 0.0


class DataCollector:
    def __init__(self, client: BinanceClient, settings: Settings):
        self._client = client
        self._settings = settings

    def collect(self, symbol: str, limit: int = 250) -> MarketSnapshot:
        snap = MarketSnapshot(symbol=symbol)

        try:
            snap.ohlcv = self._fetch_ohlcv(symbol, limit)
        except Exception as e:
            log.error("OHLCV fetch failed for %s: %s", symbol, e)
            return snap

        try:
            snap.orderbook = self._fetch_orderbook(symbol)
        except Exception as e:
            log.warning("Orderbook fetch failed for %s: %s", symbol, e)

        try:
            funding = self._client.get_funding_rate_history(symbol, limit=5)
            if funding:
                # Binance returns list with 'fundingRate' and 'fundingTime'
                snap.funding_rate = float(funding[-1].get("fundingRate", 0))
                snap.next_funding_ts = int(funding[-1].get("fundingTime", 0))
        except Exception as e:
            log.warning("Funding fetch failed for %s: %s", symbol, e)

        try:
            oi_period = _OI_PERIOD_MAP.get(self._settings.SIGNAL_TIMEFRAME, "5m")
            oi_records = self._client.get_open_interest(symbol, period=oi_period, limit=5)
            if len(oi_records) >= 2:
                # Binance OI hist: 'sumOpenInterest', 'timestamp'
                snap.open_interest = float(oi_records[-1].get("sumOpenInterest", 0))
                snap.oi_prev = float(oi_records[-2].get("sumOpenInterest", 0))
                if snap.oi_prev > 0:
                    snap.oi_change_pct = (snap.open_interest - snap.oi_prev) / snap.oi_prev
            elif len(oi_records) == 1:
                snap.open_interest = float(oi_records[0].get("sumOpenInterest", 0))
        except Exception as e:
            log.warning("OI fetch failed for %s: %s", symbol, e)

        if not snap.ohlcv.empty:
            snap.last_price = float(snap.ohlcv["close"].iloc[-1])

        try:
            snap.trend_ohlcv = self._fetch_trend_ohlcv(symbol)
        except Exception as e:
            log.warning("Trend OHLCV fetch failed for %s: %s", symbol, e)

        if not snap.ohlcv.empty:
            snap.vwap = self._compute_vwap(snap.ohlcv, self._settings.VWAP_SESSION_BARS)
            if snap.vwap > 0 and snap.last_price > 0:
                snap.vwap_distance_pct = (snap.last_price - snap.vwap) / snap.vwap

        return snap

    def collect_for_scalp(self, symbol: str, limit: int = 500) -> MarketSnapshot:
        """
        스캘핑 전략용 데이터 수집.
        - LTF (3m): 진입 신호용
        - HTF (15m): 추세 바이어스용
        """
        snap = MarketSnapshot(symbol=symbol)

        # LTF (3m) OHLCV — 진입 신호
        try:
            ltf_interval = _INTERVAL_MAP.get(self._settings.SCALP_PURE_LTF_TIMEFRAME, "3m")
            raw_ltf = self._client.get_kline(symbol, interval=ltf_interval, limit=limit)
            if raw_ltf:
                snap.ohlcv = self._raw_to_df(raw_ltf)
        except Exception as e:
            log.error("Scalp LTF OHLCV fetch failed for %s: %s", symbol, e)
            return snap

        # HTF (15m) OHLCV — 추세 바이어스
        try:
            htf_interval = _INTERVAL_MAP.get(self._settings.SCALP_PURE_HTF_TIMEFRAME, "15m")
            raw_htf = self._client.get_kline(symbol, interval=htf_interval, limit=300)
            if raw_htf:
                snap.trend_ohlcv = self._raw_to_df(raw_htf)
        except Exception as e:
            log.warning("Scalp HTF OHLCV fetch failed for %s: %s", symbol, e)

        # 오더북
        try:
            snap.orderbook = self._fetch_orderbook(symbol)
        except Exception as e:
            log.warning("Scalp orderbook fetch failed for %s: %s", symbol, e)

        # 펀딩 비율
        try:
            funding = self._client.get_funding_rate_history(symbol, limit=5)
            if funding:
                snap.funding_rate = float(funding[-1].get("fundingRate", 0))
                snap.next_funding_ts = int(funding[-1].get("fundingTime", 0))
        except Exception as e:
            log.warning("Scalp funding fetch failed for %s: %s", symbol, e)

        # OI (5m 기준)
        try:
            oi_records = self._client.get_open_interest(symbol, period="5m", limit=5)
            if len(oi_records) >= 2:
                snap.open_interest = float(oi_records[-1].get("sumOpenInterest", 0))
                snap.oi_prev = float(oi_records[-2].get("sumOpenInterest", 0))
                if snap.oi_prev > 0:
                    snap.oi_change_pct = (snap.open_interest - snap.oi_prev) / snap.oi_prev
            elif len(oi_records) == 1:
                snap.open_interest = float(oi_records[0].get("sumOpenInterest", 0))
        except Exception as e:
            log.warning("Scalp OI fetch failed for %s: %s", symbol, e)

        # 현재가 및 VWAP (HTF 기준 — 세션 VWAP)
        if not snap.ohlcv.empty:
            snap.last_price = float(snap.ohlcv["close"].iloc[-1])

        if snap.trend_ohlcv is not None and not snap.trend_ohlcv.empty:
            snap.vwap = self._compute_vwap(snap.trend_ohlcv, self._settings.VWAP_SESSION_BARS)
        elif not snap.ohlcv.empty:
            snap.vwap = self._compute_vwap(snap.ohlcv, self._settings.VWAP_SESSION_BARS)

        if snap.vwap > 0 and snap.last_price > 0:
            snap.vwap_distance_pct = (snap.last_price - snap.vwap) / snap.vwap

        return snap

    def _fetch_ohlcv(self, symbol: str, limit: int) -> pd.DataFrame:
        interval = _INTERVAL_MAP.get(self._settings.SIGNAL_TIMEFRAME, "5m")
        raw = self._client.get_kline(symbol, interval=interval, limit=limit)
        if not raw:
            return pd.DataFrame()
        return self._raw_to_df(raw)

    def _fetch_trend_ohlcv(self, symbol: str, limit: int = 250) -> pd.DataFrame:
        interval = _INTERVAL_MAP.get(self._settings.TREND_TIMEFRAME, "1h")
        raw = self._client.get_kline(symbol, interval=interval, limit=limit)
        if not raw:
            return pd.DataFrame()
        return self._raw_to_df(raw)

    def _raw_to_df(self, raw: List) -> pd.DataFrame:
        """
        Binance kline format:
        [open_time, open, high, low, close, volume, close_time,
         quote_vol, num_trades, taker_buy_base, taker_buy_quote, ignore]
        """
        df = pd.DataFrame(raw, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
        df = df.astype({
            "ts": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        # Binance ts is already in ms
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["turnover"] = df["close"] * df["volume"]  # approximation
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _compute_vwap(self, ohlcv: pd.DataFrame, period: int) -> float:
        if ohlcv.empty or len(ohlcv) < period:
            return 0.0
        vwap_series = compute_vwap_series(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], period
        )
        val = vwap_series.iloc[-1]
        return float(val) if not (val != val) else 0.0

    def _fetch_orderbook(self, symbol: str) -> OrderBookSnapshot:
        book = self._client.get_orderbook(symbol, limit=20)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            return OrderBookSnapshot()

        # Binance orderbook: [[price_str, qty_str], ...]
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        spread_pct = spread / mid if mid > 0 else 0

        bid_depth = sum(float(b[0]) * float(b[1]) for b in bids[:5])
        ask_depth = sum(float(a[0]) * float(a[1]) for a in asks[:5])
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0

        return OrderBookSnapshot(
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_pct=spread_pct,
            bid_depth_top5=bid_depth,
            ask_depth_top5=ask_depth,
            imbalance=imbalance,
        )
