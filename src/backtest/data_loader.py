"""
Historical data loader for backtesting.

Downloads up to 5 years of OHLCV data from Binance Futures (max 1500 bars per call)
and caches locally as CSV.

Binance limits:
  - Max 1500 bars per klines call
  - 15-min bars: 5 years = ~175,000 bars → ~117 API calls per symbol
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from binance.client import Client

from src.logger import get_logger

log = get_logger(__name__)

_CACHE_DIR = Path("data_cache")
_CACHE_DIR.mkdir(exist_ok=True)

# Binance interval string mapping (config "5" → "5m", "60" → "1h")
_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h",
    "480": "8h", "720": "12h", "1440": "1d",
}


def load_ohlcv(
    symbol: str,
    interval: str = "15",
    years: int = 5,
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load OHLCV for `symbol` covering the last `years` years.
    Uses cache if available and not older than 1 hour.
    """
    cache_file = _CACHE_DIR / f"{symbol}_{interval}.csv"

    # Check cache
    if not force_refresh and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 1.0:
            log.info("Loading %s from cache (%s)", symbol, cache_file)
            df = pd.read_csv(cache_file, parse_dates=["datetime"])
            df = df.sort_values("ts").reset_index(drop=True)
            return df

    log.info("Downloading %s history (interval=%s, years=%d)...", symbol, interval, years)

    client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
    binance_interval = _INTERVAL_MAP.get(str(interval), f"{interval}m")

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=365 * years)).timestamp() * 1000
    )

    interval_ms = _interval_to_ms(interval)
    all_bars = []
    current_start = start_ms

    while current_start < end_ms:
        try:
            bars = client.futures_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=1500,
                startTime=current_start,
                endTime=min(current_start + 1500 * interval_ms - 1, end_ms),
            )
            if not bars:
                break

            all_bars.extend(bars)
            # bars[-1][0] = open_time of last bar
            last_ts = int(bars[-1][0])
            current_start = last_ts + interval_ms

            if last_ts >= end_ms - interval_ms:
                break

            time.sleep(0.05)  # rate limit friendly

        except Exception as e:
            log.warning("Download error for %s at %d: %s", symbol, current_start, e)
            time.sleep(2)
            continue

    if not all_bars:
        log.error("No data downloaded for %s", symbol)
        return pd.DataFrame()

    df = _bars_to_df(all_bars)
    df = df[df["ts"] >= start_ms]
    df = df[df["ts"] <= end_ms]
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(subset=["ts"], inplace=True)

    # Save cache
    df.to_csv(cache_file, index=False)
    log.info(
        "Downloaded %d bars for %s | %s → %s",
        len(df),
        symbol,
        df["datetime"].iloc[0].date() if not df.empty else "?",
        df["datetime"].iloc[-1].date() if not df.empty else "?",
    )
    return df


def load_multi(
    symbols: list,
    interval: str = "15",
    years: int = 5,
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV for multiple symbols."""
    result = {}
    for sym in symbols:
        df = load_ohlcv(sym, interval, years, api_key, api_secret, testnet)
        if not df.empty:
            result[sym] = df
        else:
            log.warning("No data for %s — excluded from backtest", sym)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bars_to_df(raw_bars: list) -> pd.DataFrame:
    """
    Convert raw Binance klines list to DataFrame.
    Binance kline format (12 columns):
      [open_time, open, high, low, close, volume, close_time,
       quote_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
    """
    df = pd.DataFrame(
        raw_bars,
        columns=[
            "ts", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df = df[["ts", "open", "high", "low", "close", "volume", "quote_volume"]].copy()
    df = df.astype({
        "ts": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "quote_volume": "float64",
    })
    # turnover = quote_volume (USDT value, same as Bybit's turnover)
    df.rename(columns={"quote_volume": "turnover"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def _interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    mapping = {
        "1": 60_000,
        "3": 180_000,
        "5": 300_000,
        "15": 900_000,
        "30": 1_800_000,
        "60": 3_600_000,
        "120": 7_200_000,
        "240": 14_400_000,
        "360": 21_600_000,
        "480": 28_800_000,
        "720": 43_200_000,
        "1440": 86_400_000,
        "D": 86_400_000,
        "W": 604_800_000,
    }
    return mapping.get(str(interval), 900_000)
