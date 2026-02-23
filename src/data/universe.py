"""
Universe manager — maintains the top-N USDT perpetual contracts
by 24-hour quote volume, refreshed periodically.
Adapted for Binance Futures API.
"""
from __future__ import annotations

import time
from typing import List

from src.api.binance_client import BinanceClient
from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)

# Tokens to skip regardless of volume rank
_EXCLUDE_PATTERNS = [
    "USDC", "USDT", "BUSD", "DAI", "TUSD", "USDP", "FDUSD",
    "3L", "3S", "5L", "5S", "10L", "10S",  # leveraged tokens
    "UP", "DOWN", "BULL", "BEAR",           # leveraged tokens (alt names)
]


class UniverseManager:
    def __init__(self, client: BinanceClient, settings: Settings):
        self._client = client
        self._settings = settings
        self._symbols: List[str] = []
        self._last_refresh: float = 0.0

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    def needs_refresh(self) -> bool:
        elapsed_h = (time.time() - self._last_refresh) / 3600
        return elapsed_h >= self._settings.UNIVERSE_REFRESH_INTERVAL_HOURS

    def refresh(self) -> List[str]:
        """Fetch latest tickers and pick top-N by 24h USDT quote volume."""
        try:
            tickers = self._client.get_tickers()
        except Exception as exc:
            log.error("Universe refresh failed: %s", exc)
            return self._symbols

        candidates = []
        for t in tickers:
            symbol: str = t.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            base = symbol[:-4]
            if any(pat in base for pat in _EXCLUDE_PATTERNS):
                continue
            # Binance ticker has 'quoteVolume' for USDT-denominated volume
            volume = float(t.get("quoteVolume", 0) or 0)
            if volume < 1_000_000:
                continue
            candidates.append((symbol, volume))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_n = [sym for sym, _ in candidates[: self._settings.TOP_N_COINS]]

        if top_n != self._symbols:
            log.info("Universe updated: %s", top_n)
        else:
            log.debug("Universe unchanged: %s", top_n)

        self._symbols = top_n
        self._last_refresh = time.time()
        return top_n

    def get_or_refresh(self) -> List[str]:
        if self.needs_refresh() or not self._symbols:
            return self.refresh()
        return self._symbols
