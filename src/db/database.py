"""
SQLite database — persists trades, equity curve, and bot state.
Uses aiosqlite for async access.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

_KST = timezone(timedelta(hours=9))


def _today_kst() -> str:
    """Return today's date in KST (Korea Standard Time) as ISO string."""
    return datetime.now(tz=_KST).date().isoformat()

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)


class Database:
    def __init__(self, settings: Settings):
        self._path = settings.DATABASE_PATH
        self._init()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    side        TEXT    NOT NULL,
                    entry_price REAL    NOT NULL,
                    exit_price  REAL,
                    qty         REAL    NOT NULL,
                    leverage    INTEGER DEFAULT 1,
                    pnl         REAL,
                    fee         REAL    DEFAULT 0,
                    exit_reason TEXT,
                    entry_ts    REAL    NOT NULL,
                    exit_ts     REAL,
                    mode        TEXT    DEFAULT 'dry-run',
                    extra       TEXT    DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS equity_curve (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        REAL    NOT NULL,
                    equity    REAL    NOT NULL,
                    mode      TEXT    DEFAULT 'dry-run'
                );

                CREATE TABLE IF NOT EXISTS bot_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    date        TEXT PRIMARY KEY,
                    pnl         REAL DEFAULT 0,
                    trades      INTEGER DEFAULT 0,
                    wins        INTEGER DEFAULT 0,
                    losses      INTEGER DEFAULT 0,
                    mode        TEXT DEFAULT 'dry-run'
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_entry_ts ON trades(entry_ts);
                CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts);
            """)
        log.info("Database initialised at %s", self._path)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        # WAL mode: allows concurrent reads + writes, prevents blocking during backtest
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Trade CRUD ────────────────────────────────────────────────────────────

    def insert_trade_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        leverage: int,
        mode: str,
        extra: Dict = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (symbol, side, entry_price, qty, leverage, entry_ts, mode, extra)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (symbol, side, entry_price, qty, leverage,
                 time.time(), mode, json.dumps(extra or {})),
            )
            return cur.lastrowid

    def update_trade_close(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        fee: float,
        reason: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades SET exit_price=?, pnl=?, fee=?, exit_reason=?, exit_ts=?
                   WHERE id=?""",
                (exit_price, pnl, fee, reason, time.time(), trade_id),
            )
        self._update_daily_stats(pnl)

    def _update_daily_stats(self, pnl: float) -> None:
        today = _today_kst()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO daily_stats(date, pnl, trades, wins, losses)
                   VALUES(?, ?, 1, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                   pnl=pnl+excluded.pnl,
                   trades=trades+1,
                   wins=wins+excluded.wins,
                   losses=losses+excluded.losses""",
                (today, pnl, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0),
            )

    def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 1000,
        mode: Optional[str] = None,
    ) -> List[Dict]:
        with self._conn() as conn:
            sql = "SELECT * FROM trades WHERE exit_ts IS NOT NULL"
            params: List[Any] = []
            if symbol:
                sql += " AND symbol=?"; params.append(symbol)
            if mode:
                sql += " AND mode=?"; params.append(mode)
            sql += " ORDER BY entry_ts DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    # ── Equity Curve ──────────────────────────────────────────────────────────

    def record_equity(self, equity: float, mode: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_curve(ts, equity, mode) VALUES(?,?,?)",
                (time.time(), equity, mode),
            )

    def get_equity_curve(self, mode: Optional[str] = None, limit: int = 5000) -> List[Dict]:
        with self._conn() as conn:
            sql = "SELECT * FROM equity_curve"
            params: List[Any] = []
            if mode:
                sql += " WHERE mode=?"; params.append(mode)
            sql += " ORDER BY ts DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ── Bot State (KV store) ──────────────────────────────────────────────────

    def set_state(self, key: str, value: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bot_state(key, value) VALUES(?,?)",
                (key, json.dumps(value)),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key=?", (key,)
            ).fetchone()
            if row:
                return json.loads(row[0])
            return default

    # ── Daily Stats ───────────────────────────────────────────────────────────

    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?", (days,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_today_pnl(self) -> float:
        today = _today_kst()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pnl FROM daily_stats WHERE date=?", (today,)
            ).fetchone()
            return float(row[0]) if row else 0.0

    def get_weekly_pnl(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT SUM(pnl) FROM daily_stats
                   WHERE date >= date('now', '-7 days')"""
            ).fetchone()
            return float(row[0] or 0)

    def get_peak_equity(self, mode: Optional[str] = None) -> float:
        with self._conn() as conn:
            sql = "SELECT MAX(equity) FROM equity_curve"
            params: List[Any] = []
            if mode:
                sql += " WHERE mode=?"; params.append(mode)
            row = conn.execute(sql, params).fetchone()
            return float(row[0] or 0)

    def get_total_pnl(self, mode: Optional[str] = None) -> float:
        with self._conn() as conn:
            sql = "SELECT SUM(pnl) FROM trades WHERE exit_ts IS NOT NULL"
            params: List[Any] = []
            if mode:
                sql += " AND mode=?"; params.append(mode)
            row = conn.execute(sql, params).fetchone()
            return float(row[0] or 0)

    def get_trades_count_today(self, symbol: Optional[str] = None) -> int:
        today_kst = datetime.now(tz=_KST).replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = today_kst.timestamp()
        with self._conn() as conn:
            if symbol:
                row = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE entry_ts >= ? AND symbol=?",
                    (today_start, symbol),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE entry_ts >= ?", (today_start,)
                ).fetchone()
            return int(row[0] or 0)

    def get_rolling_win_stats(self, window: int = 50) -> Optional[Dict]:
        """최근 N개 거래의 승률/평균수익/평균손실 — Kelly 산출용."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pnl FROM trades WHERE exit_ts IS NOT NULL "
                "ORDER BY exit_ts DESC LIMIT ?",
                (window,),
            ).fetchall()
            if len(rows) < 10:
                return None
            pnls = [float(r[0]) for r in rows]
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            n = len(pnls)
            wr = len(wins) / n if n > 0 else 0.5
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            return {
                "n_trades": n,
                "win_rate": wr,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
            }

    def get_win_rate(self, days: int = 30) -> float:
        """Win rate of closed trades over the last N days."""
        cutoff = __import__("time").time() - days * 86400
        with self._conn() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE exit_ts IS NOT NULL AND exit_ts >= ?",
                (cutoff,),
            ).fetchone()
            total = int(total_row[0] or 0)
            if total == 0:
                return 0.0
            win_row = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE exit_ts IS NOT NULL AND exit_ts >= ? AND pnl > 0",
                (cutoff,),
            ).fetchone()
            wins = int(win_row[0] or 0)
            return wins / total

    def get_consecutive_losses(self, symbol: Optional[str] = None) -> int:
        """Count consecutive losses from the most recent trades."""
        with self._conn() as conn:
            sql = "SELECT pnl FROM trades WHERE exit_ts IS NOT NULL"
            params: List[Any] = []
            if symbol:
                sql += " AND symbol=?"; params.append(symbol)
            sql += " ORDER BY exit_ts DESC LIMIT 20"
            rows = conn.execute(sql, params).fetchall()
            count = 0
            for row in rows:
                if float(row[0] or 0) < 0:
                    count += 1
                else:
                    break
            return count
