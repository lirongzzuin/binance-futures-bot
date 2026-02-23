"""
Central configuration via pydantic-settings.
All values sourced from environment variables / .env file.
"""
from __future__ import annotations

from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Trading Mode ──────────────────────────────────────────
    TRADING_MODE: Literal["dry-run", "paper", "live"] = "dry-run"

    # ── Binance API ────────────────────────────────────────────
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET: bool = True
    BINANCE_HEDGE_MODE: bool = True   # True = hedge mode (long+short simultaneously)

    # ── Capital & Risk ────────────────────────────────────────
    PAPER_CAPITAL_USDT: float = 10000.0
    MAX_RISK_PER_TRADE_PCT: float = 0.02       # 2% equity risk per trade
    MAX_DAILY_LOSS_PCT: float = 0.08           # 8% → halt trading for today
    MAX_WEEKLY_LOSS_PCT: float = 0.15          # 15% → halt for the week
    MAX_DRAWDOWN_PCT: float = 0.30             # 30% → switch to conservative
    MAX_LEVERAGE_CAP: int = 10                 # 10× max leverage
    MAX_CONCURRENT_POSITIONS: int = 3
    MAX_TOTAL_EXPOSURE_PCT: float = 0.75       # 75% of equity exposed
    MAX_SINGLE_EXPOSURE_PCT: float = 0.30      # 30% per symbol

    # ── Signal Settings ───────────────────────────────────────
    SIGNAL_ENTRY_THRESHOLD: float = 0.42
    SIGNAL_EXIT_THRESHOLD: float = 0.35
    SIGNAL_TIMEFRAME: str = "5"                # 5-minute scalping
    TREND_TIMEFRAME: str = "60"                # higher timeframe for regime detection
    UNIVERSE_REFRESH_INTERVAL_HOURS: int = 4
    TOP_N_COINS: int = 10

    # ── Scalping Parameters ───────────────────────────────────────
    VWAP_SESSION_BARS: int = 96
    SCALP_STOP_ATR: float = 1.0
    SCALP_TP1_ATR: float = 1.5
    SCALP_TP2_ATR: float = 3.0
    SCALP_MAX_BARS: int = 12
    SCALP_PARTIAL_EXIT_PCT: float = 0.60

    # ── Trend Ride Parameters ─────────────────────────────────────
    TREND_STOP_ATR: float = 2.0
    TREND_TP1_ATR: float = 3.0
    TREND_TP2_ATR: float = 6.0
    TREND_MAX_BARS: int = 120
    TREND_PARTIAL_EXIT_PCT: float = 0.40
    TREND_RIDE_SCORE_THRESHOLD: int = 2

    # ── Signal Quality Filters ─────────────────────────────────────
    SIGNAL_FILTER_ER_THRESHOLD: float = 0.06
    SIGNAL_FILTER_HURST_MIN: float = 0.40
    SIGNAL_FILTER_MIN_ATR_PCT: float = 0.001

    # ── Advanced Risk Management ──────────────────────────────────
    VOLATILITY_TARGET_PCT: float = 0.15
    USE_KELLY_FRACTION: float = 0.25

    # ── Indicators ────────────────────────────────────────────
    ATR_PERIOD: int = 14
    STOP_ATR_MULTIPLIER: float = 2.0
    TP_ATR_MULTIPLIER: float = 3.0
    TRAILING_STOP_ATR_MULTIPLIER: float = 1.0
    ADX_PERIOD: int = 14
    ADX_TREND_THRESHOLD: float = 25.0
    ADX_RANGE_THRESHOLD: float = 20.0
    EMA_FAST: int = 9
    EMA_MID: int = 21
    EMA_SLOW: int = 50
    VOLATILITY_LOOKBACK: int = 20
    RSI_PERIOD: int = 14
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    KELTNER_PERIOD: int = 20
    KELTNER_ATR_MULT: float = 1.5
    HULL_MA_PERIOD: int = 16

    # ── Advanced Indicators ───────────────────────────────────
    STOCHRSI_RSI_PERIOD: int = 14
    STOCHRSI_STOCH_PERIOD: int = 14
    STOCHRSI_SMOOTH_K: int = 3
    STOCHRSI_SMOOTH_D: int = 3
    WILLIAMS_R_PERIOD: int = 14
    CMF_PERIOD: int = 20
    OBV_EMA_PERIOD: int = 20
    SQUEEZE_BB_PERIOD: int = 20
    SQUEEZE_KC_PERIOD: int = 20
    RSI_DIVERGENCE_LOOKBACK: int = 14

    # ── ICT Settings ─────────────────────────────────────────
    ICT_SWING_PERIOD: int = 5
    ICT_FVG_LOOKBACK: int = 50
    ICT_RANGE_PERIOD: int = 20

    # ── Order Execution ───────────────────────────────────────
    LIMIT_ORDER_TIMEOUT_SECONDS: int = 30
    LIMIT_ORDER_MAX_RETRIES: int = 3
    REENTRY_COOLDOWN_SECONDS: int = 300
    SPREAD_TAKER_THRESHOLD: float = 0.003

    # ── Backtest ─────────────────────────────────────────────
    BACKTEST_START: str = ""
    BACKTEST_END: str = ""
    BACKTEST_FEE_TAKER: float = 0.0004    # Binance taker fee 0.04%
    BACKTEST_FEE_MAKER: float = 0.0002    # Binance maker fee 0.02%
    BACKTEST_SLIPPAGE_PCT: float = 0.0002
    BACKTEST_INITIAL_CAPITAL: float = 10000.0
    BACKTEST_FUNDING_RATE_AVG: float = 0.0001

    # ── Quality Gate ─────────────────────────────────────────
    QUALITY_GATE_SHARPE: float = 1.0
    QUALITY_GATE_MDD: float = 0.30
    QUALITY_GATE_WIN_RATE: float = 0.45
    QUALITY_GATE_PROFIT_FACTOR: float = 1.5
    QUALITY_GATE_MIN_TRADES: int = 100
    QUALITY_GATE_PASSED: bool = False

    # ── Profit Protection ─────────────────────────────────────
    PROFIT_PROTECT_ENABLED: bool = True
    PROFIT_PROTECT_THRESHOLD_PCT: float = 0.20
    PROFIT_PROTECT_TRANSFER_PCT: float = 0.20   # 20% 이체, 80% 복리 재투자
    PROFIT_PROTECT_MIN_CAPITAL_PCT: float = 1.10
    PROFIT_PROTECT_CHECK_HOURS: int = 24
    # 복리 성장 단계: equity < 이 금액이면 이체 건너뜀 (100% 복리)
    PROFIT_PROTECT_MIN_EQUITY_USDT: float = 10000.0

    # ── Macro Regime (Market Mode Switching) ─────────────────
    # BULL: multi-coin, MAX_LEVERAGE_CAP; BEAR/SIDEWAYS: BTC+ETH only
    BULL_MODE_SYMBOLS: str = ""              # empty = TOP_N_COINS auto
    BEAR_MODE_SYMBOLS: str = "BTCUSDT,ETHUSDT"
    BEAR_MAX_LEVERAGE_CAP: int = 50          # 하락장/횡보장 최대 레버리지
    MACRO_REGIME_CHECK_HOURS: int = 4        # 매크로 레짐 체크 주기 (시간)
    MACRO_BULL_CONFIRM_COUNT: int = 3        # BULL 전환 확정 연속 횟수
    MACRO_BEAR_CONFIRM_COUNT: int = 2        # BEAR/SIDEWAYS 전환 확정 연속 횟수

    # ── Slack ─────────────────────────────────────────────────
    SLACK_WEBHOOK_URL: str = ""
    SLACK_DAILY_REPORT_HOUR: int = 9
    SLACK_DASHBOARD_INTERVAL_MINUTES: int = 120
    SLACK_BOT_TOKEN: str = ""
    SLACK_APP_TOKEN: str = ""

    # ── Database ─────────────────────────────────────────────
    DATABASE_PATH: str = "bot.db"

    # ── Logging ──────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }

    # ── Helpers ───────────────────────────────────────────────
    def is_live(self) -> bool:
        return self.TRADING_MODE == "live"

    def is_paper(self) -> bool:
        return self.TRADING_MODE == "paper"

    def is_dry_run(self) -> bool:
        return self.TRADING_MODE == "dry-run"

    def check_live_allowed(self) -> None:
        """Raise if LIVE mode is requested but quality gate not passed."""
        if self.is_live() and not self.QUALITY_GATE_PASSED:
            raise RuntimeError(
                "LIVE mode requires QUALITY_GATE_PASSED=true.\n"
                "Run the backtest first: python -m src.backtest.engine --run\n"
                "Then set QUALITY_GATE_PASSED=true in .env after reviewing the report."
            )

    @field_validator("TRADING_MODE")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        return v.lower()


settings = Settings()
