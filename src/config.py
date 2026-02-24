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
    MAX_RISK_PER_TRADE_PCT: float = 0.02       # 2% equity risk per trade (aggressive compounding)
    MAX_DAILY_LOSS_PCT: float = 0.08           # 8% → Slack 알림 (거래 중지 없음)
    MAX_WEEKLY_LOSS_PCT: float = 0.15          # 15% → Slack 알림 (거래 중지 없음)
    MAX_DRAWDOWN_PCT: float = 0.30             # 30% → HALT (계좌 보호 핵심)
    MAX_LEVERAGE_CAP: int = 20                 # 20× max leverage (백테스트 최적값)
    MAX_CONCURRENT_POSITIONS: int = 2
    MAX_TOTAL_EXPOSURE_PCT: float = 0.75       # 75% of equity exposed
    MAX_SINGLE_EXPOSURE_PCT: float = 0.30      # 30% per symbol

    # ── Signal Settings ───────────────────────────────────────
    SIGNAL_ENTRY_THRESHOLD: float = 0.60       # MTF 정렬 기반 — 최고품질 진입만 허용
    SIGNAL_EXIT_THRESHOLD: float = 0.38
    SIGNAL_TIMEFRAME: str = "15"               # 15분봉 — 5분봉 대비 노이즈 3× 감소
    TREND_TIMEFRAME: str = "1440"              # 일봉 — 200일 이동평균 기반 불/베어마켓 구분
    UNIVERSE_REFRESH_INTERVAL_HOURS: int = 4
    TOP_N_COINS: int = 10                      # more symbols = more opportunities

    # ── Scalping Parameters (15분봉 기준) ──────────────────────────
    VWAP_SESSION_BARS: int = 96                # bars per session for VWAP
    SCALP_STOP_ATR: float = 1.2               # 타이트 SL → 높은 R:R 달성
    SCALP_TP1_ATR: float = 2.5                # TP1 R:R 2.08:1
    SCALP_TP2_ATR: float = 5.5                # TP2 R:R 4.58:1 (풀 트렌드 수익)
    SCALP_MAX_BARS: int = 48                   # 15분봉 × 48 = 12시간 최대
    SCALP_PARTIAL_EXIT_PCT: float = 0.55       # 55% TP1에서 청산, 45% 추세 유지

    # ── Trend Ride Parameters (15분봉 기준) ──────────────────────────
    TREND_STOP_ATR: float = 2.0               # 일봉 추세 동행 — 여유 있는 SL
    TREND_TP1_ATR: float = 4.0                # TP1 R:R 2.0:1
    TREND_TP2_ATR: float = 10.0               # TP2 R:R 5.0:1 — 풀 트렌드 수익 극대화
    TREND_MAX_BARS: int = 240                  # 15분봉 × 240 = 60시간 최대
    TREND_PARTIAL_EXIT_PCT: float = 0.45       # close 45% at TP1, keep 55% for full run
    TREND_RIDE_SCORE_THRESHOLD: int = 1        # allow trend ride more often

    # ── Signal Quality Filters (15분봉 기준 최적화) ─────────────────
    SIGNAL_FILTER_ER_THRESHOLD: float = 0.06   # 추세성 확인 (완화로 기회 확대)
    SIGNAL_FILTER_HURST_MIN: float = 0.42      # 평균회귀 구간 진입 회피
    SIGNAL_FILTER_MIN_ATR_PCT: float = 0.0012  # 최소 0.12% ATR (수수료 대비 여유)

    # ── Higher Timeframe (Daily) Trend Filter ──────────────────────
    HTF_EMA_FAST: int = 9                      # Daily EMA fast
    HTF_EMA_MID: int = 21                      # Daily EMA mid
    HTF_EMA_SLOW: int = 50                     # Daily EMA slow (추세 방향 기준)
    HTF_EMA_TREND: int = 200                   # 일봉 EMA 200 (불/베어마켓 가장 중요한 기준선)
    HTF_ADX_THRESHOLD: float = 20.0            # Daily ADX 최소값 (추세 확인)
    HTF_RSI_BULL_MIN: float = 40.0             # 롱 진입 시 Daily RSI 최소 (과매도 회피)
    HTF_RSI_BULL_MAX: float = 70.0             # 롱 진입 시 Daily RSI 최대 (과매수 회피)
    HTF_RSI_BEAR_MIN: float = 30.0             # 숏 진입 시 Daily RSI 최소
    HTF_RSI_BEAR_MAX: float = 60.0             # 숏 진입 시 Daily RSI 최대

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
    REENTRY_COOLDOWN_SECONDS: int = 900        # 15 min = 1 full bar on 15m timeframe
    SPREAD_TAKER_THRESHOLD: float = 0.003  # 0.3% spread → use market order

    # ── Backtest ─────────────────────────────────────────────
    BACKTEST_START: str = ""
    BACKTEST_END: str = ""
    BACKTEST_FEE_TAKER: float = 0.0004    # Binance taker fee 0.04%
    BACKTEST_FEE_MAKER: float = 0.0002    # Binance maker fee 0.02%
    BACKTEST_SLIPPAGE_PCT: float = 0.0002
    BACKTEST_INITIAL_CAPITAL: float = 10000.0
    BACKTEST_FUNDING_RATE_AVG: float = 0.0001

    # ── Quality Gate (MTF 전략 기준 상향) ────────────────────────
    QUALITY_GATE_SHARPE: float = 1.2
    QUALITY_GATE_MDD: float = 0.25
    QUALITY_GATE_WIN_RATE: float = 0.60        # 60%+ 승률 목표
    QUALITY_GATE_PROFIT_FACTOR: float = 1.8    # PF 1.8+ 목표
    QUALITY_GATE_MIN_TRADES: int = 50          # MTF는 진입 횟수 줄어드므로 조정
    QUALITY_GATE_PASSED: bool = False      # flipped only by backtest runner

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
