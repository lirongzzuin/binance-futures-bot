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

    # ── Strategy Selection ────────────────────────────────────
    # "swing": 15분봉 기반 스윙/추세추종 전략 (ICT + 복합 신호)
    # "scalp": 3분봉 기반 EMA Ribbon+VWAP+RSI 순수 스캘핑
    # "trend": Phase-locked direction TrendEMA 전략
    # "auto":  Dynamic Regime Router — 매 사이클 장세 자동 감지 후 최적 전략 선택
    STRATEGY_MODE: Literal["swing", "scalp", "trend", "auto"] = "swing"

    # ── Dynamic Regime Router (STRATEGY_MODE=auto) ────────────
    # ADX 강세 임계값 → STRONG_BULL/STRONG_BEAR (TREND_RIDE 모드)
    REGIME_ROUTER_ADX_STRONG: float = 27.0
    # ADX 보통 추세 임계값 → BULL/BEAR (SCALP 모드)
    REGIME_ROUTER_ADX_MODERATE: float = 20.0
    # ATR 백분위 임계값 — 이 이상이면 VOLATILE (포지션 50% 축소)
    REGIME_ROUTER_ATR_VOLATILE_PCT: float = 0.82
    # 레짐 그룹 간 전환 확정에 필요한 연속 감지 횟수 (그룹 내 전환은 즉시)
    REGIME_ROUTER_CONFIRM_BARS: int = 3
    # 모멘텀 품질 최소값 — 이하 시 risk_multiplier 50% 감소
    REGIME_ROUTER_MIN_MOMENTUM_QUALITY: float = 0.3
    # BB 수축 백분위 — 이하 시 breakout 대기 (compression 감지)
    REGIME_ROUTER_BB_COMPRESSION_PCT: float = 0.20

    # ── Binance API ──────────────────────────────────────────
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET: bool = True
    BINANCE_HEDGE_MODE: bool = True   # True = hedge mode (long+short simultaneously)

    # ── Capital & Risk ────────────────────────────────────────
    PAPER_CAPITAL_USDT: float = 10000.0
    MAX_RISK_PER_TRADE_PCT: float = 0.02       # 2% equity risk per trade (aggressive compounding)
    MAX_DAILY_LOSS_PCT: float = 0.08           # 8% → Slack 알림 (거래 중지 없음)
    MAX_WEEKLY_LOSS_PCT: float = 0.15          # 15% → Slack 알림 (거래 중지 없음)
    MAX_DRAWDOWN_PCT: float = 0.30             # 30% → REDUCE 트리거 (포지션 비중 축소 시작)
    PAUSE_DRAWDOWN_PCT: float = 0.42           # 42% → PAUSE (신규 진입 중단, 기존 포지션 관리)
    HALT_DRAWDOWN_PCT: float = 0.50            # 50% → HALT (긴급 전면 중단 — 계좌 보호 핵심)
    MAX_LEVERAGE_CAP: int = 20                 # 20× max leverage (검증된 전략 + 추세 전용)
    MAX_CONCURRENT_POSITIONS: int = 2
    MAX_TOTAL_EXPOSURE_PCT: float = 15.0       # 1500% — 레버리지 선물 기준 (notional/equity)
    MAX_SINGLE_EXPOSURE_PCT: float = 0.30      # 30% per symbol

    # ── Signal Settings ───────────────────────────────────────
    SIGNAL_ENTRY_THRESHOLD: float = 0.80       # 고확신 진입만 허용 (강한 신호만 거래)
    SIGNAL_EXIT_THRESHOLD: float = 0.38
    SIGNAL_TIMEFRAME: str = "15"               # 15분봉 — 5분봉 대비 노이즈 3× 감소
    TREND_TIMEFRAME: str = "1440"              # 일봉 — 200일 이동평균 기반 불/베어마켓 구분
    UNIVERSE_REFRESH_INTERVAL_HOURS: int = 4
    TOP_N_COINS: int = 10                      # more symbols = more opportunities

    # ── Scalping Parameters (15분봉 기준) ──────────────────────────
    VWAP_SESSION_BARS: int = 96                # bars per session for VWAP
    SCALP_STOP_ATR: float = 1.5               # SL: 1.5× ATR (확대 — 15분봉 노이즈 대응)
    SCALP_TP1_ATR: float = 2.0                # TP1 R:R 1.33:1 — 빠른 수익 확보
    SCALP_TP2_ATR: float = 4.5                # TP2 R:R 3.0:1 — 합리적 타겟
    SCALP_MAX_BARS: int = 48                   # 15분봉 × 48 = 12시간 최대
    SCALP_PARTIAL_EXIT_PCT: float = 0.40       # 40% TP1에서 청산, 60% 추세 유지 (수익 극대화)

    # ── Trend Ride Parameters (15분봉 기준) ──────────────────────────
    TREND_STOP_ATR: float = 2.5               # SL: 2.5× ATR (확대 — 추세 타이트 스탑 방지)
    TREND_TP1_ATR: float = 3.0                # TP1 R:R 1.2:1 — 확실한 수익 확보
    TREND_TP2_ATR: float = 8.0                # TP2 R:R 3.2:1 — 추세 극대화
    TREND_MAX_BARS: int = 240                  # 15분봉 × 240 = 60시간 최대
    TREND_PARTIAL_EXIT_PCT: float = 0.30       # close 30% at TP1, keep 70% for full run (추세 수익 극대화)
    TREND_RIDE_SCORE_THRESHOLD: int = 1        # allow trend ride more often

    # ── Pure Scalping Strategy Parameters (STRATEGY_MODE="scalp") ──────────
    # LTF (진입 신호용) / HTF (추세 바이어스용) 타임프레임
    SCALP_PURE_LTF_TIMEFRAME: str = "3"       # 3분봉 — 스캘핑 진입 신호
    SCALP_PURE_HTF_TIMEFRAME: str = "15"      # 15분봉 — HTF 추세 바이어스 필터
    # EMA 파라미터 (LTF에서 사용)
    SCALP_PURE_EMA_FAST: int = 9
    SCALP_PURE_EMA_MID: int = 21
    SCALP_PURE_EMA_SLOW: int = 50
    # RSI 진입 구간
    SCALP_PURE_RSI_PERIOD: int = 14
    SCALP_PURE_RSI_LONG_MIN: float = 45.0     # 롱: RSI 45 이상 (모멘텀 확인, 경계성 과매도 제거)
    SCALP_PURE_RSI_LONG_MAX: float = 65.0     # 롱: RSI 65 이하 (과매수 회피, 상단 여유)
    SCALP_PURE_RSI_SHORT_MIN: float = 42.0    # 숏: RSI 42 이상 (과매도 반등 회피, 상향 조정)
    SCALP_PURE_RSI_SHORT_MAX: float = 55.0    # 숏: RSI 55 이하 (하락 모멘텀 확인 구간만)
    # 진입 조건
    SCALP_PURE_VOLUME_MULT: float = 1.3       # 볼륨 확인: 평균 대비 1.3배 이상 (품질 강화)
    SCALP_PURE_EMA_ZONE_ATR: float = 1.5      # EMA 존 범위: EMA21 ± 1.5×ATR (더 깊은 풀백 포착)
    SCALP_PURE_ENTRY_MIN_SCORE: int = 8       # 진입 최소 점수 (11점 만점, 모멘텀 확인 추가)
    SCALP_PURE_ENTRY_URGENT_SCORE: int = 9    # 긴급 진입 점수 → 시장가 선호 (최고 점수만)
    # 리스크 파라미터 (3분봉 ATR 기준, 15분봉보다 타이트)
    SCALP_PURE_STOP_ATR: float = 1.2          # SL: 1.2× ATR (확대 — 3분봉 노이즈 대응)
    SCALP_PURE_TP1_ATR: float = 2.0           # TP1: 2.0× ATR → 플로어 적용 시 스케일업
    SCALP_PURE_TP2_ATR: float = 4.5           # TP2: 4.5× ATR → 플로어 적용 시 스케일업
    SCALP_PURE_MAX_BARS: int = 30             # 최대 보유: 3분봉×30 = 90분
    SCALP_PURE_PARTIAL_EXIT_PCT: float = 0.40 # TP1에서 40% 청산, 60% 유지

    # ── Minimum Stop Distance (노이즈 터짐 방지) ─────────────────────────
    # ATR × 배수가 너무 작을 때 스탑이 즉시 터지는 것 방지 (가격 대비 최소 %)
    MIN_STOP_PCT_PURE_SCALP: float = 0.004   # 최소 0.4% — 3분봉 노이즈 흡수
    MIN_STOP_PCT_SCALP: float = 0.007        # 최소 0.7% — 15분봉 노이즈 흡수
    MIN_STOP_PCT_TREND_RIDE: float = 0.012   # 최소 1.2% — 추세 큰 노이즈 흡수

    # ── Trend EMA Strategy Parameters (STRATEGY_MODE="trend") ──────────
    # Phase-locked direction: BULL_TREND=LONG만, BEAR_TREND=SHORT만, SIDEWAYS=양방향
    TREND_EMA_ADX_THRESHOLD: float = 22.0      # HTF ADX ≥ 22: 중강도 이상 추세만 허용 (노이즈 감소)
    TREND_EMA_MIN_SCORE: int = 4               # 추세 진입 최소 점수 (5점 만점, 강한 컨펌 요구)
    TREND_EMA_RANGE_MIN_SCORE: int = 4         # 횡보 진입 최소 점수 (5점 만점, 보수적)
    TREND_EMA_RSI_BULL_MIN: float = 40.0       # BULL_TREND 롱 RSI 하한 (모멘텀 확인)
    TREND_EMA_RSI_BULL_MAX: float = 65.0       # BULL_TREND 롱 RSI 상한 (상단 여유)
    TREND_EMA_RSI_BEAR_MIN: float = 40.0       # BEAR_TREND 숏 RSI 하한 (과매도 반등 회피)
    TREND_EMA_RSI_BEAR_MAX: float = 60.0       # BEAR_TREND 숏 RSI 상한 (중립 이하 확인)
    TREND_EMA_RSI_RANGE_LONG_MAX: float = 35.0 # SIDEWAYS 롱: RSI 35 이하 (강한 과매도)
    TREND_EMA_RSI_RANGE_SHORT_MIN: float = 65.0 # SIDEWAYS 숏: RSI 65 이상 (강한 과매수)
    TREND_EMA_ZONE_ATR_MULT: float = 2.0       # EMA21 ±2.0×ATR = 풀백/반등 진입 존 (확장)
    TREND_EMA_VOLUME_MULT: float = 1.1         # 거래량 ≥ 20봉 평균 × 1.1 (약간 완화)

    # ── Signal Quality Filters (15분봉 기준 최적화) ─────────────────
    SIGNAL_FILTER_ER_THRESHOLD: float = 0.06   # 추세성 확인 (완화로 기회 확대)
    SIGNAL_FILTER_HURST_MIN: float = 0.42      # 평균회귀 구간 진입 회피
    SIGNAL_FILTER_MIN_ATR_PCT: float = 0.0012  # 최소 0.12% ATR (수수료 대비 여유)

    # ── Higher Timeframe (4H) Trend Filter ──────────────────────
    HTF_EMA_FAST: int = 9                      # 4H EMA fast
    HTF_EMA_MID: int = 21                      # 4H EMA mid
    HTF_EMA_SLOW: int = 50                     # 4H EMA slow (추세 방향 기준)
    HTF_EMA_TREND: int = 200                   # 일봉 EMA 200 (불/베어마켓 가장 중요한 기준선)
    HTF_ADX_THRESHOLD: float = 20.0            # 4H ADX 최소값 (추세 확인)
    HTF_RSI_BULL_MIN: float = 40.0             # 롱 진입 시 4H RSI 최소 (과매도 회피)
    HTF_RSI_BULL_MAX: float = 70.0             # 롱 진입 시 4H RSI 최대 (과매수 회피)
    HTF_RSI_BEAR_MIN: float = 30.0             # 숏 진입 시 4H RSI 최소
    HTF_RSI_BEAR_MAX: float = 60.0             # 숏 진입 시 4H RSI 최대

    # ── Advanced Risk Management ──────────────────────────────────
    VOLATILITY_TARGET_PCT: float = 0.15         # AQR vol-targeting (15% annualized)
    USE_KELLY_FRACTION: float = 0.50            # Half-Kelly position size (adaptive)
    KELLY_MIN_TRADES: int = 20                   # Kelly 적용 최소 거래 수 (30→20 완화)

    # ── Indicators ────────────────────────────────────────────
    ATR_PERIOD: int = 14
    STOP_ATR_MULTIPLIER: float = 2.0
    TP_ATR_MULTIPLIER: float = 3.0
    TRAILING_STOP_ATR_MULTIPLIER: float = 1.5  # 1.5× ATR trailing — 적당한 룸 (추세 수익 캡처)
    TRAILING_PROFIT_RATCHET_PCT: float = 0.35  # 최대 수익 대비 35% 트레일 (Chandelier ratchet)
    TRAILING_MIN_PROFIT_LOCK_PCT: float = 0.003  # TP1 후 최소 0.3% 수익 잠금
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
    # Keltner channels (Chester Keltner / Linda Raschke)
    KELTNER_PERIOD: int = 20
    KELTNER_ATR_MULT: float = 1.5
    # Hull Moving Average (Alan Hull, 2005)
    HULL_MA_PERIOD: int = 16

    # ── Advanced Indicators ───────────────────────────────────
    STOCHRSI_RSI_PERIOD: int = 14          # RSI period for StochRSI
    STOCHRSI_STOCH_PERIOD: int = 14        # Stochastic period
    STOCHRSI_SMOOTH_K: int = 3             # K line smoothing
    STOCHRSI_SMOOTH_D: int = 3             # D signal line smoothing
    WILLIAMS_R_PERIOD: int = 14            # Williams %R period
    CMF_PERIOD: int = 20                   # Chaikin Money Flow period
    OBV_EMA_PERIOD: int = 20               # OBV trend EMA
    SQUEEZE_BB_PERIOD: int = 20            # Squeeze momentum BB period
    SQUEEZE_KC_PERIOD: int = 20            # Squeeze momentum KC period
    RSI_DIVERGENCE_LOOKBACK: int = 14      # RSI divergence lookback window

    # ── ICT Settings ─────────────────────────────────────────
    ICT_SWING_PERIOD: int = 5              # bars each side for swing pivot
    ICT_FVG_LOOKBACK: int = 50
    ICT_RANGE_PERIOD: int = 20             # bars for premium/discount range

    # ── Order Execution ───────────────────────────────────────
    LIMIT_ORDER_TIMEOUT_SECONDS: int = 15   # 15s/시도 (스캘핑: 빠른 진입이 중요)
    LIMIT_ORDER_MAX_RETRIES: int = 2        # 2회 후 시장가 폴백 (총 최대 ~30s)
    REENTRY_COOLDOWN_SECONDS: int = 1800       # 30 min — 손절 후 재진입 대기 (노이즈 회피)
    REENTRY_SAME_DIR_COOLDOWN_SECONDS: int = 2700  # 45 min — 동일 방향 손절 후 대기
    MAX_TRADES_PER_DAY_PER_SYMBOL: int = 3  # 일일 심볼당 최대 거래 횟수
    SPREAD_TAKER_THRESHOLD: float = 0.003  # 0.3% spread → use market order
    # 신호 강도가 이 값 초과 시 지정가 대신 시장가(즉시 체결 우선)
    SIGNAL_URGENT_THRESHOLD: float = 0.90  # > 0.90 → market order (urgency)

    # ── Stale Order Management ────────────────────────────────
    # 미체결 지정가 주문이 이 시간(초) 이상 지속되면 취소 대상
    STALE_ORDER_TIMEOUT_SECONDS: int = 300     # 5분 이상 미체결 → 취소
    # 미체결 주문 감사 주기 (분)
    STALE_ORDER_AUDIT_INTERVAL_MINUTES: int = 5

    # ── Backtest ─────────────────────────────────────────────
    BACKTEST_START: str = ""               # ISO date; empty = 5 years ago
    BACKTEST_END: str = ""                 # ISO date; empty = today
    BACKTEST_FEE_TAKER: float = 0.0004     # Binance taker fee 0.04%
    BACKTEST_FEE_MAKER: float = 0.0002     # Binance maker fee 0.02%
    BACKTEST_SLIPPAGE_PCT: float = 0.0002  # realistic slippage for BTC perp (0.02% per side)
    BACKTEST_INITIAL_CAPITAL: float = 10000.0
    BACKTEST_FUNDING_RATE_AVG: float = 0.0001  # avg funding rate per 8h interval

    # ── Quality Gate (MTF 전략 기준 상향) ────────────────────────
    QUALITY_GATE_SHARPE: float = 1.0
    QUALITY_GATE_MDD: float = 0.25
    QUALITY_GATE_WIN_RATE: float = 0.45        # 추세추종 현실적 승률 (45%+, R:R 보상)
    QUALITY_GATE_PROFIT_FACTOR: float = 1.2    # PF 1.2+ (실질 수익성 확보)
    QUALITY_GATE_MIN_TRADES: int = 50          # MTF는 진입 횟수 줄어드므로 조정
    QUALITY_GATE_PASSED: bool = False      # flipped only by backtest runner

    # ── Profit Protection ─────────────────────────────────────
    PROFIT_PROTECT_ENABLED: bool = True
    # 최소 계좌 잔고: 이 금액 이상일 때만 이체 실행 (복리 성장 단계 보호)
    PROFIT_PROTECT_MIN_EQUITY_USDT: float = 10000.0
    PROFIT_PROTECT_THRESHOLD_PCT: float = 0.20
    PROFIT_PROTECT_TRANSFER_PCT: float = 0.20
    PROFIT_PROTECT_MIN_CAPITAL_PCT: float = 1.10
    PROFIT_PROTECT_CHECK_HOURS: int = 24

    # ── Macro Regime (시장 상황 자동 전환) ────────────────────────────────────
    BULL_MODE_SYMBOLS: str = ""
    BEAR_MODE_SYMBOLS: str = "BTCUSDT,ETHUSDT"
    BEAR_MAX_LEVERAGE_CAP: int = 20
    MACRO_REGIME_CHECK_HOURS: int = 4
    MACRO_BULL_CONFIRM_COUNT: int = 3
    MACRO_BEAR_CONFIRM_COUNT: int = 2

    # ── Slack ─────────────────────────────────────────────────
    SLACK_WEBHOOK_URL: str = ""
    SLACK_DAILY_REPORT_HOUR: int = 9       # KST
    SLACK_DASHBOARD_INTERVAL_MINUTES: int = 120  # 0 = disabled

    # Slack Bot (Socket Mode) — for on-demand slash commands
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
