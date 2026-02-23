# Binance USDT Perpetual Futures Trading Bot

> **주의**: 선물 거래는 레버리지로 인해 원금 전액 손실 위험이 있습니다.
> 이 소프트웨어는 수익을 보장하지 않으며, 사용에 따른 투자 손실의 책임은 사용자 본인에게 있습니다.

---

Python으로 설계·구현한 **Binance USDT 무기한 선물 자동매매 시스템**입니다.
시장 국면(Regime)을 실시간으로 분류하고, 15가지 기술 지표 기반 시그널을 레짐별로 가중 합산해 진입·청산 결정을 내립니다.
Fly.io 클라우드에 컨테이너로 배포 가능하며, Slack으로 실시간 알림과 대화형 명령어 조회를 지원합니다.

Bybit 버전(`bybit-futures-bot`)과 **완전히 동일한 전략 로직**을 사용하며, Binance Futures API에 맞게 적응됩니다.

---

## 프로젝트 하이라이트

| 항목 | 내용 |
|------|------|
| **구현 언어** | Python 3.13 |
| **전략 유형** | 시장 국면 분류 기반 멀티 시그널 알고리즘 트레이딩 |
| **지표 수** | 15가지 기술 지표 → 9개 카테고리 가중 합산 |
| **리스크 관리** | 동적 레버리지, 다층 손실 가드, 청산 버퍼 검증 |
| **수수료 최적화** | 진입·익절: Maker(0.020%), 손절·긴급청산: Taker(0.040%) |
| **테스트** | pytest 29개 테스트 케이스 |
| **배포** | Docker → Fly.io (공유 CPU 1× / 256MB, 월 ≈ $2.17) |
| **모니터링** | Slack Webhook + Socket Mode 명령어 봇 |

---

## Binance vs Bybit 주요 차이점

| 항목 | Bybit | Binance |
|------|-------|---------|
| **라이브러리** | `pybit 5.8.0` | `python-binance 1.0.19` |
| **Hedge Mode** | `positionIdx=1` (Long) / `2` (Short) | `positionSide="LONG"` / `"SHORT"` |
| **PostOnly 주문** | `timeInForce="PostOnly"` | `timeInForce="GTX"` (Good Till Crossing) |
| **손절 주문** | `set_trading_stop()` | `type="STOP_MARKET"` 별도 주문 |
| **익절 주문** | `set_trading_stop()` | `type="TAKE_PROFIT"` (Limit) 별도 주문 |
| **Maker 수수료** | 0.020% | 0.020% |
| **Taker 수수료** | 0.055% | 0.040% |
| **자금 이체** | `POST /v5/asset/transfer` | `futures_transfer(type=2)` |
| **데이터** | `session.get_kline()` 7컬럼 | `futures_klines()` 12컬럼 |
| **유니버스** | `turnover24h` | `quoteVolume` |

---

## 목차

1. [아키텍처 개요](#1-아키텍처-개요)
2. [데이터 수집 및 유니버스 관리](#2-데이터-수집-및-유니버스-관리)
3. [시장 국면(Regime) 분류](#3-시장-국면regime-분류)
4. [시그널 생성 — 15가지 지표](#4-시그널-생성--15가지-지표)
5. [시그널 조합 — 레짐별 가중 합산](#5-시그널-조합--레짐별-가중-합산)
6. [진입 품질 필터](#6-진입-품질-필터)
7. [트레이드 모드 선택](#7-트레이드-모드-선택)
8. [포지션 사이징 및 레버리지 계산](#8-포지션-사이징-및-레버리지-계산)
9. [익절 & 손절 전략](#9-익절--손절-전략)
10. [리스크 가드 시스템](#10-리스크-가드-시스템)
11. [주문 실행 시스템](#11-주문-실행-시스템)
12. [백테스트 & Quality Gate](#12-백테스트--quality-gate)
13. [Slack 연동](#13-slack-연동)
14. [데이터베이스 설계](#14-데이터베이스-설계)
15. [프로젝트 구조](#15-프로젝트-구조)
16. [설치 및 실행](#16-설치-및-실행)
17. [Fly.io 배포](#17-flyio-배포)
18. [주요 설정 파라미터](#18-주요-설정-파라미터)
19. [기술 스택](#19-기술-스택)
20. [기술적 의사결정 기록](#20-기술적-의사결정-기록)

---

## 1. 아키텍처 개요

**매 5분마다** 아래 파이프라인이 순서대로 실행됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        5-Minute Cycle                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Binance REST API]                                             │
│   └─ OHLCV 250봉 + 오더북 + 펀딩비 + 미결제약정(OI)            │
│          │                                                      │
│          ▼                                                      │
│  [Universe Manager]   4시간마다 갱신                            │
│   └─ 24h 거래량(quoteVolume) 기준 상위 10개 코인 자동 선별      │
│          │                                                      │
│          ▼                                                      │
│  [Regime Detector]   5분봉 + 60분봉 듀얼 타임프레임             │
│   └─ TRENDING_UP / TRENDING_DOWN / RANGING / HIGH_VOLATILITY    │
│          │                                                      │
│          ▼                                                      │
│  [Signal Generator]  15가지 지표 병렬 계산                      │
│   └─ 각 지표 → (direction: +1/0/-1, confidence: 0~1)           │
│          │                                                      │
│          ▼                                                      │
│  [Signal Combiner]   레짐별 카테고리 가중 합산                  │
│   └─ 최종 (direction, strength: 0~1) 출력                       │
│          │                                                      │
│          ▼                                                      │
│  [Quality Filters]   저품질 시그널 제거                         │
│   └─ Kaufman ER + Hurst Exponent + Min ATR 검사                 │
│          │                                                      │
│  ┌───────┴───────────┐                                          │
│  │ 포지션 보유 중?   │                                          │
│  │  YES → 청산 조건  │  NO → 진입 조건 검사                     │
│  │  순서 점검        │    ├─ Risk Guards 통과?                   │
│  │  (SL/TP/Trailing/ │    ├─ Risk Engine 계산                   │
│  │   Regime/Signal/  │    └─ Trade Mode 결정 (SCALP/TREND_RIDE) │
│  │   Time)           │         → Order Manager 주문 발송        │
│  └───────────────────┘                                          │
│          │                                                      │
│          ▼                                                      │
│  [Position Sync]  거래소 실제 포지션과 로컬 상태 비교            │
│   └─ 강제 청산(Liquidation) 감지 시 즉시 FLAT 처리              │
│          │                                                      │
│          ▼                                                      │
│  [SQLite DB + Slack]  거래 기록 저장 + 실시간 알림              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 데이터 수집 및 유니버스 관리

### 데이터 수집 (`src/data/collector.py`)

매 사이클마다 각 심볼에 대해 다음을 수집합니다:

| 데이터 | 용도 |
|--------|------|
| OHLCV 250봉 (5분봉) | 기술 지표 계산 기반 데이터 |
| OHLCV 60봉 (60분봉) | 상위 타임프레임 레짐 감지 |
| 오더북 스냅샷 | 매수/매도 호가 불균형 계산 |
| 펀딩비 | 군중 포지션 편향 파악 |
| 미결제약정(OI) | 추세 강도 확인 |

### 유니버스 자동 선별 (`src/data/universe.py`)

- 4시간마다 Binance 전체 USDT 선물 종목을 스캔
- 24시간 거래량(`quoteVolume`) 기준 상위 `TOP_N_COINS`(기본 **10개**) 자동 선별
- 스테이블코인 및 레버리지 토큰 자동 제외 (`USDT`, `USDC`, `BUSD`, `FDUSD`, `UP`, `DOWN`, `BULL`, `BEAR`)

---

## 3. 시장 국면(Regime) 분류

| 국면 | 판단 기준 | 전략 |
|------|-----------|------|
| **TRENDING_UP** | ADX ≥ 25, EMA 정배열(9 > 21 > 50), Hurst > 0.50 | 추세 추종 Long |
| **TRENDING_DOWN** | ADX ≥ 25, EMA 역배열, Hurst > 0.50 | 추세 추종 Short |
| **RANGING** | ADX < 20 | 지지/저항 평균 회귀 |
| **HIGH_VOLATILITY** | 변동성 백분위 > 85% | 포지션 축소, 보수적 운용 |

---

## 4. 시그널 생성 — 15가지 지표

| 카테고리 | 지표 | Long 신호 조건 (예시) |
|----------|------|----------------------|
| **trend** | EMA Cross (9/21) | 단기 EMA가 중기 EMA 위로 교차 |
| **trend** | EMA Alignment (9/21/50) | 3선 완전 정배열 |
| **trend** | MACD Histogram (12/26/9) | 히스토그램 양전환 |
| **trend** | Hull MA (16) | 기울기 상향 전환 |
| **reversion** | RSI (14) | RSI < 30 과매도 구간 |
| **reversion** | Bollinger Band (20, 2σ) | 가격이 하단 밴드 하회 |
| **reversion** | StochRSI (14) | K선이 D선을 **30 이하**에서 상향 교차 (과매도 확정) |
| **reversion** | Williams %R (14) | -80 이하 과매도 |
| **reversion** | RSI Divergence | 가격 신저가 + RSI 상대 고저 (강세 다이버전스) |
| **volume** | CMF — Chaikin Money Flow (20) | CMF > 0.1 매수 압력 |
| **volume** | OBV — On-Balance Volume | OBV 상승 추세 + 가격 방향 일치 |
| **volatility** | ATR / Keltner Channel | 변동성 확대 + 방향 확인 |
| **squeeze** | Squeeze Momentum (존 카터) | BB가 KC 안에 갇혔다가 상향 이탈 |
| **vwap** | VWAP (세션 96봉) | 가격이 VWAP 위로 복귀 |
| **orderbook** | 호가 불균형 | 매수 호가 우세 (±15% 이상) |
| **ict** | ICT 구조 분석 | BOS / CHoCH / FVG / 유동성 스윕 |

---

## 5. 시그널 조합 — 레짐별 가중 합산

```
┌──────────────┬────────────────┬─────────┬──────────────────┐
│  카테고리    │  추세장        │ 횡보장  │ 고변동성장       │
├──────────────┼────────────────┼─────────┼──────────────────┤
│ ict          │ 30%  ←최고    │ 22%     │ 25%  ←최고      │
│ trend        │ 27%           │  5%     │ 16%              │
│ volume       │ 12%           │ 18%     │ 18%              │
│ vwap         │  8%           │ 10%     │  9%              │
│ squeeze      │  8%           │  6%     │  8%              │
│ momentum     │  6%           │  6%     │  4%              │
│ volatility   │  6%           │  4%     │ 12%              │
│ reversion    │  3%           │ 25% ←최고 │  4%            │
│ orderbook    │  0%           │  4%     │  4%              │
└──────────────┴────────────────┴─────────┴──────────────────┘
```

---

## 6. 진입 품질 필터

| 필터 | 기준값 | 목적 |
|------|--------|------|
| **Kaufman ER** | ER > 0.06 | 방향성 없는 노이즈 구간 진입 차단 |
| **Hurst Exponent** | H > 0.40 | 순수 평균회귀 구간에서 모멘텀 진입 차단 |
| **Min ATR** | ATR > 가격의 0.10% | 변동성이 극히 낮은 구간 진입 차단 (수수료 감안) |

---

## 7. 트레이드 모드 선택

| 항목 | SCALP 모드 | TREND_RIDE 모드 |
|------|-----------|----------------|
| 손절 (SL) | ATR × 1.0 | ATR × 2.0 |
| 1차 익절 (TP1) | ATR × 1.5 (R:R 1.5) | ATR × 3.0 (R:R 1.5) |
| 2차 익절 (TP2) | ATR × 3.0 (R:R 3.0) | ATR × 6.0 (R:R 3.0) |
| TP1 부분청산 비율 | 60% | 40% |
| 최대 보유 시간 | 12봉 (1시간) | 120봉 (10시간) |
| 트레일링 스탑 | 1.0×ATR (TP1 이후 발동) | 1.0×ATR (TP1 이후 발동) |

---

## 8. 포지션 사이징 및 레버리지 계산

레버리지는 **입력값이 아닌 결과값**입니다.

```
Step 1. 손절 거리 계산
        stop_distance = ATR × stop_atr_mult
Step 2. 기본 레버리지 (리스크 역산)
        base_leverage = MAX_RISK_PER_TRADE_PCT(2%) / stop_distance_pct
Step 3. 5가지 감쇠 계수 적용
        × vol_modifier × liq_modifier × fund_modifier × signal_modifier × health_modifier
Step 4. 하드캡 적용 (MAX_LEVERAGE_CAP=10)
Step 5. 포지션 크기 계산 (리스크 기반 / AQR 변동성 타게팅 / Quarter-Kelly 중 최소값)
Step 6. 청산 버퍼 검증: liq_distance > stop_distance × 3.0
```

---

## 9. 익절 & 손절 전략

```
우선순위  조건                              동작
────────────────────────────────────────────────────────────────
  0순위  TP1 도달 (tp1_hit = False)        → 부분청산 + 손절→진입가로 이동
  1순위  손절가(SL) 터치                   → 전량 시장가 즉시 청산
  2순위  TP2 도달                          → 잔여 전량 청산
  3순위  트레일링 스탑 발동 (tp1_hit = True) → 잔여 전량 청산
  4순위  레짐 역전                          → 전량 청산
  5순위  역방향 강시그널                    → 전량 청산
  6순위  시그널 강도 약화 (< 0.35)          → 전량 청산
  7순위  최대 보유 시간 초과                → 전량 청산
```

### 주문 유형별 수수료 전략

| 청산 이유 | 주문 유형 | 수수료 | 이유 |
|-----------|----------|--------|------|
| TP1, TP2 | 지정가 (Maker) | 0.020% | 여유 있는 청산 |
| SL, 가드, 청산 감지 | 시장가 (Taker) | 0.040% | 즉각 실행 필요 |
| 레짐/시그널/트레일링/시간 | 지정가 (Maker) | 0.020% | 속도보다 비용 중시 |

### 거래소 사전예약 주문 (Binance SL/TP)

Binance는 Bybit의 `set_trading_stop()`과 달리 SL과 TP를 **별도 주문**으로 등록합니다.

| 항목 | 설정 | 이유 |
|------|------|------|
| **SL 주문** | `STOP_MARKET` (closePosition=true) | 즉각 체결 필수 |
| **TP 주문** | `TAKE_PROFIT` Limit (closePosition=true) | Maker 수수료 절약 |
| **TP 등록 가격** | **TP2 가격** | TP1 부분청산은 봇 내부 로직 처리, TP1에 등록 시 전량 시장가 청산 방지 |
| **GTX 주문 거절** | `-5022` 에러 → Market으로 즉시 폴백 | 호가 변동으로 PostOnly 거절 시 체결 보장 |

---

## 10. 리스크 가드 시스템

| 가드 | 조건 | 조치 |
|------|------|------|
| **일간 손실** | 당일 손실 ≥ 8% | 당일 신규 진입 중단 (PAUSE) |
| **주간 손실** | 주간 손실 ≥ 15% | 당주 신규 진입 중단 (PAUSE) |
| **최대 낙폭 경고** | 낙폭 ≥ 21% (한도의 70%) | 레버리지 자동 축소 (REDUCE) |
| **최대 낙폭 한도** | 낙폭 ≥ 30% | 전체 포지션 강제 청산 + Slack 경보 (HALT) |
| **변동성 스파이크** | 변동성 백분위 ≥ 97% | 포지션 50% 축소 (REDUCE) |
| **스프레드 이상** | 스프레드 ≥ 1% | 신규 진입 일시 중단 (PAUSE) |
| **총 노출도 한도** | 전체 포지션 노출도 ≥ 75% | 신규 진입 차단 |
| **심볼별 노출도** | 단일 심볼 노출도 ≥ 30% | 해당 심볼 신규 진입 차단 |

---

## 11. 주문 실행 시스템

### 진입 주문 전략 (`src/execution/order_manager.py`)

```
스프레드 > 0.3% 또는 시그널 강도 > 0.90
  → 시장가 주문 (즉각 체결, Taker 수수료)

그 외
  → GTX(PostOnly) 지정가 주문 (Maker 수수료 절약)
     └─ LIMIT_ORDER_TIMEOUT_SECONDS(30초) 내 미체결
        → 재시도 (최대 3회)
        → 그래도 미체결 → 시장가로 전환
     └─ -5022 GTX 거절 → 즉시 시장가로 폴백
```

### 서킷 브레이커 (`src/api/binance_client.py`)

- 연속 5회 API 오류 → 60초간 신규 주문 차단
- Rate limit (`-1003` 또는 HTTP 429) → 지수 백오프 재시도 (최대 10회)
- 중복 취소 (`-2011`, `-2013`) → 정상 처리로 분류 (멱등성 보장)

---

## 12. 백테스트 & Quality Gate

### 실제 거래 환경과 동일한 비용 구조

| 비용 항목 | 값 | 적용 조건 |
|-----------|-----|-----------|
| Maker 수수료 | 0.020% | 진입(GTX/PostOnly), TP1, TP2, 트레일링, 시간 만료 청산 |
| Taker 수수료 | 0.040% | SL, 가드, 청산 감지 (Market 주문) |
| 슬리피지 | 0.020% (편도) | 모든 시장가 주문 |
| 펀딩비 | 0.010% (8시간 평균) | 8시간마다 부과 |

### Walk-Forward 검증

- **In-Sample (80%)**: 전략 파라미터 검증
- **Out-of-Sample (20%)**: 실제 성과 측정

### Quality Gate — 라이브 전환 조건

| 지표 | 기준 | 의미 |
|------|------|------|
| **Sharpe Ratio** | ≥ 1.0 | 위험 대비 수익이 충분한가 |
| **최대 낙폭(MDD)** | ≤ 30% | 최악의 경우 손실이 허용 범위인가 |
| **승률** | ≥ 45% | 수익 구조가 성립하는가 |
| **Profit Factor** | ≥ 1.5 | 총 이익이 총 손실의 1.5배 이상인가 |
| **최소 거래 횟수** | ≥ 100회 | 통계적으로 유의미한 샘플인가 |

```bash
# 합성 데이터 백테스트 (API 키 불필요)
python scripts/backtest_btc.py --days 365 --capital 10000

# 실제 과거 데이터 백테스트
python -m src.backtest.engine --symbols BTCUSDT ETHUSDT --years 3
```

---

## 13. Slack 연동

### 실시간 자동 알림

| 이벤트 | 알림 내용 |
|--------|-----------|
| 포지션 진입 | 심볼, 방향, 레버리지, 진입가, SL/TP 가격, 트레이드 모드 |
| 1차 익절(TP1) | 부분청산 수량, 실현 손익, 잔여 포지션 |
| 전량 청산 | 청산 이유, 실현 손익, 누적 손익 |
| 강제 청산 감지 | 긴급 경보 (즉각 대응 필요) |
| 일간 손실 경보 | 현재 낙폭, 잔여 한도 |
| **4시간 간격 정기 브리핑** | 09:00 → 13:00 → 17:00 → 21:00 → 01:00 → 05:00 (KST) |

> 모든 타임스탬프는 **KST(한국 표준시, UTC+9)** 기준으로 표시됩니다.

### 대화형 Slack 명령어 (Socket Mode)

| 명령어 | 내용 |
|--------|------|
| `/dashboard` | 자산 현황, 일간/누적 PnL, 오픈 포지션 수 |
| `/positions` | 현재 포지션 목록 (진입가, 현재가, 레버리지, 미실현 손익) |
| `/pnl` | 오늘 / 이번 주 / 전체 PnL + 승률 + 거래 횟수 |
| `/status` | 봇 상태 (모드, 가동 시간, 마지막 사이클, 서킷 브레이커) |

---

## 14. 데이터베이스 설계

SQLite를 WAL(Write-Ahead Logging) 모드로 운영합니다.

```sql
-- 거래 기록
trades:
  id, symbol, side (Buy/Sell)
  entry_price, exit_price, qty, leverage
  pnl, fee, pnl_net
  exit_reason
  entry_ts, exit_ts
  mode (dry-run / paper / live)

-- 자산 추이 (시간별)
equity_curve:
  ts, equity, mode

-- 일별 통계
daily_stats:
  date, pnl, trades, wins, losses, mode

-- 봇 상태 (Key-Value)
bot_state:
  key, value
```

---

## 15. 프로젝트 구조

```
binance-futures-bot/
├── src/
│   ├── main.py                  # 메인 루프, APScheduler (5분 주기)
│   ├── config.py                # 환경변수 → Pydantic Settings
│   ├── logger.py                # structlog 구조화 로깅
│   │
│   ├── api/
│   │   ├── binance_client.py    # Binance Futures REST API 래퍼 (서킷 브레이커)
│   │   ├── slack.py             # Slack Webhook 알림 (KST 타임스탬프)
│   │   └── slack_bot.py         # Slack Socket Mode 명령어 봇
│   │
│   ├── data/
│   │   ├── collector.py         # OHLCV + 오더북 + 펀딩비 + OI 수집
│   │   └── universe.py          # 거래 유니버스 자동 선별 (4시간 갱신)
│   │
│   ├── strategy/
│   │   ├── indicators.py        # ATR, EMA, RSI, StochRSI, OBV, CMF 등
│   │   ├── regime.py            # 시장 국면 분류 (ADX + EMA + Hurst)
│   │   ├── signals.py           # 15가지 시그널 생성기
│   │   ├── combiner.py          # 레짐별 가중 합산 → (direction, strength)
│   │   ├── filters.py           # 품질 필터 (ER, Hurst, Min ATR)
│   │   ├── ict.py               # ICT 개념 구현 (BOS, CHoCH, FVG, Liquidity)
│   │   └── scalp.py             # 트레이드 모드 판별 (SCALP / TREND_RIDE)
│   │
│   ├── risk/
│   │   ├── engine.py            # 동적 레버리지 + 포지션 사이징
│   │   ├── guards.py            # 손실 한도 / 낙폭 / 변동성 가드
│   │   └── profit_protector.py  # 수익금 자동 Spot 계좌 이전
│   │
│   ├── execution/
│   │   ├── state_machine.py     # 포지션 FSM (FLAT / LONG / SHORT)
│   │   └── order_manager.py     # 주문 생성/재시도/취소, 멱등성 보장
│   │
│   ├── db/
│   │   └── database.py          # SQLite WAL 모드, CRUD 메서드
│   │
│   ├── backtest/
│   │   ├── engine.py            # 봉 단위 시뮬레이션
│   │   ├── analyzer.py          # Sharpe, MDD, PF, 승률 계산
│   │   ├── synthetic.py         # 합성 데이터 생성 (GBM)
│   │   ├── data_loader.py       # Binance futures_klines 히스토리 로더
│   │   └── report.py            # HTML 보고서 생성 + Quality Gate 판정
│   │
│   └── reporting/
│       └── daily_briefing.py    # 4시간 간격 정기 브리핑
│
├── tests/                       # pytest 29개 테스트 케이스
├── scripts/
│   └── backtest_btc.py          # 합성 데이터 백테스트 스크립트
├── fly.toml                     # Fly.io 배포 설정
├── Dockerfile
├── requirements.txt
└── .env                         # 환경변수 (git 미포함)
```

---

## 16. 설치 및 실행

### 사전 요구사항

- Python 3.11 이상
- Binance 계정 (Futures API 활성화, Hedge Mode 활성화 필요)
- Slack App (선택 사항)

### Binance Hedge Mode 활성화

```
Binance Futures → 설정 → 포지션 모드 → 헤지 모드 선택
또는 봇이 자동으로 활성화: BINANCE_HEDGE_MODE=true (기본값)
```

### 로컬 실행

```bash
git clone https://github.com/lirongzzuin/binance-futures-bot.git
cd binance-futures-bot

python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에서 BINANCE_API_KEY, BINANCE_API_SECRET 등 입력

# 테스트 실행
pytest tests/ -v

# 합성 데이터 백테스트 (API 키 불필요)
python scripts/backtest_btc.py --days 365 --quiet

# Dry-run 모드 (주문 없음, API 불필요)
TRADING_MODE=dry-run python -m src.main

# Paper 트레이딩 (실제 시그널, 가상 자금)
TRADING_MODE=paper python -m src.main
```

### 운영 모드

| 모드 | API 연결 | 실제 주문 | 권장 용도 |
|------|----------|-----------|-----------|
| `dry-run` | 불필요 | 없음 | 코드 동작 확인, 로직 검증 |
| `paper` | 필요 | 없음 | 2~4주 시그널 품질 검증 |
| `live` | 필요 | **있음** | Quality Gate 통과 후만 허용 |

---

## 17. Fly.io 배포

```bash
# Fly CLI 설치
brew install flyctl
fly auth login

# 앱 및 퍼시스턴트 볼륨 생성 (최초 1회)
fly apps create binance-futures-bot
fly volumes create bot_data --size 1 --region nrt   # 도쿄 리전

# 환경변수 (시크릿) 등록
fly secrets set BINANCE_API_KEY=your_key \
               BINANCE_API_SECRET=your_secret \
               SLACK_WEBHOOK_URL=your_webhook

# 배포
fly deploy

# 실시간 로그 확인
fly logs
```

**운영 비용**: shared-cpu-1x 256MB ≈ $2.02/월 + 1GB 볼륨 $0.15/월 = **약 $2.17/월**

---

## 18. 주요 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `TRADING_MODE` | `dry-run` | `dry-run` / `paper` / `live` |
| `BINANCE_API_KEY` | `` | Binance API 키 |
| `BINANCE_API_SECRET` | `` | Binance API 시크릿 |
| `BINANCE_TESTNET` | `true` | Testnet 사용 여부 |
| `BINANCE_HEDGE_MODE` | `true` | Hedge Mode 사용 여부 |
| `PAPER_CAPITAL_USDT` | `10000.0` | Paper 모드 초기 자본 (USDT) |
| `MAX_RISK_PER_TRADE_PCT` | `0.02` | 거래당 최대 손실 (자본의 2%) |
| `MAX_DAILY_LOSS_PCT` | `0.08` | 일간 손실 한도 8% |
| `MAX_LEVERAGE_CAP` | `10` | 최대 레버리지 하드캡 |
| `SIGNAL_ENTRY_THRESHOLD` | `0.42` | 진입 시그널 강도 임계값 |
| `TOP_N_COINS` | `10` | 모니터링 코인 수 |
| `BACKTEST_FEE_TAKER` | `0.0004` | Binance Taker 수수료 |
| `BACKTEST_FEE_MAKER` | `0.0002` | Binance Maker 수수료 |

---

## 19. 기술 스택

| 분류 | 라이브러리 / 도구 | 선택 이유 |
|------|------------------|-----------|
| **거래소 연동** | `python-binance 1.0.19` | Binance 공식 SDK, Futures 전용 메서드 |
| **데이터 처리** | `pandas 2.2`, `numpy 1.26`, `scipy 1.14` | 벡터 연산 + 통계 분석 |
| **설정 관리** | `pydantic-settings 2.6`, `python-dotenv` | 타입 안전 설정, 환경변수 자동 파싱 |
| **스케줄링** | `APScheduler 3.10` | 정확한 5분 주기 실행 |
| **데이터베이스** | SQLite WAL 모드 | 경량, 서버리스, 동시 읽기/쓰기 |
| **로깅** | `structlog 24.4` | 구조화 로그 (JSON 형태) |
| **알림** | `slack-bolt 1.21` (Socket Mode) | 실시간 양방향 Slack 통신 |
| **시각화** | `matplotlib 3.9`, `jinja2 3.1` | HTML 백테스트 보고서 |
| **테스트** | `pytest 8.3`, `pytest-asyncio` | 29개 테스트, 커버리지 측정 |
| **배포** | Fly.io, Docker (`python:3.11-slim`) | 컨테이너 기반 무중단 배포 |

---

## 20. 기술적 의사결정 기록

### Binance GTX (PostOnly) 주문 → 거절 시 Market 폴백

Binance의 `timeInForce="GTX"` 주문은 Maker 수수료를 보장하지만,
호가 변동으로 인해 즉시 Taker가 될 경우 에러 코드 `-5022`로 거절됩니다.

**해결**: `-5022` 에러 감지 시 즉시 시장가 주문으로 전환합니다.
지연이 허용되지 않는 시장 상황에서도 체결을 보장합니다.

### Binance SL/TP를 별도 주문으로 관리하는 이유

Bybit은 `set_trading_stop()` 하나의 API로 SL과 TP를 함께 등록하지만,
Binance는 `STOP_MARKET`(SL)과 `TAKE_PROFIT`(TP)을 각각 별도 주문으로 등록해야 합니다.

**SL**: `STOP_MARKET + closePosition=true` → 즉각 체결 (Taker)
**TP**: `TAKE_PROFIT + Limit + closePosition=true` → Maker 수수료 절약

### 거래소 TP를 TP2 가격으로 설정하는 이유

TP1 가격으로 등록하면 거래소가 도달 시 **전량 시장가 청산**합니다.
이로 인해 봇의 부분청산(60%/40%) 로직이 무력화됩니다.

**해결**: 거래소 TP는 TP2 가격으로 지정가 등록합니다.
TP1 부분청산은 봇의 5분 사이클이 처리하고,
봇 다운 시에는 TP2에서 안전하게 청산됩니다.

### StochRSI 임계값 표준화 (30/70)

초기 구현에서 `k < 50` 조건으로 과매도 신호를 발생시켜
RSI 50 이하 전체 구간(중립 포함)에서 반전 신호가 나오는 문제가 있었습니다.

**수정**: 과매도 `k < 30`, 과매수 `k > 70`으로 표준화.
Confidence = `(30 - k) / 30`으로 극단값에 가까울수록 높은 확신도 부여.
False signal이 대폭 감소하고 실제 과매도/과매수 구간에서만 신호 발생.

### 트레일링 스탑을 TP1 이후에만 활성화하는 이유

TP1 전에 트레일링을 활성화하면 ATR 노이즈 수준의 되돌림에서 손절이 발동됩니다.
TP1 이후 손절을 브레이크이븐으로 이동시킨 후 트레일링을 활성화하면
원금 리스크가 제거된 상태에서 추가 수익만 보호합니다.

---

## 주의사항

- **API 키 권한**: 발급 시 **출금(Withdrawal) 권한은 절대 부여하지 마십시오.**
- **API 시크릿 보안**: `.env` 파일은 절대 git에 커밋하지 마십시오.
- **단계적 전환**: `dry-run` → `paper` (최소 2주 검증) → `live` 순서로 전환하십시오.
- **Quality Gate 필수**: `TRADING_MODE=live` 전환은 반드시 백테스트 Quality Gate 통과 후 설정하십시오.
- **Hedge Mode**: Binance Futures 계정에서 Hedge Mode가 활성화되어 있어야 Long/Short 동시 포지션이 가능합니다.

---

## License

MIT
