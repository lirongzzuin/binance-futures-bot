# Binance USDT Perpetual Futures Trading Bot

자동화된 암호화폐 선물 거래 시스템. Binance USDT 무기한 선물 시장에서 **Dynamic Regime Router**를 통해 장세를 실시간 감지하고, 멀티 타임프레임(MTF) 신호 분석과 동적 레버리지 조절로 체계적인 스캘핑 및 트렌드 추종 거래를 수행합니다.

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| **Dynamic Regime Router** | 7개 장세 자동 감지 (STRONG_BULL/BULL/SIDEWAYS/BEAR/STRONG_BEAR/VOLATILE/TRANSITION) → 전략 자동 선택 |
| **4가지 전략 모드** | `auto` / `trend` / `scalp` / `swing` — 환경변수로 선택 |
| **멀티 타임프레임 분석** | 3분봉(LTF) + 15분봉(HTF) 신호 + 60분봉 리샘플 + 일봉 HTF 필터 |
| **동적 레버리지** | 신호 강도에 따라 2×~20× 자동 조절 + Kelly 사이징 |
| **3단계 드로우다운 방어** | REDUCE(21%) → PAUSE(42%) → HALT(50%) 점진적 리스크 축소 |
| **최소 손절 바닥(Min Stop Floor)** | ATR 기반 손절이 너무 타이트할 때 최소 거리 보장 |
| **헤지 모드** | Binance 헤지 모드 활용 — 롱/숏 동시 포지션 가능 |
| **상관관계 가드** | BTC/ETH 동시 포지션 노출 제한 (CorrelationGuard) |
| **방향별 쿨다운** | 손절 후 같은 방향 45분, 반대 방향 30분 대기 |
| **Slack 실시간 알림** | 진입/청산/일일 브리핑 — 모든 타임스탬프는 KST |

---

## 전략 모드 (`STRATEGY_MODE`)

### `auto` (기본값, 권장)

**Dynamic Regime Router** — 매 사이클마다 60분봉 리샘플 후 EMA21/50/200 + ADX + ATR 백분위로 7개 장세를 감지하고 최적 전략을 자동 선택합니다.

| 장세 | 조건 | 전략 | 방향 | 리스크 | 레버리지 상한 |
|------|------|------|------|--------|-------------|
| STRONG_BULL | EMA↑ ADX>27 | TREND_RIDE | LONG only | 100% | 20× |
| BULL | EMA↑ ADX≥20 | SCALP | LONG only | 90% | 15× |
| SIDEWAYS | 혼합 EMA | PURE_SCALP | 양방향 | 80% | 10× |
| BEAR | EMA↓ ADX≥20 | SCALP | SHORT only | 90% | 15× |
| STRONG_BEAR | EMA↓ ADX>27 | TREND_RIDE | SHORT only | 100% | 20× |
| VOLATILE | ATR pct>0.82 | PURE_SCALP | 양방향 | 50% | 8× |
| TRANSITION | 전환 확정 중 | — | 진입 없음 | — | — |

- 같은 그룹(BULL↔STRONG_BULL) 전환: 즉시 확정
- 다른 그룹 전환: 3회 연속 확인 후 확정 (디바운스)
- macro_bear=True 시 STRONG_BULL→BEAR, BULL→SIDEWAYS 오버라이드

### `trend`

**Phase-locked Direction System** — EMA21/50/200 + ADX(≥22) + DI+/DI- 3중 필터로 추세 방향을 확정하고, 확정된 방향으로만 진입합니다.
- BULL 페이즈: LONG only
- BEAR 페이즈: SHORT only
- SIDEWAYS: 양방향

### `scalp`

순수 스캘핑 전략 — 3분봉(LTF) + 15분봉(HTF) 기반. PURE_SCALP(타이트) + SCALP(넓은) 두 가지 모드를 자동 선택합니다.

### `swing`

기존 15분봉 기반 스윙/추세추종 전략 — ICT + 복합 신호 9개 카테고리 조합.

---

## 신호 감지 시스템 (9개 카테고리)

15분봉을 기준으로 아래 9가지 유형의 신호를 수집해 **방향(롱/숏)** 과 **강도(0~1)** 를 계산합니다.

| 카테고리 | 의미 | 지표 예시 |
|---------|------|----------|
| **추세(Trend)** | 현재 가격이 상승 추세인가? | EMA 9/21/50, ADX, 헐 이동평균 |
| **반전(Reversion)** | 과매수/과매도로 반전 임박? | RSI, StochRSI, Williams %R, RSI 다이버전스 |
| **변동성(Volatility)** | 변동성이 충분한가? | 볼린저밴드, 켈트너채널, ATR |
| **볼륨(Volume)** | 거래량이 방향을 지지하는가? | CMF(차이킨 머니플로우), OBV, 자금 유입/유출 |
| **ICT** | 스마트머니가 움직이는가? | 공정가치 갭(FVG), 오더블록, 시장 구조 변화 |
| **VWAP** | 현재 가격이 VWAP 대비 어디에 있나? | 거래량 가중 평균 가격 대비 위치 |
| **오더북** | 매수/매도 잔량 불균형이 있나? | 상위 5단계 매수/매도 잔량 비율 |
| **스퀴즈(Squeeze)** | 변동성 압축 후 폭발 직전인가? | 볼린저밴드 + 켈트너채널 스퀴즈 |
| **모멘텀(Momentum)** | 단기 모멘텀 방향은? | MACD, StochRSI 모멘텀 |

이 신호들을 **시장 상황(추세장/횡보장/고변동성)에 따라 가중치를 다르게** 결합해 최종 점수를 계산합니다.

### 일봉 HTF 필터 — "나침반" 역할

15분봉 신호가 나왔더라도, 일봉 차트의 방향과 반대라면 진입하지 않습니다.

```
롱 진입 가능 조건 :
  - 일봉 EMA50 > EMA200 (골든크로스 구간)
  - 일봉 RSI가 40~70 사이 (과매도/과매수 아닐 때)

숏 진입 가능 조건 :
  - 일봉 EMA50 < EMA200 (데드크로스 구간)
  - 일봉 RSI가 30~60 사이
```

---

## 거래 모드 (SCALP / TREND RIDE / PURE SCALP)

| 구분 | PURE_SCALP | SCALP | TREND RIDE |
|------|-----------|-------|-----------|
| 목표 | 초단기 빠른 수익 | 단기 수익 | 큰 추세 전체 포착 |
| 손절 | 1.2 × ATR | 1.5 × ATR | 2.5 × ATR |
| 1차 목표 | 2.0 × ATR | 2.0 × ATR | 3.0 × ATR |
| 2차 목표 | 4.5 × ATR | 4.5 × ATR | 8.0 × ATR |
| 1차 목표 청산 | 40% | 50% | 30% |
| 최대 보유 | 30봉 | 48봉 | 240봉 |
| 최소 손절 바닥 | 0.4% | 0.7% | 1.2% |

**최소 손절 바닥(Min Stop Floor)**: ATR이 매우 낮을 때 stop distance가 수수료 이하로 좁아지는 것을 방지합니다. 바닥이 적용되면 TP도 비례 확대하여 R:R 비율을 유지합니다.

---

## 동적 레버리지

```
신호 강도 ≥ 0.85 : stop 0.68× → 레버리지 ~15-20× (리스크 12%)
신호 강도 ≥ 0.80 : stop 0.73× → 레버리지 ~10-12× (리스크 10%)
신호 강도 ≥ 0.75 : stop 0.80× → 레버리지 ~7-9×  (리스크 7%)
신호 강도 ≥ 0.65 : stop 0.88× → 레버리지 ~4-5×  (리스크 5%)
기본값          : stop 1.00× → 레버리지 ~2-3×  (리스크 3%)
```

레버리지는 `base_leverage = trade_risk_pct / stop_distance_pct` 공식으로 계산됩니다.

### 변동성 적응형 스톱 (vol_stop_mult)

ATR < 0.3% 저변동성 구간에서 stop이 너무 좁아지는 것을 방지하기 위해 최대 2.0× 확장 적용.

### 적응형 Half-Kelly 사이징

최근 50개 거래 기반 승률/평균수익/평균손실로 Kelly Criterion을 산출하고, Half-Kelly(50%)를 적용합니다.
- 최소 20개 거래 이상이어야 활성화
- 신호 강도에 비례하여 스케일링

### 드로우다운 레버리지 축소

| 드로우다운 | 조건 | 동작 |
|-----------|------|------|
| ≤15% | 정상 | 레버리지 에스컬레이션 허용 |
| >15% | 주의 | 에스컬레이션 비활성화 |
| >21% | REDUCE | 리스크 0~75% 선형 축소 |
| >30% | Heavy REDUCE | 리스크 75~85% 축소 |
| >42% | PAUSE | 신규 진입 중단 |
| ≥50% | HALT | 전면 거래 중단 |

---

## 리스크 관리

| 규칙 | 값 | 동작 |
|------|-----|------|
| 1거래당 최대 리스크 | 자산의 2% | 레버리지 자동 조정 |
| 일일 손실 한도 | 8% | Slack 경고 |
| 주간 손실 한도 | 15% | Slack 경고 |
| 3단계 드로우다운 | 21%/42%/50% | REDUCE → PAUSE → HALT |
| 최대 동시 포지션 | 2개 | 자동 제한 |
| 최대 총 노출도 | 자산의 1500% | 레버리지 명목가 기준 |
| 심볼당 최대 노출 | 자산의 30% | 자동 제한 |
| 일일 심볼당 최대 거래 | 3회 | 과거래 방지 |
| 상관관계 가드 | BTC+ETH | 동시 노출 제한 |
| 스프레드 임계값 | 0.3% | 시장가 주문으로 전환 |

### 방향별 재진입 쿨다운

손절 발생 시:
- **같은 방향 재진입**: 45분 대기
- **반대 방향 재진입**: 30분 대기

### 수수료 모델

| 구분 | 수수료 | 용도 |
|------|--------|------|
| Maker (PostOnly) | 0.02% | 진입 (기본), TP 청산 |
| Taker (Market) | 0.04% | 손절, 긴급청산, 스프레드 넓을 때 |

- PostOnly 진입 → round_trip 약 0.080% → min TP1 = 0.160%
- Market 진입 → round_trip 약 0.120% → min TP1 = 0.240%
- PURE_SCALP: 수수료 안전계수 1.5×, SCALP/TREND: 2.0×

---

## 기술 아키텍처

```
src/
├── api/
│   ├── binance_client.py   # Binance API 래퍼 (Circuit Breaker 포함)
│   ├── slack.py            # Slack 알림 (KST 타임존)
│   └── slack_bot.py        # Slack Socket Mode (슬래시 커맨드)
├── data/
│   ├── collector.py        # OHLCV + 오더북 + 펀딩비 + OI 수집
│   └── universe.py         # 거래량 상위 코인 자동 선정
├── strategy/
│   ├── regime_router.py    # Dynamic Regime Router (7개 장세 감지)
│   ├── trend_ema.py        # TrendEMA 전략 (Phase-locked Direction)
│   ├── signals.py          # 9개 카테고리 신호 생성
│   ├── combiner.py         # 시장 상황별 가중치 결합
│   ├── regime.py           # 추세/횡보/고변동성 시장 판단
│   ├── macro_regime.py     # BTC 일봉 기반 매크로 레짐 감지
│   ├── scalp.py            # 스캘핑 vs 트렌드 모드 선택
│   ├── indicators.py       # 모든 기술적 지표 함수
│   └── ict.py              # ICT 스마트머니 분석
├── execution/
│   ├── state_machine.py    # 포지션 상태 관리 (방향별 쿨다운 포함)
│   └── order_manager.py    # Binance 주문 실행 (PostOnly/Market)
├── risk/
│   ├── engine.py           # 동적 레버리지 + Kelly 사이징 + Min Stop Floor
│   ├── guards.py           # 3단계 드로우다운 방어 시스템
│   ├── correlation.py      # 상관관계 가드 (BTC/ETH 동시 노출 제한)
│   └── profit_protector.py # 수익 자동 이체 (Spot 계좌)
├── reporting/
│   └── daily_briefing.py   # 하루 2회 브리핑 (09:00, 21:00 KST)
├── db/
│   └── database.py         # SQLite (WAL 모드) 거래 기록 + Kelly 통계
├── backtest/
│   └── engine.py           # 백테스트 엔진
└── main.py                 # 봇 오케스트레이터 (4-way 전략 라우팅)
```

### 핵심 설계 원칙

**Circuit Breaker**: API 실패 5회 연속 → 60초 쿨다운. 거래소 장애 시 주문 폭주 방지.

**PostOnly 주문**: 진입 시 메이커(0.02%) 수수료 우선 사용. 즉시 체결 시 새 client_id로 재시도. 스프레드 > 0.3% 또는 signal > 0.90 → 처음부터 시장가.

**트레일링 스톱**: TP1(1차 목표) 도달 후에만 활성화. `tp1_hit` 플래그 조건으로 조기 청산 방지.

**3분 사이클**: auto/trend/scalp 모드 기본 실행 간격 3분.

---

## 설정 (주요 환경변수)

```bash
# .env 파일
TRADING_MODE=live              # dry-run | paper | live
STRATEGY_MODE=auto             # auto | trend | scalp | swing
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_TESTNET=false
BINANCE_HEDGE_MODE=true

# 시그널 설정
SIGNAL_ENTRY_THRESHOLD=0.65
SIGNAL_EXIT_THRESHOLD=0.38
TREND_TIMEFRAME=1440           # 일봉 (HTF 필터)
TOP_N_COINS=10

# 자본 및 리스크
PAPER_CAPITAL_USDT=4000.0
MAX_RISK_PER_TRADE_PCT=0.02
MAX_LEVERAGE_CAP=20
BEAR_MAX_LEVERAGE_CAP=20
MAX_CONCURRENT_POSITIONS=2
MAX_TOTAL_EXPOSURE_PCT=15.0    # 레버리지 명목가 기준
MAX_SINGLE_EXPOSURE_PCT=0.30

# 손절/이익 실현
SCALP_STOP_ATR=1.5
SCALP_TP1_ATR=2.0
TREND_STOP_ATR=2.5
TREND_TP1_ATR=3.0

# 쿨다운
REENTRY_COOLDOWN_SECONDS=1800
REENTRY_SAME_DIR_COOLDOWN_SECONDS=2700

# Slack
SLACK_WEBHOOK_URL=https://...

QUALITY_GATE_PASSED=true
```

---

## 실행 방법

```bash
# 의존성 설치
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 드라이런 (실제 주문 없음)
TRADING_MODE=dry-run python -m src.main

# 페이퍼 트레이딩
TRADING_MODE=paper python -m src.main

# 실전 거래 (QUALITY_GATE_PASSED=true 필수)
TRADING_MODE=live python -m src.main
```

---

## Slack 슬래시 커맨드

| 커맨드 | 기능 |
|--------|------|
| `/dashboard` | 현재 포지션 + 잔고 + 수익 현황 |
| `/positions` | 열린 포지션 목록 |
| `/pnl` | 오늘/주간/전체 수익 |
| `/status` | 봇 상태 (업타임, 사이클, 회로차단기) |

---

## Slack 자동 알림 일정

| 알림 | 시간/조건 |
|------|----------|
| 브리핑 | 09:00 KST + 21:00 KST (하루 2회) |
| 포지션 진입 | 진입 즉시 |
| 포지션 청산 | 완전 청산 시 |
| 매크로 레짐 전환 | BULL ↔ BEAR/SIDEWAYS 전환 시 |
| 리스크 경고 | 일일/주간 손실 한도 초과 시 |
| 시스템 에러 | 예외 발생 시 |

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.13 |
| 거래소 SDK | python-binance 1.0.19 |
| 데이터 처리 | pandas 2.2, numpy 1.26 |
| 스케줄링 | APScheduler 3.10 |
| 설정 관리 | pydantic-settings 2.6 |
| 데이터베이스 | SQLite (WAL 모드) |
| 알림 | Slack Webhook + Socket Mode (Bot) |
| 배포 | fly.io (Docker) |
| 테스트 | pytest 8.3 (198개 테스트) |

---

## 테스트

```bash
python -m pytest tests/ -v          # 198개 테스트 실행
python -m pytest tests/ --cov=src   # 커버리지 포함
```

주요 테스트 범위:
- `test_order.py` — StateMachine 상태 전환, TP1/TP2, 트레일링, 방향별 쿨다운 (48개)
- `test_risk.py` — RiskEngine 레버리지 계산, 3단계 드로우다운, Min Stop Floor, Kelly (36개+)
- `test_scalping_strategy.py` — 스캘핑 전략 파라미터, 신호 스코어링 (50개+)
- `test_macro_regime.py` — 매크로 레짐 감지, 심볼 필터링 (8개)
