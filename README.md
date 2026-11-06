# Binance Futures Bot

Bybit 선물 봇(`bybit-futures-bot`)의 전략·리스크 엔진을 Binance USDT-M 무기한 선물로 옮긴 Python 자동매매 봇입니다. 거래소 계층(클라이언트·주문 타입·수수료·오류 코드)을 Binance 에 맞게 다시 작성하고, 나머지 구조는 공유합니다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![python-binance](https://img.shields.io/badge/python--binance-F0B90B?logo=binance&logoColor=black)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-198%20passed-0A9EDC?logo=pytest&logoColor=white)

> **투자 조언이 아닙니다.** 개인 학습·포트폴리오용 코드입니다. 레버리지 선물 거래는 원금 전액을 잃을 수 있으며, 이 코드는 수익을 보장하지 않습니다.

**상태:** 이식 작업 단계의 프로토타입 (커밋 4개, 2026-03 이후 업데이트 없음). 실거래 운영 기록은 이 저장소에 없습니다.

---

## 무엇을 해결하나

같은 전략을 다른 거래소에서 돌리려면 거래소마다 다른 부분만 갈아 끼울 수 있어야 합니다. 이 저장소는 그 경계를 실제로 나눠 본 결과물입니다.

- **거래소에 따라 바뀌는 것:** API 클라이언트, 주문 타입(Binance 의 `GTX` = PostOnly), 수수료율(Taker 0.04% / Maker 0.02%), 헤지 모드 설정, 오류 코드 처리
- **그대로 쓰는 것:** 장세 분류(Regime Router·매크로 레짐), 신호·필터, 리스크 엔진, 낙폭 가드, 포지션 상태 머신, 백테스트·품질 게이트

## 아키텍처

```
APScheduler
 └─ TradingBot.run_cycle()                        src/main.py
     ├─ MacroRegimeDetector / RegimeRouter        장세 분류 → 전략·허용 방향 선택
     ├─ trend · scalp · swing · auto 전략          src/strategy/
     ├─ RiskGuards / RiskEngine / CorrelationGuard src/risk/
     ├─ OrderManager                              GTX(PostOnly) 우선 → 거부 시 시장가 폴백
     ├─ StateMachine                              포지션 상태·부분 청산·거래소 동기화
     ├─ BinanceClient                             src/api/binance_client.py
     │    재시도(지수 백오프) + 서킷 브레이커 + 무해한 오류 코드 분류
     └─ SQLite + Slack Webhook / Socket Mode 명령어
```

## 기술적으로 집중한 문제

### 1. 거래소 클라이언트의 장애 격리
- **문제:** 거래소 API 가 연속으로 실패할 때 매 사이클 재시도를 쌓으면 레이트 리밋에 걸리고 상태가 꼬입니다. 반대로 "이미 체결된 주문 취소" 처럼 정상적인 상황에서 나는 오류까지 장애로 세면 불필요하게 멈춥니다.
- **해결:** `BinanceClient` 는 python-binance 를 감싸 레이트 리밋 오류를 지수 백오프로 재시도하고, 연속 5회 실패하면 60초 동안 호출을 막는 서킷 브레이커를 둡니다. `-2011`(알 수 없는 주문 취소), `-4046`(이미 헤지 모드) 같은 코드는 무해한 오류로 분류해 서킷 브레이커에 반영하지 않습니다.

### 2. 주문 타입별 수수료를 정확히 반영
- **문제:** 진입은 지정가로 했는데 청산은 시장가인 경우처럼 진입·청산 수수료가 다르면, 하나의 수수료율로 계산한 손익은 틀립니다.
- **해결:** 진입 수수료를 포지션에 기록해 두고 청산 사유별(손절·트레일링은 시장가, 익절·신호 청산은 지정가)로 수수료를 적용하며, 부분 청산 뒤 전량 청산 때는 진입 수수료를 비율로 나눠 계산합니다. `tests/test_fee_logic.py` 가 이 규칙을 하나씩 검증합니다.

### 3. 청산 뒤 잔여 주문 정리
- 포지션이 정상 청산된 것을 확인한 뒤에만 거래소에 남은 SL/TP 주문을 일괄 취소하고 내부 추적 상태도 정리합니다. 청산이 실패하면 SL/TP 보호를 그대로 둡니다. dry-run 모드에서는 API 를 호출하지 않습니다(`tests/test_stale_orders.py`).

### 4. Bybit 봇에서 가져온 리스크 관리
- 손절 거리로 레버리지를 역산하고 변동성·스프레드·펀딩비·계좌 낙폭으로 감쇠
- 수수료 손익분기 미달 진입 거부, 청산가가 손절 거리의 3배 이내면 진입 거부
- 낙폭 구간별 REDUCE → PAUSE → HALT 단계적 가드
- `TRADING_MODE=live` 는 백테스트 품질 게이트 통과(`QUALITY_GATE_PASSED=true`) 후에만 허용

## 품질 / 테스트

- `pytest` 6개 파일, **198개 테스트 통과** (Python 3.11 에서 실행해 확인).
- 대상: 리스크 엔진·가드, 주문 관리, 수수료 로직, 미체결 주문 정리, 스캘핑 전략, 매크로 레짐.

## 실행 방법

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # 기본값은 TRADING_MODE=dry-run, BINANCE_TESTNET=true

python -m pytest tests/ -q
python scripts/backtest_btc.py --days 365 --seed 42                    # 합성 데이터 백테스트
python -m src.backtest.engine --symbols BTCUSDT ETHUSDT --years 1      # 거래소 시세 백테스트
python -m src.main                                                     # 봇 실행
```

`Dockerfile` 과 `fly.toml` 로 컨테이너 배포를 할 수 있습니다.

## 설정

모든 설정은 환경변수로 주입되고 `src/config.py` 의 pydantic `Settings` 가 검증합니다. 항목별 설명은 `.env.example` 에 있습니다.

| 변수 | 설명 |
|---|---|
| `TRADING_MODE` | `dry-run` / `paper` / `live` |
| `STRATEGY_MODE` | `swing`(기본) / `scalp` / `trend` / `auto` |
| `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET`, `BINANCE_HEDGE_MODE` | Binance API (출금 권한은 주지 마세요) |
| `MAX_RISK_PER_TRADE_PCT`, `MAX_LEVERAGE_CAP`, `MAX_DRAWDOWN_PCT` 등 | 리스크 한도 |
| `QUALITY_GATE_*` | 백테스트 품질 기준과 실거래 잠금 |
| `SLACK_WEBHOOK_URL`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | 알림과 Slack 명령어 |

## 개발 방식

Claude Code 와 함께 설계·구현했습니다 (커밋의 `Co-Authored-By` 참고).
