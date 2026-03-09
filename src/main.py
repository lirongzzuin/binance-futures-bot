"""
Main bot entry point — Binance USDT Perpetual Futures.

Running modes:
  dry-run : no real orders, all logic simulated
  paper   : paper trading (fake capital, real signals)
  live    : real orders on Binance (requires quality gate)

Usage:
  python -m src.main
  TRADING_MODE=paper python -m src.main
  TRADING_MODE=live  python -m src.main     # only if QUALITY_GATE_PASSED=true
"""
from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

_KST = timezone(timedelta(hours=9))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.api.binance_client import BinanceClient, CircuitOpenError
from src.api.slack import SlackNotifier
from src.config import settings
from src.data.collector import DataCollector
from src.data.universe import UniverseManager
from src.db.database import Database
from src.execution.order_manager import OrderManager
from src.execution.state_machine import StateMachine
from src.logger import get_logger
from src.reporting.daily_briefing import DailyBriefing
from src.risk.correlation import CorrelationGuard
from src.risk.engine import RiskEngine
from src.risk.guards import GuardAction, RiskGuards
from src.risk.profit_protector import ProfitProtector
from src.strategy.combiner import SignalCombiner
from src.strategy.ict import ICTAnalyzer
from src.strategy.indicators import volatility_percentile
from src.strategy.macro_regime import MacroRegimeDetector, MacroRegime
from src.strategy.regime import RegimeDetector, Regime
from src.strategy.scalp import TradeModeDetector
from src.strategy.scalping import ScalpingStrategy
from src.strategy.trend_ema import TrendEMAStrategy
from src.strategy.signals import SignalGenerator
from src.strategy.regime_router import RegimeRouter, DynamicRegime

log = get_logger(__name__)


class TradingBot:
    def __init__(self):
        settings.check_live_allowed()

        self._client = BinanceClient(settings)
        self._slack = SlackNotifier(settings.SLACK_WEBHOOK_URL)
        self._db = Database(settings)
        self._universe = UniverseManager(self._client, settings)
        self._collector = DataCollector(self._client, settings)
        self._regime = RegimeDetector(settings)
        self._ict = ICTAnalyzer(settings)
        self._sig_gen = SignalGenerator(settings)
        self._combiner = SignalCombiner(settings)
        self._mode_detector = TradeModeDetector(settings)
        self._scalp_strategy = ScalpingStrategy(settings)
        self._trend_strategy = TrendEMAStrategy(settings)
        self._regime_router  = RegimeRouter(settings)
        self._risk_engine = RiskEngine(settings)
        self._guards = RiskGuards(settings)
        self._correlation = CorrelationGuard(settings.MAX_TOTAL_EXPOSURE_PCT)
        self._state_machine = StateMachine(settings)
        self._order_mgr = OrderManager(self._client, settings, self._slack)
        self._briefing = DailyBriefing(
            settings, self._client, self._db, self._slack, self._state_machine
        )
        self._profit_protector = ProfitProtector(
            client=self._client,
            settings=settings,
            slack=self._slack,
            initial_capital=settings.PAPER_CAPITAL_USDT,
        )

        self._macro_detector = MacroRegimeDetector(self._client, settings)
        self._macro_mode: str = "BEAR_SIDEWAYS"  # conservative default
        self._macro_bull_streak: int = 0
        self._macro_bear_streak: int = 0

        # Auto regime router per-symbol cache (STRATEGY_MODE=auto)
        self._auto_risk_multiplier: Dict[str, float] = {}
        self._auto_leverage_cap: Dict[str, int] = {}
        self._auto_direction_bias: Dict[str, int] = {}

        self._paper_equity = settings.PAPER_CAPITAL_USDT
        self._start_ts: float = time.time()
        self._last_cycle_ts: float = 0.0
        self._daily_loss_alert_sent_date: Optional[str] = None

        log.info(
            "TradingBot ready | mode=%s testnet=%s",
            settings.TRADING_MODE,
            settings.BINANCE_TESTNET,
        )

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self) -> None:
        try:
            self._cycle()
            self._last_cycle_ts = time.time()
        except CircuitOpenError:
            log.error("Circuit breaker open — skipping cycle")
        except Exception as e:
            log.exception("Unexpected error in cycle: %s", e)

    def _cycle(self) -> None:
        symbols = self._universe.get_or_refresh()
        if not symbols:
            log.warning("Empty universe — skipping cycle")
            return

        symbols = self._filter_symbols_by_macro_regime(symbols)

        equity = self._get_equity()
        daily_pnl = self._db.get_today_pnl()
        weekly_pnl = self._db.get_weekly_pnl()
        peak_equity = self._db.get_peak_equity(settings.TRADING_MODE) or equity

        if settings.is_live():
            self._sync_positions(symbols)

        for symbol in symbols:
            try:
                self._process_symbol(
                    symbol=symbol,
                    equity=equity,
                    daily_pnl=daily_pnl,
                    weekly_pnl=weekly_pnl,
                    peak_equity=peak_equity,
                )
            except Exception as e:
                log.error("Error processing %s: %s", symbol, e)
                continue

        try:
            transferred = self._profit_protector.check_and_protect(equity)
            if transferred:
                log.info("Profit protection: %.2f USDT moved to Spot Account", transferred)
                if settings.is_paper() or settings.is_dry_run():
                    self._paper_equity -= transferred
        except Exception as e:
            log.error("Profit protector error: %s", e)

        self._db.record_equity(equity, settings.TRADING_MODE)

    def _process_symbol(
        self,
        symbol: str,
        equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        peak_equity: float,
    ) -> None:
        # ── 전략 모드에 따라 데이터 수집 및 신호 생성 분기 ────────────────────
        if settings.STRATEGY_MODE == "scalp":
            result = self._collect_and_signal_scalp(symbol)
        elif settings.STRATEGY_MODE == "trend":
            result = self._collect_and_signal_trend(symbol)
        elif settings.STRATEGY_MODE == "auto":
            result = self._collect_and_signal_auto(symbol)
        else:
            result = self._collect_and_signal_swing(symbol)

        if result is None:
            return

        snap, combined, atr_val, regime_r, trade_mode_str = result
        if snap.ohlcv.empty or snap.last_price <= 0:
            return

        current_price = snap.last_price

        close = snap.ohlcv["close"]
        vol_pct_s = volatility_percentile(close)
        vol_pct = float(vol_pct_s.iloc[-1]) if not vol_pct_s.empty else 0.5

        guard = self._guards.check(
            equity=equity,
            peak_equity=peak_equity,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            initial_capital=settings.PAPER_CAPITAL_USDT,
            spread_pct=snap.orderbook.spread_pct,
            vol_percentile=vol_pct,
        )

        if guard.action == GuardAction.HALT:
            log.warning("Guard HALT: %s — closing all positions", guard.reason)
            self._emergency_close_all(symbols=[symbol])
            return

        # 일일 손실 한도 초과 시 Slack 알림 (하루 1회, 거래는 계속)
        if guard.reason and "Daily loss" in guard.reason:
            today_kst = datetime.now(tz=_KST).date().isoformat()
            if self._daily_loss_alert_sent_date != today_kst:
                self._daily_loss_alert_sent_date = today_kst
                loss_pct = abs(daily_pnl) / max(equity, 1) * 100
                self._slack.send_daily_loss_limit(loss_pct)

        pos = self._state_machine.get_position(symbol)

        if not pos.is_flat:
            # 포지션 보유 중 — 바 수 계산: trade_mode에 따라 타임프레임 결정
            if pos.trade_mode == "PURE_SCALP":
                bar_seconds = int(settings.SCALP_PURE_LTF_TIMEFRAME) * 60
            else:
                bar_seconds = int(settings.SIGNAL_TIMEFRAME) * 60
            bars_in_trade = int((time.time() - pos.entry_ts) / bar_seconds)

            regime_flip = (
                (pos.is_long and regime_r.regime == Regime.TRENDING_DOWN) or
                (pos.is_short and regime_r.regime == Regime.TRENDING_UP)
            )

            exit_check = self._state_machine.check_exit(
                symbol=symbol,
                current_price=current_price,
                current_atr=atr_val,
                signal_direction=combined.direction,
                regime_flip=regime_flip,
                bars_in_trade=bars_in_trade,
                signal_strength=combined.strength,
                exit_threshold=settings.SIGNAL_EXIT_THRESHOLD,
            )

            if exit_check.should_exit:
                if exit_check.partial:
                    self._execute_partial_exit(symbol, pos, current_price, exit_check, snap)
                else:
                    self._execute_exit(symbol, pos, current_price, exit_check.reason.value, snap)
            return

        if guard.action == GuardAction.PAUSE:
            log.debug("Guard PAUSE: %s — 신규 진입 건너뜀", guard.reason)
            return

        # ── Recovery mode: REDUCE 구간에서 저품질 신호 거부 ─────────────────
        if guard.action == GuardAction.REDUCE and guard.reduce_pct > 0.60:
            if combined.strength < 0.77:
                log.debug(
                    "Recovery REDUCE mode: skipping weak signal "
                    "(strength=%.2f reduce_pct=%.2f)",
                    combined.strength, guard.reduce_pct,
                )
                return

        if combined.direction == 0:
            return

        # ── 방향 인식 쿨다운 체크 ────────────────────────────────────────
        if not self._state_machine.can_enter(symbol, direction=combined.direction):
            return

        # ── 일일 거래 횟수 제한 ──────────────────────────────────────────
        daily_trades = self._db.get_trades_count_today(symbol=symbol)
        if daily_trades >= settings.MAX_TRADES_PER_DAY_PER_SYMBOL:
            log.debug(
                "Daily trade cap reached for %s (%d/%d)",
                symbol, daily_trades, settings.MAX_TRADES_PER_DAY_PER_SYMBOL,
            )
            return

        sym_info = self._client.get_symbol_info(symbol)
        if not sym_info:
            return

        consec = self._db.get_consecutive_losses(symbol)

        # ── Rolling win stats for adaptive Kelly ─────────────────────────
        win_stats = self._db.get_rolling_win_stats(window=50)

        is_bear_mode = (self._macro_mode == "BEAR_SIDEWAYS")
        risk_params = self._risk_engine.calculate(
            equity=equity,
            entry_price=current_price,
            side="Buy" if combined.direction == 1 else "Sell",
            atr=atr_val,
            signal_strength=combined.strength,
            spread_pct=snap.orderbook.spread_pct,
            funding_rate=snap.funding_rate,
            vol_percentile=vol_pct,
            drawdown_pct=(peak_equity - equity) / peak_equity if peak_equity > 0 else 0,
            consecutive_losses=consec,
            symbol_max_leverage=sym_info.get("max_leverage", 125),
            symbol_info=sym_info,
            current_positions=self._state_machine.open_count(),
            trade_mode=trade_mode_str,
            bear_mode=is_bear_mode,
            win_stats=win_stats,
        )

        if not risk_params.risk_ok:
            log.debug("Risk check failed for %s: %s", symbol, risk_params.reason)
            return

        # ── Correlation Guard: 상관 자산 과다 노출 방지 ──────────────────
        corr_allowed, corr_scale = self._correlation.check_entry(
            symbol=symbol,
            direction=combined.direction,
            notional=risk_params.position_size_usdt,
            positions=self._state_machine.all_positions(),
            equity=equity,
        )
        if not corr_allowed:
            log.debug("Correlation guard blocked %s entry", symbol)
            return
        if corr_scale < 1.0:
            risk_params.qty_base *= corr_scale
            risk_params.position_size_usdt *= corr_scale
            risk_params.margin_usdt *= corr_scale
            risk_params.max_loss_usdt *= corr_scale

        # ── Auto mode: 레짐 라우터 리스크 스케일 적용 ────────────────────────
        if settings.STRATEGY_MODE == "auto":
            auto_bias = self._auto_direction_bias.get(symbol, 0)
            if auto_bias == +1 and combined.direction == -1:
                log.debug("Auto regime: SHORT 신호 차단 (LONG-only 레짐) %s", symbol)
                return
            elif auto_bias == -1 and combined.direction == +1:
                log.debug("Auto regime: LONG 신호 차단 (SHORT-only 레짐) %s", symbol)
                return

            rm = self._auto_risk_multiplier.get(symbol, 1.0)
            lc = self._auto_leverage_cap.get(symbol, int(settings.MAX_LEVERAGE_CAP))

            if rm < 1.0:
                risk_params.qty_base           *= rm
                risk_params.position_size_usdt *= rm
                risk_params.margin_usdt        *= rm
                risk_params.max_loss_usdt      *= rm

            if lc < risk_params.leverage:
                risk_params.leverage = lc

        # ── Recovery mode: 포지션 비중 축소 적용 ────────────────────────────
        if guard.action == GuardAction.REDUCE and guard.reduce_pct > 0:
            scale = max(0.10, 1.0 - guard.reduce_pct)
            risk_params.qty_base *= scale
            risk_params.position_size_usdt *= scale
            risk_params.margin_usdt *= scale
            risk_params.max_loss_usdt *= scale
            log.info(
                "Recovery REDUCE mode: %s position scaled to %.0f%% "
                "(dd_reduce_pct=%.2f)",
                symbol, scale * 100, guard.reduce_pct,
            )

        log.info(
            "Signal %s %s strength=%.2f regime=%s strategy=%s",
            symbol,
            "Buy" if combined.direction == 1 else "Sell",
            combined.strength,
            regime_r.regime.value,
            settings.STRATEGY_MODE,
        )

        self._execute_entry(symbol, combined.direction, risk_params, snap, sym_info, combined.strength)

    # ── 전략별 신호 수집 메서드 ───────────────────────────────────────────────

    def _collect_and_signal_swing(self, symbol: str):
        """스윙 전략: 15분봉 신호 + 일봉 추세 필터."""
        snap = self._collector.collect(symbol, limit=250)
        if snap.ohlcv.empty or len(snap.ohlcv) < 100:
            log.warning("Insufficient data for %s (%d bars)", symbol, len(snap.ohlcv))
            return None

        df = snap.ohlcv
        regime_df = snap.trend_ohlcv if (
            snap.trend_ohlcv is not None and len(snap.trend_ohlcv) >= 60
        ) else df
        regime_r = self._regime.detect(regime_df, oi_change_pct=snap.oi_change_pct)
        ict_r = self._ict.analyze(df)
        sig_bundle = self._sig_gen.generate(
            df, ict_r,
            funding_rate=snap.funding_rate,
            oi_change_pct=snap.oi_change_pct,
            spread_pct=snap.orderbook.spread_pct,
            vwap=snap.vwap,
            vwap_distance_pct=snap.vwap_distance_pct,
            orderbook_imbalance=snap.orderbook.imbalance,
            htf_df=snap.trend_ohlcv,
        )
        combined = self._combiner.combine(sig_bundle, regime_r)

        mode_result = self._mode_detector.detect(
            regime=regime_r,
            ict_result=ict_r,
            bundle=sig_bundle,
            funding_rate=snap.funding_rate,
            spread_pct=snap.orderbook.spread_pct,
            direction=combined.direction,
        )
        trade_mode_str = mode_result.mode.value

        return snap, combined, sig_bundle.atr_val, regime_r, trade_mode_str

    def _collect_and_signal_scalp(self, symbol: str):
        """스캘핑 전략: 3분봉 신호 + 15분봉 추세 필터."""
        from src.strategy.regime import RegimeResult

        snap = self._collector.collect_for_scalp(symbol, limit=500)
        if snap.ohlcv.empty or len(snap.ohlcv) < 50:
            log.warning("Scalp: insufficient LTF data for %s (%d bars)", symbol, len(snap.ohlcv))
            return None

        combined, atr_val = self._scalp_strategy.generate(
            ltf_df=snap.ohlcv,
            htf_df=snap.trend_ohlcv,
            vwap_val=snap.vwap,
            funding_rate=snap.funding_rate,
            spread_pct=snap.orderbook.spread_pct,
            oi_change_pct=snap.oi_change_pct,
        )

        regime_r = RegimeResult(
            regime=combined.regime,
            adx_val=0.0,
            plus_di=0.0,
            minus_di=0.0,
            vol_percentile=0.5,
            ema_aligned_up=(combined.direction == 1),
            ema_aligned_down=(combined.direction == -1),
            volume_ratio=1.0,
            trend_score=combined.long_score - combined.short_score,
            confidence=combined.strength,
        )

        return snap, combined, atr_val, regime_r, "PURE_SCALP"

    def _collect_and_signal_trend(self, symbol: str):
        """
        Trend EMA 전략: Phase-locked Direction System.
        3분봉(LTF) + 15분봉(HTF) 데이터로 장세 판단 후 방향 고정 진입.
        """
        from src.strategy.regime import RegimeResult, Regime

        snap = self._collector.collect_for_scalp(symbol, limit=500)
        if snap.ohlcv.empty or len(snap.ohlcv) < 50:
            log.warning("Trend: Insufficient LTF data for %s (%d bars)", symbol, len(snap.ohlcv))
            return None

        combined, atr_val = self._trend_strategy.generate(
            ltf_df=snap.ohlcv,
            htf_df=snap.trend_ohlcv,
            vwap_val=snap.vwap,
            funding_rate=snap.funding_rate,
            spread_pct=snap.orderbook.spread_pct,
            oi_change_pct=snap.oi_change_pct,
        )

        regime_r = RegimeResult(
            regime=combined.regime,
            confidence=combined.strength,
            adx_val=0.0,
            plus_di=0.0,
            minus_di=0.0,
            ema_aligned_up=(combined.direction == 1),
            ema_aligned_down=(combined.direction == -1),
            vol_percentile=0.5,
            volume_ratio=1.0,
            trend_score=combined.long_score - combined.short_score,
        )

        trade_mode_str = "SCALP" if combined.regime != Regime.RANGING else "PURE_SCALP"
        return snap, combined, atr_val, regime_r, trade_mode_str

    def _collect_and_signal_auto(self, symbol: str):
        """
        Dynamic Regime Router 모드 — 매 사이클 장세를 자동 감지하고 최적 전략을 선택.
        """
        from src.strategy.regime import RegimeResult, Regime

        snap = self._collector.collect_for_scalp(symbol, limit=500)
        if snap.ohlcv.empty or len(snap.ohlcv) < 50:
            log.warning("Auto: LTF 데이터 부족 %s (%d봉)", symbol, len(snap.ohlcv))
            return None

        # ── 레짐 감지 ────────────────────────────────────────────────────────
        is_macro_bear = (self._macro_mode == "BEAR_SIDEWAYS")
        router_result = self._regime_router.detect(
            symbol=symbol,
            mtf_df=snap.trend_ohlcv,
            macro_bear=is_macro_bear,
        )

        log.info(
            "Auto regime %s: %s → strategy=%s mode=%s bias=%+d "
            "risk_mult=%.2f lev_cap=%d (conf=%.2f | %s)",
            symbol, router_result.regime.value,
            router_result.strategy, router_result.trade_mode,
            router_result.direction_bias,
            router_result.risk_multiplier, router_result.leverage_cap,
            router_result.confidence, router_result.reason,
        )

        # TRANSITION: 레짐 전환 확정 중 → 진입 없음
        if router_result.strategy == "skip":
            return None

        # ── 심볼별 리스크 파라미터 저장 ─────────────────────────────────────
        self._auto_risk_multiplier[symbol] = router_result.risk_multiplier
        self._auto_leverage_cap[symbol]    = router_result.leverage_cap
        self._auto_direction_bias[symbol]  = router_result.direction_bias

        # ── 전략 라우팅 ───────────────────────────────────────────────────────
        if router_result.strategy == "trend":
            combined, atr_val = self._trend_strategy.generate(
                ltf_df=snap.ohlcv,
                htf_df=snap.trend_ohlcv,
                vwap_val=snap.vwap,
                funding_rate=snap.funding_rate,
                spread_pct=snap.orderbook.spread_pct,
                oi_change_pct=snap.oi_change_pct,
            )
        else:
            combined, atr_val = self._scalp_strategy.generate(
                ltf_df=snap.ohlcv,
                htf_df=snap.trend_ohlcv,
                vwap_val=snap.vwap,
                funding_rate=snap.funding_rate,
                spread_pct=snap.orderbook.spread_pct,
                oi_change_pct=snap.oi_change_pct,
            )

        # ── 방향 바이어스 1차 필터 ────────────────────────────────────────────
        if router_result.direction_bias != 0 and combined.direction != 0:
            if combined.direction != router_result.direction_bias:
                log.debug(
                    "Auto: %s %s 신호 차단 (레짐=%s, bias=%+d)",
                    symbol,
                    "LONG" if combined.direction == 1 else "SHORT",
                    router_result.regime.value,
                    router_result.direction_bias,
                )
                return None

        # ── regime_r 생성 (check_exit 레짐플립 감지용) ───────────────────────
        if combined.direction == 1:
            regime_obj = Regime.TRENDING_UP
        elif combined.direction == -1:
            regime_obj = Regime.TRENDING_DOWN
        else:
            regime_obj = Regime.RANGING

        regime_r = RegimeResult(
            regime=regime_obj,
            confidence=router_result.confidence,
            adx_val=0.0,
            plus_di=0.0,
            minus_di=0.0,
            ema_aligned_up=(router_result.direction_bias == +1),
            ema_aligned_down=(router_result.direction_bias == -1),
            vol_percentile=0.5,
            volume_ratio=1.0,
            trend_score=float(router_result.direction_bias),
        )

        return snap, combined, atr_val, regime_r, router_result.trade_mode

    # ── Order Execution ───────────────────────────────────────────────────────

    def _execute_entry(
        self, symbol, direction, risk_params, snap, sym_info, signal_strength: float = 0.7
    ) -> None:
        order = self._order_mgr.open_position(
            symbol=symbol,
            direction=direction,
            risk_params=risk_params,
            spread_pct=snap.orderbook.spread_pct,
            signal_strength=signal_strength,
            current_price=snap.last_price,
            symbol_info=sym_info,
        )

        if order and order.status.value in ("FILLED", "PENDING"):
            entry_price = order.avg_price or snap.last_price
            qty = float(order.qty)

            # 진입 수수료: MARKET(시장가/GTX폴백)=Taker 0.04%, LIMIT(GTX성공)=Maker 0.02%
            entry_fee_rate = (
                settings.BACKTEST_FEE_TAKER if order.order_type == "MARKET"
                else settings.BACKTEST_FEE_MAKER
            )
            entry_fee = qty * entry_price * entry_fee_rate

            trade_id = self._db.insert_trade_open(
                symbol=symbol,
                side="Buy" if direction == 1 else "Sell",
                entry_price=entry_price,
                qty=qty,
                leverage=risk_params.leverage,
                mode=settings.TRADING_MODE,
            )
            self._state_machine.enter(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                qty=qty,
                leverage=risk_params.leverage,
                stop_price=risk_params.stop_price,
                take_profit_price=risk_params.take_profit_price,
                trailing_distance=risk_params.trailing_stop_atr,
                order_id=order.client_id,
                reason=f"signal_strength={signal_strength:.2f}",
                tp1_price=risk_params.tp1_price,
                tp2_price=risk_params.tp2_price,
                trade_mode=risk_params.trade_mode,
                mode_max_bars=risk_params.mode_max_bars,
                partial_exit_pct=risk_params.partial_exit_pct,
                entry_fee=entry_fee,
            )
            self._db.set_state(f"trade_id:{symbol}", trade_id)

            self._slack.send_position_opened(
                symbol=symbol,
                side="Buy" if direction == 1 else "Sell",
                entry_price=entry_price,
                qty=float(order.qty),
                leverage=risk_params.leverage,
                stop_price=risk_params.stop_price,
                take_profit_price=risk_params.take_profit_price,
                signal_strength=signal_strength,
                order_type=order.order_type,
            )

    def _execute_partial_exit(self, symbol, pos, current_price, exit_signal, snap) -> None:
        qty_to_close = pos.qty * pos.partial_exit_pct
        sym_info = self._client.get_symbol_info(symbol)
        direction = 1 if pos.is_long else -1

        order = self._order_mgr.close_position(
            symbol=symbol,
            direction=direction,
            qty=qty_to_close,
            current_price=current_price,
            symbol_info=sym_info or {},
            reason="PARTIAL_TP1",
        )

        if order:
            exit_price = order.avg_price or current_price
            partial_pct = pos.partial_exit_pct

            # entry_fee pro-rata: 닫히는 비율만큼의 진입 수수료
            entry_fee_for_partial = pos.entry_fee * partial_pct

            pnl = self._state_machine.partial_exit(
                symbol, qty_to_close, exit_price, partial_pct=partial_pct
            )

            notional = qty_to_close * exit_price
            exit_fee_rate = (
                settings.BACKTEST_FEE_TAKER if order.order_type == "MARKET"
                else settings.BACKTEST_FEE_MAKER
            )
            exit_fee = notional * exit_fee_rate
            total_fee = entry_fee_for_partial + exit_fee
            pnl_net = pnl - total_fee

            trade_id = self._db.get_state(f"trade_id:{symbol}")
            if trade_id:
                partial_trade_id = self._db.insert_trade_open(
                    symbol=symbol,
                    side="Buy" if direction == 1 else "Sell",
                    entry_price=pos.entry_price,
                    qty=qty_to_close,
                    leverage=pos.leverage,
                    mode=settings.TRADING_MODE,
                )
                self._db.update_trade_close(
                    trade_id=partial_trade_id,
                    exit_price=exit_price,
                    pnl=pnl_net,
                    fee=total_fee,
                    reason="PARTIAL_TP1",
                )

            log.info(
                "Partial TP1 exit %s | closed=%.4f gross_pnl=%.4f fee=%.4f net_pnl=%.4f stop→breakeven",
                symbol, qty_to_close, pnl, total_fee, pnl_net,
            )
            # PARTIAL_TP1은 Slack 알림 없음 (노이즈 감소 — /dashboard로 확인)

    def _execute_exit(self, symbol, pos, current_price, reason, snap) -> None:
        sym_info = self._client.get_symbol_info(symbol)
        direction = 1 if pos.is_long else -1
        order = self._order_mgr.close_position(
            symbol=symbol,
            direction=direction,
            qty=pos.qty,
            current_price=current_price,
            symbol_info=sym_info or {},
            reason=reason,
        )

        if order:
            exit_price = order.avg_price or current_price
            entry_fee = pos.entry_fee   # 남아있는 entry_fee (partial exit 후 pro-rata)
            entry = pos.entry_price
            remaining_qty = pos.qty
            pos_leverage = pos.leverage   # capture before exit() resets to 1

            pnl = self._state_machine.exit(symbol, exit_price)

            # 포지션이 정상 청산된 후에만 거래소 잔여 SL/TP 주문 정리
            # (close_position 성공 여부 확인 후 호출 — 실패 시엔 SL/TP 보호 유지)
            self._order_mgr.cancel_stale_orders(symbol)

            notional = remaining_qty * exit_price
            # 실제 체결된 주문 타입 기반: MARKET=Taker 0.04%, LIMIT=Maker 0.02%
            # (STOP_LOSS/TRAILING_STOP/RISK_GUARD → market, TP/SIGNAL_FLIP → limit)
            fee_rate = (
                settings.BACKTEST_FEE_TAKER if order.order_type == "MARKET"
                else settings.BACKTEST_FEE_MAKER
            )
            exit_fee = notional * fee_rate
            total_fee = entry_fee + exit_fee
            pnl_net = pnl - total_fee

            trade_id = self._db.get_state(f"trade_id:{symbol}")
            if trade_id:
                self._db.update_trade_close(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    pnl=pnl_net,
                    fee=total_fee,
                    reason=reason,
                )

            side_str = "Buy" if direction == 1 else "Sell"
            margin_used = (entry * remaining_qty) / pos_leverage if pos_leverage > 0 else (entry * remaining_qty)
            pnl_pct = pnl_net / margin_used if margin_used > 0 else 0
            log.info(
                "Closed %s %s ×%d | entry=%.4f exit=%.4f pnl_net=%.4f pnl_pct=%.2f%%",
                symbol, side_str, pos_leverage, entry, exit_price, pnl_net, pnl_pct * 100,
            )
            self._slack.send_position_closed(
                symbol=symbol,
                side=side_str,
                reason=reason,
                pnl=pnl_net,
                pnl_pct=pnl_pct,
                entry_price=entry,
                exit_price=exit_price,
                leverage=pos_leverage,
            )

    # ── Macro Regime ──────────────────────────────────────────────────────────

    def _filter_symbols_by_macro_regime(self, symbols: List[str]) -> List[str]:
        """매크로 레짐에 따라 허용 심볼 필터링."""
        if self._macro_mode == "BULL":
            bull_syms = [s.strip() for s in settings.BULL_MODE_SYMBOLS.split(",") if s.strip()]
            if bull_syms:
                return bull_syms
            return symbols  # BULL_MODE_SYMBOLS 미설정 → 전체 유니버스

        # BEAR_SIDEWAYS: BTC+ETH + 오픈 포지션 심볼
        allowed = [s.strip() for s in settings.BEAR_MODE_SYMBOLS.split(",") if s.strip()]
        open_syms = [p.symbol for p in self._state_machine.all_positions()]
        result = list(dict.fromkeys(allowed + open_syms))  # 중복 제거, 순서 유지

        # 유니버스에 없어도 BTC/ETH는 강제 포함
        for force_sym in allowed:
            if force_sym not in result:
                result.append(force_sym)

        log.debug(
            "MacroRegime BEAR_SIDEWAYS: allowed=%s (universe was %d symbols)",
            result, len(symbols),
        )
        return result

    def update_macro_regime(self) -> None:
        """매크로 레짐 업데이트 (스케줄러에서 주기적 호출)."""
        try:
            result = self._macro_detector.detect()
            is_bull = (result.regime == MacroRegime.BULL)

            if is_bull:
                self._macro_bull_streak += 1
                self._macro_bear_streak = 0
            else:
                self._macro_bear_streak += 1
                self._macro_bull_streak = 0

            prev_mode = self._macro_mode

            if is_bull and self._macro_bull_streak >= settings.MACRO_BULL_CONFIRM_COUNT:
                new_mode = "BULL"
            elif not is_bull and self._macro_bear_streak >= settings.MACRO_BEAR_CONFIRM_COUNT:
                new_mode = "BEAR_SIDEWAYS"
            else:
                new_mode = self._macro_mode  # 아직 전환 기준 미충족

            if new_mode != prev_mode:
                self._macro_mode = new_mode
                lev_cap = settings.MAX_LEVERAGE_CAP if new_mode == "BULL" else settings.BEAR_MAX_LEVERAGE_CAP
                log.info(
                    "MacroRegime 전환: %s → %s (streak=%d) lev_cap=%d×",
                    prev_mode, new_mode,
                    self._macro_bull_streak if is_bull else self._macro_bear_streak,
                    lev_cap,
                )
                self._slack.send_market_mode_changed(
                    from_mode=prev_mode,
                    to_mode=result.regime.value,
                    reason=result.reason,
                    symbols=result.symbols,
                    leverage_cap=lev_cap,
                )
        except Exception as e:
            log.error("update_macro_regime error: %s", e)

    def _emergency_close_all(self, symbols: List[str]) -> None:
        for symbol in symbols:
            pos = self._state_machine.get_position(symbol)
            if pos.is_flat:
                continue
            snap = self._collector.collect(symbol, limit=5)
            price = snap.last_price or pos.entry_price
            sym_info = self._client.get_symbol_info(symbol)
            self._execute_exit(symbol, pos, price, "RISK_GUARD_HALT", snap)
            # _execute_exit 내부에서 cancel_stale_orders 호출됨

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def get_dashboard_data(self) -> Dict:
        equity = self._get_equity()
        initial_capital = settings.PAPER_CAPITAL_USDT
        daily_pnl = self._db.get_today_pnl()
        weekly_pnl = self._db.get_weekly_pnl()
        total_pnl = self._db.get_total_pnl(mode=settings.TRADING_MODE)
        trades_today = self._db.get_trades_count_today()
        win_rate = self._db.get_win_rate(days=30)

        positions: List[Dict] = []
        if settings.is_live():
            try:
                for p in self._client.get_positions():
                    if float(p.get("size", 0)) <= 0:
                        continue
                    positions.append({
                        "symbol": p.get("symbol", ""),
                        "side": p.get("side", "Buy"),
                        "qty": float(p.get("size", 0)),
                        "entry_price": float(p.get("avgPrice", 0)),
                        "mark_price": float(p.get("markPrice", 0)),
                        "unrealised_pnl": float(p.get("unrealisedPnl", 0)),
                        "leverage": int(float(p.get("leverage", 1))),
                    })
            except Exception as e:
                log.error("Dashboard: failed to fetch positions: %s", e)
        else:
            for symbol in (self._universe.get_or_refresh() or []):
                pos = self._state_machine.get_position(symbol)
                if pos.is_flat:
                    continue
                positions.append({
                    "symbol": symbol,
                    "side": "Buy" if pos.is_long else "Sell",
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "mark_price": pos.entry_price,
                    "unrealised_pnl": pos.unrealised_pnl,
                    "leverage": pos.leverage,
                })

        return {
            "equity": equity,
            "initial_capital": initial_capital,
            "daily_pnl": daily_pnl,
            "weekly_pnl": weekly_pnl,
            "total_pnl": total_pnl,
            "positions": positions,
            "mode": settings.TRADING_MODE,
            "trades_today": trades_today,
            "win_rate": win_rate,
            "uptime_seconds": time.time() - self._start_ts,
            "last_cycle_ts": self._last_cycle_ts,
            "circuit_breaker_open": self._client._circuit.is_open,
        }

    def send_dashboard(self) -> None:
        try:
            data = self.get_dashboard_data()
            self._slack.send_dashboard(
                equity=data["equity"],
                initial_capital=data["initial_capital"],
                daily_pnl=data["daily_pnl"],
                total_pnl=data["total_pnl"],
                positions=data["positions"],
                mode=data["mode"],
            )
        except Exception as e:
            log.error("send_dashboard failed: %s", e)

    # ── Equity ────────────────────────────────────────────────────────────────

    def _get_equity(self) -> float:
        if settings.is_dry_run() or settings.is_paper():
            total_pnl = self._db.get_total_pnl(mode=settings.TRADING_MODE)
            return settings.PAPER_CAPITAL_USDT + total_pnl
        try:
            return self._client.get_usdt_equity()
        except Exception as e:
            log.error("Failed to get equity: %s", e)
            return settings.PAPER_CAPITAL_USDT

    # ── Position Sync ─────────────────────────────────────────────────────────

    def _sync_positions(self, symbols: List[str]) -> None:
        """Sync local state with exchange every cycle to detect liquidations."""
        try:
            exchange_positions = self._client.get_positions()
            exchange_by_symbol = {
                p.get("symbol"): p
                for p in exchange_positions
                if float(p.get("size", 0)) > 0
            }

            for symbol in symbols:
                local_pos = self._state_machine.get_position(symbol)
                sym_pos = [p for p in exchange_positions if p.get("symbol") == symbol]
                self._state_machine.sync_from_exchange(symbol, sym_pos)

                if not local_pos.is_flat and symbol not in exchange_by_symbol:
                    log.error(
                        "LIQUIDATION DETECTED for %s — local=%s but exchange has no position",
                        symbol, local_pos.state.value,
                    )
                    trade_id = self._db.get_state(f"trade_id:{symbol}")
                    if trade_id:
                        liq_price = local_pos.stop_price or local_pos.entry_price
                        pnl = (liq_price - local_pos.entry_price) * local_pos.qty
                        if local_pos.is_short:
                            pnl = (local_pos.entry_price - liq_price) * local_pos.qty
                        self._db.update_trade_close(
                            trade_id=trade_id,
                            exit_price=liq_price,
                            pnl=pnl,
                            fee=local_pos.qty * liq_price * settings.BACKTEST_FEE_TAKER,
                            reason="LIQUIDATION",
                        )
                    self._state_machine.force_flat(symbol)
                    self._slack.send_risk_alert(
                        f"🚨 LIQUIDATION: `{symbol}` position was liquidated by exchange!"
                    )
        except Exception as e:
            log.warning("Position sync failed: %s", e)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Binance Futures Bot — Starting")
    log.info("  Mode: %s | Testnet: %s", settings.TRADING_MODE, settings.BINANCE_TESTNET)
    log.info("=" * 60)

    bot = TradingBot()
    slack = SlackNotifier(settings.SLACK_WEBHOOK_URL)
    slack.send_startup(settings.TRADING_MODE)

    if settings.SLACK_BOT_TOKEN and settings.SLACK_APP_TOKEN:
        from src.api.slack_bot import SlackBotHandler
        slack_bot = SlackBotHandler(
            bot_token=settings.SLACK_BOT_TOKEN,
            app_token=settings.SLACK_APP_TOKEN,
            data_provider=bot.get_dashboard_data,
        )
        slack_bot.start()
    else:
        log.info(
            "Slack Bot disabled — set SLACK_BOT_TOKEN and SLACK_APP_TOKEN to enable slash commands"
        )

    scheduler = BackgroundScheduler(timezone="Asia/Seoul")

    # 전략 모드에 따라 사이클 인터벌 결정
    # scalp/trend/auto: 3분봉 기반 → 3분 주기, swing: 15분봉 기반 → 15분 주기
    if settings.STRATEGY_MODE in ("scalp", "trend", "auto"):
        cycle_interval = int(settings.SCALP_PURE_LTF_TIMEFRAME)  # 3
    else:
        cycle_interval = int(settings.SIGNAL_TIMEFRAME) if int(settings.SIGNAL_TIMEFRAME) <= 60 else 5

    log.info(
        "Strategy mode: %s | Cycle interval: %d min",
        settings.STRATEGY_MODE,
        cycle_interval,
    )

    scheduler.add_job(
        bot.run_cycle,
        "interval",
        minutes=cycle_interval,
        id="bot_cycle",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )

    scheduler.add_job(
        bot._universe.refresh,
        "interval",
        hours=settings.UNIVERSE_REFRESH_INTERVAL_HOURS,
        id="universe_refresh",
    )

    scheduler.add_job(
        bot.update_macro_regime,
        "interval",
        hours=settings.MACRO_REGIME_CHECK_HOURS,
        id="macro_regime_check",
        max_instances=1,
        coalesce=True,
    )

    # Briefing: 09:00 KST + 21:00 KST (하루 2회만 — 노이즈 감소)
    scheduler.add_job(
        bot._briefing.send,
        CronTrigger(hour="9,21", minute=0, timezone="Asia/Seoul"),
        id="daily_briefing",
    )

    # 미체결 주문 감사: 주기적으로 실행하여 잔여 SL/TP 및 시간 초과 주문 정리
    scheduler.add_job(
        lambda: bot._order_mgr.audit_stale_orders(
            {pos.symbol for pos in bot._state_machine.all_positions()}
        ),
        "interval",
        minutes=settings.STALE_ORDER_AUDIT_INTERVAL_MINUTES,
        id="stale_order_audit",
        max_instances=1,
        coalesce=True,
    )
    # 자동 dashboard 스케줄러: 제거 (노이즈 감소 — /dashboard 슬래시 커맨드로 수동 확인)

    scheduler.start()
    log.info("Scheduler started. Running first cycle immediately...")

    # 초기 매크로 레짐 감지 (첫 사이클 전에)
    try:
        bot.update_macro_regime()
    except Exception as e:
        log.exception("Initial macro regime detection failed: %s", e)

    try:
        bot.run_cycle()
    except Exception as e:
        log.exception("First cycle failed: %s", e)

    def _shutdown(signum, frame):
        log.info("Shutdown signal received")
        try:
            if scheduler.running:
                scheduler.shutdown(wait=False)
        except Exception:
            pass
        slack.send_shutdown("signal")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Bot running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
