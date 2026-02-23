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
from src.risk.engine import RiskEngine
from src.risk.guards import GuardAction, RiskGuards
from src.risk.profit_protector import ProfitProtector
from src.strategy.combiner import SignalCombiner
from src.strategy.ict import ICTAnalyzer
from src.strategy.indicators import volatility_percentile
from src.strategy.macro_regime import MacroRegimeDetector, MacroRegime
from src.strategy.regime import RegimeDetector, Regime
from src.strategy.scalp import TradeModeDetector
from src.strategy.signals import SignalGenerator

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
        self._risk_engine = RiskEngine(settings)
        self._guards = RiskGuards(settings)
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

        if not settings.is_dry_run():
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
        snap = self._collector.collect(symbol, limit=250)
        if snap.ohlcv.empty or len(snap.ohlcv) < 100:
            log.warning("Insufficient data for %s (%d bars)", symbol, len(snap.ohlcv))
            return

        df = snap.ohlcv
        current_price = snap.last_price

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
        )
        combined = self._combiner.combine(sig_bundle, regime_r)

        close = df["close"]
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
                self._slack.send_risk_alert(
                    f"⚠️ {guard.reason}\n복리 모드: 거래를 계속 진행합니다."
                )

        pos = self._state_machine.get_position(symbol)

        if not pos.is_flat:
            bars_in_trade = int((time.time() - pos.entry_ts) /
                                (int(settings.SIGNAL_TIMEFRAME) * 60))

            regime_flip = (
                (pos.is_long and regime_r.regime == Regime.TRENDING_DOWN) or
                (pos.is_short and regime_r.regime == Regime.TRENDING_UP)
            )

            exit_check = self._state_machine.check_exit(
                symbol=symbol,
                current_price=current_price,
                current_atr=sig_bundle.atr_val,
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
            return

        if combined.direction == 0:
            return

        if not self._state_machine.can_enter(symbol):
            return

        sym_info = self._client.get_symbol_info(symbol)
        if not sym_info:
            return

        consec = self._db.get_consecutive_losses(symbol)

        mode_result = self._mode_detector.detect(
            regime=regime_r,
            ict_result=ict_r,
            bundle=sig_bundle,
            funding_rate=snap.funding_rate,
            spread_pct=snap.orderbook.spread_pct,
            direction=combined.direction,
        )

        is_bear_mode = (self._macro_mode == "BEAR_SIDEWAYS")
        risk_params = self._risk_engine.calculate(
            equity=equity,
            entry_price=current_price,
            side="Buy" if combined.direction == 1 else "Sell",
            atr=sig_bundle.atr_val,
            signal_strength=combined.strength,
            spread_pct=snap.orderbook.spread_pct,
            funding_rate=snap.funding_rate,
            vol_percentile=vol_pct,
            drawdown_pct=(peak_equity - equity) / peak_equity if peak_equity > 0 else 0,
            consecutive_losses=consec,
            symbol_max_leverage=sym_info.get("max_leverage", 125),
            symbol_info=sym_info,
            current_positions=self._state_machine.open_count(),
            trade_mode=mode_result.mode.value,
            bear_mode=is_bear_mode,
        )

        if not risk_params.risk_ok:
            log.debug("Risk check failed for %s: %s", symbol, risk_params.reason)
            return

        log.info(
            "Signal %s %s strength=%.2f regime=%s",
            symbol,
            "Buy" if combined.direction == 1 else "Sell",
            combined.strength,
            regime_r.regime.value,
        )

        self._execute_entry(symbol, combined.direction, risk_params, snap, sym_info, combined.strength)

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

            # 진입 수수료 계산 (GTX=Maker 0.020%)
            entry_fee_rate = settings.BACKTEST_FEE_MAKER
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
            exit_fee = notional * settings.BACKTEST_FEE_MAKER
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

            side_str = "Buy" if direction == 1 else "Sell"
            pnl_pct = pnl_net / (pos.entry_price * qty_to_close) if pos.entry_price > 0 and qty_to_close > 0 else 0.0
            self._slack.send_position_closed(
                symbol=symbol,
                side=side_str,
                reason="PARTIAL_TP1",
                pnl=pnl_net,
                pnl_pct=pnl_pct,
            )

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
            qty = pos.qty

            pnl = self._state_machine.exit(symbol, exit_price)

            notional = qty * exit_price
            urgent_keywords = ("STOP", "GUARD", "LIQUIDATION")
            fee_rate = (
                settings.BACKTEST_FEE_TAKER
                if any(kw in reason for kw in urgent_keywords)
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
            pnl_pct = pnl_net / (entry * qty) if entry > 0 and qty > 0 else 0
            self._slack.send_position_closed(
                symbol=symbol,
                side=side_str,
                reason=reason,
                pnl=pnl_net,
                pnl_pct=pnl_pct,
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

    cycle_interval = int(settings.SIGNAL_TIMEFRAME) if int(settings.SIGNAL_TIMEFRAME) <= 60 else 5
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

    # Briefing: 09:00 KST 시작, 4시간 간격
    scheduler.add_job(
        bot._briefing.send,
        CronTrigger(hour="9,13,17,21,1,5", minute=0, timezone="Asia/Seoul"),
        id="periodic_briefing",
    )

    if settings.SLACK_DASHBOARD_INTERVAL_MINUTES > 0:
        scheduler.add_job(
            bot.send_dashboard,
            "interval",
            minutes=settings.SLACK_DASHBOARD_INTERVAL_MINUTES,
            id="dashboard",
            max_instances=1,
            coalesce=True,
        )

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

    try:
        bot.send_dashboard()
    except Exception as e:
        log.exception("Initial dashboard failed: %s", e)

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
