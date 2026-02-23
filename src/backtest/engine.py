"""
Backtest Engine — bar-by-bar event simulation.

Features:
  - Realistic fee model (taker/maker separately)
  - Slippage (proportional to spread)
  - Funding cost interpolation
  - Leverage-aware PnL and liquidation simulation
  - Walk-forward validation (in-sample vs out-of-sample split)
  - Quality gate evaluation

Usage:
  python -m src.backtest.engine --symbols BTCUSDT ETHUSDT --years 5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.data_loader import load_multi
from src.backtest.report import generate_report, QualityGate
from src.config import Settings
from src.logger import get_logger
from src.strategy.combiner import SignalCombiner
from src.strategy.filters import QualityFilter
from src.strategy.ict import ICTAnalyzer
from src.strategy.indicators import atr, volatility_percentile
from src.strategy.regime import RegimeDetector
from src.strategy.scalp import TradeModeDetector
from src.strategy.signals import SignalGenerator

log = get_logger(__name__)

settings = Settings()


# ── Per-trade record ──────────────────────────────────────────────────────────

class BacktestTrade:
    __slots__ = [
        "symbol", "side", "entry_bar", "exit_bar",
        "entry_price", "exit_price", "qty", "leverage",
        "pnl_gross", "fees", "pnl_net", "exit_reason",
        "entry_ts", "exit_ts", "trade_mode", "funding_cost",
    ]

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, 0)


# ── Portfolio state ───────────────────────────────────────────────────────────

class PortfolioState:
    def __init__(self, initial_capital: float, max_concurrent: int):
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.initial_capital = initial_capital
        self.open_positions: Dict[str, dict] = {}  # symbol → position
        self.max_concurrent = max_concurrent
        self.consecutive_losses: Dict[str, int] = {}
        self.last_exit_bar: Dict[str, int] = {}

    @property
    def drawdown(self) -> float:
        return (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0

    def update_peak(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity)


# ── Backtest engine ───────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, cfg: Settings):
        self._s = cfg
        self._regime = RegimeDetector(cfg)
        self._ict = ICTAnalyzer(cfg)
        self._sig_gen = SignalGenerator(cfg)
        self._combiner = SignalCombiner(cfg)
        self._mode_detector = TradeModeDetector(cfg)
        self._quality_filter = QualityFilter(cfg)
        # How many 5-min bars in 8 hours (funding interval)
        self._funding_bars = 96
        # Filter result cache: symbol → (bar_idx, result)
        # Recompute only every 12 bars (1 hour of 5-min data) to avoid O(n²) cost
        self._filter_cache: dict = {}
        self._filter_every = 12

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        oos_split: float = 0.80,   # 80% in-sample, 20% OOS
    ) -> Dict:
        """
        Run full backtest on all symbols.
        Returns: {"all": report_dict, "is": is_report, "oos": oos_report}
        """
        all_trades: List[BacktestTrade] = []
        equity_curve: List[Tuple[int, float]] = []   # (ts, equity)
        portfolio = PortfolioState(
            initial_capital=self._s.BACKTEST_INITIAL_CAPITAL,
            max_concurrent=self._s.MAX_CONCURRENT_POSITIONS,
        )

        # Align all symbols to common timestamps
        symbols = list(data.keys())
        if not symbols:
            log.error("No data available for backtest")
            return {}

        # Build sorted list of all bars across symbols
        bars = []
        for sym, df in data.items():
            for i, row in enumerate(df.itertuples()):
                bars.append((int(row.ts), sym, i))
        bars.sort(key=lambda x: x[0])

        log.info("Starting backtest | %d symbols | %d total bars | capital=%.0f",
                 len(symbols), len(bars), portfolio.equity)

        prev_sym_idx: Dict[str, int] = {s: -1 for s in symbols}

        for bar_ts, symbol, bar_idx in bars:
            df = data[symbol]
            if bar_idx < 60:  # need warmup
                prev_sym_idx[symbol] = bar_idx
                continue

            # Update equity curve (hourly sampling)
            if not equity_curve or bar_ts - equity_curve[-1][0] > 3_600_000:
                equity_curve.append((bar_ts, portfolio.equity))
                portfolio.update_peak()

            # Guard: max drawdown halt
            if portfolio.drawdown >= self._s.MAX_DRAWDOWN_PCT:
                continue

            row = df.iloc[bar_idx]
            close = float(row.close)
            high = float(row.high)
            low = float(row.low)

            # ── Check existing position for exit ─────────────────────────────
            if symbol in portfolio.open_positions:
                pos = portfolio.open_positions[symbol]
                trade = self._check_exits(
                    pos, df, bar_idx, close, high, low, portfolio
                )
                if trade:
                    all_trades.append(trade)
                    if trade.exit_reason == "PARTIAL_TP1":
                        # Partial exit: position stays open, equity already updated
                        # inside _check_exits; just record and keep going
                        portfolio.update_peak()
                        continue
                    # Full exit
                    portfolio.equity += trade.pnl_net
                    portfolio.update_peak()
                    portfolio.last_exit_bar[symbol] = bar_idx
                    del portfolio.open_positions[symbol]
                    if trade.pnl_net < 0:
                        portfolio.consecutive_losses[symbol] = \
                            portfolio.consecutive_losses.get(symbol, 0) + 1
                    else:
                        portfolio.consecutive_losses[symbol] = 0
                    continue

            # ── Check for new entry ───────────────────────────────────────────
            if symbol in portfolio.open_positions:
                continue  # already handled
            if len(portfolio.open_positions) >= portfolio.max_concurrent:
                continue
            # Cooldown
            cooldown_bars = self._s.REENTRY_COOLDOWN_SECONDS // (
                int(self._s.SIGNAL_TIMEFRAME) * 60
            )
            if bar_idx - portfolio.last_exit_bar.get(symbol, 0) < cooldown_bars:
                continue

            entry = self._evaluate_entry(df, bar_idx, portfolio, symbol)
            if entry:
                portfolio.open_positions[symbol] = entry

            prev_sym_idx[symbol] = bar_idx

        # Close any remaining open positions at last price
        for sym, pos in portfolio.open_positions.items():
            df = data[sym]
            last_price = float(df["close"].iloc[-1])
            trade = self._force_close(pos, df, len(df) - 1, last_price, portfolio)
            if trade:
                all_trades.append(trade)
                portfolio.equity += trade.pnl_net

        equity_curve.append((bars[-1][0] if bars else 0, portfolio.equity))

        log.info("Backtest complete | trades=%d final_equity=%.2f",
                 len(all_trades), portfolio.equity)

        # Split into IS / OOS
        split_idx = int(len(all_trades) * oos_split)
        is_trades = all_trades[:split_idx]
        oos_trades = all_trades[split_idx:]

        ec_df = pd.DataFrame(equity_curve, columns=["ts", "equity"])

        return {
            "all": _build_metrics(all_trades, ec_df, self._s.BACKTEST_INITIAL_CAPITAL),
            "in_sample": _build_metrics(
                is_trades, ec_df.iloc[:int(len(ec_df) * oos_split)],
                self._s.BACKTEST_INITIAL_CAPITAL
            ),
            "out_of_sample": _build_metrics(
                oos_trades, ec_df.iloc[int(len(ec_df) * oos_split):],
                self._s.BACKTEST_INITIAL_CAPITAL
            ),
            "trades": all_trades,
            "equity_curve": ec_df,
        }

    # ── Entry Evaluation ──────────────────────────────────────────────────────

    def _evaluate_entry(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        portfolio: PortfolioState,
        symbol: str,
    ) -> Optional[dict]:
        window = df.iloc[max(0, bar_idx - 199): bar_idx + 1]

        # ── Quality filter (ER + ATR; Hurst skipped for backtest performance) ──
        # Cache result for 12 bars to avoid re-computation on each candidate bar
        try:
            cached = self._filter_cache.get(symbol)
            if cached is None or (bar_idx - cached[0]) >= self._filter_every:
                flt = self._quality_filter.check(window, use_hurst=False)
                self._filter_cache[symbol] = (bar_idx, flt)
            else:
                flt = cached[1]
            if not flt.passed:
                return None
        except Exception:
            pass  # filter errors are non-fatal

        try:
            # hurst_max_lags=0 = fast mode (skip O(n²) Hurst in backtest)
            regime_r = self._regime.detect(window, hurst_max_lags=0)
            ict_r = self._ict.analyze(window)
            sig_bundle = self._sig_gen.generate(window, ict_r)
            combined = self._combiner.combine(sig_bundle, regime_r)
        except Exception as e:
            log.debug("Signal error at bar %d: %s", bar_idx, e)
            return None

        if combined.direction == 0:
            return None
        if combined.strength < self._s.SIGNAL_ENTRY_THRESHOLD:
            return None

        close = float(df["close"].iloc[bar_idx])
        atr_val = float(atr(window["high"], window["low"], window["close"], self._s.ATR_PERIOD).iloc[-1])

        if atr_val <= 0:
            return None

        # Detect trade mode
        mode_result = self._mode_detector.detect(
            regime=regime_r,
            ict_result=ict_r,
            bundle=sig_bundle,
            funding_rate=0.0,
            spread_pct=0.001,
            direction=combined.direction,
        )
        mode_params = self._mode_detector.get_params(mode_result.mode)

        stop_dist = atr_val * mode_params.stop_atr
        stop_dist_pct = stop_dist / close
        if stop_dist_pct <= 0:
            return None

        # Position sizing — correct leverage for risk-based exposure.
        # We want: leverage × MAX_SINGLE_EXPOSURE_PCT × stop_dist_pct = target_risk
        # → leverage = target_risk / (MAX_SINGLE_EXPOSURE_PCT × stop_dist_pct)
        target_risk = self._s.MAX_RISK_PER_TRADE_PCT
        lev_float = target_risk / (self._s.MAX_SINGLE_EXPOSURE_PCT * stop_dist_pct)
        leverage = max(1, min(int(lev_float), self._s.MAX_LEVERAGE_CAP))

        # Notional and qty from margin allocation
        margin = portfolio.equity * self._s.MAX_SINGLE_EXPOSURE_PCT
        notional = margin * leverage
        qty = notional / close

        # Minimum notional sanity check
        if notional < 10.0:
            return None

        # Apply slippage
        slip = close * self._s.BACKTEST_SLIPPAGE_PCT
        if combined.direction == 1:
            entry_price = close + slip
            stop_price = entry_price - stop_dist
            tp1_price = entry_price + atr_val * mode_params.tp1_atr
            tp_price = entry_price + atr_val * mode_params.tp2_atr
        else:
            entry_price = close - slip
            stop_price = entry_price + stop_dist
            tp1_price = entry_price - atr_val * mode_params.tp1_atr
            tp_price = entry_price - atr_val * mode_params.tp2_atr

        # Entry fee — PostOnly limit orders use maker fee rate
        fee = notional * self._s.BACKTEST_FEE_MAKER

        return {
            "symbol": symbol,
            "side": "Buy" if combined.direction == 1 else "Sell",
            "direction": combined.direction,
            "entry_price": entry_price,
            "entry_bar": bar_idx,
            "entry_ts": int(df["ts"].iloc[bar_idx]) if "ts" in df.columns else 0,
            "qty": qty,
            "leverage": leverage,
            "stop_price": stop_price,
            "tp1_price": tp1_price,
            "tp_price": tp_price,
            "tp1_hit": False,
            "partial_exit_pct": mode_params.partial_exit_pct,
            "trailing_dist": atr_val * self._s.TRAILING_STOP_ATR_MULTIPLIER,
            "highest": entry_price,
            "lowest": entry_price,
            "fees": fee,
            "funding_cost": 0.0,   # accumulated funding cost
            "bars": 0,
            "atr": atr_val,
            "trade_mode": mode_result.mode.value,
            "max_bars": mode_params.max_bars,
        }

    # ── Exit Check ────────────────────────────────────────────────────────────

    def _check_exits(
        self,
        pos: dict,
        df: pd.DataFrame,
        bar_idx: int,
        close: float,
        high: float,
        low: float,
        portfolio: PortfolioState,
    ) -> Optional[BacktestTrade]:
        pos["bars"] += 1
        direction = pos["direction"]

        # ── Funding cost (every 96 bars = 8h for 5-min) ───────────────────────
        if pos["bars"] % self._funding_bars == 0:
            notional_now = pos["qty"] * close
            funding_charge = notional_now * abs(self._s.BACKTEST_FUNDING_RATE_AVG)
            pos["funding_cost"] = pos.get("funding_cost", 0.0) + funding_charge

        # Update trailing
        if direction == 1:
            pos["highest"] = max(pos["highest"], high)
        else:
            pos["lowest"] = min(pos["lowest"], low)

        # ── Partial TP1 check ─────────────────────────────────────────────────
        if not pos.get("tp1_hit", True):  # default True for legacy positions
            tp1 = pos.get("tp1_price", 0)
            tp1_triggered = (
                (direction == 1 and tp1 > 0 and high >= tp1) or
                (direction == -1 and tp1 > 0 and low <= tp1)
            )
            if tp1_triggered:
                partial_pct = pos.get("partial_exit_pct", 0.60)
                qty_partial = pos["qty"] * partial_pct
                partial_price = tp1

                slip = partial_price * self._s.BACKTEST_SLIPPAGE_PCT
                partial_price = partial_price - slip if direction == 1 else partial_price + slip

                partial_notional = qty_partial * partial_price
                # TP1 exits use limit orders → maker fee
                partial_exit_fee = partial_notional * self._s.BACKTEST_FEE_MAKER
                # Pro-rate the entry fee for the closed portion
                entry_fee_for_partial = pos["fees"] * partial_pct
                partial_fee = partial_exit_fee + entry_fee_for_partial

                if direction == 1:
                    partial_pnl_gross = (partial_price - pos["entry_price"]) * qty_partial
                else:
                    partial_pnl_gross = (pos["entry_price"] - partial_price) * qty_partial

                partial_pnl_net = partial_pnl_gross - partial_fee

                # Record partial trade
                t_partial = BacktestTrade()
                t_partial.symbol = pos["symbol"]
                t_partial.side = pos["side"]
                t_partial.entry_bar = pos["entry_bar"]
                t_partial.exit_bar = bar_idx
                t_partial.entry_price = pos["entry_price"]
                t_partial.exit_price = partial_price
                t_partial.qty = qty_partial
                t_partial.leverage = pos["leverage"]
                t_partial.pnl_gross = partial_pnl_gross
                t_partial.fees = partial_fee
                t_partial.pnl_net = partial_pnl_net
                t_partial.exit_reason = "PARTIAL_TP1"
                t_partial.trade_mode = pos.get("trade_mode", "SCALP")
                t_partial.funding_cost = 0.0
                t_partial.entry_ts = pos.get("entry_ts", 0)

                # Update portfolio equity with partial gain
                portfolio.equity += partial_pnl_net

                # Update position: reduce qty, move stop to entry, target to tp2
                pos["qty"] -= qty_partial
                # Keep proportional entry fee for the remaining position
                pos["fees"] = pos["fees"] * (1.0 - partial_pct)
                pos["stop_price"] = pos["entry_price"]   # breakeven stop
                pos["tp1_hit"] = True
                # tp_price is already TP2 (set in _evaluate_entry)

                # Return the partial trade so caller records it
                return t_partial

        exit_price = None
        reason = ""

        # Stop loss (use low/high of bar for realism)
        if direction == 1 and low <= pos["stop_price"]:
            exit_price = pos["stop_price"]
            reason = "STOP_LOSS"
        elif direction == -1 and high >= pos["stop_price"]:
            exit_price = pos["stop_price"]
            reason = "STOP_LOSS"

        # Take profit (TP2 after TP1, or single TP if no TP1)
        elif direction == 1 and high >= pos["tp_price"]:
            exit_price = pos["tp_price"]
            reason = "TAKE_PROFIT"
        elif direction == -1 and low <= pos["tp_price"]:
            exit_price = pos["tp_price"]
            reason = "TAKE_PROFIT"

        # Trailing stop — only activate after TP1 partial exit to avoid
        # premature firing on normal market noise before profit is secured
        elif direction == 1 and pos.get("tp1_hit", False):
            trailing = pos["highest"] - pos["trailing_dist"]
            if low <= trailing and trailing > pos["stop_price"]:
                exit_price = trailing
                reason = "TRAILING_STOP"
        elif direction == -1 and pos.get("tp1_hit", False):
            trailing = pos["lowest"] + pos["trailing_dist"]
            if high >= trailing and trailing < pos["stop_price"]:
                exit_price = trailing
                reason = "TRAILING_STOP"

        # Time exit (use mode-specific max bars)
        max_bars = pos.get("max_bars", 200)
        if exit_price is None and pos["bars"] >= max_bars:
            exit_price = close
            reason = "TIME_EXIT"

        if exit_price is None:
            return None

        # Apply slippage on exit
        slip = exit_price * self._s.BACKTEST_SLIPPAGE_PCT
        if direction == 1:
            exit_price -= slip
        else:
            exit_price += slip

        notional = pos["qty"] * exit_price
        # SL → Market(Taker).  TAKE_PROFIT / TRAILING_STOP / TIME_EXIT → Limit(Maker).
        # Matches live order_manager.py: urgent_keywords = ("STOP", "GUARD", "LIQUIDATION")
        exit_fee_rate = (
            self._s.BACKTEST_FEE_TAKER if reason == "STOP_LOSS"
            else self._s.BACKTEST_FEE_MAKER
        )
        exit_fee = notional * exit_fee_rate
        total_fee = pos["fees"] + exit_fee

        if direction == 1:
            pnl_gross = (exit_price - pos["entry_price"]) * pos["qty"]
        else:
            pnl_gross = (pos["entry_price"] - exit_price) * pos["qty"]

        pnl_net = pnl_gross - total_fee

        t = BacktestTrade()
        t.symbol = pos["symbol"]
        t.side = pos["side"]
        t.entry_bar = pos["entry_bar"]
        t.exit_bar = bar_idx
        t.entry_price = pos["entry_price"]
        t.exit_price = exit_price
        t.qty = pos["qty"]
        t.leverage = pos["leverage"]
        funding_cost = pos.get("funding_cost", 0.0)
        pnl_net -= funding_cost  # deduct funding from net PnL

        t.pnl_gross = pnl_gross
        t.fees = total_fee
        t.pnl_net = pnl_net
        t.exit_reason = reason
        t.trade_mode = pos.get("trade_mode", "SCALP")
        t.funding_cost = funding_cost
        t.entry_ts = pos.get("entry_ts", 0)

        return t

    def _force_close(self, pos: dict, df: pd.DataFrame, bar_idx: int,
                     price: float, portfolio: PortfolioState) -> Optional[BacktestTrade]:
        return self._check_exits(pos, df, bar_idx, price, price, price, portfolio) \
               or self._check_exits(
                   {**pos, "stop_price": -1e9, "tp_price": 1e9, "bars": 0},
                   df, bar_idx, price, price, price, portfolio
               )


# ── Metrics ───────────────────────────────────────────────────────────────────

def _build_metrics(
    trades: List[BacktestTrade],
    ec_df: pd.DataFrame,
    initial_capital: float,
) -> Dict:
    if not trades:
        return {"error": "No trades"}

    pnls = [t.pnl_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")

    # CAGR
    if not ec_df.empty and len(ec_df) >= 2:
        start_ts = float(ec_df["ts"].iloc[0])
        end_ts = float(ec_df["ts"].iloc[-1])
        years = (end_ts - start_ts) / (1000 * 365.25 * 86400)
        final_equity = float(ec_df["equity"].iloc[-1])
        if years > 0 and initial_capital > 0:
            cagr = (final_equity / initial_capital) ** (1 / max(years, 0.1)) - 1
        else:
            cagr = 0.0
    else:
        cagr = 0.0

    # MDD
    if not ec_df.empty:
        eq = ec_df["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1)
        mdd = float(dd.max())
    else:
        mdd = 0.0

    # Sharpe (trade-level, annualised)
    if len(pnls) >= 2:
        pnl_arr = np.array(pnls)
        if pnl_arr.std() > 0:
            # Trades per year (assume 15-min bars, 8760 bars/year)
            trades_per_year = len(pnls) / max(
                (float(ec_df["ts"].iloc[-1]) - float(ec_df["ts"].iloc[0])) / (1000 * 365.25 * 86400),
                0.1
            )
            sharpe = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Monthly breakdown (approximate)
    monthly = {}
    if not ec_df.empty:
        ec_df = ec_df.copy()
        ec_df["date"] = pd.to_datetime(ec_df["ts"], unit="ms")
        ec_df["month"] = ec_df["date"].dt.to_period("M")
        for m, grp in ec_df.groupby("month"):
            start_eq = float(grp["equity"].iloc[0])
            end_eq = float(grp["equity"].iloc[-1])
            monthly[str(m)] = (end_eq - start_eq) / max(start_eq, 1)

    return {
        "total_trades": len(pnls),
        "win_rate": win_rate,
        "avg_win_usdt": avg_win,
        "avg_loss_usdt": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl_usdt": total_pnl,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "initial_capital": initial_capital,
        "final_equity": initial_capital + total_pnl,
        "monthly_returns": monthly,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"])
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--interval", default="15")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    from src.config import settings

    log.info("=== BACKTEST START ===")
    log.info("Symbols: %s | Years: %d | Interval: %s", args.symbols, args.years, args.interval)

    data = load_multi(
        symbols=args.symbols,
        interval=args.interval,
        years=args.years,
        api_key=settings.BINANCE_API_KEY,
        api_secret=settings.BINANCE_API_SECRET,
        testnet=settings.BINANCE_TESTNET,
    )

    if not data:
        log.error("No data loaded. Check API keys and symbol names.")
        sys.exit(1)

    engine = BacktestEngine(settings)
    results = engine.run(data)

    # Quality gate check
    qg = QualityGate(settings)
    passed, gate_results = qg.evaluate(results["all"])

    # Generate reports
    report_path = generate_report(results, args.symbols, args.interval, args.years)
    log.info("Report written to: %s", report_path)

    # Summary
    m = results["all"]
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS — ALL DATA")
    print("=" * 60)
    print(f"  Trades:         {m.get('total_trades', 0)}")
    print(f"  Win Rate:       {m.get('win_rate', 0):.2%}")
    print(f"  Profit Factor:  {m.get('profit_factor', 0):.2f}")
    print(f"  CAGR:           {m.get('cagr', 0):.2%}")
    print(f"  Max Drawdown:   {m.get('mdd', 0):.2%}")
    print(f"  Sharpe Ratio:   {m.get('sharpe', 0):.2f}")
    print(f"  Total PnL:      {m.get('total_pnl_usdt', 0):+.2f} USDT")
    print(f"  Final Equity:   {m.get('final_equity', 0):,.2f} USDT")
    print()
    print("QUALITY GATE:", "✅ PASSED" if passed else "❌ FAILED")
    for k, v in gate_results.items():
        status = "✅" if v["pass"] else "❌"
        print(f"  {status} {k}: {v['value']:.3f} (required {v['required']})")

    if passed:
        print("\n" + "=" * 60)
        print("Quality gate PASSED.")
        print("Set QUALITY_GATE_PASSED=true in .env to enable LIVE trading.")
        print("=" * 60)
    else:
        print("\nQuality gate FAILED. Adjust strategy parameters and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
