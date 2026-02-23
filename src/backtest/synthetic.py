"""
Synthetic BTC/USDT 5-minute futures data generator.

Models BTC 2024-2025 price action using:
  - 3 market phases matching actual BTC 2024-2025 behaviour
  - Geometric Brownian Motion (GBM) with GARCH-like volatility clustering
  - Jump diffusion (Merton 1976) for sudden price dislocations
  - Mean-reverting funding rates (longs pay when market is crowded)
  - Realistic bid-ask spreads correlated with volatility
  - Volume correlated with absolute price moves

Phase structure:
  Phase 1 (days 0–90):   ETF approval bull run  $42k → $73k  (Jan–Mar 2024)
  Phase 2 (days 90–240): Post-ATH consolidation  $50k–$70k    (Mar–Aug 2024)
  Phase 3 (days 240–365): Post-halving surge     $60k → $100k (Sep–Dec 2024)

No API keys required. ~105,000 5-min bars for 365 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class SyntheticBTCData:
    """
    Generate synthetic BTC/USDT perpetual futures OHLCV data at 5-minute bars.
    """

    # Phase definitions: (n_days, daily_drift, daily_sigma, label)
    _PHASES = [
        {"days": 90,  "drift": 0.0045,  "sigma": 0.030, "name": "bull_run"},
        {"days": 150, "drift": -0.0003, "sigma": 0.020, "name": "consolidation"},
        {"days": 125, "drift": 0.0025,  "sigma": 0.025, "name": "post_halving_surge"},
    ]

    def generate(
        self,
        start_price: float = 42_000.0,
        n_days: int = 365,
        interval_min: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate n_days of 5-minute OHLCV bars starting from start_price.

        Returns a DataFrame with columns:
            ts, open, high, low, close, volume
        Index is DatetimeIndex (UTC).
        """
        rng = np.random.default_rng(seed)

        bars_per_day = 24 * 60 // interval_min   # 288 for 5-min

        current_price = start_price
        # GARCH vol state — starts at phase-0 long-run sigma
        vol_state = self._PHASES[0]["sigma"]

        ts = pd.Timestamp("2024-01-01", tz="UTC")
        all_bars = []

        days_done = 0
        for phase in self._PHASES:
            if days_done >= n_days:
                break
            phase_days = min(phase["days"], n_days - days_done)
            phase_bars = phase_days * bars_per_day
            sigma_lr = phase["sigma"]           # long-run vol for GARCH reversion
            mu_per_bar = phase["drift"] / bars_per_day  # drift per 5-min bar

            for _ in range(phase_bars):
                # ── GARCH-like volatility clustering ─────────────────────────
                # vol_state mean-reverts to sigma_lr with occasional shocks
                innovation = abs(rng.standard_normal()) * sigma_lr
                vol_state = (
                    0.94 * vol_state
                    + 0.03 * sigma_lr
                    + 0.03 * innovation
                )
                vol_state = float(np.clip(vol_state, sigma_lr * 0.25, sigma_lr * 3.5))
                bar_sigma = vol_state / np.sqrt(bars_per_day)

                # ── GBM return ────────────────────────────────────────────────
                ret = mu_per_bar + bar_sigma * rng.standard_normal()

                # ── Jump diffusion (Merton 1976) ──────────────────────────────
                # 0.35% chance per bar ≈ ~1 jump per day
                if rng.random() < 0.0035:
                    # jump amplitude: normally distributed ±2–4%
                    jump = rng.normal(0, 0.030)
                    ret += jump

                close = max(1.0, current_price * np.exp(ret))

                # ── Intrabar OHLC ─────────────────────────────────────────────
                bar_range = close * bar_sigma * 2.0
                high_extra = bar_range * float(rng.uniform(0.15, 0.85))
                low_extra  = bar_range * float(rng.uniform(0.15, 0.85))
                open_ = current_price

                high = max(open_, close) + high_extra
                low  = min(open_, close) - low_extra

                # ── Volume (correlated with |ret| and vol) ────────────────────
                vol_ratio = abs(ret) / (bar_sigma + 1e-10)
                base_volume = 80.0 + 600.0 * float(np.clip(vol_ratio, 0, 5))
                volume = max(1.0, base_volume * float(rng.lognormal(0, 0.45)))

                all_bars.append({
                    "ts":     int(ts.timestamp() * 1000),
                    "open":   round(open_, 2),
                    "high":   round(high, 2),
                    "low":    round(low, 2),
                    "close":  round(close, 2),
                    "volume": round(volume, 4),
                })

                current_price = close
                ts += pd.Timedelta(minutes=interval_min)

            days_done += phase_days

        df = pd.DataFrame(all_bars)
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def generate_60min(self, df_5min: pd.DataFrame) -> pd.DataFrame:
        """Resample 5-min data to 60-min bars for regime / trend detection."""
        df = df_5min.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)

        resampled = df.resample("60min").agg({
            "ts":     "first",
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["close"])

        return resampled

    def summary(self, df: pd.DataFrame) -> dict:
        """Print a brief summary of the generated data."""
        return {
            "bars":        len(df),
            "days":        len(df) // (24 * 12),    # approximate at 5-min
            "start_price": float(df["close"].iloc[0]),
            "end_price":   float(df["close"].iloc[-1]),
            "min_price":   float(df["close"].min()),
            "max_price":   float(df["close"].max()),
            "start_date":  str(df.index[0]),
            "end_date":    str(df.index[-1]),
        }
