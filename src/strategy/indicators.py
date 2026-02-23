"""
Pure-function technical indicator library.
All functions operate on pandas Series / DataFrame and return pandas objects.
No side effects, no external state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Trend / Momentum ──────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


# ── Volatility ────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b}
    )


def realized_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Log-return realized volatility (annualized)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252 * 24 * 4)  # 15-min bars


def volatility_percentile(close: pd.Series, short: int = 20, long: int = 100) -> pd.Series:
    """Rolling percentile of short vol vs long vol window (0–1)."""
    rv = realized_volatility(close, short)
    return rv.rolling(long).rank(pct=True)


# ── ADX / Directional Movement ────────────────────────────────────────────────

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Returns DataFrame: adx, plus_di, minus_di
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    tr = atr(high, low, close, period)  # reuse ATR (smoothed TR)

    plus_di = 100 * plus_dm_s.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di})


# ── Volume ────────────────────────────────────────────────────────────────────

def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period).mean()


def volume_ratio(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Ratio of short-term avg volume to long-term avg volume."""
    return volume.rolling(short).mean() / volume.rolling(long).mean().replace(0, np.nan)


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Z-score of volume relative to rolling window."""
    avg = volume.rolling(period).mean()
    std = volume.rolling(period).std()
    return (volume - avg) / std.replace(0, np.nan)


# ── VWAP ──────────────────────────────────────────────────────────────────────

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 96,
) -> pd.Series:
    """
    Rolling Volume-Weighted Average Price.
    Uses a rolling window of `period` bars rather than session reset.
    Returns NaN for the first `period-1` bars.
    """
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    rolling_tp_vol = tp_vol.rolling(period).sum()
    rolling_vol = volume.rolling(period).sum()
    return rolling_tp_vol / rolling_vol.replace(0, np.nan)


# ── Efficiency Ratio (Kaufman) ────────────────────────────────────────────────

def efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Kaufman Efficiency Ratio (ER).
    ER = abs(net_change_over_period) / sum(abs(bar_changes))
    ER ≈ 1.0 → clean directional trend
    ER ≈ 0.0 → choppy, noisy, random movement
    Reference: Perry Kaufman "Trading Systems and Methods"
    """
    net_change = (series - series.shift(period)).abs()
    path_length = series.diff().abs().rolling(period).sum()
    return net_change / path_length.replace(0, np.nan)


# ── Hurst Exponent (Peters 1991) ──────────────────────────────────────────────

def hurst_exponent(series: pd.Series, max_lags: int = 20) -> float:
    """
    Hurst Exponent via Rescaled Range (R/S) analysis (Peters 1991).
    H > 0.55 → trending / persistent momentum
    H ≈ 0.50 → random walk (efficient market)
    H < 0.45 → mean-reverting / anti-persistent
    Returns a scalar for the provided series.
    Reference: Edgar Peters "Fractal Market Analysis" (1994)
    """
    if len(series) < max_lags * 4:
        return 0.5

    log_returns = np.log(series / series.shift(1)).dropna().values
    n = len(log_returns)
    if n < max_lags * 2:
        return 0.5

    rs_vals = []
    lag_vals = []

    for lag in range(2, min(max_lags, n // 2)):
        n_chunks = n // lag
        if n_chunks < 2:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = log_returns[i * lag: (i + 1) * lag]
            mean_chunk = chunk.mean()
            deviation = np.cumsum(chunk - mean_chunk)
            r = deviation.max() - deviation.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_chunk.append(r / s)

        if rs_chunk:
            rs_vals.append(np.log(np.mean(rs_chunk)))
            lag_vals.append(np.log(lag))

    if len(rs_vals) < 4:
        return 0.5

    hurst, _ = np.polyfit(lag_vals, rs_vals, 1)
    return float(np.clip(hurst, 0.0, 1.0))


# ── Hull Moving Average (Alan Hull, 2005) ─────────────────────────────────────

def hull_ma(series: pd.Series, period: int = 16) -> pd.Series:
    """
    Hull Moving Average.
    HMA = WMA(2 × WMA(n/2) − WMA(n), sqrt(n))
    Faster and smoother than EMA with significantly less lag.
    Fast vectorized implementation using numpy convolution (O(n) per call).
    Reference: Alan Hull (2005), alanhull.com
    """
    half = max(2, period // 2)
    sqrt_n = max(2, int(round(period ** 0.5)))

    def _wma(s: pd.Series, w: int) -> pd.Series:
        vals = s.values.astype(float)
        n = len(vals)
        if n < w:
            return pd.Series(np.full(n, np.nan), index=s.index)
        weights = np.arange(1, w + 1, dtype=float)
        weights /= weights.sum()
        # numpy convolve 'valid': output length = n - w + 1
        conv = np.convolve(vals, weights[::-1], mode="valid")
        result = np.full(n, np.nan)
        result[w - 1:] = conv
        return pd.Series(result, index=s.index)

    wma_half = _wma(series, half)
    wma_full = _wma(series, period)
    raw = 2.0 * wma_half - wma_full
    return _wma(raw, sqrt_n)


# ── Keltner Channels ──────────────────────────────────────────────────────────

def keltner_channel(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Keltner Channels (Chester Keltner original / Linda Raschke ATR variant).
    Mid   = EMA(close, period)
    Upper = Mid + atr_mult × ATR(period)
    Lower = Mid − atr_mult × ATR(period)
    pct_k = (close − lower) / (upper − lower)   [0..1, >1 = breakout above]
    """
    mid = ema(close, period)
    atr_vals = atr(high, low, close, period)
    upper = mid + atr_mult * atr_vals
    lower = mid - atr_mult * atr_vals
    pct_k = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower, "pct_k": pct_k})


# ── Stochastic RSI ────────────────────────────────────────────────────────────

def stoch_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """
    Stochastic RSI — applies Stochastic oscillator formula to RSI values.
    K line (smoothed): current RSI position within RSI range over stoch_period
    D line: signal line = smoothed K

    Returns DataFrame: k, d
    Interpretation:
      K > 80 → overbought; K < 20 → oversold
      K crosses D from below → bullish; from above → bearish
    Reference: Chande & Kroll (1994)
    """
    rsi_vals = rsi(close, rsi_period)
    min_rsi = rsi_vals.rolling(stoch_period).min()
    max_rsi = rsi_vals.rolling(stoch_period).max()
    range_rsi = (max_rsi - min_rsi).replace(0, np.nan)
    raw_k = 100 * (rsi_vals - min_rsi) / range_rsi
    k = sma(raw_k, smooth_k)
    d = sma(k, smooth_d)
    return pd.DataFrame({"k": k, "d": d})


# ── Williams %R ───────────────────────────────────────────────────────────────

def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Williams %R oscillator.
    Range: -100 (oversold) to 0 (overbought)
    Threshold: above -20 = overbought; below -80 = oversold
    Reference: Larry Williams (1973)
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    range_hl = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / range_hl


# ── Chaikin Money Flow (CMF) ──────────────────────────────────────────────────

def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Chaikin Money Flow — measures buying/selling pressure using volume and price.
    Range: -1 to +1
    CMF > 0.1 → buying pressure (bullish)
    CMF < -0.1 → selling pressure (bearish)
    Reference: Marc Chaikin (1980s)
    """
    hl_range = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / hl_range
    money_flow_volume = money_flow_multiplier * volume
    cmf_val = money_flow_volume.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)
    return cmf_val


# ── RSI Divergence ────────────────────────────────────────────────────────────

def rsi_divergence(
    close: pd.Series,
    rsi_series: pd.Series,
    lookback: int = 14,
) -> pd.Series:
    """
    Detect bullish and bearish RSI divergences — checks only the latest bar.
    Optimised for live trading: O(1) computation (no Python loop over all bars).
    Returns a Series where only the last value is meaningful:
       +1  = bullish divergence (price lower low, RSI higher low)
       -1  = bearish divergence (price higher high, RSI lower high)
        0  = no divergence
    """
    result = pd.Series(0, index=close.index)

    if len(close) < lookback + 2:
        return result

    # Only compute for the last bar (live trading only needs current signal)
    i = len(close) - 1
    window_close = close.iloc[i - lookback: i + 1]
    window_rsi = rsi_series.iloc[i - lookback: i + 1]

    curr_close = window_close.iloc[-1]
    curr_rsi = window_rsi.iloc[-1]
    prev_close_min = window_close.iloc[:-1].min()
    prev_close_max = window_close.iloc[:-1].max()
    prev_rsi_at_low = window_rsi.iloc[:-1].min()
    prev_rsi_at_high = window_rsi.iloc[:-1].max()

    # Bullish divergence: price makes lower low, RSI makes higher low
    if curr_close < prev_close_min and curr_rsi > prev_rsi_at_low:
        result.iloc[i] = 1
    # Bearish divergence: price makes higher high, RSI makes lower high
    elif curr_close > prev_close_max and curr_rsi < prev_rsi_at_high:
        result.iloc[i] = -1

    return result


# ── On-Balance Volume (OBV) ───────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume — cumulative volume indicator.
    OBV trending up when price rises → confirms bullish move.
    OBV trending down when price rises → bearish divergence.
    Reference: Joe Granville (1963)
    """
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    return (pd.Series(direction, index=close.index) * volume).cumsum()


# ── Squeeze Momentum (LazyBear adaptation) ────────────────────────────────────

def squeeze_momentum(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Squeeze Momentum indicator (LazyBear / John Carter).
    Detects when Bollinger Bands are inside Keltner Channels (squeeze).

    Returns DataFrame:
      momentum: linear regression of (close - midpoint) — direction of move
      squeeze_on: True when BB inside KC (low volatility, coiling)
      squeeze_off: True when BB outside KC (volatility expansion)
    Reference: John Carter "Mastering the Trade" (2005)
    """
    bb = bollinger_bands(close, bb_period, bb_std)
    kc = keltner_channel(close, high, low, kc_period, kc_atr_mult)

    squeeze_on = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])

    # Momentum value: delta from midpoint (high+low)/2 averaged with midline
    mid = (high.rolling(kc_period).max() + low.rolling(kc_period).min()) / 2
    mid_close_avg = (mid + close.ewm(span=kc_period).mean()) / 2
    delta = close - mid_close_avg

    # Vectorized linear regression endpoint (fully O(n), no Python loops)
    # For window of length L, x=[0..L-1]:
    #   sum_x  = L*(L-1)/2,  sum_x2 = (L-1)*L*(2L-1)/6  (constants)
    #   b = (L*roll_sum_xy - sum_x*roll_sum_y) / (L*sum_x2 - sum_x^2)
    #   a = (roll_sum_y - b*sum_x) / L
    #   endpoint value = a + b*(L-1)
    def _linreg_endpoint(s: pd.Series, length: int) -> pd.Series:
        L = length
        x = np.arange(L, dtype=float)
        sum_x = x.sum()                           # L*(L-1)/2
        sum_x2 = (x * x).sum()                    # (L-1)*L*(2L-1)/6
        denom = L * sum_x2 - sum_x ** 2           # constant

        weights_xy = x / denom                    # for Σxy term
        weights_y = (sum_x2 - sum_x * x) / denom  # for endpoint = a + b*(L-1)

        # endpoint = Σ( weights_y[i] * y[i] ) + last_x * Σ( weights_xy[i] * y[i] )
        # combine into single weight vector for rolling dot product
        endpoint_weights = weights_y + (L - 1) * weights_xy

        # Rolling dot product via convolution
        vals = s.values.astype(float)
        kernel = endpoint_weights[::-1]
        conv = np.convolve(vals, kernel, mode="full")[:len(vals)]
        conv[:L - 1] = np.nan
        # NaN propagation: if any value in window is NaN, output is NaN
        nan_mask = pd.Series(vals).rolling(L).apply(lambda w: np.any(np.isnan(w)), raw=True)
        result = pd.Series(conv, index=s.index)
        result[nan_mask == 1] = np.nan
        return result

    momentum = _linreg_endpoint(delta, kc_period)
    return pd.DataFrame({
        "momentum": momentum,
        "squeeze_on": squeeze_on,
        "squeeze_off": ~squeeze_on,
    })


# ── Session VWAP ──────────────────────────────────────────────────────────────

def session_vwap(df: pd.DataFrame, session_start_utc_hour: int = 0) -> pd.Series:
    """
    True session VWAP that resets at session_start_utc_hour each day (UTC).
    df must have columns: high, low, close, volume.
    If df.index is a DatetimeIndex, a proper daily reset is performed.
    If not (e.g. integer index), falls back to the rolling VWAP.
    Standard in institutional and prop-trading floor practice.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return vwap(df["high"], df["low"], df["close"], df["volume"])

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical * df["volume"]

    idx_utc = df.index
    if idx_utc.tz is None:
        idx_utc = idx_utc.tz_localize("UTC")
    shifted = idx_utc - pd.Timedelta(hours=session_start_utc_hour)
    session_key = shifted.normalize()

    cumtp_vol = tp_vol.groupby(session_key).cumsum()
    cumvol = df["volume"].groupby(session_key).cumsum()
    return cumtp_vol / cumvol.replace(0, np.nan)
