import numpy as np
import pandas as pd
from typing import Optional, List

def make_features_for_latest(
    df: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    roll_windows: List[int] = [3, 5, 10, 30, 60],
    max_lag: int = 10,
    eps: float = 1e-9,
    fillna: bool = True,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Return a single-row DataFrame containing engineered features for the latest timestamp.
    
    Parameters
    ----------
    df : pd.DataFrame
        New data. Can be one row (latest) or several rows.
        Index should be datetime-like (required for minute_of_day).
        Must contain at least ['Open','High','Low','Close','Volume'].
    history : pd.DataFrame, optional
        If df has only 1 row, you can pass prior historical data here so lags/rollings
        are computed correctly. If None, function will compute features with available data.
    roll_windows : list[int]
        Rolling windows to compute.
    max_lag : int
        Maximum lag for ret_lag_N features.
    eps : float
        Small epsilon to avoid divisions by zero.
    fillna : bool
        If True, fills NaNs with 0 in the returned single-row DataFrame.
    feature_cols : list[str], optional
        If provided, ensure final result contains these columns (in this order).
        Missing columns will be added and filled according to fillna.
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame (index = latest timestamp) with engineered features.
    """

    # Validate required base columns
    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required.issubset(df.columns) and not (history is not None and required.issubset(history.columns)):
        raise ValueError(f"Input data must contain columns: {required}")

    # Combine history + df (history may be None)
    if history is not None:
        full = pd.concat([history, df]).copy()
    else:
        full = df.copy()

    # Ensure datetime index or try to set it if Date/Datetime column is present
    if not isinstance(full.index, pd.DatetimeIndex):
        if 'Date' in full.columns:
            full = full.set_index('Date')
        elif 'Datetime' in full.columns:
            full = full.set_index('Datetime')
        else:
            # if still not datetime, try to convert index
            full.index = pd.to_datetime(full.index)

    full = full.sort_index()

    # === Feature engineering (same logic as your function) ===
    full['ret_1'] = full['Close'].pct_change()
    full['logret_1'] = np.log(full['Close']).diff()

    for lag in range(1, max_lag + 1):
        full[f'ret_lag_{lag}'] = full['Close'].pct_change(periods=lag)

    for w in roll_windows:
        full[f'roll_mean_{w}'] = full['ret_1'].rolling(window=w, min_periods=1).mean()
        full[f'roll_std_{w}'] = full['ret_1'].rolling(window=w, min_periods=1).std().fillna(0)
        full[f'roll_vol_{w}'] = full['Volume'].rolling(window=w, min_periods=1).mean()
        full[f'roll_max_{w}'] = full['Close'].rolling(window=w, min_periods=1).max()
        full[f'roll_min_{w}'] = full['Close'].rolling(window=w, min_periods=1).min()

    full['ema_5'] = full['Close'].ewm(span=5, adjust=False).mean()
    full['ema_13'] = full['Close'].ewm(span=13, adjust=False).mean()
    full['ema_diff_5_13'] = full['ema_5'] - full['ema_13']

    full['body'] = full['Close'] - full['Open']
    full['upper_shadow'] = full['High'] - full[['Close', 'Open']].max(axis=1)
    full['lower_shadow'] = full[['Close', 'Open']].min(axis=1) - full['Low']
    full['body_ratio'] = full['body'] / (full['High'] - full['Low'] + eps)

    tr1 = full['High'] - full['Low']
    tr2 = (full['High'] - full['Close'].shift(1)).abs()
    tr3 = (full['Low'] - full['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    full['atr_14'] = tr.rolling(window=14, min_periods=1).mean()

    full['vol_delta'] = full['Volume'] - full['Volume'].shift(1)
    full['vol_ratio_5'] = full['Volume'] / (full['Volume'].rolling(5, min_periods=1).mean() + eps)

    for w in [5, 15]:
        p_v = (full['Close'] * full['Volume']).rolling(window=w, min_periods=1).sum()
        vol_sum = full['Volume'].rolling(window=w, min_periods=1).sum() + eps
        full[f'vwap_{w}'] = p_v / vol_sum
        full[f'close_vwap_diff_{w}'] = full['Close'] - full[f'vwap_{w}']

    delta = full['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=7, adjust=False).mean()
    roll_down = down.ewm(span=7, adjust=False).mean()
    rs = roll_up / (roll_down + eps)
    full['rsi_7'] = 100 - (100 / (1 + rs))

    ema_fast = full['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = full['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    full['macd'] = macd
    full['macd_signal'] = signal
    full['macd_hist'] = macd - signal

    minute_of_day = full.index.hour * 60 + full.index.minute
    full['minute_of_day'] = minute_of_day
    full['minute_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    full['minute_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)

    full.replace([np.inf, -np.inf], np.nan, inplace=True)

    # === get only the last row ===
    last_row = full.iloc[[-1]].copy()

    # If user provided feature_cols, make sure we return those in that order
    if feature_cols is not None:
        for col in feature_cols:
            if col not in last_row.columns:
                last_row[col] = np.nan
        # reorder to match feature_cols
        last_row = last_row.reindex(columns=feature_cols)

    # Optionally fill NaNs (safer for model input)
    if fillna:
        last_row = last_row.fillna(0)

    # convert dtypes numeric where possible
    for c in last_row.columns:
        try:
            last_row[c] = pd.to_numeric(last_row[c])
        except Exception:
            pass

    return last_row
