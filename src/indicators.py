"""Technical indicators: MA, EMA, RSI, MACD, Bollinger Bands."""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    d["SMA_20"] = c.rolling(20).mean()
    d["SMA_50"] = c.rolling(50).mean()
    d["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    d["EMA_26"] = c.ewm(span=26, adjust=False).mean()
    d["RSI_14"] = rsi(c, 14)
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]
    std = c.rolling(20).std()
    d["BB_Upper"] = d["SMA_20"] + 2 * std
    d["BB_Lower"] = d["SMA_20"] - 2 * std
    d["Return_1d"] = c.pct_change()
    d["Return_5d"] = c.pct_change(5)
    d["Volatility_10d"] = d["Return_1d"].rolling(10).std()
    d["Volume_Chg"] = d["Volume"].pct_change().replace([np.inf, -np.inf], 0)
    return d
