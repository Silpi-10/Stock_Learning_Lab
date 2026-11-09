"""Data collection: yfinance with a synthetic fallback so the app works offline."""
import numpy as np
import pandas as pd


def _synthetic(ticker: str, days: int = 1250) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    ret = rng.normal(0.0005, 0.015, days)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, days))
    vol = rng.integers(1_000_000, 10_000_000, days)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def fetch_history(ticker: str, period: str = "5y"):
    """Return (DataFrame, source). source is 'yfinance' or 'synthetic'."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("empty response")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna(), "yfinance"
    except Exception:
        return _synthetic(ticker), "synthetic"
