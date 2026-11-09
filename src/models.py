"""Model training/evaluation: Linear Regression, Random Forest, XGBoost, LSTM.

All models predict the NEXT-DAY RETURN (not raw price) which avoids the
non-stationarity trap; predicted price = today's close * (1 + predicted return).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["SMA_20", "SMA_50", "EMA_12", "EMA_26", "RSI_14", "MACD", "MACD_Signal",
            "MACD_Hist", "Return_1d", "Return_5d", "Volatility_10d", "Volume_Chg"]
PRICE_FEATURES = {"SMA_20", "SMA_50", "EMA_12", "EMA_26", "MACD", "MACD_Signal", "MACD_Hist"}
WINDOW = 20


def prepare(df: pd.DataFrame, train_frac: float = 0.8):
    d = df.copy()
    # make price-scale features relative to close so they are stationary
    for f in PRICE_FEATURES:
        d[f] = d[f] / d["Close"]
    d["Target"] = d["Close"].shift(-1) / d["Close"] - 1
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    last_row = d.iloc[[-1]]                      # no target yet -> used for forecast
    d = d.dropna(subset=["Target"])
    split = int(len(d) * train_frac)
    return d.iloc[:split], d.iloc[split:], last_row


def evaluate(close_today, true_ret, pred_ret) -> dict:
    close_today = np.asarray(close_today)
    true_p = close_today * (1 + np.asarray(true_ret))
    pred_p = close_today * (1 + np.asarray(pred_ret))
    err = pred_p - true_p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / true_p)))
    dir_acc = float(np.mean(np.sign(pred_ret) == np.sign(true_ret)))
    return {"MAE": mae, "RMSE": rmse, "Accuracy (%)": (1 - mape) * 100,
            "Directional Accuracy (%)": dir_acc * 100}


def _sk_models():
    models = {"Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8,
                                                     min_samples_leaf=5, n_jobs=-1, random_state=42)}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    except Exception:
        models["XGBoost"] = GradientBoostingRegressor(random_state=42)  # fallback
    return models


def _windows(X, y, window):
    xs, ys = [], []
    for i in range(window, len(X) + 1):
        xs.append(X[i - window:i])
        ys.append(y[i - 1])
    return np.array(xs), np.array(ys)


def _lstm(train, test, last_row, full, epochs=25):
    """Returns (test_pred, forecast) or None if TensorFlow is unavailable."""
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        tf.random.set_seed(42)
        from tensorflow.keras import layers, models, callbacks
    except Exception:
        return None
    sx = StandardScaler().fit(train[FEATURES])
    sy = StandardScaler().fit(train[["Target"]])
    Xtr = sx.transform(train[FEATURES]); ytr = sy.transform(train[["Target"]]).ravel()
    # prepend end of train to test so every test row has a full window
    ctx = pd.concat([train.iloc[-(WINDOW - 1):], test])
    Xte = sx.transform(ctx[FEATURES])
    Xw, yw = _windows(Xtr, ytr, WINDOW)
    Xt, _ = _windows(Xte, np.zeros(len(Xte)), WINDOW)
    model = models.Sequential([layers.Input((WINDOW, len(FEATURES))), layers.LSTM(48),
                               layers.Dropout(0.2), layers.Dense(16, activation="relu"),
                               layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xw, yw, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1,
              callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    pred = sy.inverse_transform(model.predict(Xt, verbose=0)).ravel()
    tail = full[FEATURES].tail(WINDOW)
    fc = sy.inverse_transform(model.predict(sx.transform(tail)[None], verbose=0)).ravel()[0]
    return pred, float(fc)


def train_all(df: pd.DataFrame, use_lstm: bool = True, progress=None):
    """Returns (results_df, test_frame_with_predictions, next_day_forecasts)."""
    train, test, last_row = prepare(df)
    full = pd.concat([train, test, last_row])
    preds, forecasts, rows = {}, {}, []
    for name, m in _sk_models().items():
        if progress: progress(f"Training {name}…")
        m.fit(train[FEATURES], train["Target"])
        preds[name] = m.predict(test[FEATURES])
        forecasts[name] = float(m.predict(last_row[FEATURES])[0])
    if use_lstm:
        if progress: progress("Training LSTM (this may take a minute)…")
        out = _lstm(train, test, last_row, full)
        if out:
            preds["LSTM"], forecasts["LSTM"] = out
    for name, p in preds.items():
        met = evaluate(test["Close"], test["Target"], p)
        rows.append({"Model": name, **met})
    results = pd.DataFrame(rows).set_index("Model").round(3)
    tf_ = test[["Close"]].copy()
    tf_["Actual"] = tf_["Close"] * (1 + test["Target"])
    for name, p in preds.items():
        tf_[name] = tf_["Close"] * (1 + p)
    last_close = float(last_row["Close"].iloc[0])
    fc = {n: {"return_%": r * 100, "price": last_close * (1 + r)} for n, r in forecasts.items()}
    return results, tf_, fc
