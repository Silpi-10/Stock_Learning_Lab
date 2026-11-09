"""Virtual trading engine backed by SQLite."""
import sqlite3
from datetime import datetime
import pandas as pd

DB = "trading.db"
START_CASH = 100_000.0


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS accounts(user TEXT PRIMARY KEY, cash REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS holdings(user TEXT, ticker TEXT, qty INTEGER, avg_cost REAL,
                                            PRIMARY KEY(user, ticker));
        CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT,
                ts TEXT, ticker TEXT, side TEXT, qty INTEGER, price REAL, total REAL);
        CREATE TABLE IF NOT EXISTS snapshots(user TEXT, ts TEXT, equity REAL);
        """)


def ensure_account(user):
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO accounts VALUES(?,?)", (user, START_CASH))


def get_cash(user) -> float:
    with _conn() as c:
        return c.execute("SELECT cash FROM accounts WHERE user=?", (user,)).fetchone()["cash"]


def trade(user, ticker, side, qty, price):
    """Returns (ok, message)."""
    qty = int(qty)
    if qty <= 0:
        return False, "Quantity must be positive."
    total = qty * price
    with _conn() as c:
        cash = c.execute("SELECT cash FROM accounts WHERE user=?", (user,)).fetchone()["cash"]
        h = c.execute("SELECT qty, avg_cost FROM holdings WHERE user=? AND ticker=?",
                      (user, ticker)).fetchone()
        if side == "BUY":
            if total > cash:
                return False, f"Insufficient funds: need {total:,.2f}, have {cash:,.2f}."
            if h:
                nq = h["qty"] + qty
                avg = (h["qty"] * h["avg_cost"] + total) / nq
                c.execute("UPDATE holdings SET qty=?, avg_cost=? WHERE user=? AND ticker=?",
                          (nq, avg, user, ticker))
            else:
                c.execute("INSERT INTO holdings VALUES(?,?,?,?)", (user, ticker, qty, price))
            c.execute("UPDATE accounts SET cash=cash-? WHERE user=?", (total, user))
        else:
            if not h or h["qty"] < qty:
                return False, "You don't own enough shares to sell."
            if h["qty"] == qty:
                c.execute("DELETE FROM holdings WHERE user=? AND ticker=?", (user, ticker))
            else:
                c.execute("UPDATE holdings SET qty=qty-? WHERE user=? AND ticker=?",
                          (qty, user, ticker))
            c.execute("UPDATE accounts SET cash=cash+? WHERE user=?", (total, user))
        c.execute("INSERT INTO transactions(user,ts,ticker,side,qty,price,total) VALUES(?,?,?,?,?,?,?)",
                  (user, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticker, side, qty, price, total))
    return True, f"{side} {qty} × {ticker} @ {price:,.2f} executed."


def holdings(user) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql("SELECT ticker, qty, avg_cost FROM holdings WHERE user=?", c, params=(user,))


def transactions(user) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql("SELECT ts, ticker, side, qty, price, total FROM transactions "
                           "WHERE user=? ORDER BY id DESC", c, params=(user,))


def snapshot(user, equity):
    with _conn() as c:
        c.execute("INSERT INTO snapshots VALUES(?,?,?)",
                  (user, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), equity))


def snapshots(user) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql("SELECT ts, equity FROM snapshots WHERE user=? ORDER BY ts", c, params=(user,))


def reset(user):
    with _conn() as c:
        for t in ("holdings", "transactions", "snapshots"):
            c.execute(f"DELETE FROM {t} WHERE user=?", (user,))
        c.execute("UPDATE accounts SET cash=? WHERE user=?", (START_CASH, user))
