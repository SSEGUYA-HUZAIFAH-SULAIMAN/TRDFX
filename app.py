import datetime
import os
import sqlite3
import threading
import time
import traceback

from flask import Flask, render_template_string
import numpy as np
import pandas as pd
import requests
from smartmoneyconcepts import smc
from xgboost import XGBClassifier
import yfinance as yf

app = Flask(__name__)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
REFRESH_SECONDS = 300          # retrain/refresh every 5 minutes, not every request
TICKER = "EURUSD=X"
INTERVAL = "15m"
PERIOD = "5d"
RISK_REWARD_RATIO = 2.0        # take-profit distance = risk distance * this
SL_BUFFER_PIPS = 0.0005        # small buffer beyond the swing point (5 pips on EURUSD)
WALK_FORWARD_FOLDS = 4
FOLD_SIZE = 50                 # each fold ~12.5 hours of M15 candles
SIGNAL_HORIZON_CANDLES = 8     # matches the 8-candle (~2hr) prediction target

# Higher-timeframe trend filter
TREND_INTERVAL = "1h"
TREND_PERIOD = "90d"
TREND_FAST_EMA = 50
TREND_SLOW_EMA = 200

# Persistent signal log. NOTE: on Render's free tier the filesystem is
# ephemeral — this file (and everything logged in it) is wiped on every
# redeploy and likely on every spin-down/wake cycle. If you later add a
# Render Disk (paid) or an external DB, just point DB_PATH at that mounted
# path / connection instead — everything else here stays the same.
DB_PATH = os.environ.get("DB_PATH", "trading_log.db")

# ----------------------------------------------------------------------
# SHARED STATE (what every web request actually reads — instant, no I/O)
# ----------------------------------------------------------------------
_state_lock = threading.Lock()
_state = {
    "status": "WARMING UP",
    "confidence": "N/A",
    "backtest_accuracy": "N/A",
    "ob": False,
    "fvg": False,
    "choch": False,
    "time": None,
    "next_candle": "N/A",
    "horizon": "M15 candles, ~2 hour horizon (8 candles ahead)",
    "entry_price": "N/A",
    "stop_loss": "N/A",
    "take_profit": "N/A",
    "risk_reward": f"1:{RISK_REWARD_RATIO:g}",
    "htf_trend": "N/A",
    "track_record": "No signals logged yet.",
    "error": "First data/model refresh is still running. Refresh the page shortly.",
}

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Engine</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0d1117; color: #c9d1d9; text-align: center; padding: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px; max-width: 480px; margin: 30px auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
        .buy { color: #3fb950; font-size: 2.2rem; margin: 15px 0; }
        .wait { color: #d29922; font-size: 2.2rem; margin: 15px 0; }
        .error-box { background: #210d10; border: 1px solid #7d1a1d; color: #f85149; padding: 12px; border-radius: 6px; font-size: 0.85rem; text-align: left; overflow-x: auto; white-space: pre-wrap; margin-top: 15px; }
        .badge { background: #21262d; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .stale { color: #8b949e; font-size: 0.75rem; margin-top: 10px; }
        .track { background: #0d1520; border: 1px solid #1f6feb44; color: #79c0ff; padding: 10px; border-radius: 6px; font-size: 0.85rem; text-align: left; margin-top: 10px; }
        a { color: #58a6ff; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🤖 AI Trading Engine</h2>
        <p><strong>Last refreshed:</strong> {{ data.time or "pending..." }}</p>
        <hr style="border-color: #30363d;">

        <h1 class="{{ 'buy' if 'GO AHEAD' in data.status else 'wait' }}">{{ data.status }}</h1>
        <p><strong>Signal Confidence:</strong> <span class="badge">{{ data.confidence }}</span></p>
        <p><strong>Backtested Accuracy:</strong> <span class="badge">{{ data.backtest_accuracy }}</span></p>

        {% if data.error %}
            <div class="error-box"><strong>Notice:</strong> {{ data.error }}</div>
        {% endif %}

        <hr style="border-color: #30363d;">
        <h3>SMC Confluences</h3>
        <p>Order Block (OB): {{ '✅ Present' if data.ob else '❌ None' }}</p>
        <p>Fair Value Gap (FVG): {{ '✅ Present' if data.fvg else '❌ None' }}</p>
        <p>CHOCH Reversal: {{ '✅ Present' if data.choch else '❌ None' }}</p>
        <p>H1 Trend Filter: <strong>{{ data.htf_trend }}</strong></p>

        <hr style="border-color: #30363d;">
        <h3>Trade Plan</h3>
        <p>Entry (next candle open): <strong>{{ data.entry_price }}</strong></p>
        <p>Stop Loss (beyond recent swing): <strong style="color:#f85149;">{{ data.stop_loss }}</strong></p>
        <p>Take Profit ({{ data.risk_reward }} R:R): <strong style="color:#3fb950;">{{ data.take_profit }}</strong></p>

        <hr style="border-color: #30363d;">
        <h3>Trade Timing</h3>
        <p>Next M15 candle open (UTC): <strong>{{ data.next_candle }}</strong></p>
        <p>{{ data.horizon }}</p>

        <hr style="border-color: #30363d;">
        <h3>Live Track Record</h3>
        <div class="track">{{ data.track_record }}</div>
        <p class="stale"><a href="/history">View full signal history →</a></p>

        <p class="stale">Refreshed automatically every {{ refresh_seconds }}s in the background. Track record resets if this instance restarts (free-tier disk is not persistent).</p>
    </div>
</body>
</html>
"""

HISTORY_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal History</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
        table { border-collapse: collapse; width: 100%; max-width: 900px; margin: 0 auto; font-size: 0.85rem; }
        th, td { border: 1px solid #30363d; padding: 6px 8px; text-align: left; }
        th { background: #161b22; }
        .win { color: #3fb950; font-weight: bold; }
        .loss { color: #f85149; font-weight: bold; }
        .pending { color: #d29922; }
        .timeout { color: #8b949e; }
        h2 { text-align: center; }
        a { color: #58a6ff; display:block; text-align:center; margin-bottom: 15px; }
    </style>
</head>
<body>
    <h2>📜 Signal History</h2>
    <a href="/">← Back to dashboard</a>
    <table>
        <tr>
            <th>Logged (UTC)</th>
            <th>Direction</th>
            <th>Confidence</th>
            <th>Entry</th>
            <th>SL</th>
            <th>TP</th>
            <th>H1 Trend</th>
            <th>Outcome</th>
        </tr>
        {% for row in rows %}
        <tr>
            <td>{{ row.created_at }}</td>
            <td>{{ row.direction }}</td>
            <td>{{ row.confidence }}</td>
            <td>{{ row.entry_price }}</td>
            <td>{{ row.stop_loss }}</td>
            <td>{{ row.take_profit }}</td>
            <td>{{ row.htf_trend }}</td>
            <td class="{{ row.outcome_class }}">{{ row.outcome }}</td>
        </tr>
        {% endfor %}
    </table>
    {% if not rows %}<p style="text-align:center;">No signals logged yet.</p>{% endif %}
</body>
</html>
"""


# ----------------------------------------------------------------------
# PERSISTENT SIGNAL LOG (SQLite)
# ----------------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                backtest_accuracy TEXT,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                htf_trend TEXT,
                entry_candle_iso TEXT NOT NULL,
                outcome TEXT NOT NULL DEFAULT 'PENDING',
                outcome_price REAL,
                resolved_at TEXT
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_entry_candle "
            "ON signals(entry_candle_iso, direction)"
        )
        conn.commit()


def log_signal(direction, confidence, backtest_accuracy_str, entry_price,
                stop_loss, take_profit, htf_trend, entry_candle_iso):
    """
    Insert a new actionable signal. Deduplicates on (entry_candle_iso,
    direction) so re-evaluating the same still-open candle every refresh
    cycle doesn't create repeat rows.
    """
    try:
        with get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO signals
                (created_at, direction, confidence, backtest_accuracy, entry_price,
                 stop_loss, take_profit, htf_trend, entry_candle_iso, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """,
                (
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    direction,
                    confidence,
                    backtest_accuracy_str,
                    entry_price,
                    stop_loss,
                    take_profit,
                    htf_trend,
                    entry_candle_iso,
                ),
            )
            conn.commit()
    except Exception:
        # Logging must never break the live pipeline.
        pass


def resolve_pending_signals(df):
    """
    Walk every still-PENDING signal and check the actual M15 candles since
    its entry time: whichever of stop-loss / take-profit was touched first
    determines WIN/LOSS. If neither is touched within the horizon, mark it
    TIMEOUT rather than leaving it pending forever.
    """
    try:
        with get_conn() as conn:
            pending = conn.execute(
                "SELECT id, direction, stop_loss, take_profit, entry_candle_iso "
                "FROM signals WHERE outcome = 'PENDING'"
            ).fetchall()

        if not pending:
            return

        now_utc = pd.Timestamp.now(tz="UTC")

        for row in pending:
            try:
                entry_time = pd.Timestamp(row["entry_candle_iso"])
                if entry_time.tzinfo is None:
                    entry_time = entry_time.tz_localize("UTC")
                else:
                    entry_time = entry_time.tz_convert("UTC")
            except Exception:
                continue

            future_candles = df[df.index >= entry_time]
            if future_candles.empty:
                continue

            horizon_candles = future_candles.iloc[:SIGNAL_HORIZON_CANDLES]
            direction = row["direction"]
            stop_loss = row["stop_loss"]
            take_profit = row["take_profit"]

            outcome = None
            outcome_price = None

            for _, candle in horizon_candles.iterrows():
                high = float(candle["high"])
                low = float(candle["low"])
                if direction == "BUY":
                    hit_tp = high >= take_profit
                    hit_sl = low <= stop_loss
                else:
                    hit_tp = low <= take_profit
                    hit_sl = high >= stop_loss

                if hit_sl:
                    # If both TP and SL fall inside the same candle we can't
                    # know which was touched first from OHLC alone — assume
                    # the conservative (loss) case rather than overstate performance.
                    outcome, outcome_price = "LOSS", stop_loss
                    break
                elif hit_tp:
                    outcome, outcome_price = "WIN", take_profit
                    break

            if outcome is None:
                enough_time_passed = (
                    now_utc - entry_time
                ) >= pd.Timedelta(minutes=15 * SIGNAL_HORIZON_CANDLES)
                enough_candles = len(horizon_candles) >= SIGNAL_HORIZON_CANDLES
                if enough_time_passed or enough_candles:
                    outcome = "TIMEOUT"
                    outcome_price = float(horizon_candles.iloc[-1]["close"])
                else:
                    continue  # still genuinely pending

            with get_conn() as conn:
                conn.execute(
                    "UPDATE signals SET outcome = ?, outcome_price = ?, resolved_at = ? WHERE id = ?",
                    (
                        outcome,
                        outcome_price,
                        datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        row["id"],
                    ),
                )
                conn.commit()
    except Exception:
        # Resolution must never break the live pipeline.
        pass


def get_track_record_summary():
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT outcome, COUNT(*) as n FROM signals GROUP BY outcome"
            ).fetchall()
        counts = {r["outcome"]: r["n"] for r in rows}
    except Exception:
        return "No signals logged yet."

    wins = counts.get("WIN", 0)
    losses = counts.get("LOSS", 0)
    timeouts = counts.get("TIMEOUT", 0)
    pending = counts.get("PENDING", 0)
    resolved = wins + losses
    total = wins + losses + timeouts + pending

    if total == 0:
        return "No signals logged yet. A row is added each time a GO AHEAD signal fires."

    if resolved > 0:
        win_rate = wins / resolved * 100
        win_rate_str = f"{win_rate:.1f}% win rate ({wins}W / {losses}L)"
    else:
        win_rate_str = "Not enough resolved trades yet for a win rate"

    return (
        f"{win_rate_str}. {timeouts} timed out without hitting SL/TP, "
        f"{pending} still open. {total} signals logged in total since this "
        f"instance started."
    )


def get_history_rows(limit=50):
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    except Exception:
        return []

    outcome_class_map = {"WIN": "win", "LOSS": "loss", "PENDING": "pending", "TIMEOUT": "timeout"}
    formatted = []
    for r in rows:
        formatted.append(
            {
                "created_at": r["created_at"],
                "direction": r["direction"],
                "confidence": f"{r['confidence'] * 100:.1f}%",
                "entry_price": f"{r['entry_price']:.5f}",
                "stop_loss": f"{r['stop_loss']:.5f}",
                "take_profit": f"{r['take_profit']:.5f}",
                "htf_trend": r["htf_trend"] or "N/A",
                "outcome": r["outcome"],
                "outcome_class": outcome_class_map.get(r["outcome"], ""),
            }
        )
    return formatted


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def fetch_market_data(interval=INTERVAL, period=PERIOD):
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        ticker = yf.Ticker(TICKER, session=session)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            df = yf.download(
                tickers=TICKER,
                period=period,
                interval=interval,
                progress=False,
                ignore_tz=True,
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]

        if len(df) < 50:
            return None, f"Insufficient data rows returned ({len(df)} rows)."

        return df, None
    except Exception as e:
        return None, str(e)


def determine_htf_trend(trend_df):
    """
    Simple, transparent higher-timeframe trend filter using EMA50 vs EMA200
    on H1 candles: price above both EMAs with EMA50 above EMA200 = uptrend;
    the mirror image = downtrend; anything else = neutral/no clear trend.
    Returns (trend_str, direction) where direction is 'BUY', 'SELL', or None.
    """
    if trend_df is None or len(trend_df) < TREND_SLOW_EMA + 5:
        return "Unavailable (insufficient H1 history)", None

    closes = trend_df["close"]
    ema_fast = closes.ewm(span=TREND_FAST_EMA, adjust=False).mean()
    ema_slow = closes.ewm(span=TREND_SLOW_EMA, adjust=False).mean()

    last_close = float(closes.iloc[-1])
    last_fast = float(ema_fast.iloc[-1])
    last_slow = float(ema_slow.iloc[-1])

    if last_close > last_fast > last_slow:
        return f"UPTREND (H1, EMA{TREND_FAST_EMA}>EMA{TREND_SLOW_EMA})", "BUY"
    elif last_close < last_fast < last_slow:
        return f"DOWNTREND (H1, EMA{TREND_FAST_EMA}<EMA{TREND_SLOW_EMA})", "SELL"
    else:
        return "RANGING / NO CLEAR TREND (H1)", None


def compute_smc_features(df, max_trim=8):
    """
    smartmoneyconcepts can raise IndexError('positional indexers are
    out-of-bounds') when a swing point lands too close to the end of the
    dataframe (common with thin/weekend data). Retry on a trimmed copy
    until it succeeds instead of crashing the refresh cycle.
    """
    last_err = None
    for trim in range(0, max_trim + 1):
        d = df.iloc[: len(df) - trim].copy() if trim else df.copy()
        if len(d) < 50:
            break
        try:
            fvg_df = smc.fvg(d)
            d["fvg_signal"] = fvg_df["FVG"]

            swing_df = smc.swing_highs_lows(d, swing_length=10)
            bos_choch = smc.bos_choch(d, swing_df)
            d["bos_signal"] = bos_choch["BOS"]
            d["choch_signal"] = bos_choch["CHOCH"]

            ob_df = smc.ob(d, swing_df)
            d["order_block"] = ob_df["OB"]

            return d, swing_df, None
        except IndexError as e:
            last_err = e
            continue
    return None, None, last_err or IndexError("SMC feature computation failed after trimming")


def find_recent_swing_level(swing_df, direction, lookback=30):
    """
    Look back through the swing_highs_lows output for the most recent
    confirmed swing point in the given direction, to anchor a stop-loss.
    direction: -1 for swing low (used for BUY stop-loss), 1 for swing high
    (used for SELL stop-loss). Returns the price level, or None if not found.
    Handles the library's actual column names defensively since exact
    column naming can vary between versions.
    """
    if swing_df is None or swing_df.empty:
        return None

    type_col = "HighLow" if "HighLow" in swing_df.columns else None
    level_col = "Level" if "Level" in swing_df.columns else None
    if type_col is None or level_col is None:
        return None

    recent = swing_df.tail(lookback)
    matches = recent[recent[type_col] == direction]
    if matches.empty:
        return None
    return float(matches[level_col].iloc[-1])


# ----------------------------------------------------------------------
# PIPELINE (runs in the background, not per-request)
# ----------------------------------------------------------------------
def run_pipeline():
    current_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    df, err = fetch_market_data(interval=INTERVAL, period=PERIOD)

    # Higher-timeframe trend is fetched independently. If it fails, we don't
    # block the whole pipeline — we just fall back to no trend filtering
    # (documented clearly in the UI) rather than crashing the signal.
    trend_df, trend_err = fetch_market_data(interval=TREND_INTERVAL, period=TREND_PERIOD)
    htf_trend_label, htf_trend_direction = determine_htf_trend(trend_df if not trend_err else None)

    if err or df is None:
        return {
            "status": "DATA PAUSED",
            "confidence": "N/A",
            "backtest_accuracy": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": current_utc,
            "next_candle": "N/A",
            "horizon": "N/A",
            "entry_price": "N/A",
            "stop_loss": "N/A",
            "take_profit": "N/A",
            "risk_reward": f"1:{RISK_REWARD_RATIO:g}",
            "htf_trend": htf_trend_label,
            "track_record": get_track_record_summary(),
            "error": f"Market data provider issue: {err}",
        }

    # Resolve any past pending signals against fresh candle data before
    # generating a new one. Never fatal to the live pipeline.
    resolve_pending_signals(df)

    try:
        df, swing_df, smc_err = compute_smc_features(df)
        if df is None:
            raise smc_err

        df["support_zone"] = df["low"].rolling(50).min()
        df["resistance_zone"] = df["high"].rolling(50).max()
        df["dist_from_support"] = (df["close"] - df["support_zone"]) / df["close"]
        df["dist_from_resistance"] = (df["resistance_zone"] - df["close"]) / df["close"]
        df.fillna(0, inplace=True)

        feature_cols = [
            "fvg_signal",
            "bos_signal",
            "choch_signal",
            "order_block",
            "dist_from_support",
            "dist_from_resistance",
        ]

        X = df[feature_cols]
        df["target"] = (df["close"].shift(-8) > df["close"]).astype(int)

        # Drop the last 8 rows: their target is unknown (shift(-8) looks into
        # the future that hasn't happened yet), so they can't be used for
        # training or backtesting — only for generating today's live signal.
        usable = df.iloc[:-8] if len(df) > 8 else df.iloc[:0]

        BACKTEST_WINDOW = WALK_FORWARD_FOLDS * FOLD_SIZE
        if len(usable) > BACKTEST_WINDOW + 50:
            initial_train_size = len(usable) - BACKTEST_WINDOW
            fold_accuracies = []

            for fold_i in range(WALK_FORWARD_FOLDS):
                fold_train_end = initial_train_size + fold_i * FOLD_SIZE
                fold_test_start = fold_train_end
                fold_test_end = fold_train_end + FOLD_SIZE

                fold_X_train = X.iloc[:fold_train_end]
                fold_y_train = usable["target"].iloc[:fold_train_end]
                fold_X_test = X.iloc[fold_test_start:fold_test_end]
                fold_y_test = usable["target"].iloc[fold_test_start:fold_test_end]

                if len(fold_X_test) == 0:
                    continue

                fold_model = XGBClassifier(
                    n_estimators=50,
                    learning_rate=0.03,
                    max_depth=3,
                    eval_metric="logloss",
                )
                fold_model.fit(fold_X_train, fold_y_train)
                fold_preds = fold_model.predict(fold_X_test)
                fold_accuracies.append(float((fold_preds == fold_y_test.values).mean()))

            train_size = initial_train_size + BACKTEST_WINDOW
            X_train, y_train = X.iloc[:train_size], usable["target"].iloc[:train_size]

            if fold_accuracies:
                mean_acc = float(np.mean(fold_accuracies))
                std_acc = float(np.std(fold_accuracies))
                backtest_accuracy_str = (
                    f"{mean_acc * 100:.1f}% \u00b1 {std_acc * 100:.1f}% "
                    f"(walk-forward, {len(fold_accuracies)} folds)"
                )
            else:
                backtest_accuracy_str = "N/A (not enough history yet)"
        else:
            train_size = len(usable)
            X_train, y_train = X.iloc[:train_size], usable["target"].iloc[:train_size]
            backtest_accuracy_str = "N/A (not enough history yet)"

        model = XGBClassifier(
            n_estimators=50,
            learning_rate=0.03,
            max_depth=3,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        latest_row = df.iloc[[-1]][feature_cols]
        prediction = model.predict(latest_row)[0]

        proba = model.predict_proba(latest_row)
        confidence = float(proba[0][1]) if proba.ndim == 2 else float(proba[1])

        has_fvg = df.iloc[-1]["fvg_signal"] != 0
        has_ob = df.iloc[-1]["order_block"] != 0
        has_choch = df.iloc[-1]["choch_signal"] != 0

        direction = "BUY" if prediction == 1 else "SELL"

        base_signal_ok = confidence >= 0.68 and (has_ob or has_fvg or has_choch)
        trend_aligned = (htf_trend_direction is None) or (htf_trend_direction == direction)

        if base_signal_ok and trend_aligned:
            status = f"GO AHEAD: {direction}"
        elif base_signal_ok and not trend_aligned:
            status = f"WAIT / NO TRADE (counter-trend: model says {direction}, H1 trend disagrees)"
        else:
            status = "WAIT / NO TRADE"

        # Trade plan: entry is the next candle's open (unknown yet, so we
        # use the current close as the best available estimate). Stop-loss
        # anchors to the most recent opposing swing point (SMC-style), with
        # a small buffer since price often wicks slightly past a swing
        # before reversing. Take-profit is derived from the configured
        # risk:reward ratio, not a separate prediction.
        entry_estimate = float(df.iloc[-1]["close"])

        if direction == "BUY":
            swing_level = find_recent_swing_level(swing_df, direction=-1)
            if swing_level is not None:
                stop_loss = swing_level - SL_BUFFER_PIPS
            else:
                stop_loss = entry_estimate - float(df.iloc[-1]["dist_from_support"]) * entry_estimate - SL_BUFFER_PIPS
            risk = entry_estimate - stop_loss
            take_profit = entry_estimate + risk * RISK_REWARD_RATIO
        else:
            swing_level = find_recent_swing_level(swing_df, direction=1)
            if swing_level is not None:
                stop_loss = swing_level + SL_BUFFER_PIPS
            else:
                stop_loss = entry_estimate + float(df.iloc[-1]["dist_from_resistance"]) * entry_estimate + SL_BUFFER_PIPS
            risk = stop_loss - entry_estimate
            take_profit = entry_estimate - risk * RISK_REWARD_RATIO

        valid_levels = risk > 0 and np.isfinite(risk)

        if not valid_levels:
            entry_price_str = f"{entry_estimate:.5f}"
            stop_loss_str = "N/A (no valid swing reference)"
            take_profit_str = "N/A"
        else:
            entry_price_str = f"{entry_estimate:.5f}"
            stop_loss_str = f"{stop_loss:.5f}"
            take_profit_str = f"{take_profit:.5f}"

        # Entry timing: the candle this signal is based on already closed,
        # so the actionable entry point is the OPEN of the next M15 candle.
        last_candle_time = df.index[-1]
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.tz_localize("UTC")
        else:
            last_candle_time = last_candle_time.tz_convert("UTC")
        next_candle_time = last_candle_time + pd.Timedelta(minutes=15)
        next_candle_str = next_candle_time.strftime("%Y-%m-%d %H:%M UTC")

        # Persist actionable signals only (skip WAIT states — they aren't trades).
        if status.startswith("GO AHEAD") and valid_levels:
            log_signal(
                direction=direction,
                confidence=confidence,
                backtest_accuracy_str=backtest_accuracy_str,
                entry_price=entry_estimate,
                stop_loss=stop_loss,
                take_profit=take_profit,
                htf_trend=htf_trend_label,
                entry_candle_iso=next_candle_time.isoformat(),
            )

        return {
            "status": status,
            "confidence": f"{confidence * 100:.2f}%",
            "backtest_accuracy": backtest_accuracy_str,
            "ob": bool(has_ob),
            "fvg": bool(has_fvg),
            "choch": bool(has_choch),
            "time": current_utc,
            "next_candle": next_candle_str,
            "horizon": "Based on M15 candles \u2014 signal targets price movement ~2 hours ahead (8 candles). Not a scalping signal.",
            "entry_price": entry_price_str,
            "stop_loss": stop_loss_str,
            "take_profit": take_profit_str,
            "risk_reward": f"1:{RISK_REWARD_RATIO:g}",
            "htf_trend": htf_trend_label,
            "track_record": get_track_record_summary(),
            "error": None,
        }
    except Exception:
        return {
            "status": "PROCESSING ERROR",
            "confidence": "N/A",
            "backtest_accuracy": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": current_utc,
            "next_candle": "N/A",
            "horizon": "N/A",
            "entry_price": "N/A",
            "stop_loss": "N/A",
            "take_profit": "N/A",
            "risk_reward": f"1:{RISK_REWARD_RATIO:g}",
            "htf_trend": "N/A",
            "track_record": get_track_record_summary(),
            "error": f"Pipeline failure:\n{traceback.format_exc()}",
        }


# ----------------------------------------------------------------------
# BACKGROUND REFRESH LOOP
# ----------------------------------------------------------------------
def refresh_loop():
    while True:
        try:
            result = run_pipeline()
            with _state_lock:
                _state.update(result)
        except Exception:
            with _state_lock:
                _state["error"] = f"Refresh loop crashed:\n{traceback.format_exc()}"
        time.sleep(REFRESH_SECONDS)


def start_background_refresh():
    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()


# ----------------------------------------------------------------------
# ROUTES — instant, read-only, no blocking work
# ----------------------------------------------------------------------
@app.route("/")
def index():
    with _state_lock:
        data = dict(_state)
    return render_template_string(HTML_LAYOUT, data=data, refresh_seconds=REFRESH_SECONDS)


@app.route("/history")
def history():
    rows = get_history_rows(limit=50)
    return render_template_string(HISTORY_LAYOUT, rows=rows)


@app.route("/healthz")
def healthz():
    # Cheap endpoint for Render/uptime pings — avoids triggering the pipeline.
    return {"ok": True}, 200


@app.errorhandler(Exception)
def handle_any_error(e):
    with _state_lock:
        data = dict(_state)
    data["status"] = "SERVICE ERROR"
    data["error"] = f"{type(e).__name__}: {e}"
    return render_template_string(HTML_LAYOUT, data=data, refresh_seconds=REFRESH_SECONDS), 200


# Initialize DB and start the background loop once, at import time (works
# under gunicorn too, as long as you run a single worker — see note below).
init_db()
start_background_refresh()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)