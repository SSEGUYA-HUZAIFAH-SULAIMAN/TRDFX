import datetime
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

# Higher-timeframe trend filter
TREND_INTERVAL = "1h"
TREND_PERIOD = "90d"
TREND_FAST_EMA = 50
TREND_SLOW_EMA = 200

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
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px; max-width: 450px; margin: 30px auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
        .buy { color: #3fb950; font-size: 2.2rem; margin: 15px 0; }
        .wait { color: #d29922; font-size: 2.2rem; margin: 15px 0; }
        .error-box { background: #210d10; border: 1px solid #7d1a1d; color: #f85149; padding: 12px; border-radius: 6px; font-size: 0.85rem; text-align: left; overflow-x: auto; white-space: pre-wrap; margin-top: 15px; }
        .badge { background: #21262d; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .stale { color: #8b949e; font-size: 0.75rem; margin-top: 10px; }
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

        <p class="stale">Refreshed automatically every {{ refresh_seconds }}s in the background.</p>
    </div>
</body>
</html>
"""


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

    # Common column names in smartmoneyconcepts output
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
            "error": f"Market data provider issue: {err}",
        }

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
            # Not enough history yet for a clean walk-forward split — train
            # on everything usable and skip backtesting this cycle.
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

        # Guard against a degenerate/zero risk distance (flat swing data)
        if risk <= 0 or not np.isfinite(risk):
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


# Start the background loop once, at import time (works under gunicorn too,
# as long as you run a single worker — see note below).
start_background_refresh()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)