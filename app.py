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

# ----------------------------------------------------------------------
# SHARED STATE (what every web request actually reads — instant, no I/O)
# ----------------------------------------------------------------------
_state_lock = threading.Lock()
_state = {
    "status": "WARMING UP",
    "confidence": "N/A",
    "ob": False,
    "fvg": False,
    "choch": False,
    "time": None,
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
        <p><strong>Model Confidence:</strong> <span class="badge">{{ data.confidence }}</span></p>

        {% if data.error %}
            <div class="error-box"><strong>Notice:</strong> {{ data.error }}</div>
        {% endif %}

        <hr style="border-color: #30363d;">
        <h3>SMC Confluences</h3>
        <p>Order Block (OB): {{ '✅ Present' if data.ob else '❌ None' }}</p>
        <p>Fair Value Gap (FVG): {{ '✅ Present' if data.fvg else '❌ None' }}</p>
        <p>CHOCH Reversal: {{ '✅ Present' if data.choch else '❌ None' }}</p>

        <p class="stale">Refreshed automatically every {{ refresh_seconds }}s in the background.</p>
    </div>
</body>
</html>
"""


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def fetch_market_data():
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        ticker = yf.Ticker(TICKER, session=session)
        df = ticker.history(period=PERIOD, interval=INTERVAL)

        if df.empty:
            df = yf.download(
                tickers=TICKER,
                period=PERIOD,
                interval=INTERVAL,
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

            return d, None
        except IndexError as e:
            last_err = e
            continue
    return None, last_err or IndexError("SMC feature computation failed after trimming")


# ----------------------------------------------------------------------
# PIPELINE (runs in the background, not per-request)
# ----------------------------------------------------------------------
def run_pipeline():
    current_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    df, err = fetch_market_data()

    if err or df is None:
        return {
            "status": "DATA PAUSED",
            "confidence": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": current_utc,
            "error": f"Market data provider issue: {err}",
        }

    try:
        df, smc_err = compute_smc_features(df)
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

        train_size = max(10, len(df) - 50)
        X_train, y_train = X.iloc[:train_size], df["target"].iloc[:train_size]

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
        status = (
            f"GO AHEAD: {direction}"
            if (confidence >= 0.68 and (has_ob or has_fvg or has_choch))
            else "WAIT / NO TRADE"
        )

        return {
            "status": status,
            "confidence": f"{confidence * 100:.2f}%",
            "ob": bool(has_ob),
            "fvg": bool(has_fvg),
            "choch": bool(has_choch),
            "time": current_utc,
            "error": None,
        }
    except Exception:
        return {
            "status": "PROCESSING ERROR",
            "confidence": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": current_utc,
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