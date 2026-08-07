import datetime
import traceback
from flask import Flask, render_template_string
import numpy as np
import pandas as pd
import requests
from smartmoneyconcepts import smc
from xgboost import XGBClassifier
import yfinance as yf

app = Flask(__name__)

# Basic Inline HTML Template to render directly without needing templates/ folder
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Engine Debugger</title>
    <style>
        body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; max-width: 600px; margin: 0 auto; }
        .success { color: #3fb950; font-size: 1.5rem; }
        .warning { color: #d29922; font-size: 1.5rem; }
        .error { color: #f85149; background: #210d10; border: 1px solid #7d1a1d; padding: 15px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }
        .step { background: #21262d; padding: 8px 12px; border-radius: 4px; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🤖 AI Trading Signal - Debug Console</h2>
        <p><strong>Timestamp:</strong> {{ data.time }}</p>
        <hr style="border-color: #30363d;">

        <h3>Execution Status</h3>
        <p class="{{ 'success' if 'GO AHEAD' in data.status else ('warning' if 'WAIT' in data.status else 'error') }}">
            {{ data.status }}
        </p>

        {% if data.debug_logs %}
            <h3>Execution Trace Logs</h3>
            {% for log in data.debug_logs %}
                <div class="step">ℹ️ {{ log }}</div>
            {% endfor %}
        {% endif %}

        {% if data.error %}
            <h3>Traceback Error Details</h3>
            <div class="error">{{ data.error }}</div>
        {% endif %}

        {% if not data.error %}
            <hr style="border-color: #30363d;">
            <p><strong>Confidence:</strong> {{ data.confidence }}</p>
            <p><strong>Order Block:</strong> {{ '✅' if data.ob else '❌' }} | <strong>FVG:</strong> {{ '✅' if data.fvg else '❌' }} | <strong>CHOCH:</strong> {{ '✅' if data.choch else '❌' }}</p>
        {% endif %}
    </div>
</body>
</html>
"""


def fetch_forex_data(debug_logs):
    """Fetches candle data while recording each step into debug_logs."""
    try:
        debug_logs.append("Initiating yfinance request for EURUSD=X...")
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        ticker = yf.Ticker("EURUSD=X", session=session)
        df = ticker.history(period="5d", interval="15m")

        if df.empty:
            debug_logs.append(
                "ticker.history returned empty. Attempting yf.download fallback..."
            )
            df = yf.download(
                tickers="EURUSD=X", period="5d", interval="15m", progress=False
            )

        debug_logs.append(f"Successfully fetched Dataframe with shape: {df.shape}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]
        return df, None
    except Exception as e:
        error_msg = f"Data Fetch Failed:\n{traceback.format_exc()}"
        return None, error_msg


def analyze_market():
    debug_logs = []
    current_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    df, fetch_error = fetch_forex_data(debug_logs)

    if fetch_error:
        return {
            "status": "DATA FETCH ERROR",
            "time": current_utc,
            "error": fetch_error,
            "debug_logs": debug_logs,
        }

    if df is None or len(df) < 50:
        return {
            "status": "INSUFFICIENT DATA",
            "time": current_utc,
            "error": f"Dataframe contains only {len(df) if df is not None else 0} rows. Required >= 50.",
            "debug_logs": debug_logs,
        }

    try:
        debug_logs.append("Processing Smart Money Concepts (FVG, BOS, OB)...")
        fvg_df = smc.fvg(df)
        df["fvg_signal"] = fvg_df["FVG"]

        swing_df = smc.swing_highs_lows(df, swing_length=10)
        bos_choch = smc.bos_choch(df, swing_df)
        df["bos_signal"] = bos_choch["BOS"]
        df["choch_signal"] = bos_choch["CHOCH"]

        ob_df = smc.ob(df, swing_df)
        df["order_block"] = ob_df["OB"]

        debug_logs.append("Calculating Support & Resistance features...")
        df["support_zone"] = df["low"].rolling(50).min()
        df["resistance_zone"] = df["high"].rolling(50).max()
        df["dist_from_support"] = (
            df["close"] - df["support_zone"]
        ) / df["close"]
        df["dist_from_resistance"] = (
            df["resistance_zone"] - df["close"]
        ) / df["close"]
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

        debug_logs.append("Training XGBClassifier Model...")
        train_size = max(10, len(df) - 50)
        X_train, y_train = X.iloc[:train_size], df["target"].iloc[:train_size]

        model = XGBClassifier(
            n_estimators=50,
            learning_rate=0.03,
            max_depth=3,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        latest_row = df.iloc[[-1]]
        features = latest_row[feature_cols]
        prediction = model.predict(features)[0]
        confidence = float(model.predict_proba(features)[0][1])

        has_fvg = latest_row["fvg_signal"].values[0] != 0
        has_ob = latest_row["order_block"].values[0] != 0
        has_choch = latest_row["choch_signal"].values[0] != 0

        direction = "BUY" if prediction == 1 else "SELL"
        status = (
            f"GO AHEAD: {direction}"
            if (confidence >= 0.68 and (has_ob or has_fvg or has_choch))
            else "WAIT / NO TRADE"
        )

        debug_logs.append("Analysis completed successfully!")

        return {
            "status": status,
            "confidence": f"{confidence * 100:.2f}%",
            "ob": has_ob,
            "fvg": has_fvg,
            "choch": has_choch,
            "time": current_utc,
            "error": None,
            "debug_logs": debug_logs,
        }

    except Exception as e:
        pipeline_error = f"Pipeline Processing Error:\n{traceback.format_exc()}"
        return {
            "status": "PIPELINE ERROR",
            "time": current_utc,
            "error": pipeline_error,
            "debug_logs": debug_logs,
        }


@app.route("/")
def index():
    data = analyze_market()
    return render_template_string(HTML_TEMPLATE, data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)