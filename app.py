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

# Inline HTML template so no templates/ folder or index.html file is needed
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Signal Engine</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0d1117; color: #c9d1d9; text-align: center; padding: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px; max-width: 450px; margin: 30px auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
        .buy { color: #3fb950; font-size: 2.2rem; margin: 15px 0; }
        .wait { color: #d29922; font-size: 2.2rem; margin: 15px 0; }
        .error-box { background: #210d10; border: 1px solid #7d1a1d; color: #f85149; padding: 12px; border-radius: 6px; font-size: 0.85rem; text-align: left; overflow-x: auto; white-space: pre-wrap; margin-top: 15px; }
        .badge { background: #21262d; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🤖 AI Trading Engine</h2>
        <p><strong>Updated:</strong> {{ data.time }}</p>
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
    </div>
</body>
</html>
"""


def fetch_market_data():
    """Fetches EUR/USD candles with session headers to bypass basic cloud blocks."""
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        ticker = yf.Ticker("EURUSD=X", session=session)
        df = ticker.history(period="5d", interval="15m")

        if df.empty:
            df = yf.download(
                tickers="EURUSD=X",
                period="5d",
                interval="15m",
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


def run_pipeline():
    current_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    df, err = fetch_market_data()

    if err or df is None:
        return {
            "status": "DATA PAUSED",
            "confidence": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": current_utc,
            "error": f"Market data provider rate-limited by Yahoo. Details: {err}",
        }

    try:
        fvg_df = smc.fvg(df)
        df["fvg_signal"] = fvg_df["FVG"]

        swing_df = smc.swing_highs_lows(df, swing_length=10)
        bos_choch = smc.bos_choch(df, swing_df)
        df["bos_signal"] = bos_choch["BOS"]
        df["choch_signal"] = bos_choch["CHOCH"]

        ob_df = smc.ob(df, swing_df)
        df["order_block"] = ob_df["OB"]

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

        return {
            "status": status,
            "confidence": f"{confidence * 100:.2f}%",
            "ob": has_ob,
            "fvg": has_fvg,
            "choch": has_choch,
            "time": current_utc,
            "error": None,
        }
    except Exception as e:
        return {
            "status": "PROCESSING ERROR",
            "confidence": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": current_utc,
            "error": f"Pipeline execution failed:\n{traceback.format_exc()}",
        }


@app.route("/")
def index():
    data = run_pipeline()
    return render_template_string(HTML_LAYOUT, data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)