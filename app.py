import datetime
from flask import Flask, render_template
import numpy as np
import pandas as pd
import requests
from smartmoneyconcepts import smc
from xgboost import XGBClassifier
import yfinance as yf

app = Flask(__name__)


def fetch_forex_data():
    """Fetches EUR/USD candles with custom User-Agent headers to bypass cloud rate-limits."""
    try:
        # Create a session with browser headers to avoid IP blocks on Render
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        ticker = yf.Ticker("EURUSD=X", session=session)
        df = ticker.history(period="5d", interval="15m")

        if df.empty:
            # Fallback direct download
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
            return None

        return df
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None


def analyze_market():
    df = fetch_forex_data()

    # Safety Guard: Return friendly error message if data provider fails
    if df is None or len(df) < 50:
        return {
            "status": "DATA PAUSED",
            "confidence": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "error": "Market data provider rate-limited by Yahoo. Refresh in 1-2 minutes.",
        }

    try:
        # SMC Features
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

        # Dynamic train size based on available rows
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
            "time": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "error": None,
        }
    except Exception as e:
        print(f"Analysis Pipeline Error: {e}")
        return {
            "status": "PROCESSING ERROR",
            "confidence": "N/A",
            "ob": False,
            "fvg": False,
            "choch": False,
            "time": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "error": str(e),
        }


@app.route("/")
def index():
    data = analyze_market()
    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)