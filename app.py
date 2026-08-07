import datetime
from flask import Flask, render_template
import numpy as np
import pandas as pd
from smartmoneyconcepts import smc
from xgboost import XGBClassifier
import yfinance as yf

app = Flask(__name__)


def analyze_market():
    # Download EURUSD 15m data using yfinance for cloud compatibility
    df = yf.download(
        tickers="EURUSD=X", period="5d", interval="15m", progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

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
    df["dist_from_support"] = (df["close"] - df["support_zone"]) / df["close"]
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

    # Train Model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        eval_metric="logloss",
    )
    model.fit(X.iloc[:-200], df["target"].iloc[:-200])

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
    }


@app.route("/")
def index():
    data = analyze_market()
    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)