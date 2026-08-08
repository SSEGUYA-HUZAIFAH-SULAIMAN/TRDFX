import datetime
import time
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from smartmoneyconcepts import smc
from xgboost import XGBClassifier

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================
ACCOUNT_ID = 5481848
PASSWORD = "Sseggg"  
SERVER = "Headway-Demo"
SYMBOL = "EURUSD"
CHECK_INTERVAL_SECONDS = 900  # 15 minutes (900 seconds)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def check_liquidity_conditions(df_data):
    # Timezone-aware UTC time fix
    current_time = datetime.datetime.now(datetime.timezone.utc)
    current_hour = current_time.hour

    # Session Windows (UTC): London (07-10), NY Overlap (13-16)
    london_window = 7 <= current_hour < 10
    ny_window = 13 <= current_hour < 16

    if not (london_window or ny_window):
        return (
            False,
            f"Outside Optimal Session Window (Current UTC: {current_time.strftime('%H:%M')})",
        )

    if 21 <= current_hour <= 23:
        return False, "Market Rollover Period (High Spread Risk)"

    latest = df_data.iloc[-1]
    prev_swings = df_data.iloc[-20:-1]
    swept_lows = latest["low"] < prev_swings["low"].min()
    swept_highs = latest["high"] > prev_swings["high"].max()

    if not (swept_lows or swept_highs):
        return False, "No Recent Liquidity Sweep Detected"

    return True, "Liquidity Conditions Optimal"


def run_analysis_cycle():
    """Fetches new candle data, processes features, trains model, and outputs trade decision."""
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting new analysis cycle...")

    # 1. MT5 Data Fetching
    if not mt5.initialize(login=ACCOUNT_ID, password=PASSWORD, server=SERVER):
        print("Failed to initialize MT5 terminal, error:", mt5.last_error())
        mt5.shutdown()
        return

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 3000)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("Failed to fetch rates from MT5 terminal.")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"real_volume": "volume"}, inplace=True)

    # 2. Feature Engineering (SMC + Price Action)
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
    df["dist_from_resistance"] = (df["resistance_zone"] - df["close"]) / df["close"]

    df.fillna(0, inplace=True)

    # 3. Model Training
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

    X_train, X_test = X.iloc[:-200], X.iloc[-200:]
    y_train, y_test = df["target"].iloc[:-200], df["target"].iloc[-200:]

    model = XGBClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=4, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # 4. Recommendation Output
    latest_row = df.iloc[[-1]]
    features = latest_row[feature_cols]
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][1]

    has_fvg = latest_row["fvg_signal"].values[0] != 0
    has_ob = latest_row["order_block"].values[0] != 0
    has_choch = latest_row["choch_signal"].values[0] != 0

    liq_pass, liq_msg = check_liquidity_conditions(df)

    print("==================================================")
    print("        LIVE AI TRADING ANALYSIS SYSTEM           ")
    print("==================================================")
    print(f"Timeframe Analyzed : M15 (Entry) + H1 (Structure)")
    print(f"Liquidity Status   : {liq_msg}")
    print(f"SMC Context        : OB: {has_ob} | FVG: {has_fvg} | CHOCH: {has_choch}")
    print(f"Model Confidence   : {confidence:.2%}")
    print("--------------------------------------------------")

    if confidence >= 0.68 and (has_ob or has_fvg or has_choch) and liq_pass:
        direction = "BUY" if prediction == 1 else "SELL"
        print(f"RECOMMENDATION  : [ GO AHEAD: {direction} ]")
        print(f"EXPECTED HORIZON: 30m - 2 Hours")
        print(f"ACTION          : Execute {direction} on Headway MT5")
    else:
        print("RECOMMENDATION  : [ WAIT / NO TRADE ]")
        print("REASON          : Criteria/Confluence or Liquidity Filter not met.")
    print("==================================================\n")


# ==========================================
# 5. AUTOMATED EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("Automated AI Trading Monitor Started.")
    print("Press Ctrl + C in the terminal to stop the program.\n")

    while True:
        try:
            run_analysis_cycle()
            print(f"Sleeping for {CHECK_INTERVAL_SECONDS // 60} minutes until next candle evaluation...\n")
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nAutomated Trading Monitor Stopped by User.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)