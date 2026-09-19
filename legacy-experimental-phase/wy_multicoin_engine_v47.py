import websocket
import json
import threading
from datetime import datetime
import os
import pandas as pd
import time
from wy_function_winners_list_v03 import start_websocket, get_updated_symbols
import wy_function_winners_list_v03
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from wy_autosell_module_test import load_trades_from_file, run, get_conditions
from telegram import Bot
import asyncio
from dotenv import load_dotenv
import subprocess
import math
# ================== V47 =========================
# ================== CONFIGURATION ==========================
# BUY LOGIC: SIMPLE - SELL LOGIC: BOLLINGER LATERAL ; WITH LOSERS ; V6 REAL
REAL_TRADES = False  # True => Real Orders ; False => Simulation
# .env
load_dotenv()
api_key = os.getenv("BINANCE_API_TEST_KEY")
api_secret = os.getenv("BINANCE_API_TEST_SECRET")
client = Client(api_key, api_secret)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Configuration files
TRADE_HISTORY_FILE = "wy_multicoin__v47.json"
OPEN_TRADES_FILE = "wy_multicoin__v47_open_trades.json"
DATAFRAMES_PATH = "dataframes__v47/"

# Strategy parameters
STOP_LOSS_PERCENT = 0.02       # 2% initial stop-loss
TRAILING_STOP_PERCENT = 0.05   # 5% pullback from the peak
MIN_ADX = 25
RSI_BUY_THRESHOLD = 30         # Require RSI>30 to avoid "excessive oversold"
BOLLINGER_SLOPE_THRESHOLD = 0  # Slope >= 0

ENABLE_LIVE_STOPLOSS = False          # Live stoploss based on price data
ENABLE_LIVE_TRAILING_STOP_1 = False   # First trailing stop (drop 2% from new high)
ENABLE_LIVE_TRAILING_STOP_2 = False   # Second trailing stop (2% over the new *profit* only)

MAX_TRADES_OPEN = 1           # Up to 10 total open positions
CAPITAL = 3000.0

# ================== GLOBAL VARIABLES ==================
trade_history = []  # All recorded operations (buys/sells)
open_trades = []    # List of open trades
dataframes = {}     # Stores candles for each symbol
latest_prices = {}  # Global dict to store the most recent trade price per symbol

# ==================  kline API ================== 

def fetch_historical_klines(symbol, interval="1m", limit=150):
    """
    Downloads the most recent 'limit' historical candles for 'symbol' using the Binance REST API.
    Returns a DataFrame with columns: [timestamp, open, high, low, close, volume].
    """

    try:
        # Using the python-binance library (Client)
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        # Each element of klines is a list with ~12 fields,
        # the ones we mainly care about are:
        # 0 => open time,
        # 1 => open,
        # 2 => high,
        # 3 => low,
        # 4 => close,
        # 5 => volume,
        # 6 => close time, ...

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "ignore1", "ignore2", "ignore3", "ignore4", "ignore5"
        ])

        # Keep only what matters and convert to numeric types
        df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        df["open_time"] = df["open_time"].astype(int)
        df["open"]      = df["open"].astype(float)
        df["high"]      = df["high"].astype(float)
        df["low"]       = df["low"].astype(float)
        df["close"]     = df["close"].astype(float)
        df["volume"]    = df["volume"].astype(float)

        # Rename "open_time" => "timestamp" to stay consistent with the rest of the code
        df.rename(columns={"open_time": "timestamp"}, inplace=True)
        return df

    except Exception as e:
        print(f"[ERROR] Could not fetch klines for {symbol}: {e}")
        return pd.DataFrame()  # Empty if there was an error

def load_initial_historical_data(symbols, interval="1m", limit=150):
    """
    For each symbol in 'symbols', downloads 'limit' recent candles
    and stores it in dataframes[symbol].
    """
    for sym in symbols:
        df = fetch_historical_klines(sym, interval=interval, limit=limit)
        if df.empty:
            print(f"[WARNING] Could not load candles for {sym}.")
            continue

        # Sort by timestamp in case Binance doesn't return them in order (usually it does, but just in case).
        df.sort_values("timestamp", inplace=True)

        # In this pipeline we store timestamp, open, high, low, close, volume.
        # It's sometimes worth resetting the index:
        df.reset_index(drop=True, inplace=True)

        # Assign it to the global dataframes dict
        dataframes[sym] = df

        print(f"[INFO] Loaded {len(df)} initial candles for {sym}")


# ==================  MACD DEA CROSSOVER 2 ================== 

def get_macd_crossover_signal(df, short_window=12, long_window=26, signal_window=9):
    """
    Returns:
      - "bullish" if the last candle has MACD > DEA and the previous one MACD < DEA (bullish crossover)
      - "bearish" if the last candle has MACD < DEA and the previous one MACD > DEA (bearish crossover)
      - "none"    if there's no crossover or not enough data
    """
    if len(df) < long_window + 2:
        return "none"

    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    macd_series = short_ema - long_ema
    dea_series  = macd_series.ewm(span=signal_window, adjust=False).mean()

    # Take the last 2 values of MACD and DEA
    macd_prev = macd_series.iloc[-2]
    macd_now  = macd_series.iloc[-1]
    dea_prev  = dea_series.iloc[-2]
    dea_now   = dea_series.iloc[-1]
    
    if macd_prev < dea_prev and macd_now > dea_now:
        return "bullish"
    elif macd_prev > dea_prev and macd_now < dea_now:
        return "bearish"
    else:
        return "none"


def check_macd_slope_condition(df, bars=3, mode="<=0"):
    """
    Checks whether the MACD slope satisfies the `mode` condition
    for the last `bars` consecutive candles.

    `mode` can be "<0", "==0", "<=0", ">0", etc.

    Returns True if ALL of the last `bars` slopes satisfy the condition.
    """
    if len(df) < 26 + bars:
        return False

    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    macd_series = short_ema - long_ema

    # slope[i] = macd[i] - macd[i-1]
    slope = macd_series.diff(1)

    # Take the last `bars` slopes
    last_slopes = slope.iloc[-bars:]

    for val in last_slopes:
        if mode == "<0":
            if not (val < 0):
                return False
        elif mode == "==0":
            # allow a margin for floats
            if not (abs(val) < 1e-12):
                return False
        elif mode == "<=0":
            if not (val <= 0):
                return False
        elif mode == ">0":
            if not (val > 0):
                return False
        # Add more cases as needed

    return True


# ==================  FILE BUILDER HEAD, CUT-IN OR CUT-AWAY STRATEGY ==================
def generate_and_run_crypto_script(symbol, base_script="wy_base_v05.py", output_dir="crypto_scripts"):
    # Verify the base file exists
    if not os.path.exists(base_script):
        print(f"[ERROR] Base file {base_script} does not exist.")
        return

    # Try reading the base file as utf-8, falling back to latin-1 if that fails
    try:
        with open(base_script, "r", encoding="utf-8") as base:
            script_content = base.read()
    except UnicodeDecodeError:
        print("[WARNING] Problem reading as UTF-8. Trying latin-1...")
        with open(base_script, "r", encoding="latin-1") as base:
            script_content = base.read()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a dedicated file for the coin
    crypto_script = os.path.join(output_dir, f"wy_{symbol}_05.py")

    # Replace the symbol in the script
    script_per_crypto = script_content.replace('SYMBOL = "XRPUSDT"', f'SYMBOL = "{symbol}"')

    # Save the modified script
    with open(crypto_script, "w", encoding="utf-8") as crypto_file:
        crypto_file.write(script_per_crypto)

    print(f"[INFO] Script generated: {crypto_script}")

    # Run the script in a separate process
    try:
        subprocess.Popen(["python", crypto_script])
        print(f"[INFO] Script running: {crypto_script}")
    except Exception as e:
        print(f"[ERROR] Could not run script {crypto_script}: {e}")

# ==================  CLOSE TRADE LIVE ==================

def check_live_stops(trade, current_price):
    """
    Applies the various live stop/trailing-stop logics
    if they are active in 'trade'.
    """
    # 1) Update 'highest_price' if we see a new high
    if current_price > trade["highest_price"]:
        trade["highest_price"] = current_price

    entry_price = trade["entry_price"]
    highest_price = trade["highest_price"]
    symbol = trade["symbol"]

    # ================== LIVE STOPLOSS (simple) ==================
    if trade.get("live_stoploss_active"):
        # Example: 5% below entry => just an example
        live_stop_loss_threshold = entry_price * 0.95
        if current_price <= live_stop_loss_threshold:
            print(f"[LIVE STOPLOSS TRIGGER] {symbol} => price={current_price} <= {live_stop_loss_threshold}")
            close_position(trade, current_price,  {}, reason="LiveStopLoss")
            return  # Stop checking after closing

    # ================== TRAILING STOP #1 ==================
    if trade.get("trailing_stop_1_active"):
        # Example: if price rises 10% from the entry => new high is entry*1.10 or more
        # then if it drops 2% from that high => close
        # We'll assume the user only wants to check this once 'highest_price' >= entry*1.10
        if highest_price >= entry_price * 1.10:
            # compute 2% drop from the new high
            trailing_stop_1_threshold = highest_price * (1 - 0.02)  # 2% drop
            if current_price <= trailing_stop_1_threshold:
                print(f"[TRAILING STOP #1 TRIGGER] {symbol} => price={current_price} <= {trailing_stop_1_threshold}")
                close_position(trade, current_price,  {}, reason="TrailingStop1")
                return  # Stop checking after closing

    # ================== TRAILING STOP #2 (2% of new profit) ==================
    if trade.get("trailing_stop_2_active"):
        # If entry=100, highest_price=110 => profit=10 => 2% of profit=0.2 => 
        # stop = 110 - 0.2 => 109.8
        profit_so_far = highest_price - entry_price
        if profit_so_far > 0:
            stop_loss_amount = profit_so_far * 0.02  # 2% of profit
            trailing_stop_2_threshold = highest_price - stop_loss_amount
            if current_price <= trailing_stop_2_threshold:
                print(f"[TRAILING STOP #2 TRIGGER] {symbol} => price={current_price} <= {trailing_stop_2_threshold}")
                close_position(trade, current_price,  {}, reason="TrailingStop2")
                return  # Stop checking after closing


# ==================  ATR STOP ==================

def calculate_atr(df, period=14):
    """
    Average True Range (ATR) measures volatility.
    """
    if len(df) < period:
        return None  # Not enough data
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    # ATR = smoothed average of TR
    atr = tr.rolling(window=period).mean().iloc[-1]
    return atr

def set_atr_stop_loss(trade, df, multiplier=2.0):
    """
    Example: For a newly opened trade, you set an ATR-based stop.
    If you want to keep updating it dynamically, you can recalculate and 
    tighten the stop if your position is in profit.
    """
    atr_value = calculate_atr(df, period=14)
    if atr_value is None:
        return  # Not enough data, skip

    current_price = df['close'].iloc[-1]
    if trade["type"] == "buy":
        # ATR-based stop for a long position => current_price - multiplier * ATR
        atr_stop = current_price - multiplier * atr_value
        trade["atr_stop"] = atr_stop
    else:
        # If you have short logic, you'd do current_price + multiplier*ATR
        pass

def update_atr_stop_loss(trade, df, multiplier=2.0):
    """
    Called periodically to update the ATR-based stop if the trade is in profit,
    or if you want to re-tighten the stop as volatility changes.
    """
    if "atr_stop" not in trade:
        return  # We never set an ATR stop for this trade

    atr_value = calculate_atr(df, period=14)
    if atr_value is None:
        return
    current_price = df['close'].iloc[-1]

    # For a buy trade, we only move the stop up if the new ATR-based stop is higher
    new_atr_stop = current_price - multiplier * atr_value
    if new_atr_stop > trade["atr_stop"]:
        trade["atr_stop"] = new_atr_stop

# ==================  PROFIT CALCULATION ==================

def calculate_total_profit(file_path):
    # Check whether the file exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return None

    # Read the JSON file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print("File is empty or has invalid JSON format.")
        return None

    # Validate and sum the profits
    total_profit = 0.0
    for item in data:
        profit = item.get('profit', {}).get('usdt', None)
        if isinstance(profit, (int, float)):
            total_profit += profit

    return total_profit

# Usage example
file_path = "TRADE_HISTORY_FILE"
result = calculate_total_profit(file_path)
if result is not None:
    print(f"Total profit is: {result}")

# ==================  TELEGRAM BOT ==================

async def send_telegram_message(message):
    """
    Sends a message through a Telegram bot asynchronously.

    Args:
        message (str): Message to send to the Telegram chat.
    """
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        print(f"[INFO] Message sent to Telegram: {message}")
    except Exception as e:
        print(f"[ERROR] Could not send message to Telegram: {e}")

def send_telegram_message_sync(message):
    """
    Sends a message synchronously using the async function.
    """
    try:
        asyncio.run(send_telegram_message(message))
    except Exception as e:
        print(f"[ERROR] Error sending message in sync mode: {e}")

# ==================  JSON PERSISTENCE ==================

# Create the dataframes directory if it doesn't exist
if not os.path.exists(DATAFRAMES_PATH):
    os.makedirs(DATAFRAMES_PATH)

# Initialize JSON files if they don't exist
for file_path in [TRADE_HISTORY_FILE, OPEN_TRADES_FILE]:
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump([], file)  # Initialize as an empty list

# Load trade_history data
try:
    with open(TRADE_HISTORY_FILE, "r") as file:
        trade_history = json.load(file)
except json.JSONDecodeError:
    print(f"[ERROR] {TRADE_HISTORY_FILE} is corrupt. Initializing empty list.")
    trade_history = []

# Load open_trades data
try:
    with open(OPEN_TRADES_FILE, "r") as file:
        open_trades = json.load(file)
except json.JSONDecodeError:
    print(f"[ERROR] {OPEN_TRADES_FILE} is corrupt. Initializing empty list.")
    open_trades = []

# ================== FUNCTIONS ==================

def save_open_trades(file_path=OPEN_TRADES_FILE):
    """
    Saves the list of open trades to a JSON file.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(open_trades, file, indent=4)
        print(f"[INFO] Open trades saved to {file_path}.")
    except Exception as e:
        print(f"[ERROR] Could not save open trades to {file_path}: {e}")

def save_dataframes():
    """
    Saves all DataFrames to CSV files inside the DATAFRAMES_PATH directory,
    limiting each DataFrame to the last 100 rows.
    """
    os.makedirs(DATAFRAMES_PATH, exist_ok=True)
    for symbol, df in dataframes.items():
        try:
            file_path = f"{DATAFRAMES_PATH}{symbol}.csv"
            # Keep the last 100 rows
            df = df.tail(150)
            df.to_csv(file_path, index=False)
            print(f"[INFO] DataFrame saved for {symbol} at {file_path}.")
        except Exception as e:
            print(f"[ERROR] Could not save DataFrame for {symbol}: {e}")


def load_dataframes():
    """
    Loads all DataFrames from CSV files inside the DATAFRAMES_PATH directory.
    """
    if not os.path.exists(DATAFRAMES_PATH):
        print("[INFO] DataFrames directory not found. Initializing empty.")
        return

    for filename in os.listdir(DATAFRAMES_PATH):
        if filename.endswith(".csv"):
            try:
                symbol = filename.replace(".csv", "")
                file_path = f"{DATAFRAMES_PATH}{filename}"
                dataframes[symbol] = pd.read_csv(file_path)
                print(f"[INFO] DataFrame loaded for {symbol} from {file_path}.")
            except Exception as e:
                print(f"[ERROR] Could not load DataFrame from {filename}: {e}")

# ================== LIVE EMERGENCY BRAKE ==================

file_path = TRADE_HISTORY_FILE
trades = load_trades_from_file(file_path)
# Function to poll conditions periodically
def monitor_conditions():
    while True:
        conditions = get_conditions()
        for symbol, condition in conditions.items():
            print(f"{symbol}: Stop-Loss={condition['stop_loss']}, Trailing-Stop={condition['trailing_stop']}")

            # Check if there are open positions for this symbol
            for trade in open_trades[:]:
                if trade["symbol"] == symbol and not trade.get("sell"):
                    # If the condition is met, close the position
                    if condition["stop_loss"]:
                        print(f"[MONITOR] Triggering Stop-Loss for {symbol}")
                        close_position(trade, trade["price"], {}, "Stop-Loss triggered")
                    elif condition["trailing_stop"]:
                        print(f"[MONITOR] Triggering Trailing-Stop for {symbol}")
                        close_position(trade, trade["price"], {}, "Trailing-Stop triggered")
        time.sleep(5)  # Adjust this interval as needed



# ================== GET SYMBOLS ==================

def get_symbols(base_symbols, winners, losers):
    """Dynamically builds the SYMBOLS list based on winners and losers."""
    symbols = base_symbols.copy()

    # Add winners if not already present
    for symbol in winners:
        if symbol not in symbols:
            symbols.append(symbol)

    # Remove losers
    symbols = [symbol for symbol in symbols if symbol not in losers]

    print(f"[DEBUG] Dynamically generated SYMBOLS: {symbols}")
    return symbols

# ================== MACD AND DEA ==================

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculates MACD, Signal Line (DEA) and Histogram from the closing prices."""
    if len(df) < long_window:
        return None, None, None  # Not enough data for the calculation

    # Calculate the EMAs
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()

    # MACD line
    macd = short_ema - long_ema

    # Signal line (DEA)
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()

    # Histogram (optional)
    histogram = macd - signal_line

    return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1] if not macd.empty else (None, None, None)


# ================== HISTORY INITIALIZATION ==================

if os.path.exists(TRADE_HISTORY_FILE):
    try:
        with open(TRADE_HISTORY_FILE, "r") as file:
            trade_history = json.load(file)
    except json.JSONDecodeError:
        print("[ERROR] Corrupt JSON file. Starting with empty history.")
        trade_history = []
else:
    trade_history = []  # If the file doesn't exist, start with empty history


def save_trade_history():
    """Saves 'trade_history' to disk."""
    with open(TRADE_HISTORY_FILE, "w") as file:
        json.dump(trade_history, file, indent=4)

def add_new_trade(new_trade):
    """
    Adds a trade to 'trade_history'.
    Then, if it exceeds 3 elements, removes the oldest one.
    Finally, saves to disk.
    """
    trade_history.append(new_trade)
    while len(trade_history) > 30000:
        trade_history.pop(0)  # Remove the oldest element
    save_trade_history()

# ================== INDICATOR FUNCTIONS ==================

def calculate_adx_components(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = abs(minus_dm.rolling(window=period).sum())

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    if not adx.empty:
        return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
    return None, None, None

def calculate_rsi(df, period=14):
    """Calculates RSI with pandas. Returns None if df is too short."""
    if len(df) < period:
        return None
    deltas = df['close'].diff()
    gains = deltas.where(deltas > 0, 0).rolling(window=period).mean()
    losses = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def calculate_bollinger(df, period=20):
    """Returns (sma_series, upper_band, lower_band)."""
    if len(df) < period:
        return None, None, None
    sma = df['close'].rolling(window=period).mean()
    std_dev = df['close'].rolling(window=period).std()
    if sma.isna().iloc[-1] or std_dev.isna().iloc[-1]:
        return None, None, None
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return sma, upper_band, lower_band

def bollinger_slope(series):
    """Slope between the last 2 SMA values."""
    if len(series) < 2:
        return 0
    return series.iloc[-1] - series.iloc[-2]

## ================== SCORE AND RANKING ==================

def calculate_score(
    adx,
    plus_di,
    minus_di,
    rsi,
    boll_slope,
    volume,
    avg_volume,
    atr,
    macd,
    signal_line,
    min_adx=25
):
    """
    Calculates a score based on the sample ranking logic, but
    extended with volume, ATR and MACD.
    """
    score = 1.0  # Start at 1 to avoid multiplying by zero

    # 1) Evaluate ADX
    # -----------------------------------------------------
    if adx:
        if adx > 50:
            score *= 3
        elif adx > 30:
            score *= 2
        elif adx > 25:
            score *= 1
        else:
            score *= 0  # If ADX is too low, not worth continuing

    # 2) Evaluate Plus DI vs Minus DI
    # -----------------------------------------------------
    if plus_di and minus_di:
        if plus_di > minus_di:
            score *= 2
        else:
            score *= 0  # If there's no positive crossover, discard

    # 3) Evaluate RSI
    # -----------------------------------------------------
    if rsi:
        if 30 <= rsi <= 70:
            score *= 2  # Neutral (good zone)
        elif rsi > 70 or rsi < 30:
            score *= 0.5  # Penalty for overbought/oversold
        else:
            score *= 0

    # 4) Evaluate the Bollinger slope
    # -----------------------------------------------------
    # If the slope is strongly positive:
    if boll_slope > 2:
        # Check that ADX is strong and RSI isn't overbought
        if adx > min_adx and rsi < 70:
            score *= 2
        else:
            return 0.0  # If it doesn't qualify, discard
    # If the slope is moderately positive
    elif 0 < boll_slope <= 2:
        if adx > min_adx:
            score *= 1.5
        else:
            return 0.0
    else:
        # If the slope is zero or negative, discard
        return 0.0

    # 5) Bonuses (example)
    # -----------------------------------------------------
    # Bonus if ADX > 30 and plus_di > minus_di
    if adx > 30 and plus_di > minus_di:
        score += 1

    # Bonus if RSI is in a "healthy" zone and Bollinger slope is positive
    if 30 <= rsi <= 70 and boll_slope > 0:
        score += 1

    # 6) Extra adjustments using volume, ATR, MACD
    # -----------------------------------------------------
    # Volume: if current volume exceeds the average by 20%, give a bonus
    if avg_volume and volume > avg_volume * 1.2:
        score += 1

    # MACD: Simple example -> if MACD > Signal and MACD is above 0, boost the score
    if macd is not None and signal_line is not None:
        if macd > signal_line and macd > 0:
            score *= 1.2  # Small upward multiplier
        elif macd < 0:
            score *= 0.9  # Slight penalty if MACD is negative

    # ATR: Could be used to filter out excessive volatility
    # (Example: if ATR is very high, apply a small penalty)
    if atr and atr > 0:
        # Adjust the threshold based on your market
        if atr > 3:
            score *= 0.95  # Slight penalty

    final_score = round(score, 2)
    return final_score


def rank_cryptos(symbols, dataframes, min_score=7):
    """
    Ranks cryptos using the 'sample ranking' logic, but also
    factors in other indicators (volume, ATR, MACD).
    """
    rankings = []
    for symbol in symbols:
        df = dataframes.get(symbol)
        if df is None or len(df) < 26:
            # We need at least 26 periods for MACD,
            # and a minimum number of candles for ADX/RSI/Bollinger
            continue

        # ----------------- Indicator calculation -----------------
        # ADX, +DI, -DI
        adx, plus_di, minus_di = calculate_adx_components(df, period=14)

        # RSI
        rsi = calculate_rsi(df, period=14)

        # Bollinger
        sma_series, upper_b, lower_b = calculate_bollinger(df, period=20)
        slope = bollinger_slope(sma_series) if sma_series is not None else 0

        # Volume
        if len(df) >= 20:
            avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        else:
            avg_vol = 0
        curr_vol = df['volume'].iloc[-1]

        # ATR
        atr_val = calculate_atr(df, period=14)

        # MACD, DEA (Signal), Hist
        macd_val, dea_val, hist_val = calculate_macd(df)

        # ----------------- Score calculation -----------------
        score = calculate_score(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            rsi=rsi,
            boll_slope=slope,
            volume=curr_vol,
            avg_volume=avg_vol,
            atr=atr_val,
            macd=macd_val,
            signal_line=dea_val,  # DEA is the "signal line"
        )

        # Only add it to the ranking if score > 0
        if score > 0:
            rankings.append((symbol, score))

    # Sort from highest to lowest score
    rankings.sort(key=lambda x: x[1], reverse=True)

    # Filter out those that don't reach the minimum score
    final_rank = [(sym, sc) for sym, sc in rankings if sc >= min_score]
    return final_rank


# ================== TRADE UTILS (BUY/SELL) ==================

def get_sell_quantity(symbol):
    """
    Gets the available quantity to sell for an asset, adjusting it to the minimum lot
    size and rounding it down according to Binance's restrictions.

    Args:
        symbol (str): Trading pair (example: ADAUSDT).

    Returns:
        float: Available quantity to sell, adjusted and rounded down.
    """
    asset = symbol.replace("USDT", "")  # Extract the base asset (example: ADA from ADAUSDT)

    try:
        # Get the asset balance
        balance = client.get_asset_balance(asset=asset)
        if not balance:
            return 0.0

        free_balance = float(balance['free'])  # Free balance available

        # Get the minimum lot size for the symbol
        min_lot = get_minimum_lot_size(symbol)

        # Adjust the balance to the minimum lot and round down
        free_balance = math.floor(free_balance / min_lot) * min_lot

        return free_balance if free_balance >= min_lot else 0.0  # Return 0 if it doesn't meet the minimum lot

    except Exception as e:
        print(f"[ERROR] Could not get balance for {asset}: {e}")
        return 0.0


def get_minimum_lot_size(symbol):
    """
    Gets the minimum lot size allowed by Binance for a symbol.

    Args:
        symbol (str): Trading pair (example: ADAUSDT).

    Returns:
        float: Minimum lot size.
    """
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            return 1.0
        for filt in info['filters']:
            if filt['filterType'] == 'LOT_SIZE':
                return float(filt['minQty'])
        return 1.0
    except Exception as e:
        print(f"[ERROR] Could not get symbol info for {symbol}: {e}")
        return 1.0


def log_failed_operation(error_details, file_name="real_operations_errors.json"):
    """
    Logs a failed operation to a JSON file.

    Args:
        error_details (dict): Details of the failed operation.
        file_name (str): Name of the file where errors will be saved.
    """
    if os.path.exists(file_name):
        try:
            with open(file_name, "r") as file:
                error_log = json.load(file)
        except json.JSONDecodeError:
            error_log = []
    else:
        error_log = []

    # Add error details
    error_log.append(error_details)

    # Save the updated file
    with open(file_name, "w") as file:
        json.dump(error_log, file, indent=4)
    print(f"[INFO] Error logged to {file_name}.")

# Function to round the quantity to the precision allowed by Binance
def round_quantity(symbol, quantity):
    """
    Rounds the quantity to the number of decimals allowed for the symbol.
    """
    try:
        # Get the market restrictions for the symbol
        market_info = client.get_symbol_info(symbol)
        step_size = float(next(filter(lambda f: f['filterType'] == 'LOT_SIZE', market_info['filters']))['stepSize'])
        precision = int(round(-math.log(step_size, 10), 0))  # Calculate the decimal precision
        return round(quantity, precision)
    except Exception as e:
        print(f"[ERROR] Could not round the quantity: {e}")
        return quantity

# Function to execute a trading operation
def execute_trade(side, symbol, quantity=None, max_retries=2, retry_delay=2):

    for attempt in range(max_retries):
        try:
            # If the operation is SELL and no quantity is given, get the available balance
            if side == SIDE_SELL and quantity is None:
                quantity = get_sell_quantity(symbol.split('USDT')[0])  # Extract "BTC" from "BTCUSDT"

            # Round the quantity to the format accepted by Binance

            # Validate against the minimum lot size
            min_lot = get_minimum_lot_size(symbol)  # Make sure this function is implemented
            if quantity < min_lot:
                print(f"[ERROR] Calculated quantity {quantity} < minLot {min_lot} for {symbol}")
                return None

            if REAL_TRADES:
                if side == SIDE_BUY:
                    order = client.order_market_buy(symbol=symbol, quantity=quantity)
                    print(f"[REAL TRADE] BUY {symbol} => {order}")
                    # generate_and_run_crypto_script(symbol)
                    # Send Telegram notification
                    message = (
                        f"✅ BUY ORDER:\n"
                        f"- SYMBOL: {symbol}\n"
                        f"- QTY: {quantity}\n"
                        f"- TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram_message_sync(message)
                else:
                    quantity = get_sell_quantity(symbol.split('USDT')[0])
                    order = client.order_market_sell(symbol=symbol, quantity=quantity)
                    print(f"[REAL TRADE] SELL {symbol} => {order}")
                    # Send Telegram notification
                    message = (
                        f"🚨 SELL ORDER:\n"
                        f"- SYMBOL: {symbol}\n"
                        f"- QTY: {quantity}\n"
                        f"- TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram_message_sync(message)
                return order
            else:
                # Simulated mode
                # generate_and_run_crypto_script(symbol)
                fake_order = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "status": "FILLED",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "isSimulated": True
                }
                send_telegram_message_sync(f"[SIMULATED TRADE] {fake_order}")
                print(f"[SIMULATED TRADE] {fake_order}")
                return fake_order
        except Exception as e:
            error_type = "BUY" if side == SIDE_BUY else "SELL"
            print(f"[ERROR] Error on REAL order ({error_type}, attempt {attempt + 1}/{max_retries}): {e}")
            print(f"[DETAILS] Symbol: {symbol}, Quantity: {quantity}")

            # Build the error message for Telegram
            telegram_message = (
                f"❌ Error on {error_type} attempt ({attempt + 1}/{max_retries}):\n"
                f"- Symbol: {symbol}\n"
                f"- Quantity: {quantity}\n"
                f"- Error: {str(e)}"
            )
            send_telegram_message_sync(telegram_message)

            # Retry if the max retry count hasn't been reached yet
            if attempt < max_retries - 1:
                print(f"[RETRYING] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                # Log the error to the JSON file
                error_details = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": error_type,
                    "symbol": symbol,
                    "quantity": quantity,
                    "error_message": str(e),
                    "attempts": max_retries
                }
                log_failed_operation(error_details)
                print("[FAILED] Could not complete the operation after multiple attempts.")
                return None


def open_position(symbol, current_price, indicators, reason, score):
    """
    Opens a buy position if no other one exists for this symbol
    and MAX_TRADES_OPEN isn't exceeded.
    """
    # Check whether there's already an open position for this symbol
    for t in open_trades:
        if t["symbol"] == symbol and not t.get("sell"):
            print(f"[DEBUG] There's already an open position for {symbol}. Not opening another.")
            return

    if len(open_trades) >= MAX_TRADES_OPEN:
        print(f"[DEBUG] Already {len(open_trades)} open trades (limit={MAX_TRADES_OPEN}). Not buying.")
        return

    # Simple example: fixed bankroll of 900 USDT => 300 USDT per trade
    # file_path2 = "TRADE_HISTORY_FILE"
    # result2 = calculate_total_profit(file_path2)
    used_usdt = CAPITAL / MAX_TRADES_OPEN
    fee = 0.015
    after_fee = used_usdt * (1 - fee)

    min_lot = get_minimum_lot_size(symbol)
    qty = int(after_fee / current_price)
    if qty < min_lot:
        print(f"[ERROR] Calculated quantity {qty} < minLot {min_lot} for {symbol}")
        return

    order = execute_trade(SIDE_BUY, symbol, qty)
    if order:
        buy_dict = {
            "id": len(trade_history) + 1,
            "symbol": symbol,
            "type": "buy",
            "price": current_price,
            "entry_price": current_price,
            "quantity": qty,
            "total_usdt": current_price * qty,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "reason": reason,
            "indicators": indicators,
            "sell": None,
            "profit": None,
            "is_real": REAL_TRADES,
            "highest_price": current_price,
            "atr_stop": None,
            "live_stoploss_active": ENABLE_LIVE_STOPLOSS,
            "trailing_stop_1_active": ENABLE_LIVE_TRAILING_STOP_1,
            "trailing_stop_2_active": ENABLE_LIVE_TRAILING_STOP_2,
        }
                # (A) Retrieve the symbol's DataFrame
        df = dataframes.get(symbol)

        # (B) If we have enough data, set the ATR-based stop
        if df is not None:
            set_atr_stop_loss(buy_dict, df, multiplier=2.0)
            # => Now buy_dict["atr_stop"] holds the initial ATR-based stop
            #    (unless there's insufficient data).
        # Instead of trade_history.append(...), we use add_new_trade(...) to cap the history size
        add_new_trade(buy_dict)
        open_trades.append(buy_dict)
        save_open_trades(file_path=OPEN_TRADES_FILE)
def close_position(trade, current_price, indicators, reason):
    """Closes the position (sells)."""
    symbol = trade["symbol"]
    qty = trade["quantity"]
    buy_usdt = trade["total_usdt"]
    sell_usdt = current_price * qty
    profit_usdt = sell_usdt - buy_usdt
    profit_pct = (profit_usdt / buy_usdt) * 100 if buy_usdt != 0 else 0

    order = execute_trade(SIDE_SELL, symbol, qty)
    if order:
        sell_dict = {
            "price": current_price,
            "quantity": qty,
            "total_usdt": sell_usdt,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "indicators": indicators  # Make sure MACD and DEA are included
        }
        # Update within the same object
        trade["sell"] = sell_dict
        trade["profit"] = {
            "usdt": profit_usdt,
            "percent": profit_pct
        }
        # Save to history
        save_trade_history()

        # Remove from open_trades
        open_trades.remove(trade)
        save_open_trades(file_path=OPEN_TRADES_FILE)
        # Detailed log
        print(f"[CLOSE POSITION] {symbol}: SOLD at {current_price} USDT")
        print(f"  > Profit: {profit_usdt:.2f} USDT ({profit_pct:.2f}%)")
        print(f"  > Reason: {reason}")
        if "macd" in indicators and "signal_line" in indicators:
            print(f"  > MACD: {indicators['macd']:.6f}, DEA: {indicators['signal_line']:.6f}")

    else:
        print(f"[ERROR] Could not execute the sell order for {symbol}")


# ================== STOP-LOSS / TRAILING STOP / ATR STOP==================

def check_stops_for_trade(trade, current_price):
    """
    Existing:
      - STOP_LOSS_PERCENT (simple)
      - TRAILING_STOP_PERCENT (pullback from highest_price)
    Adds:
      - ATR-based stop if 'atr_stop' is set in the trade object
    """
    if trade.get("sell"):
        return  # Already closed

    buy_price = trade["price"]
    highest_price = trade.get("highest_price", buy_price)

    # Update highest price
    if current_price > highest_price:
        trade["highest_price"] = current_price
        # Optionally re-save the trade to JSON
        save_trade_history()

    # 1) Fixed Stop-Loss
    stop_loss_threshold = buy_price * (1 - STOP_LOSS_PERCENT)
    if current_price <= stop_loss_threshold:
        print(f"[STOP-LOSS] {trade['symbol']} => {current_price} <= {stop_loss_threshold}")
        close_position(trade, current_price, {}, reason="Stop-Loss")
        return

    # 2) Trailing Stop
    trailing_threshold = highest_price * (1 - TRAILING_STOP_PERCENT)
    if current_price <= trailing_threshold:
        print(f"[TRAILING STOP] {trade['symbol']} => {current_price} <= {trailing_threshold}")
        close_position(trade, current_price, {}, reason="Trailing-Stop")
        return

    # 3) ATR Stop
    # atr_stop = trade.get("atr_stop")
    # if atr_stop is not None:
        # if current_price <= atr_stop:
            # print(f"[ATR STOP] {trade['symbol']} => {current_price} <= {atr_stop}")
            # close_position(trade, current_price, {}, reason="ATR-Stop")
            # return


# ================== BUY/SELL LOGIC ==================

def strategy_check(symbol, df):
    """
    Main strategy logic for each symbol:

    - Updates/checks stops (ATR, Stop Loss, Trailing Stop)
    - Calculates indicators (ADX, RSI, Bollinger, MACD, ATR, Volume) -> ranking / info
    - Opens a position (BUY) when a bullish MACD crossover is detected (MACD goes from <DEA to >DEA).
    - Closes a position (SELL) when MACD < DEA
      or the MACD slope <= 0 for X consecutive candles (optional).
    """

    # 1) Check that there's enough data for the indicators
    if df.empty or len(df) < 26:
        return

    current_price = df["close"].iloc[-1]

    # 2) Update stops (ATR Stop, Stop Loss, Trailing Stop) for any open trade on this symbol
    for t in open_trades:
        if t["symbol"] == symbol and not t.get("sell"):
            update_atr_stop_loss(t, df, multiplier=2.0)
            check_stops_for_trade(t, current_price)

    # 3) Calculate the "classic" indicators (in case they're needed for ranking or logs)
    adx, plus_di, minus_di = calculate_adx_components(df, period=14)
    rsi_6  = calculate_rsi(df, period=6)
    rsi_12 = calculate_rsi(df, period=12)
    rsi_24 = calculate_rsi(df, period=24)

    sma_series, upper_b, lower_b = calculate_bollinger(df, period=20)
    slope_boll = bollinger_slope(sma_series) if sma_series is not None else 0

    # MACD and DEA (signal_line)
    macd_val, signal_line, _ = calculate_macd(df)  # MACD, DEA, Hist
    atr_val = calculate_atr(df, period=14)         # ATR

    if "volume" in df.columns:
        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(14).mean().iloc[-1]
    else:
        volume = 0
        avg_volume = 0

    # 4) If a 'score' is used for ranking, calculate it here:
    score = calculate_score(
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        rsi=rsi_6,
        boll_slope=slope_boll,
        volume=volume,
        avg_volume=avg_volume,
        atr=atr_val,
        macd=macd_val,
        signal_line=signal_line
    )

    # Debug log
    print(f"Symbol: {symbol}, Score: {score}")

    # Store the indicators in case they need to be passed to 'close_position'
    indicators = {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "rsi_6": rsi_6,
        "rsi_12": rsi_12,
        "rsi_24": rsi_24,
        "boll_slope": slope_boll,
        "macd": macd_val,
        "signal_line": signal_line,
        "upper_band": upper_b.iloc[-1] if upper_b is not None else None,
        "middle_band": sma_series.iloc[-1] if sma_series is not None else None,
        "lower_band": lower_b.iloc[-1] if lower_b is not None else None
    }

    # 5) Check global conditions (manual Stop, manual Trailing) in case the module requires it
    conditions = get_conditions()
    symbol_conditions = conditions.get(symbol, {"stop_loss": False, "trailing_stop": False})
    for t in open_trades[:]:
        if t["symbol"] == symbol and not t.get("sell"):
            if symbol_conditions["stop_loss"]:
                print(f"[STOP-LOSS] {symbol}: Triggering manual Stop-Loss.")
                close_position(t, current_price, indicators, "Manual Stop-Loss")
                continue
            if symbol_conditions["trailing_stop"]:
                print(f"[TRAILING STOP] {symbol}: Triggering manual Trailing-Stop.")
                close_position(t, current_price, indicators, "Manual Trailing-Stop")
                continue

    # 6) Exit (sell) if there's already an open position and MACD<DEA or the MACD slope condition is met
    # ----------------------------------------------------------------------------
    # 2) If there's an open trade, apply the sell conditions
    for t in open_trades[:]:
        if t["symbol"] == symbol and not t.get("sell"):
            # Make sure the required indicators are present
            if (
                sma_series is not None and upper_b is not None and lower_b is not None and
                adx is not None and rsi_12 is not None
            ):
                # 1. Bollinger condition: price touches the upper band
                if current_price >= upper_b.iloc[-1]:
                    print(f"[TAKE PROFIT] {symbol} Price touched the Bollinger upper band => selling")
                    close_position(t, current_price, indicators, "Bollinger Upper Band")

                # 2. RSI condition: overbought (RSI > 70)
                elif rsi_12 > 70:
                    print(f"[TAKE PROFIT] {symbol} RSI > 70 (overbought) => selling")
                    close_position(t, current_price, indicators, "RSI > 70")

                # 3. ADX condition: lack of trend (ADX < 20) and price nearing support
                elif adx < 20 and current_price <= sma_series.iloc[-1]:
                    print(f"[TAKE PROFIT] {symbol} ADX < 20 and price near support (SMA) => selling")
                    close_position(t, current_price, indicators, "ADX < 20 and SMA support")

                # 4. RSI condition: bearish divergence (optional)
                # Example: RSI decreases while price rises
                # if rsi_12 < rsi_6 and current_price >= upper_b.iloc[-1]:
                #     print(f"[TAKE PROFIT] {symbol} Bearish RSI divergence => selling")
                #     close_position(t, current_price, indicators, "RSI Divergence")

    # 7) Entry (buy) if there's NO open position and we have a bullish MACD>DEA crossover
    # ----------------------------------------------------------------------------
    already_open = any(t["symbol"] == symbol and not t.get("sell") for t in open_trades)
    if not already_open:
        # Detect the crossover using get_macd_crossover_signal
        crossover = get_macd_crossover_signal(df)
        if crossover == "bullish":
            reason = "Bullish MACD crossover (MACD from <DEA to >DEA)"
            open_position(symbol, current_price, indicators, reason, score)

    # (Optional) If you want to keep part of the Bollinger/RSI logic for closes,
    # you could merge it here or leave it commented out.
    #
    # Example if you wanted to keep selling on RSI overbought >70, etc.:
    #
    # for t in open_trades[:]:
    #     if t["symbol"] == symbol and not t.get("sell"):
    #         if rsi_12 is not None and rsi_12 > 70:
    #             print(f"[TAKE PROFIT] {symbol}: RSI > 70 => closing position.")
    #             close_position(t, current_price, indicators, "RSI > 70")
    #             continue
    #
    # (and so on)

# ================== WEBSOCKET (MULTI-STREAM) ==================

def build_multi_stream_url(symbols):
    """
    Builds a multi-stream endpoint that includes BOTH 1m kline and trade streams
    for each symbol in the 'symbols' list.
    
    e.g. btcusdt@kline_1m/btcusdt@trade/ethusdt@kline_1m/ethusdt@trade
    """
    stream_list = []
    for sym in symbols:
        stream_list.append(f"{sym.lower()}@kline_1m")  # 1-min klines
        stream_list.append(f"{sym.lower()}@trade")     # tick-by-tick trades

    # Join them with "/"
    stream_part = "/".join(stream_list)
    return f"wss://stream.binance.com:9443/stream?streams={stream_part}"


def on_open(ws):
    print("[DEBUG] Multi-stream WS opened.")
    if REAL_TRADES:
        print("[INFO] REAL orders.")
    else:
        print("[INFO] SIMULATED mode.")

def on_message(ws, message):
    try:
        msg = json.loads(message)

        # 1) If the message is a list, we update SYMBOLS from wy_function_winners_list_v03
        if isinstance(msg, list):
            global SYMBOLS
            SYMBOLS = wy_function_winners_list_v03.get_updated_symbols()
            print(f"[INFO] SYMBOLS dynamically updated: {SYMBOLS}")
            return

        # 2) Otherwise, we expect a dict with "stream" and "data"
        if "stream" not in msg or "data" not in msg:
            return  # Unknown message structure

        stream_type = msg["stream"]  # e.g. "btcusdt@trade", "btcusdt@kline_1m"
        data = msg["data"]

        # 3) If it's a kline event => parse candle data
        if "@kline_" in stream_type:
            kline = data["k"]  # the candle payload
            symbol = kline["s"]
            is_final = kline["x"]         # True if candle is closed
            open_t = kline["t"]
            close_price = float(kline["c"])
            high_price = float(kline["h"])
            low_price = float(kline["l"])
            o_price = float(kline["o"])
            volume = float(kline["v"])

            # Ensure we have a DataFrame for this symbol
            if symbol not in dataframes:
                cols = ["timestamp", "open", "high", "low", "close", "volume"]
                dataframes[symbol] = pd.DataFrame(columns=cols)

            df = dataframes[symbol]

            # If last row is the same timestamp, update it
            if not df.empty and df["timestamp"].iloc[-1] == open_t:
                df.at[df.index[-1], "open"] = o_price
                df.at[df.index[-1], "high"] = high_price
                df.at[df.index[-1], "low"] = low_price
                df.at[df.index[-1], "close"] = close_price
                df.at[df.index[-1], "volume"] = volume
            else:
                # Insert a new row
                new_row = {
                    "timestamp": open_t,
                    "open": o_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume
                }
                df.loc[len(df)] = new_row

            # Limit size to last 150 rows
            if len(df) > 150:
                dataframes[symbol] = df.tail(150)

            dataframes[symbol] = df  # Overwrite global reference

            # If candle just closed => run candle-based logic
            if is_final:
                print(f"[KLINE] Candle closed: {symbol} => close={close_price}")

                # Example: re-rank cryptos
                rankings = rank_cryptos(SYMBOLS, dataframes, min_score=7)
                print(f"[INFO] Updated rankings: {rankings}")

                # Possibly run your strategy on the top 3
                for top_symbol, score in rankings[:3]:
                    if symbol == top_symbol:
                        strategy_check(symbol, df)

        # 4) If it's a trade event => parse trade data for real-time price
        elif "@trade" in stream_type:
            # Example trade event data:
            # {
            #   "e": "trade",  "E": 123456789,  "s": "BNBBTC", "t": 12345,
            #   "p": "0.001",  "q": "100",       "T": 123456785, "m": true, "M": true
            # }
            symbol = data["s"]
            price = float(data["p"])
            quantity = float(data["q"])
            event_time = data["E"]

            # Update live price dict, then do immediate trailing stop
            latest_prices[symbol] = price
            for trade in open_trades[:]:
                if trade["symbol"] == symbol:
                    check_live_stops(trade, price)

            # Debug: print(f"[TRADE] {symbol} => price={price}, qty={quantity}")

        else:
            # Possibly a ticker or unknown stream
            pass

    except Exception as e:
        print("[ERROR] on_message:", e)


def save_state_on_failure():
    """
    Saves the current state (open_trades, trade_history, dataframes) to their respective files.
    """
    try:
        save_open_trades()
        save_trade_history()
        save_dataframes()
        print("[INFO] State saved successfully after failure.")
    except Exception as e:
        print(f"[ERROR] Could not save state after failure: {e}")

def on_error(ws, error):
    save_state_on_failure()
    print("[ERROR] WebSocket error:", error)

def close_all_open_trades(reason="WebSocket Failure"):
    """
    Closes all open positions and logs the reason.
    """
    global open_trades
    for trade in open_trades[:]:
        current_price = trade["price"]  # Use the entry price as a reference
        indicators = {}  # No specific indicators in this case
        close_position(trade, current_price, indicators, reason)
    print("[INFO] All open positions have been closed.")

ERROR_LOG_FILE = "error_log.json"

def log_error_to_file(error_message):
    """
    Saves an error with its timestamp to a JSON file.
    """
    error_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": error_message
    }
    if os.path.exists(ERROR_LOG_FILE):
        try:
            with open(ERROR_LOG_FILE, "r") as file:
                error_log = json.load(file)
        except json.JSONDecodeError:
            error_log = []
    else:
        error_log = []
    error_log.append(error_entry)
    with open(ERROR_LOG_FILE, "w") as file:
        json.dump(error_log, file, indent=4)
    print(f"[INFO] Error logged: {error_message}")

MAX_RETRIES = 60  # Maximum number of retries allowed
retries = 0  # Retry counter

def on_close(ws, close_status_code, close_msg):
    """
    Handles the WebSocket close event.
    Saves the state, closes open positions, and logs the error.
    """
    global retries

    # Log the close event to the error log file
    error_message = f"WebSocket closed. Code: {close_status_code}, Message: {close_msg}"
    log_error_to_file(error_message)

    # Save the current state to disk
    save_state_on_failure()

    # Close all open positions
    #close_all_open_trades(reason="WebSocket Failure")

    # Attempt to reconnect or terminate
    if retries < MAX_RETRIES:
        retries += 1
        print(f"[DEBUG] WebSocket closed. Attempting reconnection ({retries}/{MAX_RETRIES})...")
        time.sleep(5)  # Wait 5 seconds before reconnecting
        run_bot()  # Retry connecting
    else:
        print("[ERROR] Reconnection limit reached. Terminating the program.")

# ================== MAIN: RUNNING MULTI-STREAM ==================

def run_bot():
    # 1) Start the auxiliary WS (winners/losers) if needed
    threading.Thread(target=start_websocket, daemon=True).start()

    time.sleep(5)

    global SYMBOLS
    SYMBOLS = get_updated_symbols()  # E.g.: top winners, etc.
    print("[INFO] Initial SYMBOLS list updated:", SYMBOLS)

    # 2) Load initial historical data
    load_initial_historical_data(SYMBOLS, interval="1m", limit=150)
    # (dataframes[symbol] now has ~150 prior candles)

    # 3) Build the multi-stream URL
    url = build_multi_stream_url(SYMBOLS)
    print("[DEBUG] Connecting to:", url)

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()


file_name = os.path.basename(__file__)

if __name__ == "__main__":
    # Startup message
    start_bot_message = f">>>Starting operations {file_name}"
    send_telegram_message_sync(start_bot_message)

    # Run the bot
    run_bot()