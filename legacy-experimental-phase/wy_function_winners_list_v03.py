import websocket
import json
import threading
import time

# Configuration
FILTER_USDT = True  # True => only rank/track USDT pairs
TOP_N = 20           # Number of top-volume coins to track

# Fallback list, used only until the first ticker snapshot arrives
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "PEPEUSDT", "GMTUSDT", "PHAUSDT",
    "STRAXUSDT", "ADAUSDT", "TRXUSDT", "HBARUSDT", "LQTYUSDT", "LUNAUSDT", "ALGOUSDT", "ZECUSDT",
    "IOTAUSDT", "DASHUSDT", "ATAUSDT", "SCRTUSDT", "XVGUSDT", "IDEXUSDT", "OGNUSDT", "MDTUSDT",
    "LITUSDT", "RLCUSDT", "ACXUSDT", "VIBUSDT", "MOVEUSDT", "MEUSDT", "FIROUSDT", "AGLDUSDT",
    "VELODROMEUSDT"
]

# Top TOP_N symbols ranked by 24h quote volume, highest first:
# [{'symbol': ..., 'quoteVolume': ...}, ...]
RANKED_BY_VOLUME = []

# Rank symbols by 24h quote volume and keep the top TOP_N
def process_data(data):
    volumes = []
    for item in data:
        try:
            symbol = item['s']
            if FILTER_USDT and "USDT" not in symbol:
                continue
            quote_volume = float(item['q'])  # 24h quote asset volume (USDT, for USDT pairs)
            volumes.append({'symbol': symbol, 'quoteVolume': quote_volume})
        except (KeyError, ValueError):
            continue

    # Sort by quote volume, highest first
    sorted_by_volume = sorted(volumes, key=lambda x: x['quoteVolume'], reverse=True)

    global RANKED_BY_VOLUME
    RANKED_BY_VOLUME = sorted_by_volume[:TOP_N]

    # Update SYMBOLS
    update_symbols()

# Replace SYMBOLS with the current top-TOP_N coins by volume
def update_symbols():
    global SYMBOLS
    if RANKED_BY_VOLUME:
        SYMBOLS = [item['symbol'] for item in RANKED_BY_VOLUME]
    # print("[INFO] Updated SYMBOLS list:", SYMBOLS)

# Export the updated list
def get_updated_symbols():
    return SYMBOLS

# Export the current top-TOP_N ranking (symbol + quote volume)
def get_ranked_by_volume():
    return RANKED_BY_VOLUME

# Callback to process WebSocket messages
def on_message(ws, message):
    data = json.loads(message)
    if isinstance(data, list):
        process_data(data)

# Callback for WebSocket errors
def on_error(ws, error):
    print(f"Error: {error}")

# Callback for WebSocket close
def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

# Callback for WebSocket open
def on_open(ws):
    # Subscribe to the 24-hour price change channel
    payload = {
        "method": "SUBSCRIBE",
        "params": ["!ticker@arr"],
        "id": 1
    }
    ws.send(json.dumps(payload))

# Function to start the WebSocket
def start_websocket():
    url = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# Function to run the update every hour
def update_every_hour():
    while True:
        time.sleep(3600)  # Wait 1 hour (3600 seconds)
        global SYMBOLS
        SYMBOLS = get_updated_symbols()
        # print("[INFO] SYMBOLS updated (hourly):", SYMBOLS)

# Run only if called directly
if __name__ == "__main__":
    # Start WebSocket in a separate thread
    websocket_thread = threading.Thread(target=start_websocket)
    websocket_thread.daemon = True
    websocket_thread.start()

    # Run the hourly update on the main thread
    update_every_hour()
