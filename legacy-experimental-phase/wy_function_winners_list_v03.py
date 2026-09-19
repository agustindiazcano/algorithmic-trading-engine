import websocket
import json
import threading
import time

# Configuration
FILTER_USDT = True  # True ONLY SYMBOLS WITH USDT
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "PEPEUSDT", "GMTUSDT", "PHAUSDT",
    "STRAXUSDT", "ADAUSDT", "TRXUSDT", "HBARUSDT", "LQTYUSDT", "LUNAUSDT", "ALGOUSDT", "ZECUSDT",
    "IOTAUSDT", "DASHUSDT", "ATAUSDT", "SCRTUSDT", "XVGUSDT", "IDEXUSDT", "OGNUSDT", "MDTUSDT",
    "LITUSDT", "RLCUSDT", "ACXUSDT", "VIBUSDT", "MOVEUSDT", "MEUSDT", "FIROUSDT", "AGLDUSDT",
    "VELODROMEUSDT"
]
WINNERS = []
LOSERS = []

# Process percentage changes to identify winners and losers
def process_data(data):
    changes = []
    for item in data:
        try:
            symbol = item['s']
            price_change_percent = float(item['P'])  # Percentage change
            changes.append({'symbol': symbol, 'priceChangePercent': price_change_percent})
        except KeyError:
            continue

    # Sort by percentage change
    sorted_changes = sorted(changes, key=lambda x: x['priceChangePercent'], reverse=True)

    # Update winners and losers
    global WINNERS, LOSERS
    WINNERS = [item['symbol'] for item in sorted_changes[:10]]  # Top 10 winners
    LOSERS = [item['symbol'] for item in sorted_changes[-10:]]  # Last 10 (losers)

    # Update SYMBOLS
    update_symbols()

# Update the SYMBOLS array
def update_symbols():
    global SYMBOLS
    # Add winners if not already present
    for symbol in WINNERS:
        if symbol not in SYMBOLS:
            SYMBOLS.append(symbol)
    # Filter out symbols that don't include "USDT" if the filter is enabled
    if FILTER_USDT:
        SYMBOLS = [symbol for symbol in SYMBOLS if "USDT" in symbol]
    # print("[INFO] Updated SYMBOLS list:", SYMBOLS)

# Export the updated list
def get_updated_symbols():
    return SYMBOLS

# Export the updated losers list
def get_losers():
    return LOSERS

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
