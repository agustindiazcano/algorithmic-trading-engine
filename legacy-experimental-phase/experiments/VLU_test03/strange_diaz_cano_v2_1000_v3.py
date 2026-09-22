import websocket
import json
import pandas as pd
import numpy as np
import requests
import time
from collections import deque

# ==========================================
# 🔒 CONSTANTES DEL CEREBRO
# ==========================================
SYMBOL = "pepeusdt" # Minúsculas para el stream
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
DF_MIN = 1.40
DF_MAX = 1.65

# Buffer de memoria (Guardamos las últimas 1000 velas para contexto)
MEMORY_SIZE = 1000
candle_buffer = deque(maxlen=MEMORY_SIZE)

# ==========================================
# 1. FÍSICA VASCULAR (Cálculos)
# ==========================================
def calculate_higuchi_df_live(prices, k_max=10):
    # Versión optimizada para listas puras
    x = np.array(prices)
    N = len(x)
    if N < 20: return 1.5
    
    L = []
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            # Optimización vectorial simple
            idxs = np.arange(m, N, k)
            if len(idxs) < 2: continue
            diffs = np.abs(np.diff(x[idxs]))
            L_m = np.sum(diffs)
            norm = (N - 1) / (len(idxs) * k)
            Lk.append((L_m * norm) / k)
        if Lk: L.append(np.mean(Lk))
    
    if len(L) < 2: return 1.5
    try:
        # Pendiente log-log
        coeffs = np.polyfit(np.log(1/np.arange(1, len(L)+1)), np.log(L), 1)
        return coeffs[0]
    except: return 1.5

def calculate_metrics(df_live):
    # 1. Física de Fluidos (Presión)
    df_live['delta_p'] = df_live['close'].diff().abs()
    df_live['flow'] = df_live['delta_p'] * (df_live['volume'] ** ALPHA_UNIVERSAL)
    
    # Promedio móvil de 100 periodos para normalizar
    flow_avg = df_live['flow'].rolling(window=100).mean().iloc[-1]
    current_flow = df_live['flow'].iloc[-1]
    
    dcpi = (current_flow / (flow_avg + 1e-9)) / K_UNIVERSAL
    
    # 2. Umbral Dinámico (Contexto de 1000 velas)
    # Calculamos el DCPI histórico para saber qué es "alto" hoy
    historical_dcpi = (df_live['flow'] / (df_live['flow'].rolling(100).mean() + 1e-9)) / K_UNIVERSAL
    dynamic_threshold = max(6.0, historical_dcpi.quantile(0.99))
    
    # 3. Fractalidad (Últimas 60 velas)
    last_60_closes = df_live['close'].tail(60).values
    df_val = calculate_higuchi_df_live(last_60_closes)
    
    return dcpi, df_val, dynamic_threshold

# ==========================================
# 2. GESTIÓN DE DATOS (REST + WS)
# ==========================================
def init_historical_data():
    print(" [1/3] Descargando memoria a largo plazo (Snapshot REST)...")
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval=1m&limit=1000"
    res = requests.get(url).json()
    
    for k in res:
        # Formato Binance: [t, o, h, l, c, v, ...]
        candle = {
            't': k[0], 'o': float(k[1]), 'h': float(k[2]), 
            'l': float(k[3]), 'c': float(k[4]), 'v': float(k[5])
        }
        candle_buffer.append(candle)
    print(f" [OK] Memoria cargada con {len(candle_buffer)} recuerdos.")

def on_message(ws, message):
    try:
        json_msg = json.loads(message)
        k = json_msg['k'] # Objeto K-Line
        
        # Extraemos datos en vivo (Formato corto de Binance)
        live_candle = {
            't': k['t'], 'o': float(k['o']), 'h': float(k['h']), 
            'l': float(k['l']), 'c': float(k['c']), 'v': float(k['v'])
        }
        
        # Copiamos el buffer para no romperlo
        # Convertimos la lista de diccionarios a DataFrame
        temp_data = list(candle_buffer)
        temp_data.append(live_candle)
        
        df = pd.DataFrame(temp_data)
        
        # 🚨 EL FIX: Traducimos las columnas de 'c' a 'close'
        df.rename(columns={'c': 'close', 'v': 'volume', 'h': 'high', 'l': 'low', 'o': 'open'}, inplace=True)
        
        # --- CEREBRO PENSANDO ---
        # Ahora sí va a encontrar 'close' y 'volume'
        dcpi, df_vascular, umbral = calculate_metrics(df)
        price = live_candle['c']
        
        # --- VISUALIZACIÓN ---
        status_presion = "🔴 ALTA" if dcpi > umbral else "🟢 NORMAL"
        
        # Ajustamos el estado fractal para que sea legible
        if DF_MIN < df_vascular < DF_MAX:
            status_fractal = "🧠 CEREBRO"
        elif df_vascular <= DF_MIN:
            status_fractal = "💤 PLANO"
        else:
            status_fractal = "⚡ CAOS"
        
        # Imprimimos en la misma línea (\r) para efecto Matrix
        print(f"\r P: {price:.8f} | DCPI: {dcpi:5.2f} (Umb: {umbral:4.1f}) | Df: {df_vascular:.3f} | {status_presion} | {status_fractal}", end="")
        
        # GESTIÓN DE MEMORIA
        # Si la vela cerró, actualizamos el buffer oficial
        if k['x']:
            candle_buffer.append(live_candle)
            # Limpiamos línea para el log
            print(f"\n [VELA CERRADA] Memoria Actualizada. Buffer: {len(candle_buffer)}")
            
    except Exception as e:
        print(f"\n [ERROR] {e}")

def on_error(ws, error): print(error)
def on_close(ws, close_status_code, close_msg): print(" ### Desconectado ### ")

def start_live_feed():
    init_historical_data()
    print(" [2/3] Conectando al Sistema Nervioso de Binance (WebSocket)...")
    print(" [3/3] MONITOR ACTIVO. Presiona Ctrl+C para salir.\n")
    
    socket_url = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m"
    ws = websocket.WebSocketApp(socket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

if __name__ == "__main__":
    start_live_feed()