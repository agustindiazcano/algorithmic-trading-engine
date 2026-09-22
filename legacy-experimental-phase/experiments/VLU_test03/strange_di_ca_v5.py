import websocket
import json
import pandas as pd
import numpy as np
import requests
import time
from collections import deque

# ==========================================
# 🔒 CONSTANTES DEL SISTEMA DI-CA
# ==========================================
SYMBOL = "pepeusdt"  # Minúsculas para el stream
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
DF_MIN = 1.40
DF_MAX = 1.65

# Buffer de Memoria: 3000 velas (aprox. 2 días) para el "Ojo de Dios"
MEMORY_SIZE = 3000
candle_buffer = deque(maxlen=MEMORY_SIZE)
god_trend = "NEUTRO" # Estado inicial

# ==========================================
# 1. MOTOR DE CÁLCULO (FÍSICA)
# ==========================================
def calculate_higuchi_df_live(prices, k_max=10):
    """Calcula la Dimensión Fractal (HFD) de una serie de precios."""
    x = np.array(prices)
    N = len(x)
    if N < 20: return 1.5
    
    L = []
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2: continue
            diffs = np.abs(np.diff(x[idxs]))
            L_m = np.sum(diffs)
            norm = (N - 1) / (len(idxs) * k)
            Lk.append((L_m * norm) / k)
        if Lk: L.append(np.mean(Lk))
    
    if len(L) < 2: return 1.5
    try:
        coeffs = np.polyfit(np.log(1/np.arange(1, len(L)+1)), np.log(L), 1)
        return coeffs[0]
    except: return 1.5

def calculate_metrics(df_live):
    """Calcula DCPI, Umbral Dinámico y Fractalidad en tiempo real."""
    # 1. Física de Fluidos (Presión)
    # Calculamos la diferencia de precio y el flujo volumétrico
    df_live['delta_p'] = df_live['close'].diff().abs()
    df_live['flow'] = df_live['delta_p'] * (df_live['volume'] ** ALPHA_UNIVERSAL)
    
    # Promedio móvil de 100 periodos para normalizar la presión actual
    # Usamos fillna(0) para evitar errores en el arranque
    flow_rolling = df_live['flow'].rolling(window=100).mean()
    flow_avg = flow_rolling.iloc[-1] if not pd.isna(flow_rolling.iloc[-1]) else 1.0
    
    current_flow = df_live['flow'].iloc[-1]
    
    # DCPI: Índice de Presión de Di-Ca
    dcpi = (current_flow / (flow_avg + 1e-9)) / K_UNIVERSAL
    
    # 2. Umbral Dinámico (Contexto de 1000 velas)
    # Analizamos qué tan "loco" estuvo el mercado recientemente para ajustar el gatillo
    historical_dcpi = (df_live['flow'] / (flow_rolling + 1e-9)) / K_UNIVERSAL
    # Tomamos el 99% percentile de las últimas 1000 velas
    dynamic_threshold = max(6.0, historical_dcpi.tail(1000).quantile(0.99))
    
    # 3. Fractalidad (Últimas 60 velas)
    last_60_closes = df_live['close'].tail(60).values
    df_val = calculate_higuchi_df_live(last_60_closes)
    
    return dcpi, df_val, dynamic_threshold

def update_god_eye(buffer):
    """Calcula la tendencia macro (3000 mins)"""
    if len(buffer) < 2000: return "CARGANDO..."
    
    df = pd.DataFrame(list(buffer))
    # SMA 2000 como referencia de tendencia maestra
    sma_god = df['c'].rolling(2000).mean().iloc[-1]
    current_price = df['c'].iloc[-1]
    
    if current_price > sma_god: return "ALCISTA 🚀"
    else: return "BAJISTA 🐻"

# ==========================================
# 2. GESTIÓN DE DATOS (INICIALIZACIÓN ROBUSTA)
# ==========================================
def init_historical_data():
    print(" [1/3] Descargando Ojo de Dios (3000 velas de contexto)...")
    
    # Binance solo da 1000 velas por request. Hacemos paginación hacia atrás.
    temp_data = []
    end_time = int(time.time() * 1000)
    
    while len(temp_data) < MEMORY_SIZE:
        limit = min(1000, MEMORY_SIZE - len(temp_data))
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval=1m&limit={limit}&endTime={end_time}"
        try:
            res = requests.get(url).json()
            if not res or isinstance(res, dict) and 'code' in res: break
            
            # Insertamos al principio porque vamos hacia atrás en el tiempo
            temp_data = res + temp_data
            end_time = res[0][0] - 1
            print(f"   ... Cargadas {len(temp_data)}/{MEMORY_SIZE} velas.")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error descarga inicial: {e}")
            break
            
    # Llenamos el buffer
    for k in temp_data:
        candle = {
            't': k[0], 'o': float(k[1]), 'h': float(k[2]), 
            'l': float(k[3]), 'c': float(k[4]), 'v': float(k[5])
        }
        candle_buffer.append(candle)
        
    global god_trend
    god_trend = update_god_eye(candle_buffer)
    print(f" [OK] Memoria Lista. Tendencia Maestra: {god_trend}")

# ==========================================
# 3. CONEXIÓN WEBSOCKET (TIEMPO REAL)
# ==========================================
def on_message(ws, message):
    global god_trend
    try:
        json_msg = json.loads(message)
        k = json_msg['k'] # Objeto K-Line
        
        # Datos en vivo (Formato corto de Binance)
        live_candle = {
            't': k['t'], 'o': float(k['o']), 'h': float(k['h']), 
            'l': float(k['l']), 'c': float(k['c']), 'v': float(k['v'])
        }
        
        # --- PREPARAR DATOS ---
        # Copiamos buffer + Vela Viva para calcular
        temp_data = list(candle_buffer)
        
        # Si es un tick de la misma vela, reemplazamos la última. Si es nueva, append.
        if temp_data and temp_data[-1]['t'] == live_candle['t']:
            temp_data[-1] = live_candle
        else:
            temp_data.append(live_candle)
            
        df = pd.DataFrame(temp_data)
        
        # --- CORRECCIÓN DE COLUMNAS (El fix importante) ---
        df.rename(columns={'c': 'close', 'v': 'volume', 'h': 'high', 'l': 'low', 'o': 'open'}, inplace=True)
        
        # --- CEREBRO PENSANDO (Milisegundos) ---
        dcpi, df_vascular, umbral = calculate_metrics(df)
        price = live_candle['c']
        
        # --- VISUALIZACIÓN TIPO MATRIX ---
        # Estados
        status_presion = "🔴 ALTA" if dcpi > umbral else "🟢 NORMAL"
        
        if DF_MIN < df_vascular < DF_MAX: status_fractal = "🧠 CEREBRO"
        elif df_vascular <= DF_MIN: status_fractal = "💤 PLANO"
        else: status_fractal = "⚡ CAOS"
        
        # Output en la misma línea
        print(f"\r P: {price:.8f} | DCPI: {dcpi:5.2f} (Umb: {umbral:4.1f}) | Df: {df_vascular:.3f} | {status_presion} | {status_fractal} | {god_trend}   ", end="")
        
        # --- CIERRE DE VELA (Cada 1 min) ---
        if k['x']: # Vela cerrada
            final_candle = {
                't': k['t'], 'o': float(k['o']), 'h': float(k['h']), 
                'l': float(k['l']), 'c': float(k['c']), 'v': float(k['v'])
            }
            candle_buffer.append(final_candle)
            
            # Recalculamos el Ojo de Dios (Solo una vez por minuto para ahorrar CPU)
            god_trend = update_god_eye(candle_buffer)
            
            print(f"\n [VELA CERRADA] Buffer: {len(candle_buffer)} | Tendencia actualizada: {god_trend}")

    except Exception as e:
        # A veces al inicio faltan datos para calcular, ignoramos errores de arranque
        pass

def on_error(ws, error): print(f"\n Error WS: {error}")
def on_close(ws, close_status_code, close_msg): print("\n ### Desconectado ### ")

def start_live_feed():
    init_historical_data()
    print(" [2/3] Conectando al Sistema Nervioso de Binance (WebSocket)...")
    print(" [3/3] MONITOR ACTIVO. Presiona Ctrl+C para salir.\n")
    print(" ="*40)
    
    socket_url = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m"
    ws = websocket.WebSocketApp(socket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

if __name__ == "__main__":
    start_live_feed()