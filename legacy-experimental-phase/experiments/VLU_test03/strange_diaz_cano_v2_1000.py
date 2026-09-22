import pandas as pd
import numpy as np
import requests
import time
import random

# 🔒 CONSTANTES DE DIAZ-CANO (Ajustadas para Supervivencia)
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
DF_MIN = 1.40
DF_MAX = 1.65
LEVERAGE = 1.0       # Bajamos a 20x (Estándar Industrial)
STOP_LOSS_PCT = 0.99  # Si el precio se mueve 2% en contra, CORTAMOS (40% de la equidad)
FEE_RATE = 0.0005     # 0.05% (Binance Futures Standard)

# -----------------------------------------------------------------------------
# 1. MOTOR DE DATOS ROBUSTO
# -----------------------------------------------------------------------------
def fetch_binance_robust(symbol="PEPEUSDT", total_candles=15000):
    all_data = []
    end_time = int(time.time() * 1000)
    print(f"[>] Descargando {total_candles} velas para {symbol}...")
    
    while len(all_data) < total_candles:
        limit = min(1000, total_candles - len(all_data))
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}&endTime={end_time}"
        try:
            res = requests.get(url, timeout=10).json()
            if not res or isinstance(res, dict) and 'code' in res: break
            all_data = res + all_data
            end_time = res[0][0] - 1 
            time.sleep(0.05)
        except: break
            
    if not all_data: return None
    df = pd.DataFrame(all_data, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qa', 'tr', 'tb', 'tq', 'ig'])
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# -----------------------------------------------------------------------------
# 2. FÍSICA Y SEÑALES
# -----------------------------------------------------------------------------
def calculate_higuchi_df(series, k_max=10):
    N = len(series)
    if N < 9: return 1.5
    L = []
    x = np.array(series)
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            L_m = 0
            for i in range(1, int(np.floor((N - m - 1) / k))):
                L_m += abs(x[m + i * k] - x[m + (i - 1) * k])
            norm = (N - 1) / (np.floor((N - m - 1) / k) * k)
            Lk.append((L_m * norm) / k)
        L.append(np.mean(Lk))
    if len(L) < 2: return 1.5
    try:
        coeffs = np.polyfit(np.log(1/np.arange(1, k_max + 1)), np.log(L), 1)
        return coeffs[0]
    except: return 1.5

def run_adaptive_sniper(df):
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL) 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL
    
    # Umbral Dinámico: Top 1% de presión
    presion_critica_local = df['dcpi'].quantile(0.99)
    SATURACION_DINAMICA = max(6.0, presion_critica_local)
    
    # Cálculo optimizado de Df
    df['df_vascular'] = 1.5
    close_vals = df['close'].values
    df_vals = [1.5] * len(df)
    for i in range(60, len(df)):
         # Solo calculamos si hay algo de movimiento para ahorrar CPU
         if df['dcpi'].iloc[i] > 2.0: 
            df_vals[i] = calculate_higuchi_df(close_vals[i-60:i])
    df['df_vascular'] = df_vals

    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # GATILLO
    df['signal'] = np.where(
        (df['dcpi'] > SATURACION_DINAMICA) & 
        (df['df_vascular'] > DF_MIN) & 
        (df['df_vascular'] < DF_MAX), 
        -df['trend'], 0
    )
    return df, SATURACION_DINAMICA

# -----------------------------------------------------------------------------
# 3. WALLET CON GESTIÓN DE RIESGO (STOP LOSS)
# -----------------------------------------------------------------------------
def calculate_wallet_pnl(df):
    balance = 1000.0
    position = 0 # 1: Long, -1: Short
    entry_price = 0
    trades = 0
    wins = 0
    
    # Convertimos a numpy para velocidad extrema
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    signals = df['signal'].values
    
    for i in range(len(df)):
        if balance <= 0: break # Game Over realista

        current_price = closes[i]
        sig = signals[i]
        
        # --- GESTIÓN DE POSICIÓN ABIERTA ---
        if position != 0:
            # 1. Chequeo de STOP LOSS (Intra-vela)
            # Si estamos Long, el precio baja. Si estamos Short, sube.
            stop_hit = False
            
            if position == 1: # Long
                pct_change = (lows[i] - entry_price) / entry_price
                if pct_change < -STOP_LOSS_PCT: # Tocó SL
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                    stop_hit = True
            else: # Short
                pct_change = (entry_price - highs[i]) / entry_price
                if pct_change < -STOP_LOSS_PCT: # Tocó SL
                    exit_price = entry_price * (1 + STOP_LOSS_PCT)
                    stop_hit = True
            
            # 2. Salida por Señal o SL
            if stop_hit:
                # Ejecutar SL
                pnl = -STOP_LOSS_PCT * LEVERAGE
                balance += balance * pnl
                balance -= balance * LEVERAGE * FEE_RATE
                position = 0
            
            elif (position == 1 and sig == -1) or (position == -1 and sig == 1):
                # Salida por reversión (Take Profit técnico)
                pnl = (current_price - entry_price) / entry_price * position * LEVERAGE
                balance += balance * pnl
                balance -= balance * LEVERAGE * FEE_RATE
                if pnl > 0: wins += 1
                position = 0

        # --- ENTRADA NUEVA ---
        if position == 0 and sig != 0 and balance > 10:
            position = sig
            entry_price = current_price
            balance -= balance * LEVERAGE * FEE_RATE # Fee de entrada
            trades += 1

    return balance, trades, wins

# -----------------------------------------------------------------------------
# 4. EXPERIMENTO DE MONTE CARLO
# -----------------------------------------------------------------------------
def run_experiment():
    print("\n [TEST V3: SUPERVIVENCIA] 30 Muestras | Stop Loss 2% | Lev 20x")
    df_master = fetch_binance_robust("PEPEUSDT", 20000) # 20k velas para más variedad
    
    if df_master is None or len(df_master) < 2000:
        print(" [!] Faltan datos."); return

    global_bal = 0
    samples = 30
    wins_total, trades_total = 0, 0
    
    print(f" ----------------------------------------------------------------")
    for i in range(samples):
        start = random.randint(100, len(df_master) - 1001)
        df_window = df_master.iloc[start : start+1000].copy().reset_index(drop=True)
        
        df_res, umbral = run_adaptive_sniper(df_window)
        final_bal, trades, wins = calculate_wallet_pnl(df_res)
        
        roi = ((final_bal - 1000) / 1000) * 100
        wr = (wins/trades*100) if trades > 0 else 0
        
        # Iconos de estado
        if final_bal < 600: icon = "💀"     # Rekt
        elif final_bal < 1000: icon = "🔻"  # Loss
        elif final_bal < 1500: icon = "✅"  # Profit
        else: icon = "🚀"                   # Moon
        
        print(f" > Test {i+1:02} | Umb: {umbral:5.2f} | Bal: ${final_bal:7.2f} ({roi:5.1f}%) | {icon}")
        
        global_bal += final_bal
        wins_total += wins
        trades_total += trades

    avg_bal = global_bal / samples
    avg_roi = ((avg_bal - 1000) / 1000) * 100
    
    print(" ----------------------------------------------------------------")
    print(f" 📊 RESULTADO FINAL V3:")
    print(f" 🔹 Capital Promedio: ${avg_bal:.2f}")
    print(f" 🔹 ROI Esperado: {avg_roi:.2f}% (Menos volatilidad)")
    print(f" 🔹 Trades Totales: {trades_total} | Wins: {wins_total}")
    print(" ----------------------------------------------------------------")

if __name__ == "__main__":
    run_experiment()