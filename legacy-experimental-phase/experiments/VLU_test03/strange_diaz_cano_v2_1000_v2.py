import pandas as pd
import numpy as np
import requests
import time
import random

# ==========================================
# 🔒 CONSTANTES DE DIAZ-CANO (MODO KAMIKAZE)
# ==========================================
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
DF_MIN = 1.40
DF_MAX = 1.65

# --- CONFIGURACIÓN DE RIESGO ---
LEVERAGE = 10.0        # 10x (Liquidación si el precio se mueve -10%)
TAKE_PROFIT_PCT = 0.15 # Objetivo: +15% de movimiento (+150% ROE)
FEE_RATE = 0.0005      # 0.05% Fee estándar

# ==========================================
# 1. MOTOR DE DATOS (Paginación Robusta)
# ==========================================
def fetch_binance_robust(symbol="PEPEUSDT", total_candles=15000):
    all_data = []
    end_time = int(time.time() * 1000)
    print(f"[>] Descargando flujo masivo para {symbol}...")
    
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

# ==========================================
# 2. FÍSICA VASCULAR (HFD + DCPI)
# ==========================================
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
    
    # Umbral Dinámico (Top 1% de presión local)
    presion_critica_local = df['dcpi'].quantile(0.99)
    SATURACION_DINAMICA = max(6.0, presion_critica_local)
    
    # Cálculo Vascular (Optimizado)
    df['df_vascular'] = 1.5
    close_vals = df['close'].values
    df_vals = [1.5] * len(df)
    
    # Calculamos solo si hay presión mínima para ahorrar CPU
    for i in range(60, len(df)):
         if df['dcpi'].iloc[i] > 2.0: 
            df_vals[i] = calculate_higuchi_df(close_vals[i-60:i])
    df['df_vascular'] = df_vals

    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # SEÑAL CONTRARIAN
    df['signal'] = np.where(
        (df['dcpi'] > SATURACION_DINAMICA) & 
        (df['df_vascular'] > DF_MIN) & 
        (df['df_vascular'] < DF_MAX), 
        -df['trend'], 0
    )
    return df, SATURACION_DINAMICA

# ==========================================
# 3. WALLET KAMIKAZE (SIN STOP LOSS)
# ==========================================
def calculate_wallet_pnl_no_sl(df):
    balance = 1000.0
    position = 0 
    entry_price = 0
    trades = 0
    wins = 0
    
    # Optimizacion Numpy
    lows = df['low'].values
    highs = df['high'].values
    closes = df['close'].values
    signals = df['signal'].values
    
    # Distancia matemática a la liquidación (ej: 1/10 = 0.10 -> 10%)
    liquidation_dist = 1.0 / LEVERAGE 
    
    for i in range(len(df)):
        if balance <= 10: break # Liquidado (Cuenta en cero)

        current_price = closes[i]
        sig = signals[i]
        
        # --- GESTIÓN DE POSICIÓN ---
        if position != 0:
            exit_signal = False
            pnl_pct = 0
            liquidated = False
            
            if position == 1: # Long
                # 1. Chequeo Liquidación (Low)
                curr_drawdown = (lows[i] - entry_price) / entry_price
                if curr_drawdown <= -liquidation_dist: 
                    balance = 0
                    liquidated = True
                # 2. Chequeo Take Profit (High)
                elif (highs[i] - entry_price) / entry_price >= TAKE_PROFIT_PCT:
                    pnl_pct = TAKE_PROFIT_PCT
                    exit_signal = True
                    
            else: # Short
                # 1. Chequeo Liquidación (High)
                curr_drawdown = (entry_price - highs[i]) / entry_price
                if curr_drawdown <= -liquidation_dist: 
                    balance = 0
                    liquidated = True
                # 2. Chequeo Take Profit (Low)
                elif (entry_price - lows[i]) / entry_price >= TAKE_PROFIT_PCT:
                    pnl_pct = TAKE_PROFIT_PCT
                    exit_signal = True
            
            if liquidated: break # Fin del juego
            
            # --- EJECUCIÓN DE SALIDA ---
            if exit_signal:
                balance += balance * pnl_pct * LEVERAGE
                balance -= balance * LEVERAGE * FEE_RATE
                if pnl_pct > 0: wins += 1
                position = 0
            
            # Salida natural por inversión de señal
            elif (position == 1 and sig == -1) or (position == -1 and sig == 1):
                raw_pnl = (current_price - entry_price) / entry_price * position
                balance += balance * raw_pnl * LEVERAGE
                balance -= balance * LEVERAGE * FEE_RATE
                if raw_pnl > 0: wins += 1
                position = 0

        # --- ENTRADA ---
        if position == 0 and sig != 0 and balance > 10:
            position = sig
            entry_price = current_price
            balance -= balance * LEVERAGE * FEE_RATE 
            trades += 1

    return balance, trades, wins

# ==========================================
# 4. EXPERIMENTO
# ==========================================
def run_experiment():
    print(f"\n [TEST KAMI] 30 Muestras | LEV {LEVERAGE}x | LIQ si -{100/LEVERAGE:.0f}% | TP +{TAKE_PROFIT_PCT*100:.0f}%")
    df_master = fetch_binance_robust("PEPEUSDT", 20000) 
    
    if df_master is None or len(df_master) < 2000: return

    global_bal = 0
    samples = 30
    wins_total, trades_total = 0, 0
    
    print(f" ----------------------------------------------------------------")
    for i in range(samples):
        start = random.randint(100, len(df_master) - 1001)
        df_window = df_master.iloc[start : start+1000].copy().reset_index(drop=True)
        
        df_res, umbral = run_adaptive_sniper(df_window)
        
        # USAMOS LA WALLET SIN STOP LOSS (Kamikaze)
        final_bal, trades, wins = calculate_wallet_pnl_no_sl(df_res)
        
        roi = ((final_bal - 1000) / 1000) * 100
        wr = (wins/trades*100) if trades > 0 else 0
        
        if final_bal < 100: icon = "💀"     # Liquidado
        elif final_bal < 1000: icon = "🔻"  # Pérdida flotante
        elif final_bal < 1500: icon = "✅"  # Profit
        else: icon = "🚀"                   # Moon (+50% ROI)
        
        print(f" > Test {i+1:02} | Umb: {umbral:5.2f} | Bal: ${final_bal:7.2f} ({roi:5.1f}%) | WR: {wr:4.0f}% | {icon}")
        
        global_bal += final_bal
        wins_total += wins
        trades_total += trades

    avg_bal = global_bal / samples
    avg_roi = ((avg_bal - 1000) / 1000) * 100
    
    print(" ----------------------------------------------------------------")
    print(f" 📊 RESULTADO KAMI 10x:")
    print(f" 🔹 Capital Promedio: ${avg_bal:.2f}")
    print(f" 🔹 ROI Esperado: {avg_roi:.2f}%")
    print(f" 🔹 Trades: {trades_total} | Wins: {wins_total}")

if __name__ == "__main__":
    run_experiment()