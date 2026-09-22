import pandas as pd
import numpy as np
import requests
import time
import random

# 🔒 CONSTANTES UNIVERSALES (Leyes de Diaz-Cano 2.0)
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
PEPE_SATURACION_MIN = 15.8   # El piso que te dio el 600%
PEPE_SATURACION_MAX = 21.0   # El techo para evitar el "Blow-off" (Losers promedio)
DF_MIN = 1.40              # Inicio de la zona vascular saludable
DF_MAX = 1.84              # Límite antes del caos lineal

def fetch_binance_extended(symbol="PEPEUSDT", total_candles=3000):
    all_data = []
    end_time = int(time.time() * 1000)
    print(f"[>] Descargando {total_candles} velas para {symbol}...")
    
    while len(all_data) < total_candles:
        limit = min(1000, total_candles - len(all_data))
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}&endTime={end_time}"
        try:
            res = requests.get(url, timeout=10).json()
            if not res or isinstance(res, dict): break
            all_data = res + all_data
            end_time = res[0][0] - 1 
            time.sleep(0.1)
        except: break
            
    if not all_data: return None
    df = pd.DataFrame(all_data, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qa', 'tr', 'tb', 'tq', 'ig'])
    df[['close', 'volume']] = df[['close', 'volume']].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

def calculate_higuchi_df(series, k_max=10):
    """Mide la salud fractal del fluido (Higuchi Fractal Dimension)"""
    N = len(series)
    if N < 20: return 1.5 
    L = []
    x = np.array(series)
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            indices = np.arange(m, N, k)
            if len(indices) < 2: continue
            L_m_k = np.sum(np.abs(np.diff(x[indices])))
            norm = (N - 1) / (len(indices) * k)
            Lk.append(L_m_k * norm / k)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2: return 1.5
    coeffs = np.polyfit(np.log(1/np.arange(1, k_max + 1)), np.log(L), 1)
    return coeffs[0]

def run_sniper_backtest_v2(df):
    # 1. Cálculo de Presión Volumétrica (dcpi)
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL) 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL
    
    # 2. Sensor Vascular (Df)
    # Lógica original: Loop simple respetando tu código
    df['df_vascular'] = 1.5 
    for i in range(100, len(df)):
        window = df['close'].iloc[i-60:i]
        df.at[i, 'df_vascular'] = calculate_higuchi_df(window)

    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # 3. LÓGICA VASCULAR 2.0 (ORIGINAL)
    df['signal'] = np.where(
        (df['dcpi'] > PEPE_SATURACION_MIN) &    # Tiene que haber presión
        (df['dcpi'] < PEPE_SATURACION_MAX) &    # PERO NO DEMASIADA (Evita la euforia)
        (df['df_vascular'] > DF_MIN) & 
        (df['df_vascular'] < DF_MAX), 
        -df['trend'], 0
    )
    return df

def calculate_wallet_pnl(df):
    balance, leverage, fee = 1000.0, 10.0, 0.001
    position, entry_price = 0, 0
    trades_log, balance_history = [], []
    
    # --- NUEVO: Telemetría de Presión ---
    entry_dcpi = 0
    pressure_stats = [] # Guardará tuplas (PnL, Presión de Entrada)

    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        current_dcpi = df['dcpi'].iloc[i] # Leemos la presión actual
        
        if position != 0:
            if (position == 1 and signal == -1) or (position == -1 and signal == 1):
                pnl_usd = balance * ((current_price - entry_price) / entry_price * position * leverage)
                balance += pnl_usd - (balance * leverage * fee)
                
                trades_log.append(pnl_usd)
                # Guardamos el resultado y la presión que tenía al entrar
                pressure_stats.append({'pnl': pnl_usd, 'dcpi_in': entry_dcpi})
                
                position = 0

        if position == 0 and signal != 0:
            position, entry_price = signal, current_price
            entry_dcpi = current_dcpi # Capturamos la presión del momento exacto del disparo
            balance -= (balance * leverage * fee)
        
        balance_history.append(balance)

    df['balance'] = balance_history
    return df, trades_log, balance, pressure_stats

def run_scientific_experiment(df_full):
    results = []
    all_pressure_stats = []

    print("\n [TEST 2.0] VASCULAR BACKTESTS (Ventanas de 1000m)")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for i in range(20):
        start_idx = random.randint(100, len(df_full) - 1001)
        df_window = df_full.iloc[start_idx : start_idx + 1000].copy().reset_index(drop=True)
        
        df_window = run_sniper_backtest_v2(df_window)
        # Recibimos pressure_stats también
        _, logs, final_bal, p_stats = calculate_wallet_pnl(df_window)
        
        all_pressure_stats.extend(p_stats) # Acumulamos para el reporte final

        wins = len([p for p in logs if p > 0])
        total = len(logs)
        wr = (wins / total * 100) if total > 0 else 0
        results.append(final_bal)
        print(f" > Test {i+1:02} | Bal: ${final_bal:8.2f} | WR: {wr:5.1f}% | Df Prom: {df_window['df_vascular'].mean():.2f}")

    avg = sum(results)/20
    print(f" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" [FIN 2.0] PROMEDIO GLOBAL: ${avg:.2f} ({((avg/1000)-1)*100:.1f}%)")
    
    # --- REPORTE DE AUTOPSIA DE PRESIÓN ---
    if all_pressure_stats:
        wins_p = [x['dcpi_in'] for x in all_pressure_stats if x['pnl'] > 0]
        loss_p = [x['dcpi_in'] for x in all_pressure_stats if x['pnl'] <= 0]
        
        avg_win_p = sum(wins_p)/len(wins_p) if wins_p else 0
        avg_loss_p = sum(loss_p)/len(loss_p) if loss_p else 0
        
        print(f"\n 🔬 AUTOPSIA DE PRESIÓN (DCPI en el disparo):")
        print(f" 🔹 Presión Promedio en WINNERS: {avg_win_p:.2f}")
        print(f" 🔸 Presión Promedio en LOSERS:  {avg_loss_p:.2f}")
        print(f" 📊 Diferencial de Fase: {avg_win_p - avg_loss_p:.2f}")
    else:
        print("\n ⚠️ No hubo operaciones para analizar la presión.")

if __name__ == "__main__":
    df_main = fetch_binance_extended("PEPEUSDT", total_candles=3000)
    if df_main is not None:
        run_scientific_experiment(df_main)