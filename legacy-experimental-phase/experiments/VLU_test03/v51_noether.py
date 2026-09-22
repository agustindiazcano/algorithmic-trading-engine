import pandas as pd
import numpy as np
import requests
import time
import random

# 🔒 CONSTANTES "V52: THE NOETHER NUKE"
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
PEPE_SATURACION_MIN = 15.8 
PEPE_SATURACION_MAX = 21.0 
DF_MIN = 1.40
DF_MAX = 1.84

# 🌌 FILTRO NOETHER (El Escudo)
NOETHER_TOLERANCE = 2.5  

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

def run_sniper_backtest_noether(df):
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL) 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL
    
    # SENSOR NOETHER
    theoretical_move = (df['volume'] ** ALPHA_UNIVERSAL) 
    real_move_scaled = delta_p * 1000000 
    error_raw = (real_move_scaled - (theoretical_move * K_UNIVERSAL)).abs()
    df['noether_error'] = error_raw.rolling(window=20).mean() / error_raw.rolling(window=100).mean()

    df['df_vascular'] = 1.5 
    vals = [1.5]*100
    for i in range(100, len(df)):
        window = df['close'].iloc[i-60:i]
        vals.append(calculate_higuchi_df(window))
    if len(vals) < len(df): vals += [1.5] * (len(df)-len(vals))
    df['df_vascular'] = pd.Series(vals, index=df.index[:len(vals)])

    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # SEÑAL BLINDADA
    df['signal'] = np.where(
        (df['dcpi'] > PEPE_SATURACION_MIN) & 
        (df['dcpi'] < PEPE_SATURACION_MAX) & 
        (df['df_vascular'] > DF_MIN) & 
        (df['df_vascular'] < DF_MAX) &
        (df['noether_error'] < NOETHER_TOLERANCE), 
        -df['trend'], 0
    )
    return df

def calculate_wallet_pnl(df):
    # ⚠️ MODO NUCLEAR ACTIVADO ⚠️
    balance, leverage, fee = 1000.0, 10.0, 0.001
    position, entry_price = 0, 0
    trades_log, balance_history = [], []
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        
        # --- GESTIÓN DE LIQUIDACIÓN ---
        # Con x50, si el precio se mueve 2% en contra, fuiste.
        if position != 0:
            pnl_pct = (current_price - entry_price) / entry_price * position
            
            # STOP OUT (Liquidación simulada de Binance)
            if pnl_pct <= -0.018: # 1.8% en contra = Liquidación (con buffer)
                balance = 0 # Game Over
                trades_log.append(-balance)
                position = 0
                break # Se termina el test

            if (position == 1 and signal == -1) or (position == -1 and signal == 1):
                pnl_usd = balance * pnl_pct * leverage
                balance += pnl_usd - (balance * leverage * fee)
                trades_log.append(pnl_usd)
                position = 0

        if position == 0 and signal != 0 and balance > 0:
            position, entry_price = signal, current_price
            balance -= (balance * leverage * fee)
        
        balance_history.append(balance)

    return df, trades_log, balance

def run_scientific_experiment(df_full):
    results = []

    print("\n [TEST V52] NOETHER + LEVERAGE x50 (¿El Escudo aguanta la Bomba?)")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    liquidations = 0
    
    for i in range(20):
        start_idx = random.randint(100, len(df_full) - 1001)
        df_window = df_full.iloc[start_idx : start_idx + 1000].copy().reset_index(drop=True)
        
        df_window = run_sniper_backtest_noether(df_window)
        _, logs, final_bal = calculate_wallet_pnl(df_window)
        
        if final_bal <= 1.0:
            liquidations += 1
            print(f" > Test {i+1:02} | 💀 LIQUIDADO 💀")
        else:
            wins = len([p for p in logs if p > 0])
            total = len(logs)
            wr = (wins / total * 100) if total > 0 else 0
            results.append(final_bal)
            roi = ((final_bal/1000)-1)*100
            print(f" > Test {i+1:02} | Bal: ${final_bal:8.2f} | ROI: {roi:6.1f}% | Trades: {total}")

    if results:
        avg = sum(results)/len(results)
        roi_global = ((avg/1000)-1)*100
        print(f" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f" [FIN V52] PROMEDIO (Sobrevivientes): ${avg:.2f} (ROI: {roi_global:.1f}%)")
        print(f" 💀 TASA DE LIQUIDACIÓN: {(liquidations/20)*100:.1f}%")
        
        if liquidations == 0:
            print("\n 🏆 VEREDICTO: NOETHER ES UN ESCUDO DE TITANIO.")
        else:
            print("\n ⚠️ VEREDICTO: x50 ES DEMASIADO RIESGOSO AUN CON FILTRO.")

if __name__ == "__main__":
    df_main = fetch_binance_extended("PEPEUSDT", total_candles=3000)
    if df_main is not None:
        run_scientific_experiment(df_main)