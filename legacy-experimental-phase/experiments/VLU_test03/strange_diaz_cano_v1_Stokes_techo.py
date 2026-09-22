import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime

# ==========================================
# 🔒 CONSTANTES DE INGENIERÍA (Sin Noether)
# ==========================================
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
PEPE_SATURACION_MIN = 15.8 
PEPE_SATURACION_MAX = 21.0 
DF_MIN = 1.40
DF_MAX = 1.84

# ⚙️ CONFIGURACIÓN DEL VIAJE EN EL TIEMPO
SYMBOL = "PEPEUSDT"
START_DATE = "2024-01-01"
LEVERAGE = 50.0  # ⚠️ MODO NUCLEAR ACTIVADO

def get_random_window(symbol):
    """
    Selecciona un punto aleatorio en la historia desde 2024
    y descarga 1000 velas a partir de ahí.
    """
    start_ts = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(time.time() * 1000) - (1000 * 60 * 1000) 
    
    random_start = random.randint(start_ts, end_ts)
    date_readable = datetime.fromtimestamp(random_start/1000).strftime('%Y-%m-%d %H:%M')
    print(f" [>] Viajando a: {date_readable} ...", end=" ")
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1000&startTime={random_start}"
    
    try:
        res = requests.get(url, timeout=10).json()
        if not res or isinstance(res, dict) or len(res) < 900: 
            print("❌ Error de datos")
            return None
        
        df = pd.DataFrame(res, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qa', 'tr', 'tb', 'tq', 'ig'])
        df[['close', 'volume']] = df[['close', 'volume']].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        print("✅")
        return df
    except Exception as e:
        print(f"❌ Error API: {e}")
        return None

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
        if Lk: L.append(np.mean(Lk))
    if len(L) < 2: return 1.5
    coeffs = np.polyfit(np.log(1/np.arange(1, k_max + 1)), np.log(L), 1)
    return coeffs[0]

def run_strategy_logic(df):
    # FÍSICA
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL) 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL
    
    # FRACTAL
    df['df_vascular'] = 1.5 
    vals = [1.5]*100
    close_vals = df['close'].values
    for i in range(100, len(df)):
        window = close_vals[i-60:i]
        vals.append(calculate_higuchi_df(window))
    
    if len(vals) < len(df): vals += [1.5] * (len(df)-len(vals))
    df['df_vascular'] = pd.Series(vals, index=df.index[:len(vals)])

    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # SEÑAL (SIN NOETHER - A LO MACHO)
    df['signal'] = np.where(
        (df['dcpi'] > PEPE_SATURACION_MIN) & 
        (df['dcpi'] < PEPE_SATURACION_MAX) & 
        (df['df_vascular'] > DF_MIN) & 
        (df['df_vascular'] < DF_MAX),
        -df['trend'], 0
    )
    return df

def calculate_pnl(df):
    balance = 1000.0
    fee = 0.001
    position = 0
    entry_price = 0
    trades_log = []
    
    # LIQUIDACIÓN x50 REALISTA (-1.8%)
    LIQ_THRESH = -0.09 

    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        
        if position != 0:
            pnl_pct = (current_price - entry_price) / entry_price * position
            
            # 💀 CHECK LIQUIDACIÓN
            if pnl_pct <= LIQ_THRESH:
                return 0.0, [-1000] # Murió la cuenta

            # FLIP EXIT
            if (position == 1 and signal == -1) or (position == -1 and signal == 1):
                pnl_usd = balance * pnl_pct * LEVERAGE
                balance += pnl_usd - (balance * LEVERAGE * fee)
                trades_log.append(pnl_usd)
                position = 0

        # ENTRY
        if position == 0 and signal != 0:
            position, entry_price = signal, current_price
            balance -= (balance * LEVERAGE * fee)
            
    return balance, trades_log

def run_quantum_leap():
    print(f"\n ☢️  V60: RAW POWER (No Filter) | {SYMBOL} | LEV x{int(LEVERAGE)} | {START_DATE} -> HOY")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    results = []
    liquidations = 0
    
    for i in range(20):
        df = get_random_window(SYMBOL)
        if df is None: continue
        
        df = run_strategy_logic(df)
        final_bal, logs = calculate_pnl(df)
        
        if final_bal <= 1.0:
            liquidations += 1
            print(f" > Test {i+1:02} | 💀 LIQUIDADO (Cuenta en $0)")
        else:
            wins = len([p for p in logs if p > 0])
            total = len(logs)
            wr = (wins / total * 100) if total > 0 else 0
            roi = ((final_bal/1000)-1)*100
            print(f" > Test {i+1:02} | Bal: ${final_bal:8.2f} | ROI: {roi:6.1f}% | WR: {wr:.1f}% ({total} Ops)")
            results.append(final_bal)

    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if results:
        avg = sum(results)/len(results)
        roi_global = ((avg/1000)-1)*100
        print(f" [FIN V60] PROMEDIO (Sobrevivientes): ${avg:.2f} (ROI: {roi_global:.1f}%)")
    
    print(f" 💀 TASA DE MORTALIDAD (x50 Sin Filtro): {(liquidations/20)*100:.1f}%")
    
    if liquidations > 10:
        print("\n ⚠️ VEREDICTO: NOETHER TENÍA RAZÓN. ESTO ES UN MATADERO.")
    elif liquidations == 0:
        print("\n 🏆 VEREDICTO: NOETHER ERA BASURA. LA FÍSICA SOLA ALCANZA.")
    else:
        print("\n ⚖️ VEREDICTO: RIESGO EXTREMO.")

if __name__ == "__main__":
    run_quantum_leap()