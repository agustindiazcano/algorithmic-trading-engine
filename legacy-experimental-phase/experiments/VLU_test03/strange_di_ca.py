import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import random

# 🔒 CONSTANTES UNIVERSALES (Leyes de Di-Ca)
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
PEPE_SATURACION = 12.0

def fetch_binance_extended(symbol="PEPEUSDT", total_candles=3000):
    """Baja data en bloques de 1000 para superar el límite de la API"""
    all_data = []
    end_time = int(time.time() * 1000)
    
    print(f"[>] Iniciando descarga de {total_candles} velas para {symbol}...")
    
    while len(all_data) < total_candles:
        limit = min(1000, total_candles - len(all_data))
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}&endTime={end_time}"
        
        try:
            res = requests.get(url, timeout=10).json()
            if not res or 'code' in res: break
            
            all_data = res + all_data
            # Retrocedemos en el tiempo para el próximo bloque
            end_time = res[0][0] - 1 
            print(f"   ... {len(all_data)} velas descargadas.")
            time.sleep(0.1) # Respeto al rate limit de la red
        except:
            break
            
    if not all_data: return None
    
    df = pd.DataFrame(all_data, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qa', 'tr', 'tb', 'tq', 'ig'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

def run_sniper_backtest(df):
    delta_p = df['close'].diff().abs()
    # Flujo basado en la Ley Volumétrica de potencia Alpha 
    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL) 
    # Normalización con ventana de 100 para filtrar ruido 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL
    
    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # Threshold Decision D_P(S; 12.0) 
    df['signal'] = np.where(df['dcpi'] > PEPE_SATURACION, -df['trend'], 0)
    return df

def calculate_wallet_pnl(df):
    balance = 1000.0
    leverage = 50.0
    fee = 0.001
    position, entry_price = 0, 0
    trades_log, balance_history = [], []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        
        if position != 0:
            # Lógica de Reversión Determinista 
            if (position == 1 and signal == -1) or (position == -1 and signal == 1):
                pnl_usd = balance * ((current_price - entry_price) / entry_price * position * leverage)
                balance += pnl_usd - (balance * leverage * fee) # Descuento de fricción 
                trades_log.append(pnl_usd)
                position = 0

        if position == 0 and signal != 0:
            position = signal
            entry_price = current_price
            balance -= (balance * leverage * fee) # Fee de apertura 
        
        balance_history.append(balance)

    df['balance'] = balance_history
    return df, trades_log, balance

def run_scientific_experiment(df_full):
    if len(df_full) < 1100:
        print(" [X] Error: No hay suficientes datos para el test aleatorio.")
        return
        
    results = []
    print("\n [TEST] 20 BACKTESTS ALEATORIOS (Ventanas de 1000m)")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for i in range(20):
        start_idx = random.randint(0, len(df_full) - 1001)
        df_window = df_full.iloc[start_idx : start_idx + 1000].copy().reset_index(drop=True)
        
        df_window = run_sniper_backtest(df_window)
        _, logs, final_bal = calculate_wallet_pnl(df_window)
        
        wins = len([p for p in logs if p > 0])
        total = len(logs)
        wr = (wins / total * 100) if total > 0 else 0
        
        results.append(final_bal)
        print(f" > Test {i+1:02} | Balance: ${final_bal:8.2f} | WR: {wr:5.1f}% | Trades: {total}")

    avg = sum(results)/20
    print(f" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" [FIN] PROMEDIO GLOBAL: ${avg:.2f} ({((avg/1000)-1)*100:.1f}%)")

if __name__ == "__main__":
    SYMBOL = "PEPEUSDT"
    # Ahora bajamos 3000 velas reales
    df_main = fetch_binance_extended(SYMBOL, total_candles=3000)
    
    if df_main is not None:
        run_scientific_experiment(df_main)
        
        # Análisis visual del periodo final
        df_plot = df_main.iloc[-1000:].copy().reset_index(drop=True)
        df_plot = run_sniper_backtest(df_plot)
        df_plot, _, final_balance = calculate_wallet_pnl(df_plot)
        
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 5))
        plt.plot(df_plot['ts'], df_plot['balance'], color='yellow', label='Equity Curve')
        plt.title(f"Balance Final Detallado: ${final_balance:.2f}")
        plt.grid(True, alpha=0.1)
        plt.show()