import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import requests

import time

import random



# 🔒 CONSTANTES UNIVERSALES (Leyes de Diaz-Cano 2.0)

K_UNIVERSAL = 0.2816

ALPHA_UNIVERSAL = 0.2639

R_PLANCK = 0.25            # Resolución mínima de hardware

PEPE_SATURACION = 12.0     # Umbral de presión crítica

DF_MIN = 1.40              # Inicio de la zona vascular saludable

DF_MAX = 1.65              # Límite antes del caos lineal



def fetch_binance_extended(symbol="PEPEUSDT", total_candles=3000):

    all_data = []

    end_time = int(time.time() * 1000)

    print(f"[>] Descargando {total_candles} velas para {symbol}...")

   

    while len(all_data) < total_candles:

        limit = min(1000, total_candles - len(all_data))

        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}&endTime={end_time}"

        try:

            res = requests.get(url, timeout=10).json()

            if not res or 'code' in res: break

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

    if N < 20: return 1.5 # Valor neutro si no hay data

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

    coeffs = np.polyfit(np.log(1/np.arange(1, k_max + 1)), np.log(L), 1)

    return coeffs[0]



def run_sniper_backtest_v2(df):

    # 1. Cálculo de Presión Volumétrica (dcpi)

    delta_p = df['close'].diff().abs()

    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL)

    flujo_avg = flujo_bruto.rolling(window=100).mean()

    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL

   

    # 2. Sensor Vascular (Df) - Calculamos en ventanas de 60m para ver la red

    df['df_vascular'] = 1.5 # Valor base

    for i in range(100, len(df)):

        # Tomamos una ventana de 'procesamiento neuronal' de 60 velas

        window = df['close'].iloc[i-60:i]

        df.at[i, 'df_vascular'] = calculate_higuchi_df(window)



    df['sma_5'] = df['close'].rolling(5).mean()

    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)

   

    # 3. LÓGICA VASCULAR 2.0: Presión + Geometría

    # El disparo solo ocurre si hay saturación Y la red es fractal (cerebral)

    df['signal'] = np.where(

        (df['dcpi'] > PEPE_SATURACION) &

        (df['df_vascular'] > DF_MIN) &

        (df['df_vascular'] < DF_MAX),

        -df['trend'], 0

    )

    return df



def calculate_wallet_pnl(df):

    balance, leverage, fee = 1000.0, 50.0, 0.001

    position, entry_price = 0, 0

    trades_log, balance_history = [], []



    for i in range(len(df)):

        current_price = df['close'].iloc[i]

        signal = df['signal'].iloc[i]

       

        if position != 0:

            if (position == 1 and signal == -1) or (position == -1 and signal == 1):

                pnl_usd = balance * ((current_price - entry_price) / entry_price * position * leverage)

                balance += pnl_usd - (balance * leverage * fee)

                trades_log.append(pnl_usd)

                position = 0



        if position == 0 and signal != 0:

            position, entry_price = signal, current_price

            balance -= (balance * leverage * fee)

       

        balance_history.append(balance)



    df['balance'] = balance_history

    return df, trades_log, balance



def run_scientific_experiment(df_full):

    results = []

    print("\n [TEST 2.0] VASCULAR BACKTESTS (Ventanas de 1000m)")

    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

   

    for i in range(20):

        start_idx = random.randint(100, len(df_full) - 1001)

        df_window = df_full.iloc[start_idx : start_idx + 1000].copy().reset_index(drop=True)

       

        df_window = run_sniper_backtest_v2(df_window)

        _, logs, final_bal = calculate_wallet_pnl(df_window)

       

        wins = len([p for p in logs if p > 0])

        total = len(logs)

        wr = (wins / total * 100) if total > 0 else 0

        results.append(final_bal)

        print(f" > Test {i+1:02} | Bal: ${final_bal:8.2f} | WR: {wr:5.1f}% | Df Prom: {df_window['df_vascular'].mean():.2f}")



    avg = sum(results)/20

    print(f" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f" [FIN 2.0] PROMEDIO GLOBAL: ${avg:.2f} ({((avg/1000)-1)*100:.1f}%)")



if __name__ == "__main__":

    df_main = fetch_binance_extended("PEPEUSDT", total_candles=3000)

    if df_main is not None:

        run_scientific_experiment(df_main)