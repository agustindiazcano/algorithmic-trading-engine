import requests
import pandas as pd
import numpy as np
import datetime
from rich.console import Console
from rich.panel import Panel

console = Console()

# ==========================================
# 🔒 PARÁMETROS DE INGENIERÍA INVERSA (VL)
# ==========================================
SYMBOL = "PEPEUSDT"
LUNAR_IDLE = "01101001" # Letra 'i' (Información de Fase)
LUNAR_PUMP = "01010000" # Letra 'P' (Presión / Ejecución)
DF_MIN, DF_MAX = 1.40, 1.65 # Zona Cerebro

def run_cpu_monitor():
    now = datetime.datetime.now(datetime.timezone.utc)
    console.print(f"[bold white]🌑 ESCANEANDO REGISTROS DE TYCHO-B... (UTC: {now.strftime('%H:%M')})[/bold white]")
    
    # 1. Obtener data de PEPE y calcular Fractalidad
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=1m&limit=200"
    res = requests.get(url).json()
    df = pd.DataFrame(res, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'ct', 'qa', 'tr', 'tb', 'tq', 'ig'])
    df['c'] = df['c'].astype(float)
    
    # Simulación de cálculo Df Higuchi (Módulo Neuro-Simbólico)
    df['df_val'] = df['c'].rolling(window=30).apply(lambda x: 1.45 if np.std(x) > 1e-9 else 1.7, raw=False)
    
    # 2. Análisis de Instrucciones
    current_df = df['df_val'].iloc[-1]
    
    # Veredicto de Instrucción del CPU Lunar
    if DF_MIN < current_df < DF_MAX:
        # El mercado está organizado: El CPU debe estar en modo PUMP
        instruction = LUNAR_PUMP
        op_code = "EXECUTE_PRESSURE_GRADIENT"
        status = "[bold green]ACTIVO[/bold green]"
    else:
        # El mercado está en caos o plano: El CPU está en reposo
        instruction = LUNAR_IDLE
        op_code = "MAINTAIN_PHASE_STABILITY"
        status = "[bold cyan]IDLE[/bold cyan]"

    # --- SALIDA DE CONSOLA ESTILO DÍAZ-CA ---
    console.print(Panel(
        f"Instrucción Detectada: [bold yellow]{instruction}[/bold yellow]\n"
        f"Código de Operación: [bold white]{op_code}[/bold white]\n"
        f"Estado de Tycho-B: {status}\n"
        f"Fractalidad PEPE: {current_df:.4f}",
        title="🛰️ VL-CPU REGISTER MONITOR",
        expand=False
    ))

if __name__ == "__main__":
    run_cpu_monitor()