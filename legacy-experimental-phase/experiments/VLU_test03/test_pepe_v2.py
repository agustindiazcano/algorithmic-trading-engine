import threading
import websocket
import json
import pandas as pd
import numpy as np
import requests
import time
from collections import deque
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich import box

# ==========================================
# 🔒 PARÁMETROS DE CALIBRACIÓN DINÁMICA
# ==========================================
SYMBOL = "pepeusdt"
ALPHA_UNIVERSAL = 0.2639
K_UNIVERSAL = 0.2816
# Usamos 2000 velas para tener una base estadística sólida que se mueva en vivo
MEMORY_SIZE = 2000 

LIVE_METRICS = {
    "price": 0.0, "dcpi": 0.0,
    "p90": 0.0, "p95": 0.0, "p99": 0.0,
    "pct_actual": 0.0, "count": 0
}

candle_buffer = deque(maxlen=MEMORY_SIZE)

def update_logic(df):
    # Física de Presiones
    df['delta_p'] = df['close'].diff().abs()
    df['flow'] = df['delta_p'] * (df['volume'] ** ALPHA_UNIVERSAL)
    flow_rolling = df['flow'].rolling(100).mean()
    
    # Generamos la serie histórica de presiones para calcular percentiles
    dcpi_series = ((df['flow'] / (flow_rolling + 1e-9)) / K_UNIVERSAL).dropna()
    
    if len(dcpi_series) < 10: return
    
    current_dcpi = dcpi_series.iloc[-1]
    
    # CALIBRACIÓN EN VIVO: Calculamos los cortes basados en la memoria actual
    p90 = dcpi_series.quantile(0.90)
    p95 = dcpi_series.quantile(0.95)
    p99 = dcpi_series.quantile(0.99)
    
    # Ubicación del segundo actual en la historia
    pct = (dcpi_series < current_dcpi).mean() * 100
    
    LIVE_METRICS.update({
        "dcpi": current_dcpi,
        "p90": p90, "p95": p95, "p99": p99,
        "pct_actual": pct,
        "count": len(dcpi_series)
    })

def on_message(ws, message):
    msg = json.loads(message)
    k = msg['k']
    price = float(k['c'])
    LIVE_METRICS["price"] = price
    
    # Datos para el cálculo
    new_data = {'t': k['t'], 'close': price, 'volume': float(k['v'])}
    
    # Manejo del buffer
    temp_list = list(candle_buffer)
    if temp_list and temp_list[-1]['t'] == new_data['t']:
        temp_list[-1] = new_data
    else:
        temp_list.append(new_data)
        if k['x']: candle_buffer.append(new_data) # Guardar si cerró minuto
    
    if len(temp_list) > 100:
        update_logic(pd.DataFrame(temp_list))

def start_stream():
    # Precarga de datos históricos para que los percentiles no empiecen en cero
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval=1m&limit=1000"
    res = requests.get(url).json()
    for c in res:
        candle_buffer.append({'t': c[0], 'close': float(c[4]), 'volume': float(c[5])})
    
    ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m", on_message=on_message)
    ws.run_forever()

def render_ui():
    pct = LIVE_METRICS["pct_actual"]
    color = "green"
    if pct > 90: color = "yellow"
    if pct > 95: color = "orange3"
    if pct > 99: color = "bold red"

    table = Table(title=f"SCANNED SAMPLES: {LIVE_METRICS['count']}", box=box.ROUNDED, expand=True)
    table.add_column("NIVEL", justify="left")
    table.add_column("VALOR DCPI", justify="right")
    table.add_column("PERCENTIL / RAREZA", justify="left")

    # Barra visual
    bar_size = 30
    filled = int((pct/100) * bar_size)
    bar_str = f"[{color}]{'█'*filled}[/]{'░'*(bar_size-filled)}"

    table.add_row("ACTUAL", f"[{color}]{LIVE_METRICS['dcpi']:.4f}[/]", f"{bar_str} {pct:.2f}%")
    table.add_section()
    table.add_row("P90 (Calmado)", f"{LIVE_METRICS['p90']:.4f}", "[dim]Top 10% de actividad[/]")
    table.add_row("P95 (Interesante)", f"{LIVE_METRICS['p95']:.4f}", "[orange3]Top 5% (Alta Presión)[/]")
    table.add_row("P99 (ANOMALÍA)", f"{LIVE_METRICS['p99']:.4f}", "[bold red]Top 1% (EXPLOSIÓN)[/]")

    panel = Panel(
        table,
        title=f"[bold cyan]DÍAZ-CANO LIVE PERCENTILES: {SYMBOL.upper()}[/]",
        subtitle=f"[bold white]PRECIO: {LIVE_METRICS['price']:.8f} USDT[/]"
    )
    return panel

if __name__ == "__main__":
    threading.Thread(target=start_stream, daemon=True).start()
    with Live(refresh_per_second=4, screen=True) as live:
        while True:
            live.update(render_ui())
            time.sleep(0.2)