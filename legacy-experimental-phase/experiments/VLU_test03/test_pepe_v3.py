import threading
import websocket
import json
import pandas as pd
import numpy as np
import requests
import time
from collections import deque
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich import box

# ==========================================
# 🔒 CONFIGURACIÓN TÉCNICA
# ==========================================
SYMBOL = "pepeusdt"
ALPHA_UNIVERSAL = 0.2639
K_UNIVERSAL = 0.2816
DF_MIN, DF_MAX = 1.7, 1.85
MEMORY_SIZE = 1500  # Ventana de re-calibración

# ==========================================
# 💰 MOTOR DE SIMULACIÓN (PAPER TRADING)
# ==========================================
class LiveSim:
    def __init__(self):
        self.balance = 1000.0
        self.position = 0  # 0: Fuera, 1: Comprado
        self.entry_price = 0.0
        self.pnl_pct = 0.0
        self.trades_count = 0
        self.history = []

    def update(self, price, dcpi, p95, p90, df_val):
        # 1. LÓGICA DE SALIDA
        if self.position == 1:
            self.pnl_pct = (price - self.entry_price) / self.entry_price * 100
            
            # Condición de salida: Presión muere (cae de P90) o TP/SL
            if dcpi < p90 or self.pnl_pct > 1.0 or self.pnl_pct < -0.5:
                self.balance += (self.balance * (self.pnl_pct / 100))
                self.history.append(self.pnl_pct)
                self.position = 0
                self.trades_count += 1
                return "[bold red]SELL[/]"

        # 2. LÓGICA DE ENTRADA
        elif self.position == 0:
            # Si hay explosión de presión (P95) y el fractal es sano
            if dcpi > p95 and (DF_MIN < df_val < DF_MAX):
                self.entry_price = price
                self.position = 1
                return "[bold green]BUY[/]"
        
        return None

sim = LiveSim()
LIVE_DATA = {
    "price": 0.0, "dcpi": 0.0, "df": 0.0, 
    "p95": 0.0, "p90": 0.0, "pct": 0.0, "msg": "Buscando señal..."
}
candle_buffer = deque(maxlen=MEMORY_SIZE)

# ==========================================
# 🔬 MOTOR MATEMÁTICO
# ==========================================
def calculate_higuchi(prices):
    if len(prices) < 20: return 1.5
    x = np.array(prices)
    N = len(x)
    L = []
    for k in range(1, 11):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2: continue
            diffs = np.abs(np.diff(x[idxs]))
            norm = (N - 1) / (len(idxs) * k)
            Lk.append((np.sum(diffs) * norm) / k)
        L.append(np.mean(Lk))
    return np.polyfit(np.log(1/np.arange(1, 11)), np.log(L), 1)[0]

def on_message(ws, message):
    msg = json.loads(message)
    k = msg['k']
    price = float(k['c'])
    
    temp = list(candle_buffer)
    curr = {'t': k['t'], 'c': price, 'v': float(k['v'])}
    if temp and temp[-1]['t'] == curr['t']: temp[-1] = curr
    else: temp.append(curr)

    if len(temp) > 105:
        df_p = pd.DataFrame(temp)
        df_p['flow'] = df_p['c'].diff().abs() * (df_p['v'] ** ALPHA_UNIVERSAL)
        flow_avg = df_p['flow'].rolling(100).mean()
        dcpi_s = ((df_p['flow'] / (flow_avg + 1e-9)) / K_UNIVERSAL).dropna()
        
        current_dcpi = dcpi_s.iloc[-1]
        p95 = dcpi_s.quantile(0.90)
        p90 = dcpi_s.quantile(0.90)
        df_val = calculate_higuchi(df_p['c'].tail(60).values)
        
        action = sim.update(price, current_dcpi, p95, p90, df_val)
        if action: LIVE_DATA["msg"] = action

        LIVE_DATA.update({
            "price": price, "dcpi": current_dcpi, "p95": p95, 
            "p90": p90, "df": df_val, "pct": (dcpi_s < current_dcpi).mean() * 100
        })

    if k['x']: candle_buffer.append(curr)

# ==========================================
# 🎨 DASHBOARD EN VIVO
# ==========================================
def render_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="mid", ratio=1),
        Layout(name="bot", size=8)
    )

    # Header
    layout["top"].update(Panel(Align.center(Text(f"DI-CA SIMULATOR V11 | {SYMBOL.upper()}", style="bold magenta")), border_style="bright_black"))

    # Monitor de Física
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("SENSOR")
    table.add_column("VALOR ACTUAL", justify="right")
    table.add_column("UMBRAL P95", justify="right")
    table.add_column("ESTADO", justify="center")

    color_p = "red" if LIVE_DATA["dcpi"] > LIVE_DATA["p95"] else "green"
    fractal_st = "[bold green]🧠 CEREBRO[/]" if DF_MIN < LIVE_DATA["df"] < DF_MAX else "[red]⚡ CAOS[/]"
    
    table.add_row("PRESIÓN (DCPI)", f"[{color_p}]{LIVE_DATA['dcpi']:.2f}[/]", f"{LIVE_DATA['p95']:.2f}", f"Pctil: {LIVE_DATA['pct']:.1f}%")
    table.add_row("FRACTAL (Df)", f"{LIVE_DATA['df']:.3f}", "1.40 - 1.65", fractal_st)
    table.add_row("PRECIO", f"[bold white]{LIVE_DATA['price']:.8f}[/]", "", LIVE_DATA["msg"])

    layout["mid"].update(Panel(table, title="Physics Engine"))

    # Billetera y Trades
    sim_table = Table(box=None, expand=True)
    sim_table.add_column("CASH", style="bold green")
    sim_table.add_column("POSICIÓN")
    sim_table.add_column("PNL ACTUAL", justify="right")
    sim_table.add_column("HISTORIAL (Últimos 5)")

    pos = "IDLE" if sim.position == 0 else "[bold cyan]LONG (IN)[/]"
    pnl_col = "green" if sim.pnl_pct > 0 else "red"
    hist = " ".join([f"[{'green' if x > 0 else 'red'}]{x:+.2f}%[/]" for x in sim.history[-5:]])

    sim_table.add_row(f"${sim.balance:.2f}", pos, f"[{pnl_col}]{sim.pnl_pct:+.2f}%[/{pnl_col}]", hist)
    
    layout["bot"].update(Panel(sim_table, title="Simulated Wallet (Paper Trading)"))

    return layout

from rich.align import Align
if __name__ == "__main__":
    # Precarga
    res = requests.get(f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval=1m&limit=1000").json()
    for c in res: candle_buffer.append({'t': c[0], 'c': float(c[4]), 'v': float(c[5])})

    ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m", on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

    with Live(refresh_per_second=4, screen=True) as live:
        while True:
            live.update(render_dashboard())
            time.sleep(0.2)