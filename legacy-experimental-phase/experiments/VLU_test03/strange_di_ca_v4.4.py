import threading
import websocket
import json
import pandas as pd
import numpy as np
import requests
import time
from collections import deque
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.console import Console
from rich import box

# ==========================================
# 🔒 PARÁMETROS DE INGENIERÍA
# ==========================================
SYMBOL = "pepeusdt"
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
DF_MIN, DF_MAX = 1.40, 1.65
MEMORY_SIZE = 3000

# ==========================================
# 📡 SISTEMA DE ESTADO Y SIMULACIÓN
# ==========================================
class Simulator:
    def __init__(self, name, leverage=10.0):
        self.name = name
        self.balance = 1000.0
        self.leverage = leverage
        self.position = 0 # 1: Long, -1: Short, 0: Flat
        self.entry_price = 0.0
        self.trades = [] # Historial de cierres
        self.current_pnl = 0.0

    def process(self, price, signal):
        # 1. Calcular PNL flotante
        if self.position != 0:
            self.current_pnl = (price - self.entry_price) / self.entry_price * self.position * self.leverage * 100
            
            # 2. Lógica de Salida (Cierre por señal contraria o liquidación teórica)
            if (self.position == 1 and signal == -1) or (self.position == -1 and signal == 1) or (self.current_pnl <= -80):
                pnl_final = self.current_pnl
                fee = 0.001 * self.leverage * self.balance
                self.balance += (self.balance * (pnl_final/100)) - fee
                self.trades.append({"side": "LONG" if self.position == 1 else "SHORT", "pnl": pnl_final})
                self.position = 0
                self.current_pnl = 0.0

        # 3. Lógica de Entrada
        if self.position == 0 and signal != 0:
            self.position = signal
            self.entry_price = price

# Instancias globales
SIM_90 = Simulator("STRAT_90")
SIM_95 = Simulator("STRAT_95")

LIVE_DATA = {
    "price": 0.0, "dcpi": 0.0, "df": 0.0,
    "umb_90": 0.0, "umb_95": 0.0,
    "sig_90": 0, "sig_95": 0,
    "god_trend": 0, "connected": False
}

candle_buffer = deque(maxlen=MEMORY_SIZE)

# ==========================================
# 1. CORE MATEMÁTICO
# ==========================================
def calculate_metrics(df_live):
    df_live['delta_p'] = df_live['close'].diff().abs()
    df_live['flow'] = df_live['delta_p'] * (df_live['volume'] ** ALPHA_UNIVERSAL)
    flow_rolling = df_live['flow'].rolling(100).mean()
    flow_avg = flow_rolling.iloc[-1] if not pd.isna(flow_rolling.iloc[-1]) else 1.0
    dcpi = (df_live['flow'].iloc[-1] / (flow_avg + 1e-9)) / K_UNIVERSAL
    
    hist_dcpi = (df_live['flow'] / (flow_rolling + 1e-9)) / K_UNIVERSAL
    umb_90 = max(6.0, hist_dcpi.tail(1000).quantile(0.90))
    umb_95 = max(6.0, hist_dcpi.tail(1000).quantile(0.95))
    
    last_60 = df_live['close'].tail(60).values
    df_val = 1.5
    if len(last_60) >= 20:
        x = np.array(last_60)
        L = []
        for k in range(1, 11):
            Lk = [np.mean([np.sum(np.abs(np.diff(x[m::k]))) * (len(x)-1)/(((len(x)-m-1)//k)*k) / k for m in range(k)])]
            L.append(np.mean(Lk))
        df_val = np.polyfit(np.log(1/np.arange(1, 11)), np.log(L), 1)[0]
    
    # Señales
    sma5 = df_live['close'].tail(5).mean()
    trend = 1 if df_live['close'].iloc[-1] > sma5 else -1
    
    s90 = -trend if (dcpi > umb_90 and DF_MIN < df_val < DF_MAX) else 0
    s95 = -trend if (dcpi > umb_95 and DF_MIN < df_val < DF_MAX) else 0
    
    return dcpi, df_val, umb_90, umb_95, s90, s95

# ==========================================
# 2. DASHBOARD RICH UI
# ==========================================
def make_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10)
    )
    
    # HEADER
    layout["header"].update(Panel(Align.center(Text(f"DI-CA ALGO-SIM V9 | {SYMBOL.upper()}", style="bold cyan")), border_style="blue"))
    
    # MAIN (Status en vivo)
    table = Table(box=box.MINIMAL, expand=True)
    table.add_column("SENSORS", style="dim")
    table.add_column("VALUE", justify="right")
    table.add_column("THRESHOLD 90 (0.90)", justify="center")
    table.add_column("THRESHOLD 95 (0.95)", justify="center")
    
    p_color = "red" if LIVE_DATA['dcpi'] > LIVE_DATA['umb_90'] else "green"
    df_status = "[green]🧠 CEREBRO[/]" if DF_MIN < LIVE_DATA['df'] < DF_MAX else "[red]⚡ CAOS[/]"
    
    table.add_row("PRESSURE (DCPI)", f"[{p_color}]{LIVE_DATA['dcpi']:.2f}[/]", f"{LIVE_DATA['umb_90']:.2f}", f"{LIVE_DATA['umb_95']:.2f}")
    table.add_row("FRACTAL (Df)", f"{LIVE_DATA['df']:.3f}", df_status, "")
    table.add_row("LIVE PRICE", f"[bold white]{LIVE_DATA['price']:.8f}[/]", "", "")
    
    layout["main"].update(Panel(table, title="Real-Time Physics Engine"))

    # FOOTER (Simulación)
    sim_table = Table(box=box.SIMPLE, expand=True)
    sim_table.add_column("Strategy")
    sim_table.add_column("Position")
    sim_table.add_column("Floating PNL", justify="right")
    sim_table.add_column("Wallet Balance", justify="right")
    sim_table.add_column("Last 3 Trades")

    for sim in [SIM_90, SIM_95]:
        pos_txt = "IDLE" if sim.position == 0 else ("LONG" if sim.position == 1 else "SHORT")
        pnl_style = "green" if sim.current_pnl > 0 else "red"
        last_trades = " ".join([f"[{'green' if t['pnl']>0 else 'red'}]{t['pnl']:+.1f}%[/]" for t in sim.trades[-3:]])
        
        sim_table.add_row(
            sim.name, 
            pos_txt, 
            f"[{pnl_style}]{sim.current_pnl:+.2f}%[/{pnl_style}]", 
            f"${sim.balance:.2f}",
            last_trades
        )

    layout["footer"].update(Panel(sim_table, title="Paper Trading Simulation (10x Leverage)"))
    return layout

# ==========================================
# 3. WEBSOCKET & HILOS
# ==========================================
def on_message(ws, message):
    msg = json.loads(message)
    k = msg['k']
    price = float(k['c'])
    
    temp_data = list(candle_buffer)
    live_candle = {'t':k['t'],'o':float(k['o']),'h':float(k['h']),'l':float(k['l']),'c':price,'v':float(k['v'])}
    
    if temp_data and temp_data[-1]['t'] == live_candle['t']: temp_data[-1] = live_candle
    else: temp_data.append(live_candle)
    
    df = pd.DataFrame(temp_data)
    df.rename(columns={'c':'close','v':'volume','h':'high','l':'low','o':'open'}, inplace=True)
    
    # Calcular métricas
    dcpi, df_val, u90, u95, s90, s95 = calculate_metrics(df)
    
    # Actualizar LIVE_DATA
    LIVE_DATA.update({"price": price, "dcpi": dcpi, "df": df_val, "umb_90": u90, "umb_95": u95, "sig_90": s90, "sig_95": s95, "connected": True})
    
    # Procesar Simulación
    SIM_90.process(price, s90)
    SIM_95.process(price, s95)
    
    if k['x']: candle_buffer.append(live_candle)

def run_sim():
    # Descarga inicial (Simplificada)
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval=1m&limit=1000"
    res = requests.get(url).json()
    for k in res: candle_buffer.append({'t':k[0],'o':float(k[1]),'h':float(k[2]),'l':float(k[3]),'c':float(k[4]),'v':float(k[5])})
    
    ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m", on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()
    
    with Live(screen=True, refresh_per_second=4) as live:
        while True:
            live.update(make_dashboard())
            time.sleep(0.25)

if __name__ == "__main__":
    run_sim()