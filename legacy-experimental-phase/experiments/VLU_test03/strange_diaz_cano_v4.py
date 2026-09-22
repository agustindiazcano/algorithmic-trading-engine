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
# 🔒 CONSTANTES DEL MOTOR V7
# ==========================================
SYMBOL = "pepeusdt"
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
DF_MIN = 1.40
DF_MAX = 1.65
PERCENTILE_TRIGGER = 0.95 
MEMORY_SIZE = 3000

# ==========================================
# 📡 ESTADO COMPARTIDO (EL PUENTE)
# ==========================================
# Aquí el WebSocket escribe y la UI lee
LIVE_DATA = {
    "price": 0.0,
    "dcpi": 0.0,
    "umb": 0.0,
    "df": 0.0,
    "signal": 0,
    "god_trend": 0,
    "buffer_len": 0,
    "connected": False,
    "last_update": time.time()
}

candle_buffer = deque(maxlen=MEMORY_SIZE)

# ==========================================
# 1. MOTOR DE FÍSICA (BACKGROUND)
# ==========================================
def calculate_higuchi_df_live(prices, k_max=10):
    x = np.array(prices)
    N = len(x)
    if N < 20: return 1.5
    L = []
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2: continue
            diffs = np.abs(np.diff(x[idxs]))
            L_m = np.sum(diffs)
            norm = (N - 1) / (len(idxs) * k)
            Lk.append((L_m * norm) / k)
        if Lk: L.append(np.mean(Lk))
    if len(L) < 2: return 1.5
    try: return np.polyfit(np.log(1/np.arange(1, len(L)+1)), np.log(L), 1)[0]
    except: return 1.5

def update_metrics(df_live):
    # Física
    df_live['delta_p'] = df_live['close'].diff().abs()
    df_live['flow'] = df_live['delta_p'] * (df_live['volume'] ** ALPHA_UNIVERSAL)
    flow_rolling = df_live['flow'].rolling(100).mean()
    flow_avg = flow_rolling.iloc[-1] if not pd.isna(flow_rolling.iloc[-1]) else 1.0
    current_flow = df_live['flow'].iloc[-1]
    dcpi = (current_flow / (flow_avg + 1e-9)) / K_UNIVERSAL
    
    # Umbral
    historical_dcpi = (df_live['flow'] / (flow_rolling + 1e-9)) / K_UNIVERSAL
    umb = max(6.0, historical_dcpi.tail(1000).quantile(PERCENTILE_TRIGGER))
    
    # Fractal
    last_60 = df_live['close'].tail(60).values
    df_val = calculate_higuchi_df_live(last_60)
    
    # Señal
    sma_5 = df_live['close'].tail(5).mean()
    price = df_live['close'].iloc[-1]
    trend_micro = 1 if price > sma_5 else -1
    
    signal = 0
    if (dcpi > umb) and (DF_MIN < df_val < DF_MAX):
        signal = -trend_micro 
        
    return dcpi, df_val, umb, signal

def update_god_eye(buffer):
    if len(buffer) < 2000: return 0
    df = pd.DataFrame(list(buffer))
    sma = df['c'].rolling(2000).mean().iloc[-1]
    return 1 if df['c'].iloc[-1] > sma else -1

# ==========================================
# 2. HILO DE WEBSOCKET (NO BLOQUEANTE)
# ==========================================
def init_historical_data():
    temp_data = []
    end_time = int(time.time() * 1000)
    while len(temp_data) < MEMORY_SIZE:
        limit = min(1000, MEMORY_SIZE - len(temp_data))
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval=1m&limit={limit}&endTime={end_time}"
            res = requests.get(url).json()
            if not res or isinstance(res, dict): break
            temp_data = res + temp_data
            end_time = res[0][0] - 1
        except: break
            
    for k in temp_data:
        candle_buffer.append({'t':k[0],'o':float(k[1]),'h':float(k[2]),'l':float(k[3]),'c':float(k[4]),'v':float(k[5])})
    
    LIVE_DATA["god_trend"] = update_god_eye(candle_buffer)
    LIVE_DATA["buffer_len"] = len(candle_buffer)

def on_message(ws, message):
    try:
        json_msg = json.loads(message)
        k = json_msg['k']
        live_candle = {'t':k['t'],'o':float(k['o']),'h':float(k['h']),'l':float(k['l']),'c':float(k['c']),'v':float(k['v'])}
        
        temp_data = list(candle_buffer)
        if temp_data and temp_data[-1]['t'] == live_candle['t']: temp_data[-1] = live_candle
        else: temp_data.append(live_candle)
            
        df = pd.DataFrame(temp_data)
        df.rename(columns={'c':'close','v':'volume','h':'high','l':'low','o':'open'}, inplace=True)
        
        # CÁLCULOS
        dcpi, df_val, umb, signal = update_metrics(df)
        
        # ACTUALIZAR ESTADO COMPARTIDO
        LIVE_DATA["price"] = live_candle['c']
        LIVE_DATA["dcpi"] = dcpi
        LIVE_DATA["umb"] = umb
        LIVE_DATA["df"] = df_val
        LIVE_DATA["signal"] = signal
        LIVE_DATA["connected"] = True
        LIVE_DATA["last_update"] = time.time()
        
        if k['x']: # Cierre
            candle_buffer.append(live_candle)
            LIVE_DATA["god_trend"] = update_god_eye(candle_buffer)
            LIVE_DATA["buffer_len"] = len(candle_buffer)
            
    except Exception: pass

def start_socket_thread():
    init_historical_data()
    ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m", on_message=on_message)
    ws.run_forever()

# ==========================================
# 3. INTERFAZ GRÁFICA (RICH UI)
# ==========================================
def make_header():
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right", ratio=1)
    
    # Status de conexión
    status_color = "green" if (time.time() - LIVE_DATA["last_update"] < 2) else "red"
    status_text = f"[{status_color}]● CONNECTION LIVE[/{status_color}]"
    
    # Título
    title = Text("DIAZ-CANO SNIPER V8", style="bold cyan", justify="center")
    
    # Buffer Info
    buffer_txt = f"🧠 Memory: {LIVE_DATA['buffer_len']}/{MEMORY_SIZE}"
    
    grid.add_row(status_text, title, buffer_txt)
    return Panel(grid, style="blue", box=box.ROUNDED)

def make_main_stats():
    # --- PANEL IZQUIERDO: PRECIO Y SEÑAL ---
    price_val = f"{LIVE_DATA['price']:.8f}"
    
    # Lógica Visual de Señal
    sig = LIVE_DATA["signal"]
    god = LIVE_DATA["god_trend"]
    
    if sig == 1: 
        sig_txt = Text("🔵 LONG ENTRY", style="bold white on blue", justify="center")
    elif sig == -1: 
        sig_txt = Text("🔴 SHORT ENTRY", style="bold white on red", justify="center")
    else: 
        sig_txt = Text("⚪ WAITING...", style="dim white", justify="center")
        
    # Tendencia Macro
    god_txt = "🚀 BULLISH" if god == 1 else "🐻 BEARISH"
    god_style = "green" if god == 1 else "red"
    
    # Tabla interna de datos
    table_left = Table(box=None, expand=True)
    table_left.add_column("Metric", style="dim")
    table_left.add_column("Value", justify="right", style="bold")
    
    table_left.add_row("ASSET", SYMBOL.upper())
    table_left.add_row("PRICE", price_val)
    table_left.add_row("GOD TREND", f"[{god_style}]{god_txt}[/{god_style}]")
    
    # --- FIX DEL ERROR ---
    # No encadenamos add_row, lo hacemos paso a paso
    grid_left = Table.grid(expand=True)
    grid_left.add_row(table_left)
    grid_left.add_row("") # Espaciador
    grid_left.add_row(sig_txt)

    panel_left = Panel(
        grid_left,
        title="[bold]MARKET DATA[/bold]", 
        border_style="bright_white"
    )

    # --- PANEL DERECHO: FÍSICA ---
    dcpi = LIVE_DATA["dcpi"]
    umb = LIVE_DATA["umb"]
    df_val = LIVE_DATA["df"]
    
    # Barra de presión visual
    # Evitamos error de división por cero si umb es 0
    safe_umb = umb if umb > 0 else 1.0
    bar_len = 20
    ratio = min(dcpi / (safe_umb * 1.5), 1.0)
    fill = int(ratio * bar_len)
    
    bar_color = "red" if dcpi > safe_umb else "green"
    bar_str = f"[{bar_color}]{'█'*fill}[/{bar_color}]{'░'*(bar_len-fill)}"
    
    # Estado Fractal
    if DF_MIN < df_val < DF_MAX:
        df_status = "[bold green]🧠 CEREBRO[/bold green]"
    elif df_val <= DF_MIN:
        df_status = "[dim]💤 PLANO[/dim]"
    else:
        df_status = "[bold red]⚡ CAOS[/bold red]"

    table_right = Table(box=None, expand=True)
    table_right.add_column("Sensor", style="cyan")
    table_right.add_column("Value", justify="right")
    table_right.add_column("Status", justify="center")
    
    table_right.add_row("Pressure", f"{dcpi:.2f}", f"{bar_str}")
    table_right.add_row("Threshold", f"{umb:.2f}", "🎯 LIMIT")
    table_right.add_row("Fractal (Df)", f"{df_val:.3f}", df_status)
    
    panel_right = Panel(
        table_right,
        title="[bold]PHYSICS ENGINE[/bold]",
        border_style="yellow"
    )
    
    # Unir en Layout Grid (Tabla invisible para acomodar paneles)
    layout_grid = Table.grid(expand=True, padding=1)
    layout_grid.add_column(ratio=1)
    layout_grid.add_column(ratio=2)
    layout_grid.add_row(panel_left, panel_right)
    
    return layout_grid

def run_dashboard():
    # Arrancar el hilo de datos
    t = threading.Thread(target=start_socket_thread)
    t.daemon = True
    t.start()
    
    # Esperar conexión inicial
    console = Console()
    with console.status("[bold green]Cargando Memoria y Conectando a Binance...[/bold green]"):
        while not LIVE_DATA["connected"]:
            time.sleep(0.5)

    # Layout Principal
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    
    with Live(layout, refresh_per_second=10, screen=True) as live:
        while True:
            # 1. Header
            layout["header"].update(make_header())
            
            # 2. Main (El tablero de control)
            layout["main"].update(make_main_stats())
            
            # 3. Footer (Mensaje de estado)
            if LIVE_DATA["dcpi"] > LIVE_DATA["umb"]:
                msg = "[bold white on red] 🔥 ALERTA DE PRESIÓN: BUSCANDO CONFIRMACIÓN FRACTAL... [/bold white on red]"
            elif LIVE_DATA["signal"] != 0:
                msg = "[bold white on blue] 🚀 SEÑAL ENVIADA AL EXCHANGE [/bold white on blue]"
            else:
                msg = "[dim]Escaneando flujo de órdenes en tiempo real...[/dim]"
                
            layout["footer"].update(Align.center(Text.from_markup(msg)))
            
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Apagando motores...")