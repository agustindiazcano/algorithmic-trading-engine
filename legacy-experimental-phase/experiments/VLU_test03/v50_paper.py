import websocket
import json
import pandas as pd
import numpy as np
import requests
import time
import threading
from collections import deque
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich import box

console = Console()

# ==========================================
# ☢️ CONSTANTES "V50: THE 1000% PROTOCOL"
# ==========================================
SYMBOL = "pepeusdt"
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639

# --- LA BANDA DE RESONANCIA (TECHO ACTIVADO) ---
PRESSURE_FLOOR = 15.80  # Piso de ignición
PRESSURE_CEIL  = 21.00  # Válvula de alivio (Anti-Blowoff)
FRACTAL_MIN    = 1.40
FRACTAL_MAX    = 1.84   

# --- GESTIÓN DE CAPITAL (MODO TESIS) ---
INITIAL_BALANCE = 1000.0
LEVERAGE = 50.0   # ⚠️ ZONA DE PELIGRO: -2% en precio = LIQUIDACIÓN
# SIN STOP LOSS / SIN TAKE PROFIT / SIN SALIDA POR TIEMPO

# ==========================================
# 🧠 CEREBRO MATEMÁTICO
# ==========================================
def calculate_higuchi(series, k_max=10):
    x = np.array(series)
    N = len(x)
    if N < 20: return 1.5
    L = []
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            indices = np.arange(m, N, k)
            if len(indices) < 2: continue
            diffs = np.abs(np.diff(x[indices]))
            norm = (N - 1) / (len(indices) * k)
            if len(diffs) > 0:
                Lk.append((np.sum(diffs) * norm) / k)
        if Lk: L.append(np.mean(Lk))
    
    if len(L) < 2: return 1.5
    return np.polyfit(np.log(1/np.arange(1, len(L)+1)), np.log(L), 1)[0]

# ==========================================
# 💰 CLASE TRADER (PACIENCIA INFINITA)
# ==========================================
class PaperTrader:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.position = 0 # 0: Cash, 1: Long, -1: Short
        self.entry_price = 0.0
        self.active_pnl = 0.0
        self.liq_price = 0.0
        self.last_action = "Esperando entrada de 1000 minutos..."
        self.trades_count = 0

    def check_signals(self, price, dcpi, df_val, trend):
        # 1. GENERADOR DE SEÑAL
        signal = 0
        is_pressure_good = (dcpi >= PRESSURE_FLOOR) and (dcpi <= PRESSURE_CEIL)
        is_fractal_good  = (FRACTAL_MIN <= df_val <= FRACTAL_MAX)
        
        if is_pressure_good and is_fractal_good:
            signal = -trend # Reversión Sniper
        
        # 2. GESTIÓN DE POSICIÓN (FLIP FLOP PURO)
        if self.position != 0:
            # PnL en tiempo real
            raw_pnl = (price - self.entry_price) / self.entry_price * self.position
            self.active_pnl = raw_pnl * LEVERAGE * 100

            # CONDICIÓN DE SALIDA: SOLO SI HAY SEÑAL CONTRARIA (FLIP)
            # Acá es donde se necesita la paciencia de 1000 minutos.
            if (self.position == 1 and signal == -1) or (self.position == -1 and signal == 1):
                realized_usd = self.balance * raw_pnl * LEVERAGE
                self.balance += realized_usd
                self.trades_count += 1
                
                color = "green" if realized_usd > 0 else "red"
                self.last_action = f"[bold {color}]FLIP! PnL: ${realized_usd:.2f}[/]"
                
                # Ejecutar la nueva posición inmediatamente
                self.position = signal
                self.entry_price = price
                self.active_pnl = 0.0
                # Calculamos precio de liquidación estimado (aprox)
                liq_dist = 1 / LEVERAGE
                self.liq_price = price * (1 - liq_dist) if signal == 1 else price * (1 + liq_dist)
                
        # 3. ENTRADA INICIAL
        elif self.position == 0 and signal != 0:
            self.position = signal
            self.entry_price = price
            self.last_action = f"[bold cyan]{'LONG' if signal==1 else 'SHORT'} OPEN @ {price}[/]"
            # Liq Price
            liq_dist = 1 / LEVERAGE
            self.liq_price = price * (1 - liq_dist) if signal == 1 else price * (1 + liq_dist)

# ==========================================
# 📡 MOTOR DE DATOS
# ==========================================
trader = PaperTrader()
buffer_candles = deque(maxlen=2000)
LIVE_METRICS = {"price": 0, "dcpi": 0, "df": 0}

def on_message(ws, message):
    json_msg = json.loads(message)
    k = json_msg['k']
    close = float(k['c'])
    volume = float(k['v'])
    is_closed = k['x']
    
    LIVE_METRICS["price"] = close
    
    if trader.position != 0:
        raw = (close - trader.entry_price) / trader.entry_price * trader.position
        trader.active_pnl = raw * LEVERAGE * 100

    if is_closed:
        buffer_candles.append({'c': close, 'v': volume})
        
        if len(buffer_candles) > 100:
            df = pd.DataFrame(list(buffer_candles))
            
            # Física
            delta_p = df['c'].diff().abs()
            flow = delta_p * (df['v'] ** ALPHA_UNIVERSAL)
            flow_avg = flow.rolling(100).mean()
            dcpi = (flow / (flow_avg + 1e-9) / K_UNIVERSAL).iloc[-1]
            
            # Fractal
            df_val = calculate_higuchi(df['c'].iloc[-60:].values)
            
            # Tendencia
            sma = df['c'].iloc[-5:].mean()
            trend = 1 if close > sma else -1
            
            LIVE_METRICS.update({"dcpi": dcpi, "df": df_val})
            
            # Ejecución
            trader.check_signals(close, dcpi, df_val, trend)

# ==========================================
# 🎨 DASHBOARD V50
# ==========================================
def generate_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=10)
    )
    
    # Header Termonuclear
    title = f"☢️ V50: THE 1000% PROTOCOL | {SYMBOL.upper()} | LEV x{int(LEVERAGE)}"
    layout["header"].update(Panel(Text(title, justify="center", style="bold red on black")))
    
    # Sensores
    m_table = Table(expand=True, box=box.SIMPLE)
    m_table.add_column("SENSOR", style="cyan")
    m_table.add_column("VALOR", justify="right")
    m_table.add_column("RANGO", justify="center")
    m_table.add_column("ESTADO", justify="center")
    
    dcpi = LIVE_METRICS["dcpi"]
    df_val = LIVE_METRICS["df"]
    
    c_press = "green" if PRESSURE_FLOOR <= dcpi <= PRESSURE_CEIL else "dim white"
    if dcpi > PRESSURE_CEIL: status_p = "⚠️ EUPHORIA (>21)"
    elif dcpi < PRESSURE_FLOOR: status_p = "💤 HOLDING (<15.8)"
    else: status_p = "✅ ZONA DE FUEGO"
    
    c_frac = "green" if FRACTAL_MIN <= df_val <= FRACTAL_MAX else "yellow"
    
    m_table.add_row("PRESIÓN (DCPI)", f"[{c_press}]{dcpi:.2f}[/]", f"{PRESSURE_FLOOR}-{PRESSURE_CEIL}", f"[{c_press}]{status_p}[/]")
    m_table.add_row("FRACTAL (Df)", f"[{c_frac}]{df_val:.3f}[/]", f"{FRACTAL_MIN}-{FRACTAL_MAX}", f"[{c_frac}]{'CEREBRO' if c_frac=='green' else 'CAOS'}[/]")
    m_table.add_row("PRECIO", f"{LIVE_METRICS['price']}", "-", "-")

    layout["body"].update(Panel(m_table, title="Physics Engine"))
    
    # Billetera de Alto Riesgo
    w_table = Table(expand=True, box=box.SIMPLE)
    w_table.add_column("BALANCE", style="bold green")
    w_table.add_column("POSICIÓN")
    w_table.add_column("PNL FLOTANTE")
    w_table.add_column("LIQ. PRICE")
    w_table.add_column("ÚLTIMA ACCIÓN")
    
    pos_str = "SHORT" if trader.position == -1 else ("LONG" if trader.position == 1 else "CASH")
    pnl_color = "green" if trader.active_pnl > 0 else "red"
    
    # Alerta de liquidación
    liq_str = f"{trader.liq_price:.8f}" if trader.position != 0 else "---"
    
    w_table.add_row(
        f"${trader.balance:.2f}",
        f"[bold cyan]{pos_str}[/]",
        f"[{pnl_color}]{trader.active_pnl:+.2f}%[/]",
        f"[dim red]{liq_str}[/]",
        trader.last_action
    )
    
    layout["footer"].update(Panel(w_table, title="High Voltage Wallet"))
    
    return layout

if __name__ == "__main__":
    print("[>] Cargando buffer inicial (Precalentando motores)...")
    try:
        res = requests.get(f"https://api.binance.com/api/v3/klines?symbol={SYMBOL.upper()}&interval=1m&limit=150").json()
        for c in res:
            buffer_candles.append({'c': float(c[4]), 'v': float(c[5])})
    except: pass

    ws_url = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_1m"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

    with Live(generate_dashboard(), refresh_per_second=4) as live:
        while True:
            live.update(generate_dashboard())
            time.sleep(0.25)