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
# 🔒 CONSTANTES "GOLDEN RATIO" (V31 - PURISTA)
# ==========================================
SYMBOL = "pepeusdt"
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639

# --- FILTROS DE FASE (Confirmados por Tesis) ---
PRESSURE_FLOOR = 15.80  # Piso ganador
PRESSURE_CEIL  = 21.00  # Techo anti-euforia
FRACTAL_MIN    = 1.40
FRACTAL_MAX    = 1.80

# --- GESTIÓN DE CAPITAL ---
INITIAL_BALANCE = 1000.0
LEVERAGE = 10.0 
# ¡CHAU TP/SL FIJOS! EL MERCADO DECIDE LA SALIDA.

# ==========================================
# 🧠 CEREBRO MATEMÁTICO
# ==========================================
def calculate_higuchi(series, k_max=10):
    """Cálculo fractal optimizado"""
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
# 💰 CLASE PAPER TRADER (LÓGICA BACKTEST)
# ==========================================
class PaperTrader:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.position = 0 # 0: Cash, 1: Long, -1: Short
        self.entry_price = 0.0
        self.active_pnl = 0.0
        self.last_action = "Esperando señal pura..."
        self.trades_count = 0

    def check_signals(self, price, dcpi, df_val, trend):
        # 1. GENERADOR DE SEÑAL (Exactamente igual al Backtest)
        # Solo generamos señal (-1 o 1) si las condiciones de presión y fractal se cumplen.
        # Si no, la señal es 0 (HOLD).
        
        signal = 0
        is_pressure_good = (dcpi >= PRESSURE_FLOOR) and (dcpi <= PRESSURE_CEIL)
        is_fractal_good  = (FRACTAL_MIN <= df_val <= FRACTAL_MAX)
        
        if is_pressure_good and is_fractal_good:
            signal = -trend # Contratendencia (Sniper Reversal)
        
        # 2. GESTIÓN DE POSICIÓN (FLIP FLOP)
        # Solo cerramos si tenemos posición Y la señal nueva es CONTRARIA a la actual.
        
        if self.position != 0:
            # Calculamos PnL flotante solo para mostrar
            raw_pnl = (price - self.entry_price) / self.entry_price * self.position
            self.active_pnl = raw_pnl * LEVERAGE * 100

            # Lógica de Salida: Solo si hay señal opuesta (Flip)
            if (self.position == 1 and signal == -1) or (self.position == -1 and signal == 1):
                # CIERRE DE POSICIÓN
                realized_usd = self.balance * raw_pnl * LEVERAGE
                self.balance += realized_usd
                self.trades_count += 1
                color = "green" if realized_usd > 0 else "red"
                self.last_action = f"[bold {color}]FLIP! PnL: ${realized_usd:.2f}[/]"
                
                # Inmediatamente abrimos la nueva (FLIP)
                self.position = signal
                self.entry_price = price
                self.active_pnl = 0.0
                
        # 3. GESTIÓN DE ENTRADA (Si estamos en Cash)
        elif self.position == 0 and signal != 0:
            self.position = signal
            self.entry_price = price
            self.last_action = f"[bold cyan]{'LONG' if signal==1 else 'SHORT'} OPEN @ {price}[/]"

# ==========================================
# 📡 MOTOR DE DATOS
# ==========================================
trader = PaperTrader()
buffer_candles = deque(maxlen=2000)
LIVE_METRICS = {"price": 0, "dcpi": 0, "df": 0, "trend": 0}

def on_message(ws, message):
    json_msg = json.loads(message)
    k = json_msg['k']
    close = float(k['c'])
    volume = float(k['v'])
    is_closed = k['x']
    
    LIVE_METRICS["price"] = close
    
    # Actualización visual de PnL en tiempo real
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
            # Usamos iloc[-1] para el último valor cerrado
            dcpi = (flow / (flow_avg + 1e-9) / K_UNIVERSAL).iloc[-1]
            
            # Fractal
            df_val = calculate_higuchi(df['c'].iloc[-60:].values)
            
            # Tendencia
            sma = df['c'].iloc[-5:].mean()
            trend = 1 if close > sma else -1
            
            LIVE_METRICS.update({"dcpi": dcpi, "df": df_val, "trend": trend})
            
            # Ejecución
            trader.check_signals(close, dcpi, df_val, trend)

# ==========================================
# 🎨 DASHBOARD V31
# ==========================================
def generate_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=10)
    )
    
    title = f"🦁 V31: PURIST GOLDEN RATIO | {SYMBOL.upper()} | LEV x{int(LEVERAGE)}"
    layout["header"].update(Panel(Text(title, justify="center", style="bold gold1"), style="on black"))
    
    m_table = Table(expand=True, box=box.SIMPLE)
    m_table.add_column("SENSOR", style="cyan")
    m_table.add_column("VALOR", justify="right")
    m_table.add_column("ESTADO", justify="center")
    
    dcpi = LIVE_METRICS["dcpi"]
    df_val = LIVE_METRICS["df"]
    
    c_press = "green" if PRESSURE_FLOOR <= dcpi <= PRESSURE_CEIL else "red"
    if dcpi > PRESSURE_CEIL: status_p = "⚠️ EUPHORIA (>21)"
    elif dcpi < PRESSURE_FLOOR: status_p = "💤 BAJA PRESIÓN (<15.8)"
    else: status_p = "✅ FASE DORADA"
    
    c_frac = "green" if FRACTAL_MIN <= df_val <= FRACTAL_MAX else "yellow"
    
    m_table.add_row("PRESIÓN (DCPI)", f"[{c_press}]{dcpi:.2f}[/]", f"[{c_press}]{status_p}[/]")
    m_table.add_row("FRACTAL (Df)", f"[{c_frac}]{df_val:.3f}[/]", f"[{c_frac}]{'CEREBRO' if c_frac=='green' else 'CAOS'}[/]")
    m_table.add_row("PRECIO", f"{LIVE_METRICS['price']}", "---")

    layout["body"].update(Panel(m_table, title="Physics Engine"))
    
    w_table = Table(expand=True, box=box.SIMPLE)
    w_table.add_column("BALANCE", style="bold green")
    w_table.add_column("POSICIÓN")
    w_table.add_column("PNL FLOTANTE")
    w_table.add_column("ESTRATEGIA")
    
    pos_str = "SHORT" if trader.position == -1 else ("LONG" if trader.position == 1 else "CASH")
    pnl_color = "green" if trader.active_pnl > 0 else "red"
    
    w_table.add_row(
        f"${trader.balance:.2f}",
        f"[bold cyan]{pos_str}[/]",
        f"[{pnl_color}]{trader.active_pnl:+.2f}%[/]",
        trader.last_action
    )
    
    layout["footer"].update(Panel(w_table, title="Simulated Wallet (Flip-Flop Logic)"))
    
    return layout

if __name__ == "__main__":
    print("[>] Precargando datos para calibración...")
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