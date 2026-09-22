import numpy as np
import matplotlib.pyplot as plt
import time
import requests
import json
import datetime
import websocket # pip install websocket-client
import threading

# ==============================================================================
# ⚙️ CONFIGURACIÓN DEL USUARIO
# ==============================================================================
BRAIN_FILE = "best_brain_COMPLETE.npy" # Usar estado completo (no solo genoma)
SYMBOL = "PEPEUSDT"
INTERVAL = "1m"       # 1m, 5m, 15m, 1h, 4h
MODE = "BACKTEST"     # Opciones: "BACKTEST" o "LIVE"

# SOLO PARA BACKTEST:
BACKTEST_CANDLES = 5000
START_DATE = "2026-02-01" # Formato YYYY-MM-DD (O dejar None para datos recientes)

# ==============================================================================
# 🧠 IMPORT CEREBRO Y FUNCIONES DE CARGA
# ==============================================================================

# Importar la clase y funciones del archivo principal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Definir NeuroPlasticBrain inline (simplificada para backtest)
class NeuroPlasticBrain:
    def __init__(self, initial_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.genome = np.empty((0, 9))
        self.neuron_energy = np.empty(0)
        self.neuron_age = np.empty(0, dtype=int)
        self.vault_genome = np.empty((0, 9))
        self.vault_ages = np.empty(0, dtype=int)

    def predict(self, points):
        # Inferencia Volumétrica 3D (mismo código que training)
        if len(self.genome) == 0: return 0.0
        
        centers = self.genome[:, :3]      
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 8]
        stretch = self.genome[:, 5:8]     
        
        point = points[-1].reshape(1, 3) 
        diff = point - centers 
        diff_stretched = diff / (stretch + 1e-6)
        dists = np.linalg.norm(diff_stretched, axis=1)
        
        safe_radii = np.maximum(radii, 1e-6)
        raw_overlap = np.maximum(0, 1 - dists / safe_radii)
        activations = np.minimum(1.0, raw_overlap * 3.0)
        total_activation = np.sum(activations * weights)
        
        return np.tanh(total_activation), activations

def load_brain_state(filename):
    """Carga el estado COMPLETO del cerebro."""
    state = np.load(filename, allow_pickle=True).item()
    
    brain = NeuroPlasticBrain(initial_neurons=10)
    brain.genome = state['genome']
    brain.neuron_energy = state['neuron_energy']
    brain.neuron_age = state['neuron_age']
    brain.vault_genome = state.get('vault_genome', np.empty((0, 9)))
    brain.vault_ages = state.get('vault_ages', np.empty(0, dtype=int))
    
    metadata = state.get('metadata', {})
    print(f"   📂 Estado cargado: {filename}")
    if 'generation' in metadata:
        print(f"      Gen: {metadata['generation']} | Balance: ${metadata.get('balance', 0):.2f}")
    print(f"      Neuronas: {len(brain.genome)} activas, {len(brain.vault_genome)} hibernadas")
    
    return brain, metadata

# ==============================================================================
# 🛠️ PROCESADOR DE REALIDAD (DATA -> 3D)
# ==============================================================================

def raw_to_3d_inputs(closes, volumes, highs, lows, opens):
    """
    Convierte velas crudas en coordenadas espaciales para la IA.
    """
    closes = np.array(closes)
    volumes = np.array(volumes)
    
    # 1. Velocidad (Cambio de precio)
    # Necesitamos al menos 2 velas para calcular diff
    if len(closes) < 2: return None
    p_changes = np.diff(closes) / closes[:-1]
    
    # Alineamos arrays (quitamos el primer elemento de los demás para coincidir con diff)
    vols_aligned = volumes[1:]
    highs_aligned = np.array(highs)[1:]
    lows_aligned = np.array(lows)[1:]
    opens_aligned = np.array(opens)[1:]
    
    # 2. Combustible (Volumen Relativo)
    # Usamos la media de todo el bloque para normalizar
    avg_vol = np.mean(vols_aligned) if len(vols_aligned) > 0 else 1.0
    rel_vol = vols_aligned / (avg_vol * 3 + 1e-6)
    
    # 3. Turbulencia (Volatilidad)
    volatilities = (highs_aligned - lows_aligned) / opens_aligned
    norm_volatilities = volatilities / 0.005
    
    # Empaquetar en 3D
    inputs = np.column_stack([
        p_changes / 0.005,  # X
        rel_vol,            # Y
        norm_volatilities   # Z
    ])
    
    return inputs, closes[1:] # Retornamos precios alineados

# ==============================================================================
# 📡 GESTOR DE DATOS HISTÓRICOS
# ==============================================================================

def get_historical_data(symbol, interval, limit, start_date_str=None):
    end_time = None
    if start_date_str:
        # Convertir fecha string a timestamp y sumarle tiempo para obtener las velas siguientes
        # Pero Binance pide endTime, así que es más fácil pedir desde esa fecha hacia adelante (startTime)
        dt_obj = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        start_ts = int(dt_obj.timestamp() * 1000)
        print(f"📅 Buscando datos desde: {start_date_str}")
        
        # Lógica simplificada: Bajamos por lotes desde start_ts hacia adelante
        all_data = []
        current_start = start_ts
        left = limit
        
        while left > 0:
            fetch = min(left, 1000)
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={fetch}&startTime={current_start}"
            data = requests.get(url).json()
            if not data: break
            all_data.extend(data)
            current_start = int(data[-1][0]) + 1
            left -= len(data)
            if len(data) < fetch: break # No hay más datos
            
        return parse_binance_data(all_data)

    else:
        # Modo "Reciente": Bajamos hacia atrás desde hoy
        print(f"📅 Buscando los últimos {limit} datos recientes...")
        all_data = []
        current_end = int(time.time() * 1000)
        left = limit
        
        while left > 0:
            fetch = min(left, 1000)
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={fetch}&endTime={current_end}"
            data = requests.get(url).json()
            if not data: break
            # Binance devuelve del más viejo al más nuevo. Como vamos hacia atrás, insertamos al principio.
            all_data = data + all_data 
            current_end = int(data[0][0]) - 1
            left -= len(data)
            
        return parse_binance_data(all_data)

def parse_binance_data(data):
    closes = [float(x[4]) for x in data]
    opens = [float(x[1]) for x in data]
    highs = [float(x[2]) for x in data]
    lows = [float(x[3]) for x in data]
    volumes = [float(x[5]) for x in data]
    return closes, volumes, highs, lows, opens

# ==============================================================================
# 📼 MODO 1: BACKTEST HISTÓRICO
# ==============================================================================

def run_backtest(brain):
    print(f"\n🎞️ INICIANDO BACKTEST: {SYMBOL} [{INTERVAL}]")
    c, v, h, l, o = get_historical_data(SYMBOL, INTERVAL, BACKTEST_CANDLES, START_DATE)
    
    inputs, prices = raw_to_3d_inputs(c, v, h, l, o)
    
    if inputs is None:
        print("❌ No hay suficientes datos.")
        return

    # Simulación
    balance = 10000.0
    position = 0
    entry_price = 0
    equity = []
    actions = []
    
    for t in range(len(inputs)):
        signal, _ = brain.predict([inputs[t]])  # Ahora retorna (signal, activations)
        price = prices[t]
        act = 0
        
        # Lógica simple de trading (sin fees complejos para visualización rápida)
        if position == 0:
            if signal > 0.8:
                position = 1
                entry_price = price
                act = 1
            elif signal < -0.8:
                position = -1
                entry_price = price
                act = -1
        elif position == 1 and signal < -0.2: # Cierre Long
            pnl = (price - entry_price) / entry_price
            balance *= (1 + pnl * 50) # x50 Leverage
            position = 0
        elif position == -1 and signal > 0.2: # Cierre Short
            pnl = (entry_price - price) / entry_price
            balance *= (1 + pnl * 50)
            position = 0
            
        equity.append(balance)
        actions.append(act)

    print(f"🏁 FINALIZADO. Balance Final: ${balance:.2f}")
    
    # Graficar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(prices, color='black', alpha=0.5)
    buys = [i for i, x in enumerate(actions) if x == 1]
    sells = [i for i, x in enumerate(actions) if x == -1]
    ax1.scatter(buys, prices[buys], color='green', marker='^', label='Buy')
    ax1.scatter(sells, prices[sells], color='red', marker='v', label='Sell')
    ax1.legend()
    ax1.set_title(f"Backtest {SYMBOL}")
    
    ax2.plot(equity, color='blue')
    ax2.set_title("Equity Curve")
    plt.show()

# ==============================================================================
# 📡 MODO 2: PAPER TRADING (LIVE WEBSOCKET)
# ==============================================================================

class PaperTrader:
    def __init__(self, brain):
        self.brain = brain
        self.history = {
            'closes': [], 'volumes': [], 'highs': [], 'lows': [], 'opens': []
        }
        self.position = 0 # 0, 1 (Long), -1 (Short)
        self.entry_price = 0
        self.balance = 10000.0
        
        # Inicializar con datos históricos para tener contexto (medias móviles, etc)
        print("⏳ Cargando contexto inicial (100 velas)...")
        c, v, h, l, o = get_historical_data(SYMBOL, INTERVAL, 100)
        self.history['closes'] = c
        self.history['volumes'] = v
        self.history['highs'] = h
        self.history['lows'] = l
        self.history['opens'] = o
        print("✅ Contexto cargado. Conectando a Binance Stream...")

    def on_message(self, ws, message):
        json_msg = json.loads(message)
        candle = json_msg['k']
        is_closed = candle['x']
        
        current_price = float(candle['c'])
        
        # Solo actuamos cuando la vela CIERRA
        if is_closed:
            print(f"🕯️ Vela cerrada: {current_price}")
            
            # 1. Agregar nueva vela a la memoria
            self.history['closes'].append(float(candle['c']))
            self.history['opens'].append(float(candle['o']))
            self.history['highs'].append(float(candle['h']))
            self.history['lows'].append(float(candle['l']))
            self.history['volumes'].append(float(candle['v']))
            
            # Mantener memoria corta (opcional, para no llenar RAM infinitamente)
            if len(self.history['closes']) > 500:
                for k in self.history: self.history[k].pop(0)

            # 2. Convertir a 3D (La IA necesita ver el movimiento)
            inputs, _ = raw_to_3d_inputs(
                self.history['closes'], self.history['volumes'], 
                self.history['highs'], self.history['lows'], self.history['opens']
            )
            
            if inputs is None: return

            # 3. Predecir con el último input generado
            last_input = inputs[-1] # El movimiento de la vela recién cerrada
            signal, _ = self.brain.predict([last_input])  # Ahora retorna (signal, activations)
            
            print(f"🧠 Señal IA: {signal:.4f} | Posición actual: {self.position}")
            
            # 4. Ejecutar Paper Trade
            self.execute_trade(signal, current_price)

    def execute_trade(self, signal, price):
        # Lógica de cierre
        if self.position == 1 and signal < -0.2:
            pnl = (price - self.entry_price) / self.entry_price * 50
            self.balance += (self.balance * pnl)
            print(f"📉 CIERRE LONG. PnL: {pnl*100:.2f}%. Balance: ${self.balance:.2f}")
            self.position = 0
            
        elif self.position == -1 and signal > 0.2:
            pnl = (self.entry_price - price) / self.entry_price * 50
            self.balance += (self.balance * pnl)
            print(f"📈 CIERRE SHORT. PnL: {pnl*100:.2f}%. Balance: ${self.balance:.2f}")
            self.position = 0
            
        # Lógica de apertura
        if self.position == 0:
            if signal > 0.8:
                self.position = 1
                self.entry_price = price
                print(f"🚀 OPEN LONG @ {price}")
            elif signal < -0.8:
                self.position = -1
                self.entry_price = price
                print(f"🔻 OPEN SHORT @ {price}")

    def start(self):
        socket = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_{INTERVAL}"
        ws = websocket.WebSocketApp(socket, on_message=self.on_message)
        ws.run_forever()

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1. Cargar Cerebro Completo (con energía, edad, vault)
    print("\n🧠 Cargando cerebro entrenado...")
    brain, metadata = load_brain_state(BRAIN_FILE)
    print(f"✅ Cerebro cargado correctamente.")
    print(f"   Generación: {metadata.get('generation', 'N/A')}")
    print(f"   Balance Final Entrenamiento: ${metadata.get('balance', 0):.2f}\n")
    
    if MODE == "BACKTEST":
        run_backtest(brain)
    elif MODE == "LIVE":
        trader = PaperTrader(brain)
        trader.start()
    else:
        print("Modo no reconocido.")