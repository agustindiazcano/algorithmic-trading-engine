import numpy as np
import matplotlib.pyplot as plt
import time
import requests
import json
import datetime
import websocket
import copy

# ==============================================================================
# ⚙️ CONFIGURACIÓN
# ==============================================================================
GENOME_FILE = "best_genome_EVO_PLASTIC.npy"
SYMBOL = "PEPEUSDT"
INTERVAL = "1m"       
MODE = "BACKTEST"     # "BACKTEST" o "LIVE"

# Configuración de Evolución Continua
POPULATION_SIZE = 1000   # Cuántas variantes compiten en paralelo
LEARNING_RATE = 0.9   # Qué tanto pueden mutar (5%)
EVALUATION_WINDOW = 10 # Cada cuántas velas evaluamos quién es el nuevo líder

# Backtest config
BACKTEST_CANDLES = 10000
START_DATE = "2025-05-14" # "2023-01-01" o None para reciente

# ==============================================================================
# 🧠 CEREBRO (CON CAPACIDAD DE MUTACIÓN)
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = []
        # Métricas de rendimiento para la selección natural
        self.score = 0 
        self.virtual_balance = 10000.0
        self.position = 0
        self.entry_price = 0

    def load_genome(self, filepath):
        self.genome = np.load(filepath)

    def set_genome(self, genome_array):
        self.genome = np.copy(genome_array)

    def mutate(self, intensity=0.05):
        """Crea una variación de sí mismo"""
        mutant = copy.deepcopy(self)
        mask = np.random.rand(*mutant.genome.shape) < 0.2 # 20% de los genes mutan
        noise = np.random.normal(0, intensity, size=mutant.genome.shape)
        mutant.genome[mask] += noise[mask]
        
        # Corregir valores físicos imposibles (radios negativos)
        mutant.genome[:, 3] = np.abs(mutant.genome[:, 3]) 
        mutant.genome[:, 5:8] = np.abs(mutant.genome[:, 5:8])
        
        # Resetear métricas del mutante
        mutant.score = 0
        mutant.virtual_balance = 10000.0
        mutant.position = 0
        return mutant

    def predict(self, point_3d):
        # Lógica idéntica de inferencia
        centers = self.genome[:, :3]      
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8]     
        thetas = self.genome[:, 8]        
        
        point = point_3d.reshape(1, 3) 
        diff = point - centers 
        
        c, s = np.cos(thetas), np.sin(thetas)
        dx = diff[:, 0] * c - diff[:, 1] * s
        dy = diff[:, 0] * s + diff[:, 1] * c
        dz = diff[:, 2] 
        
        diff_rotated = np.stack([dx, dy, dz], axis=1)
        diff_stretched = diff_rotated / (stretch + 1e-6)
        dists = np.linalg.norm(diff_stretched, axis=1)
        
        safe_radii = np.maximum(radii, 1e-6)
        raw_overlap = np.maximum(0, 1 - dists / safe_radii)
        plateau = np.minimum(1.0, raw_overlap * 3.0)
        total_activation = np.sum(plateau * weights)
        
        return np.tanh(total_activation)

    def update_performance(self, current_price, signal):
        """Simula trading interno para ver qué tan bueno es este cerebro"""
        # Cierre virtual
        if (signal > 0.8 and self.position == -1) or (signal < -0.8 and self.position == 1):
            pnl = (current_price - self.entry_price) / self.entry_price
            if self.position == -1: pnl *= -1
            
            # Aplicamos leverage x50 y fees simplificados
            real_pnl = (pnl * 50) - 0.002 
            self.virtual_balance *= (1 + real_pnl)
            self.position = 0
            
        # Apertura virtual
        if self.position == 0:
            if signal > 0.8:
                self.position = 1
                self.entry_price = current_price
            elif signal < -0.8:
                self.position = -1
                self.entry_price = current_price

# ==============================================================================
# 🧬 GESTOR DE LA COLMENA (HIVE MIND)
# ==============================================================================

class HiveMind:
    def __init__(self, master_genome_file):
        # 1. Cargar el Maestro
        self.master = DynamicCausalBrain()
        self.master.load_genome(master_genome_file)
        
        # Le damos identidad para rastrearlo
        self.master.name = "👑 ORIGINAL" 
        self.master.generation = 0
        
        # 2. Crear población inicial
        self.population = [self.master]
        for i in range(POPULATION_SIZE - 1):
            mutant = self.master.mutate(LEARNING_RATE)
            mutant.name = f"👽 Mutante_0_{i+1}" # Gen 0, ID i
            mutant.generation = 0
            self.population.append(mutant)
            
        print(f"🧬 COLMENA INICIADA: El Rey es {self.population[0].name}")
        
        self.candles_processed = 0
        self.best_performer_idx = 0
        self.generation_count = 0

    def process_candle(self, input_3d, price):
        signals = []
        
        # Todos piensan
        for brain in self.population:
            sig = brain.predict(input_3d)
            brain.update_performance(price, sig)
            signals.append(sig)
        
        self.candles_processed += 1
        
        # Evolución periódica
        if self.candles_processed % EVALUATION_WINDOW == 0:
            self.evolve()
            
        # ¡AQUÍ ESTÁ LA CLAVE! 
        # Opera el que sea el líder actual (sea Original o Mutante)
        return signals[self.best_performer_idx]

    def evolve(self):
        self.generation_count += 1
        
        # 1. Evaluar quién tiene más dinero virtual
        balances = [b.virtual_balance for b in self.population]
        best_idx = np.argmax(balances)
        current_leader = self.population[best_idx]
        
        # 2. Notificar si hubo Golpe de Estado
        if best_idx != self.best_performer_idx:
            old_leader = self.population[self.best_performer_idx]
            print(f"\n🚨 [GOLPE DE ESTADO] {old_leader.name} ha sido derrocado.")
            print(f"👑 NUEVO LÍDER: {current_leader.name} (Balance Virtual: ${current_leader.virtual_balance:.2f})")
            self.best_performer_idx = best_idx
        
        # 3. SELECCIÓN NATURAL (Matar a los débiles)
        sorted_indices = np.argsort(balances)[::-1] # De mejor a peor
        
        new_population = []
        survivors_names = []
        
        # Elitismo: Los 3 mejores sobreviven intactos
        for i in range(3):
            idx = sorted_indices[i]
            survivor = self.population[idx]
            # Reseteamos su balance virtual para que la competencia sea justa en la siguiente ronda
            # (Opcional: Si no lo reseteas, premia la historia a largo plazo. Si lo reseteas, premia la adaptación rápida)
            # survivor.virtual_balance = 10000.0 
            new_population.append(survivor)
            survivors_names.append(survivor.name)
            
        # 4. REPRODUCCIÓN: Rellenar el resto con hijos del LÍDER ACTUAL
        leader = self.population[best_idx]
        
        child_id = 1
        while len(new_population) < POPULATION_SIZE:
            child = leader.mutate(LEARNING_RATE)
            child.name = f"👽 Mutante_{self.generation_count}_{child_id}" # Ej: Mutante_Gen5_1
            child.generation = self.generation_count
            new_population.append(child)
            child_id += 1
            
        # Verificar si el Original murió
        original_alive = any("ORIGINAL" in p.name for p in new_population)
        if not original_alive:
            # Solo imprimimos esto una vez, cuando sucede
            pass 
            
        self.population = new_population
        self.best_performer_idx = 0 # El líder siempre queda en la posición 0 de la nueva lista

# ==============================================================================
# 🛠️ UTILIDADES DE DATOS
# ==============================================================================

def get_data_and_transform(symbol, interval, limit, start_date=None):
    # (Misma lógica de descarga que el script anterior)
    end_time = None
    if start_date:
        dt_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        start_ts = int(dt_obj.timestamp() * 1000)
        print(f"📅 Descargando histórico desde {start_date}...")
        # Lógica simplificada de descarga por lotes
        all_data = []
        curr = start_ts
        rem = limit
        while rem > 0:
            fetch = min(rem, 1000)
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={fetch}&startTime={curr}"
            d = requests.get(url).json()
            if not d: break
            all_data.extend(d)
            curr = int(d[-1][0]) + 1
            rem -= len(d)
        data = all_data
    else:
        print(f"📅 Descargando {limit} velas recientes...")
        # Descarga hacia atrás
        all_data = []
        curr_end = int(time.time() * 1000)
        rem = limit
        while rem > 0:
            fetch = min(rem, 1000)
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={fetch}&endTime={curr_end}"
            d = requests.get(url).json()
            if not d: break
            all_data = d + all_data
            curr_end = int(d[0][0]) - 1
            rem -= len(d)
        data = all_data

    # Parsear
    closes = np.array([float(x[4]) for x in data])
    opens = np.array([float(x[1]) for x in data])
    highs = np.array([float(x[2]) for x in data])
    lows = np.array([float(x[3]) for x in data])
    volumes = np.array([float(x[5]) for x in data])
    
    # Transformar a 3D
    if len(closes) < 2: return None, None
    
    p_changes = np.diff(closes) / closes[:-1]
    vols_aligned = volumes[1:]
    volatilities = (highs[1:] - lows[1:]) / opens[1:]
    
    inputs = np.column_stack([
        p_changes / 0.005,
        vols_aligned / (np.mean(vols_aligned)*3 + 1e-6),
        volatilities / 0.005
    ])
    
    return inputs, closes[1:]

# ==============================================================================
# 🚀 EJECUCIÓN
# ==============================================================================

def run_backtest_with_learning():
    hive = HiveMind(GENOME_FILE)
    inputs, prices = get_data_and_transform(SYMBOL, INTERVAL, BACKTEST_CANDLES, START_DATE)
    
    if inputs is None: return

    print(f"\n🧠 INICIANDO BACKTEST EVOLUTIVO ({len(inputs)} velas)...")
    
    # --- CONFIGURACIÓN DE RIESGO ---
    balance = 10000.0
    LEVERAGE = 50.0
    RISK_PER_TRADE = 0.20  # 0.20 significa que apuesta el 20% del balance actual
    # Si prefieres monto fijo, usa: FIXED_BET = 1000.0 y cambia la lógica abajo
    
    position = 0      # 0: Nada, 1: Long, -1: Short
    entry_price = 0
    current_bet = 0   # Aquí guardaremos cuánto dinero pusimos en la mesa
    equity = []
    
    for t in range(len(inputs)):
        # 1. La Colmena procesa la vela
        signal = hive.process_candle(inputs[t], prices[t])
        current_price = prices[t]
        
        # --- LÓGICA DE CIERRE ---
        if position != 0:
            # Verificar si hay señal de salida o cambio de tendencia
            close_trade = False
            if position == 1 and signal < -0.8: close_trade = True  # Cierre Long
            if position == -1 and signal > 0.8: close_trade = True  # Cierre Short
            
            if close_trade:
                # Calcular PnL del precio
                pnl_pct = (current_price - entry_price) / entry_price
                if position == -1: pnl_pct *= -1 # Invertir si es Short
                
                # Calcular ganancia/pérdida real en dinero
                # (Apuesta * Leverage * %Cambio) - (Apuesta * Leverage * Fee)
                gross_profit = (current_bet * LEVERAGE) * pnl_pct
                fee = (current_bet * LEVERAGE) * 0.001 # 0.1% fee salida
                
                net_profit = gross_profit - fee
                balance += net_profit
                
                position = 0
                current_bet = 0

        # --- LÓGICA DE APERTURA ---
        if position == 0:
            if signal > 0.8 or signal < -0.8:
                # Definir tamaño de la apuesta
                current_bet = balance * RISK_PER_TRADE 
                # O si quieres fijo: current_bet = min(balance, 2000.0)
                
                # Fee de entrada
                entry_fee = (current_bet * LEVERAGE) * 0.001
                balance -= entry_fee
                
                entry_price = current_price
                
                if signal > 0.8:
                    position = 1 # Long
                else:
                    position = -1 # Short

        # --- LIQUIDACIÓN (MARGIN CALL) ---
        # Si la pérdida supera la apuesta, perdemos la apuesta y cerramos
        if position != 0:
            unrealized_pnl = (current_price - entry_price) / entry_price
            if position == -1: unrealized_pnl *= -1
            
            # Si el PnL con apalancamiento es <= -100% (aprox -0.8 para seguridad)
            if unrealized_pnl * LEVERAGE <= -0.8:
                balance -= current_bet # Perdemos lo apostado
                position = 0
                current_bet = 0
        
        equity.append(balance)
        
        if t % 1000 == 0:
            print(f"   ⏳ Vela {t}/{len(inputs)} | Balance: ${balance:.2f}")

    print(f"🏁 FINAL: ${balance:.2f}")
    plt.plot(equity)
    plt.title(f"Equity Curve (Risk: {RISK_PER_TRADE*100}% per trade)")
    plt.show()

if __name__ == "__main__":
    if MODE == "BACKTEST":
        run_backtest_with_learning()
    # Para LIVE, la lógica es igual: instanciar HiveMind y llamar a hive.process_candle en cada mensaje de websocket.