import numpy as np
import matplotlib.pyplot as plt
import time
import requests 

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL (VERSIÓN 3 - SPIN/ROTACIÓN) 🌪️
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=50, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        # Genoma: [x, y, z, R, W, Sx, Sy, Sz, ÁNGULO]
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3), # Posición (x,y,z)
                [np.random.uniform(0.5, 1.5)],   # Radio
                [np.random.uniform(-1.0, 1.0)],  # Peso
                np.random.uniform(0.5, 1.5, 3),  # Stretch
                [np.random.uniform(-0.5, 0.5)]   # <--- SPIN: Rotación +/- 30 grados
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def predict(self, points):
        # Inferencia Volumétrica CON SPIN 🌪️
        if len(self.genome) == 0: return 0.0
        
        # 1. Desempaquetar
        centers = self.genome[:, :3]
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8]
        thetas = self.genome[:, 8]        # Ángulo
        
        # 2. DEFINIR EL PUNTO (FIXED)
        point = points[-1].reshape(1, 3) 
        
        # 3. Diferencia Lineal
        diff = point - centers 
        
        # 4. ROTACIÓN 2D (Ejes Precio y Volumen)
        c, s = np.cos(thetas), np.sin(thetas)
        
        # Rotamos el vector diferencia para ver si encaja con la neurona inclinada
        dx = diff[:, 0] * c - diff[:, 1] * s
        dy = diff[:, 0] * s + diff[:, 1] * c
        dz = diff[:, 2] # Eje Z (Volumen/Volatilidad) sin rotar
        
        diff_rotated = np.stack([dx, dy, dz], axis=1)
        
        # 5. Distancia sobre vector rotado
        diff_stretched = diff_rotated / (stretch + 1e-6)
        dists = np.linalg.norm(diff_stretched, axis=1)
        
        # 6. Activación
        safe_radii = np.maximum(radii, 1e-6)
        raw_overlap = np.maximum(0, 1 - dists / safe_radii)
        
        plateau = np.minimum(1.0, raw_overlap * 3.0)
        total_activation = np.sum(plateau * weights)
        
        return np.tanh(total_activation)

# ==============================================================================
# 🌍 MERCADO REAL (20,000 VELAS - 14 DÍAS)
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=20000):
    print(f"📡 Conectando a Binance: Bajando {total_candles} velas de {symbol} ({interval})...")
    print(f"⏳ Esto son aprox {total_candles/60/24:.1f} días de historia.")
    
    limit_per_call = 1000
    all_closes = []
    all_volumes = []
    all_highs = []
    all_lows = []
    all_opens = []
    
    current_end_time = int(time.time() * 1000) 
    calls = total_candles // limit_per_call
    
    for i in range(calls):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit_per_call}&endTime={current_end_time}"
        try:
            response = requests.get(url)
            data = response.json()
            if not isinstance(data, list) or len(data) == 0: break
                
            closes = [float(x[4]) for x in data]
            opens = [float(x[1]) for x in data]
            highs = [float(x[2]) for x in data]
            lows = [float(x[3]) for x in data]
            volumes = [float(x[5]) for x in data]
            
            # Orden inverso (vamos hacia el pasado)
            all_closes = closes + all_closes
            all_volumes = volumes + all_volumes
            all_highs = highs + all_highs
            all_lows = lows + all_lows
            all_opens = opens + all_opens
            
            current_end_time = int(data[0][0]) - 1
            # print(f"   ... Lote {i+1}/{calls} OK.")
            
        except Exception as e:
            print(f"❌ Error bajando lote: {e}")
            break

    closes = np.array(all_closes)
    opens = np.array(all_opens)
    highs = np.array(all_highs)
    lows = np.array(all_lows)
    volumes = np.array(all_volumes)
    volatilities = (highs - lows) / opens
    
    print(f"✅ DATA FINAL: {len(closes)} velas listas.")
    return closes, volumes, volatilities

# ==============================================================================
# ⚔️ SIMULACIÓN EVOLUTIVA
# ==============================================================================

def run_evolution(generations=1000):
    # --- DENTRO DE run_evolution ---
    INITIAL_BALANCE = 1000.0
    MAX_BET = 200.0      
    FEE = 0.001          
    LEVERAGE = 10.0  # <--- AGREGÁ ESTA LÍNEA (ej: 10.0 para x10)
    print(f"💀 INICIANDO SIMULACIÓN CON NEURONAS GIRATORIAS (SPIN)")
    
    # Bajamos 20k velas de BTC (o cambialo a PEPEUSDT si sos valiente)
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 20000)
    
    if len(market_prices) < 1000: return None, None, None, None

    # Normalización
    p_changes = np.diff(market_prices) / market_prices[:-1]
    prices_aligned = market_prices[1:]
    vols_aligned = market_vols[1:]
    volatilities_aligned = market_volatilities[1:]
    
    inputs = np.column_stack([
        p_changes / 0.001,    # Más sensible para 1min
        vols_aligned / (np.mean(vols_aligned) * 3 + 1e-6), 
        volatilities_aligned / 0.001 
    ])
    
    brain = DynamicCausalBrain(n_neurons=50, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_fitness = -np.inf
    
    INITIAL_BALANCE = 1000.0
    MAX_BET = 200.0      
    FEE = 0.001          
    
    for gen in range(generations):
        # Mutación
        mut_rate = 0.5 if gen < 100 else 0.05
        mutant = np.copy(best_genome)
        mask = np.random.rand(*mutant.shape) < 0.1
        mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
        
        # Corregir valores físicos (Radios y Stretch positivos)
        # El ángulo (index 8) puede ser negativo, así que no lo tocamos
        mutant[:, 3] = np.abs(mutant[:, 3]) 
        mutant[:, 5:8] = np.abs(mutant[:, 5:8]) 
        
        brain.genome = mutant
        
        # --- BUCLE TRADING ---
        balance = INITIAL_BALANCE
        position = 0
        entry_price = 0
        current_bet = 0
        trades = 0
        alive = True
        
        for t in range(len(inputs)):
            if balance < 10: alive = False; break
                
            signal = brain.predict([inputs[t]]) 
            current_price = prices_aligned[t]
            
            # Cierre
            if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
                pnl_pct = 0
                if position == 1: pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE 
                else: pnl_pct = (entry_price - current_price) / entry_price * LEVERAGE 
                
                profit = current_bet * pnl_pct
                balance += (profit - (current_bet * FEE))
                position = 0
            
            # Apertura
            if position == 0:
                if signal > 0.8: 
                    position = 1
                    entry_price = current_price 
                    current_bet = min(balance, MAX_BET)
                    balance -= (current_bet * FEE)
                    trades += 1
                elif signal < -0.8:
                    position = -1
                    entry_price = current_price
                    current_bet = min(balance, MAX_BET)
                    balance -= (current_bet * FEE)
                    trades += 1
            
            # Liquidación simple
            if position != 0:
                unrealized = (current_price - entry_price)/entry_price if position==1 else (entry_price - current_price)/entry_price
                if unrealized < -0.2: # Stop loss duro del 20%
                    balance -= current_bet
                    position = 0

        # Fitness
        if not alive: fitness = 0
        elif trades < 5: fitness = balance * 0.5 # Castigo por vago
        else: fitness = balance
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = np.copy(mutant)
            print(f"🏆 [Gen {gen}] SPIN: ${balance:.2f} | Trades: {trades} | Mut: {mut_rate:.2f}")

    return best_genome, prices_aligned, vols_aligned, inputs

# --- RUN ---
if __name__ == "__main__":
    winner, prices, vols, inputs = run_evolution(generations=1000)
    
    if winner is not None:
        print("✅ Simulación terminada. Si el balance es > 1000, el Spin funciona.")