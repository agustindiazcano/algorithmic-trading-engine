import numpy as np
import matplotlib.pyplot as plt
import time
import requests 
import random
import sys

# Forzar UTF-8 en consola para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURACIÓN DE TRADING
TRADING_MODE = "BOTH"  # Opciones: "LONG", "SHORT", "BOTH"

# ==============================================================================
# 🧬 NEURONA MOLECULAR (Protocolo Carbono - Sin Mitosis)
# ==============================================================================

class MolecularNeuron:
    """
    Neurona Molecular con:
    - Fase 1: Respiración Termodinámica (radio ~ volatilidad)
    - Fase 2: Geometría Elipsoidal (Mahalanobis)
    - Fase 3: Inferencia por Registro (SE(2) search)
    """
    def __init__(self, genome_slice):
        # Genoma: [t, p, vol, R_base, W, Sx, Sy, Theta]
        # t, p, vol: Centroide (tiempo relativo, precio, volatilidad)
        # R_base: Radio base
        # W: Peso
        # Sx, Sy: Estiramiento del elipsoide
        # Theta: Rotación del elipsoide
        self.params = genome_slice
    
    def activate_molecular(self, market_window, current_volatility):
        """
        Inferencia Termodinámica y Geométrica.
        
        Args:
            market_window: Array de últimos N puntos [precio, volumen, volatilidad]
            current_volatility: Temperatura actual del mercado (ATR/StdDev)
        
        Returns:
            float: Activación ponderada
        """
        # --- FASE 1: RESPIRACIÓN TERMODINÁMICA ---
        # Radio efectivo = radio_base * (1 + alpha * temperatura)
        r_base = self.params[3]
        breathing_factor = 1.0 + (current_volatility * 5.0)
        r_effective = r_base * breathing_factor
        
        # --- FASE 2: GEOMETRÍA ELIPSOIDAL (MAHALANOBIS) ---
        sx, sy = self.params[5], self.params[6]
        theta = self.params[7]
        c, s = np.cos(theta), np.sin(theta)
        
        # Matriz de Rotación 2D
        R_matrix = np.array([[c, -s], [s, c]])
        # Matriz de Escala (Inversa del tamaño)
        S_matrix = np.array([[1/(sx+1e-6), 0], [0, 1/(sy+1e-6)]])
        
        # Transformación Total: T = S * R^T
        Transformation = S_matrix @ R_matrix.T
        
        # Centro de la neurona (solo precio y volumen para 2D)
        center = self.params[1:3]  # [precio, volumen]
        
        # --- FASE 3: INFERENCIA POR REGISTRO (LA CAZA) ---
        # Escanear últimos 5 puntos para encontrar mejor encaje
        max_activation = 0.0
        
        search_window = market_window[-5:] if len(market_window) >= 5 else market_window
        
        for point in search_window:
            # Vector diferencia (solo precio y volumen)
            diff = point[0:2] - center
            
            # Aplicar transformación del elipsoide
            diff_transformed = Transformation @ diff
            
            # Distancia Mahalanobis (Euclidiana en espacio transformado)
            dist = np.linalg.norm(diff_transformed)
            
            # Función de activación volumétrica (soft-shell)
            activation = np.maximum(0, 1 - dist / (r_effective + 1e-6))
            
            # Mejor encaje
            if activation > max_activation:
                max_activation = activation
        
        # Retornar activación ponderada
        weight = self.params[4]
        return max_activation * weight

# ==============================================================================
# 🧠 CEREBRO MOLECULAR
# ==============================================================================

class MolecularBrain:
    def __init__(self, n_neurons=10):
        self.n_neurons = n_neurons
        self.neurons = []
        
        # Sistema de energía metabólica
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        
        # Gen de personalidad: Umbral de decisión
        self.decision_threshold = np.random.uniform(0.3, 0.9)
        
        # Ventana de mercado para búsqueda temporal
        self.market_window = []
        
        self.reset_genome()
    
    def reset_genome(self):
        """Crea población de neuronas moleculares con genomas aleatorios"""
        self.neurons = []
        for _ in range(self.n_neurons):
            genome = np.array([
                np.random.uniform(-1.0, 1.0),     # t (tiempo relativo)
                np.random.uniform(-1.0, 1.0),     # p (precio normalizado)
                np.random.uniform(-1.0, 1.0),     # vol (volumen normalizado)
                np.random.uniform(0.5, 1.5),      # R_base (radio base)
                np.random.uniform(-1.0, 1.0),     # W (peso)
                np.random.uniform(0.5, 2.0),      # Sx (estiramiento X)
                np.random.uniform(0.5, 2.0),      # Sy (estiramiento Y)
                np.random.uniform(-np.pi/2, np.pi/2)  # Theta (rotación)
            ])
            self.neurons.append(MolecularNeuron(genome))
    
    def extract_genome(self):
        """Extrae genomas de todas las neuronas para evolución"""
        return np.array([neuron.params for neuron in self.neurons])
    
    def load_genome(self, genome_array):
        """Carga genomas en las neuronas"""
        self.neurons = []
        for genome_slice in genome_array:
            self.neurons.append(MolecularNeuron(genome_slice))
    
    def predict(self, market_point, current_volatility):
        """
        Predicción del cerebro completo.
        
        Args:
            market_point: [precio_norm, volumen_norm, volatilidad_norm]
            current_volatility: Volatilidad actual (temperatura)
        
        Returns:
            float: Señal de trading [-1, 1]
        """
        # Actualizar ventana de mercado (últimos 10 puntos)
        self.market_window.append(market_point)
        if len(self.market_window) > 10:
            self.market_window.pop(0)
        
        # Activación de todas las neuronas
        total_activation = 0
        for neuron in self.neurons:
            total_activation += neuron.activate_molecular(self.market_window, current_volatility)
        
        # Tanh para normalizar a [-1, 1]
        return np.tanh(total_activation)

# ==============================================================================
# 🌍 MERCADO REAL (Con Volatilidad)
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=20000, custom_end_time=None):
    if custom_end_time is None:
        current_end_time = int(time.time() * 1000)
        print(f"📡 Bajando {total_candles} velas recientes de {symbol}...")
    else:
        current_end_time = int(custom_end_time)
        readable_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_end_time/1000))
        print(f"📡 Bajando {total_candles} velas HISTÓRICAS terminando en {readable_date}...")
    
    limit_per_call = 1000
    all_closes = []
    all_volumes = []
    all_highs = []
    all_lows = []
    
    calls = total_candles // limit_per_call
    
    for i in range(calls):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit_per_call}&endTime={current_end_time}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data:
                print(f"⚠️  Sin datos en llamada {i+1}")
                break
            
            closes = [float(k[4]) for k in data]
            volumes = [float(k[5]) for k in data]
            highs = [float(k[2]) for k in data]
            lows = [float(k[3]) for k in data]
            
            all_closes = closes + all_closes
            all_volumes = volumes + all_volumes
            all_highs = highs + all_highs
            all_lows = lows + all_lows
            
            current_end_time = int(data[0][0]) - 1
            
        except Exception as e:
            print(f"❌ Error en llamada {i+1}: {e}")
            break
    
    if len(all_closes) < 100:
        print(f"❌ Datos insuficientes: {len(all_closes)} velas")
        return None, None, None
    
    print(f"✅ DATA CARGADA: {len(all_closes)} velas.")
    
    prices = np.array(all_closes)
    volumes = np.array(all_volumes)
    highs = np.array(all_highs)
    lows = np.array(all_lows)
    
    # Volatilidad = (High - Low) / Close
    volatilities = (highs - lows) / prices
    
    return prices, volumes, volatilities

def prepare_inputs(prices, volumes, volatilities):
    if prices is None:
        return None, None, None
    
    # Retornos
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    # Normalización
    vol_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
    ret_norm = returns / (np.std(returns) + 1e-8)
    volatility_norm = volatilities / (np.std(volatilities) + 1e-8)
    
    # Inputs: [precio_norm, volumen_norm, volatilidad_norm]
    inputs = np.column_stack([ret_norm, vol_norm, volatility_norm])
    
    # Volatilidad rolling para "temperatura"
    window = 20
    rolling_vol = np.array([np.std(returns[max(0, i-window):i+1]) for i in range(len(returns))])
    
    return inputs, prices, rolling_vol

# ==============================================================================
# ⚔️ ENTRENAMIENTO (EVOLUCIÓN)
# ==============================================================================

def run_evolution(generations=10, population_size=50):
    print(f"\n💀 --- FASE 1: ENTRENAMIENTO (DATOS RECIENTES) ---")
    
    # 1. Datos de entrenamiento
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 5000)
    inputs, prices_aligned, rolling_volatility = prepare_inputs(market_prices, market_vols, market_volatilities)
    
    if inputs is None: return None, None
    
    brain = MolecularBrain(n_neurons=10)
    best_genome = brain.extract_genome()
    best_threshold = brain.decision_threshold
    best_fitness = -np.inf
    
    # Parámetros
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.001
    FEE = 0.001
    
    # Sistema de energía
    METABOLIC_COST_FIXED = 0.01
    ENERGY_FROM_PROFIT = 0.1
    
    initial_mut = 0.90
    final_mut = 0.001
    
    for gen in range(generations):
        if gen % 1000 == 0 and gen > 0:
            np.save(f"best_genome_molecular_gen_{gen}.npy", best_genome)
            print(f"💾 [AUTO-SAVE] Gen {gen} guardada.")
        
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        print(f"\n--- GENERACIÓN {gen+1}/{generations} (Mutación: {mut_rate:.4f}) ---")
        
        gen_best_fitness = -np.inf
        gen_best_genome = None
        gen_best_threshold = None
        
        # Evaluar población
        for i in range(population_size):
            brain = MolecularBrain(n_neurons=10)
            
            # ELITISMO
            if i == 0 and gen > 0:
                brain.load_genome(best_genome)
                brain.decision_threshold = best_threshold
            else:
                # Mutar
                mutant = np.copy(best_genome)
                mask = np.random.rand(*mutant.shape) < 0.1
                mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
                mutant[:, 3] = np.abs(mutant[:, 3])  # R_base positivo
                mutant[:, 5:7] = np.abs(mutant[:, 5:7])  # Sx, Sy positivos
                
                brain.load_genome(mutant)
                
                brain.decision_threshold = best_threshold + np.random.normal(0, mut_rate * 0.5)
                brain.decision_threshold = np.clip(brain.decision_threshold, 0.3, 0.9)
            
            balance = INITIAL_BALANCE
            position = 0
            entry_price = 0
            current_bet = 0
            trades = 0
            alive = True
            
            for t in range(len(inputs)):
                # Metabolismo
                brain.energy -= METABOLIC_COST_FIXED
                if brain.energy <= 0:
                    alive = False
                    break
                
                # Bancarrota
                if balance <= 0:
                    alive = False
                    break
                
                # Predicción molecular (con temperatura)
                signal = brain.predict(inputs[t], rolling_volatility[t])
                current_price = prices_aligned[t]
                
                # Cierre
                if (signal > brain.decision_threshold and position == -1) or \
                   (signal < -brain.decision_threshold and position == 1):
                    pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                              else (entry_price - current_price) / entry_price * LEVERAGE
                    profit = current_bet * pnl_pct
                    cost = (current_bet * LEVERAGE) * FEE
                    balance += current_bet + (profit - cost)
                    
                    # Recuperar energía
                    if profit > 0:
                        energy_boost = profit * ENERGY_FROM_PROFIT
                        brain.energy = min(brain.energy + energy_boost, brain.MAX_ENERGY)
                    
                    position = 0
                    current_bet = 0
                
                # Apertura
                if position == 0:
                    if signal > brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]:
                        position = 1
                        entry_price = current_price * (1 + SLIPPAGE)
                        current_bet = balance * BET_PERCENTAGE
                        balance -= current_bet
                        balance -= (current_bet * LEVERAGE) * FEE
                        trades += 1
                    elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]:
                        position = -1
                        entry_price = current_price * (1 - SLIPPAGE)
                        current_bet = balance * BET_PERCENTAGE
                        balance -= current_bet
                        balance -= (current_bet * LEVERAGE) * FEE
                        trades += 1
                
                # Liquidación
                if position != 0:
                    unrealized = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                                 else (entry_price - current_price) / entry_price * LEVERAGE
                    if unrealized <= -0.9:
                        balance -= current_bet
                        position = 0
                        current_bet = 0
            
            # Cierre final forzado
            if alive and position != 0:
                final_price = prices_aligned[-1]
                pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                          else (entry_price - final_price) / entry_price * LEVERAGE
                profit = current_bet * pnl_pct
                cost = (current_bet * LEVERAGE) * FEE
                balance += current_bet + (profit - cost)
            
            if not alive: fitness = 0
            elif trades < 3: fitness = 500
            else: fitness = 2000 + balance
            
            # Mejor de esta generación
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_genome = brain.extract_genome()
                gen_best_threshold = brain.decision_threshold
        
        # Actualizar mejor global
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = np.copy(gen_best_genome)
            best_threshold = gen_best_threshold
            print(f"🏆 MEJOR DE GEN {gen}: Balance ${balance:.2f} | Trades: {trades} | Energía: {brain.energy:.1f} | Umbral: {best_threshold:.2f}")
    
    np.save("best_genome_molecular_FINAL.npy", best_genome)
    np.save("best_threshold_molecular_FINAL.npy", best_threshold)
    return best_genome, best_threshold

# ==============================================================================
# 🧪 TEST (OUT OF SAMPLE)
# ==============================================================================

def run_test_simulation(genome, threshold, inputs, prices, rolling_volatility):
    brain = MolecularBrain(n_neurons=10)
    brain.load_genome(genome)
    brain.decision_threshold = threshold
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.001
    FEE = 0.001
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    actions = []
    equity_curve = []
    
    print("\n⏯️  Ejecutando simulación...")
    
    for t in range(len(inputs)):
        if balance <= 0:
            break
        
        signal = brain.predict(inputs[t], rolling_volatility[t])
        current_price = prices[t]
        act = 0
        
        # Cierre
        if (signal > brain.decision_threshold and position == -1) or \
           (signal < -brain.decision_threshold and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                      else (entry_price - current_price) / entry_price * LEVERAGE
            profit = current_bet * pnl_pct
            cost = (current_bet * LEVERAGE) * FEE
            balance += current_bet + (profit - cost)
            position = 0
            current_bet = 0
        
        # Apertura
        if position == 0:
            if signal > brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]:
                position = 1
                entry_price = current_price * (1 + SLIPPAGE)
                current_bet = balance * BET_PERCENTAGE
                balance -= current_bet
                balance -= (current_bet * LEVERAGE) * FEE
                act = 1
            elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]:
                position = -1
                entry_price = current_price * (1 - SLIPPAGE)
                current_bet = balance * BET_PERCENTAGE
                balance -= current_bet
                balance -= (current_bet * LEVERAGE) * FEE
                act = -1
        
        actions.append(act)
        equity_curve.append(balance)
    
    # Cierre final
    if position != 0:
        final_price = prices[-1]
        pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                  else (entry_price - final_price) / entry_price * LEVERAGE
        profit = current_bet * pnl_pct
        cost = (current_bet * LEVERAGE) * FEE
        balance += current_bet + (profit - cost)
    
    return balance, actions, equity_curve

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1. ENTRENAR
    winner_genome, winner_threshold = run_evolution(generations=10)
    
    if winner_genome is not None:
        print("\n" + "="*60)
        print("🎲 --- FASE 2: TEST CIEGO (OUT OF SAMPLE) ---")
        print("="*60)
        
        # 2. Fecha aleatoria
        now_ms = int(time.time() * 1000)
        six_months_ago_ms = now_ms - (180 * 24 * 60 * 60 * 1000)
        random_test_time = random.randint(six_months_ago_ms, now_ms)
        
        # 3. Datos de test
        test_prices, test_vols, test_volatilities = fetch_extended_data(
            "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
        )
        test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs(test_prices, test_vols, test_volatilities)
        
        if test_inputs is not None:
            # 4. EJECUTAR TEST
            final_bal, acts, equity = run_test_simulation(winner_genome, winner_threshold, test_inputs, test_prices_aligned, test_rolling_vol)
            
            profit_pct = ((final_bal - 10000) / 10000) * 100
            print(f"\n📊 RESULTADO DEL TEST:")
            print(f"   Balance Inicial: $10,000")
            print(f"   Balance Final:   ${final_bal:,.2f}")
            print(f"   Profit/Loss:     {profit_pct:+.2f}%")
