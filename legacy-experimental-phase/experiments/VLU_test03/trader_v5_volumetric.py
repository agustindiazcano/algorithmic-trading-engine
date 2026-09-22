import numpy as np
import matplotlib.pyplot as plt
import time
import requests 
import random
import sys
from scipy.spatial.transform import Rotation as R

# Forzar UTF-8 en consola para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURACIÓN DE TRADING
TRADING_MODE = "BOTH"  # Opciones: "LONG", "SHORT", "BOTH"

# ==============================================================================
# 🧬 NEURONA VOLUMÉTRICA 3D (Phase Space Scanner)
# ==============================================================================

class VolumetricNeuron3D:
    """
    Neurona que escanea trayectorias en el espacio de fases 3D.
    
    Ejes del espacio de fases:
    - X: Cambio de precio (retorno)
    - Y: Volumen relativo
    - Z: Aceleración/Volatilidad
    
    La neurona es un "molde" elipsoidal rotado en 3D que detecta
    cuando la trayectoria del mercado (el "gusano") pasa a través de ella.
    """
    def __init__(self, genome_slice):
        # ADN GEOMÉTRICO (11 parámetros)
        # [x, y, z, R_base, W, Sx, Sy, Sz, Euler_X, Euler_Y, Euler_Z]
        self.params = genome_slice
        self.center = self.params[0:3]  # Centro del elipsoide en 3D
        
        # Memoria Bayesiana (NO heredable)
        self.alpha = 1.0
        self.beta = 1.0
        self.confidence = 0.5
    
    def activate_3d_scan(self, market_trajectory, current_temp):
        """
        Escaneo volumétrico de la trayectoria del mercado.
        
        Args:
            market_trajectory: Lista de puntos 3D [x, y, z] (el "gusano")
            current_temp: Volatilidad actual (temperatura para respiración)
        
        Returns:
            float: Activación ponderada por peso y confianza
        """
        # --- FASE 1: RESPIRACIÓN TERMODINÁMICA ---
        r_base = self.params[3]
        r_eff = r_base * (1.0 + current_temp * 3.0)
        
        # --- FASE 2: ORIENTACIÓN ESPACIAL (Rotación 3D con Euler) ---
        euler_angles = self.params[8:11]  # [roll, pitch, yaw]
        rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
        
        # --- FASE 3: ANISOTROPÍA (Forma del elipsoide) ---
        stretch = self.params[5:8]  # [Sx, Sy, Sz]
        S_inv = np.diag(1.0 / (stretch + 1e-6))
        
        # Transformación completa: M = S_inv * R^T
        M = S_inv @ rotation_matrix.T
        
        # --- FASE 4: ESCANEO DE TRAYECTORIA (El Gusano) ---
        max_overlap = 0.0
        
        for point in market_trajectory:
            # Diferencia al centro
            diff = point - self.center
            
            # Proyectar al espacio local (rotado y escalado)
            local_diff = M @ diff
            
            # Distancia física dentro del elipsoide
            dist = np.linalg.norm(local_diff)
            
            # Activación volumétrica: ¿Cuánto del punto está dentro?
            overlap = np.maximum(0, 1 - dist / (r_eff + 1e-6))
            
            if overlap > max_overlap:
                max_overlap = overlap
        
        # --- FASE 5: FILTRO DE CONFIANZA (Bayes) ---
        self.confidence = self.alpha / (self.alpha + self.beta)
        
        # Output: Encaje físico * Peso * Confianza
        weight = self.params[4]
        return max_overlap * weight * self.confidence
    
    def learn_from_result(self, pnl):
        """Aprendizaje Bayesiano desde resultado de trade"""
        learning_rate = 0.1
        
        if pnl > 0:
            self.alpha += learning_rate
        else:
            self.beta += learning_rate
        
        # Normalización
        if self.alpha + self.beta > 20:
            self.alpha *= 0.9
            self.beta *= 0.9

# ==============================================================================
# 🧠 CEREBRO VOLUMÉTRICO
# ==============================================================================

class VolumetricBrain:
    def __init__(self, n_neurons=10):
        self.n_neurons = n_neurons
        self.neurons = []
        
        # Sistema de energía
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        
        # Gen de personalidad
        self.decision_threshold = np.random.uniform(0.3, 0.9)
        
        # Historial de trayectoria (El Gusano)
        self.worm_history = []
        
        self.reset_genome()
    
    def reset_genome(self):
        """Crea población de neuronas volumétricas"""
        self.neurons = []
        for _ in range(self.n_neurons):
            genome = np.array([
                np.random.uniform(-1.0, 1.0),     # x (centro)
                np.random.uniform(-1.0, 1.0),     # y
                np.random.uniform(-1.0, 1.0),     # z
                np.random.uniform(0.5, 1.5),      # R_base
                np.random.uniform(-1.0, 1.0),     # W (peso)
                np.random.uniform(0.5, 2.0),      # Sx
                np.random.uniform(0.5, 2.0),      # Sy
                np.random.uniform(0.5, 2.0),      # Sz
                np.random.uniform(-np.pi, np.pi), # Euler X (roll)
                np.random.uniform(-np.pi, np.pi), # Euler Y (pitch)
                np.random.uniform(-np.pi, np.pi)  # Euler Z (yaw)
            ])
            self.neurons.append(VolumetricNeuron3D(genome))
    
    def extract_genome(self):
        return np.array([neuron.params for neuron in self.neurons])
    
    def load_genome(self, genome_array):
        self.neurons = []
        for genome_slice in genome_array:
            self.neurons.append(VolumetricNeuron3D(genome_slice))
    
    def predict(self, market_point, current_volatility):
        """
        Predicción volumétrica.
        
        Args:
            market_point: Punto 3D actual [retorno, volumen, aceleración]
            current_volatility: Temperatura del mercado
        
        Returns:
            float: Señal de trading [-1, 1]
        """
        # Actualizar el gusano (trayectoria)
        self.worm_history.append(market_point)
        if len(self.worm_history) > 10:
            self.worm_history.pop(0)
        
        # Escanear con todas las neuronas
        total_activation = 0
        for neuron in self.neurons:
            total_activation += neuron.activate_3d_scan(self.worm_history, current_volatility)
        
        return np.tanh(total_activation)
    
    def feedback_all(self, pnl):
        """Propagar feedback a todas las neuronas"""
        for neuron in self.neurons:
            neuron.learn_from_result(pnl)

# ==============================================================================
# 🌍 MERCADO REAL
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
    volatilities = (highs - lows) / prices
    
    return prices, volumes, volatilities

def prepare_inputs_3d(prices, volumes, volatilities):
    """
    Prepara inputs para espacio de fases 3D.
    
    Returns:
        inputs: Array Nx3 [retorno, volumen_norm, aceleración]
        prices: Precios alineados
        rolling_vol: Volatilidad rolling (temperatura)
    """
    if prices is None:
        return None, None, None
    
    # Eje X: Retornos (cambio de precio)
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    # Eje Y: Volumen normalizado
    vol_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
    
    # Eje Z: Aceleración (segunda derivada del precio)
    acceleration = np.diff(returns)
    acceleration = np.concatenate([[0], acceleration])
    
    # Normalizar
    ret_norm = returns / (np.std(returns) + 1e-8)
    acc_norm = acceleration / (np.std(acceleration) + 1e-8)
    
    # Inputs 3D: [retorno, volumen, aceleración]
    inputs = np.column_stack([ret_norm, vol_norm, acc_norm])
    
    # Volatilidad rolling (temperatura)
    window = 20
    rolling_vol = np.array([np.std(returns[max(0, i-window):i+1]) for i in range(len(returns))])
    
    return inputs, prices, rolling_vol

# ==============================================================================
# ⚔️ ENTRENAMIENTO
# ==============================================================================

def run_evolution(generations=10, population_size=50):
    print(f"\n💀 --- FASE 1: ENTRENAMIENTO (DATOS RECIENTES) ---")
    
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 5000)
    inputs, prices_aligned, rolling_volatility = prepare_inputs_3d(market_prices, market_vols, market_volatilities)
    
    if inputs is None: return None, None
    
    brain = VolumetricBrain(n_neurons=10)
    best_genome = brain.extract_genome()
    best_threshold = brain.decision_threshold
    best_fitness = -np.inf
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.001
    FEE = 0.001
    
    METABOLIC_COST_FIXED = 0.01
    ENERGY_FROM_PROFIT = 0.1
    
    initial_mut = 0.90
    final_mut = 0.001
    
    for gen in range(generations):
        if gen % 1000 == 0 and gen > 0:
            np.save(f"best_genome_volumetric_gen_{gen}.npy", best_genome)
            print(f"💾 [AUTO-SAVE] Gen {gen} guardada.")
        
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        print(f"\n--- GENERACIÓN {gen+1}/{generations} (Mutación: {mut_rate:.4f}) ---")
        
        gen_best_fitness = -np.inf
        gen_best_genome = None
        gen_best_threshold = None
        
        for i in range(population_size):
            brain = VolumetricBrain(n_neurons=10)
            
            # ELITISMO
            if i == 0 and gen > 0:
                brain.load_genome(best_genome)
                brain.decision_threshold = best_threshold
            else:
                mutant = np.copy(best_genome)
                mask = np.random.rand(*mutant.shape) < 0.1
                mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
                mutant[:, 3] = np.abs(mutant[:, 3])  # R_base positivo
                mutant[:, 5:8] = np.abs(mutant[:, 5:8])  # Sx, Sy, Sz positivos
                
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
                brain.energy -= METABOLIC_COST_FIXED
                if brain.energy <= 0:
                    alive = False
                    break
                
                if balance <= 0:
                    alive = False
                    break
                
                # Predicción volumétrica 3D
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
                    
                    # Feedback Bayesiano
                    brain.feedback_all(profit)
                    
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
            
            # Cierre final
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
            
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_genome = brain.extract_genome()
                gen_best_threshold = brain.decision_threshold
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = np.copy(gen_best_genome)
            best_threshold = gen_best_threshold
            print(f"🏆 MEJOR DE GEN {gen}: Balance ${balance:.2f} | Trades: {trades} | Energía: {brain.energy:.1f} | Umbral: {best_threshold:.2f}")
    
    np.save("best_genome_volumetric_FINAL.npy", best_genome)
    np.save("best_threshold_volumetric_FINAL.npy", best_threshold)
    return best_genome, best_threshold

# ==============================================================================
# 🧪 TEST
# ==============================================================================

def run_test_simulation(genome, threshold, inputs, prices, rolling_volatility):
    brain = VolumetricBrain(n_neurons=10)
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
            
            brain.feedback_all(profit)
            
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
    winner_genome, winner_threshold = run_evolution(generations=10)
    
    if winner_genome is not None:
        print("\n" + "="*60)
        print("🎲 --- FASE 2: TEST CIEGO (OUT OF SAMPLE) ---")
        print("="*60)
        
        now_ms = int(time.time() * 1000)
        six_months_ago_ms = now_ms - (180 * 24 * 60 * 60 * 1000)
        random_test_time = random.randint(six_months_ago_ms, now_ms)
        
        test_prices, test_vols, test_volatilities = fetch_extended_data(
            "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
        )
        test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs_3d(test_prices, test_vols, test_volatilities)
        
        if test_inputs is not None:
            final_bal, acts, equity = run_test_simulation(winner_genome, winner_threshold, test_inputs, test_prices_aligned, test_rolling_vol)
            
            profit_pct = ((final_bal - 10000) / 10000) * 100
            print(f"\n📊 RESULTADO DEL TEST:")
            print(f"   Balance Inicial: $10,000")
            print(f"   Balance Final:   ${final_bal:,.2f}")
            print(f"   Profit/Loss:     {profit_pct:+.2f}%")
