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
TRADING_MODE = "LONG"  # Opciones: "LONG", "SHORT", "BOTH"

# ==============================================================================
# 🧬 NEURONA VOLUMÉTRICA 3D CON CUATERNIONES (S³) + FROZEN
# ==============================================================================

class QuaternionNeuron3D:
    """
    Neurona volumétrica avanzada con:
    - Rotaciones en S³ (cuaterniones) para evitar gimbal lock
    - Longitud de gusano variable (gen de escala temporal)
    - Respiración termodinámica
    - Aprendizaje Bayesiano
    - FROZEN flag para escuelita incremental
    """
    def __init__(self, genome_slice, frozen=False):
        self.params = genome_slice
        self.center = self.params[0:3]
        self.frozen = frozen
        
        # Memoria Bayesiana
        self.alpha = 10.0
        self.beta = 1.0
        self.confidence = 0.5
    
    def activate_3d_scan(self, market_trajectory, current_temp):
        """
        Escaneo volumétrico con cuaterniones y ventana adaptativa.
        """
        r_base = self.params[3]
        r_eff = r_base * (1.0 + current_temp * 3.0)
        
        quat = self.params[8:12]
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation_matrix = R.from_quat(quat_norm).as_matrix()
        
        stretch = self.params[5:8]
        S_inv = np.diag(1.0 / (stretch + 1e-6))
        M = S_inv @ rotation_matrix.T
        
        worm_length = int(np.clip(self.params[12], 3, 50))
        adaptive_trajectory = market_trajectory[-worm_length:] if len(market_trajectory) >= worm_length else market_trajectory
        
        max_overlap = 0.0
        for point in adaptive_trajectory:
            diff = point - self.center
            local_diff = M @ diff
            dist = np.linalg.norm(local_diff)
            overlap = np.maximum(0, 1 - dist / (r_eff + 1e-6))
            
            if overlap > max_overlap:
                max_overlap = overlap
        
        self.confidence = self.alpha / (self.alpha + self.beta)
        weight = self.params[4]
        
        return weight * max_overlap * self.confidence
    
    def learn_from_result(self, pnl):
        """Aprendizaje Bayesiano - SKIP si está congelada"""
        if self.frozen:
            return
            
        learning_rate = 0.1
        
        if pnl > 0:
            self.alpha += learning_rate
        else:
            self.beta += learning_rate
        
        if self.alpha + self.beta > 20:
            self.alpha *= 0.9
            self.beta *= 0.9

# ==============================================================================
# 🧠 CEREBRO CUATERNIÓNICO CON ESCUELITA
# ==============================================================================

class QuaternionBrain:
    def __init__(self, n_neurons=10):
        self.neurons = []
        self.frozen_gain = 1.0  # Multiplicador para neuronas congeladas
        self.worm_history = []
        
        # Sistema de energía
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        
        # Umbral de decisión
        self.decision_threshold = np.random.uniform(0.1, 0.3)
        
        if n_neurons > 0:
            self.reset_genome(n_neurons)
    
    def reset_genome(self, n_neurons, market_sample=None, avg_volatility=None):
        """Crea población de neuronas con cuaterniones"""
        self.neurons = []
        for _ in range(n_neurons):
            genome = make_random_genome(market_sample, avg_volatility)
            self.neurons.append(QuaternionNeuron3D(genome, frozen=False))
    
    def freeze_all(self):
        """Congela todas las neuronas existentes"""
        for n in self.neurons:
            n.frozen = True
    
    def add_neurons(self, new_neurons):
        """Agrega nuevas neuronas al cerebro"""
        self.neurons.extend(new_neurons)
    
    def extract_genome(self):
        return np.array([neuron.params for neuron in self.neurons])
    
    def load_genome(self, genome_array, frozen_flags=None):
        self.neurons = []
        for i, genome_slice in enumerate(genome_array):
            is_frozen = frozen_flags[i] if frozen_flags is not None else False
            self.neurons.append(QuaternionNeuron3D(genome_slice, frozen=is_frozen))
    
    def predict(self, market_point, current_volatility):
        """Predicción con soporte para frozen_gain"""
        self.worm_history.append(market_point)
        if len(self.worm_history) > 100:
            self.worm_history.pop(0)
        
        max_activation = 0.0
        winner_activation = 0.0
        
        for neuron in self.neurons:
            activation = neuron.activate_3d_scan(self.worm_history, current_volatility)
            
            # Reducir aporte de neuronas congeladas durante entrenamiento
            if neuron.frozen:
                activation *= self.frozen_gain
            
            if abs(activation) > abs(max_activation):
                max_activation = activation
                winner_activation = activation
        
        return np.tanh(winner_activation)
    
    def feedback_all(self, pnl):
        """Feedback solo a neuronas no congeladas"""
        for neuron in self.neurons:
            neuron.learn_from_result(pnl)

# ==============================================================================
# 🔧 HELPERS PARA GENOMAS Y KMEANS
# ==============================================================================

def random_unit_quat():
    """Genera cuaternión unitario aleatorio"""
    q = np.random.randn(4)
    return q / (np.linalg.norm(q) + 1e-8)

def make_random_genome(market_sample=None, avg_volatility=None):
    """Crea genoma aleatorio con inicialización inteligente opcional"""
    q = random_unit_quat()
    
    # Inicialización inteligente de centros
    if market_sample is not None and len(market_sample) > 0:
        idx = np.random.randint(0, len(market_sample))
        center = market_sample[idx] + np.random.randn(3) * 0.1
    else:
        center = np.random.uniform(-3.0, 3.0, 3)
    
    # Radio base adaptado a volatilidad
    if avg_volatility is not None:
        R_base = np.random.uniform(0.5, 1.5) * (1.0 + avg_volatility * 2.0)
    else:
        R_base = np.random.uniform(0.5, 1.5)
    
    return np.array([
        center[0], center[1], center[2],
        R_base,
        np.random.uniform(-1.0, 1.0),     # W
        np.random.uniform(0.5, 2.0),      # Sx
        np.random.uniform(0.5, 2.0),      # Sy
        np.random.uniform(0.5, 2.0),      # Sz
        q[0], q[1], q[2], q[3],
        np.random.uniform(3, 50)          # worm_length
    ], dtype=float)

def kmeans_centers(block_inputs, k=10, seed=42):
    """KMeans para encontrar centros en el espacio del mercado"""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        km.fit(block_inputs)
        return km.cluster_centers_
    except Exception:
        # Fallback: sample aleatorio si no hay sklearn
        idx = np.random.choice(len(block_inputs), size=min(k, len(block_inputs)), replace=False)
        return block_inputs[idx]

def make_genome_from_center(center_xyz, avg_volatility=None):
    """Crea genoma desde un centro específico (KMeans)"""
    q = random_unit_quat()
    
    if avg_volatility is not None:
        R_base = np.random.uniform(0.5, 1.5) * (1.0 + avg_volatility * 2.0)
    else:
        R_base = np.random.uniform(0.5, 1.5)
    
    return np.array([
        center_xyz[0], center_xyz[1], center_xyz[2],
        R_base,
        np.random.uniform(-1.0, 1.0),
        np.random.uniform(0.5, 2.0),
        np.random.uniform(0.5, 2.0),
        np.random.uniform(0.5, 2.0),
        q[0], q[1], q[2], q[3],
        np.random.uniform(3, 50)
    ], dtype=float)

def spawn_neurons_kmeans(block_inputs, k=10, noise=0.05, avg_volatility=None):
    """Crea k neuronas nuevas usando KMeans en el bloque"""
    centers = kmeans_centers(block_inputs, k=k)
    new_neurons = []
    for c in centers:
        c2 = c + np.random.normal(0, noise, size=3)
        g = make_genome_from_center(c2, avg_volatility)
        new_neurons.append(QuaternionNeuron3D(g, frozen=False))
    return new_neurons

# ==============================================================================
# 🌍 MERCADO REAL
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=5000, custom_end_time=None):
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
            
            # Guard contra errores de Binance (dict en vez de list)
            if not isinstance(data, list):
                print(f"⚠️  Binance error: {data}")
                break
            
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
    """Normalización rolling para evitar look-ahead bias"""
    if prices is None:
        return None, None, None
    
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    acceleration = np.diff(returns)
    acceleration = np.concatenate([[0], acceleration])
    
    window = 100
    
    ret_norm = np.zeros_like(returns)
    vol_norm = np.zeros_like(volumes)
    acc_norm = np.zeros_like(acceleration)
    
    for i in range(len(returns)):
        start_idx = max(0, i - window)
        
        ret_window = returns[start_idx:i+1]
        ret_mean = np.mean(ret_window)
        ret_std = np.std(ret_window) + 1e-8
        ret_norm[i] = (returns[i] - ret_mean) / ret_std
        
        vol_window = volumes[start_idx:i+1]
        vol_mean = np.mean(vol_window)
        vol_std = np.std(vol_window) + 1e-8
        vol_norm[i] = (volumes[i] - vol_mean) / vol_std
        
        acc_window = acceleration[start_idx:i+1]
        acc_mean = np.mean(acc_window)
        acc_std = np.std(acc_window) + 1e-8
        acc_norm[i] = (acceleration[i] - acc_mean) / acc_std
    
    inputs = np.column_stack([ret_norm, vol_norm, acc_norm])
    
    vol_window = 20
    rolling_vol = np.array([np.std(returns[max(0, i-vol_window):i+1]) for i in range(len(returns))])
    
    MIN_TEMP = 0.001
    rolling_vol = np.maximum(rolling_vol, MIN_TEMP)
    
    return inputs, prices, rolling_vol

# ==============================================================================
# 🏫 ESCUELITA: ENTRENAMIENTO INCREMENTAL POR BLOQUES
# ==============================================================================

def evolve_trainables_on_block(brain, threshold, block_inputs, block_prices, block_temp,
                                generations=10, population_size=50):
    """
    Evoluciona SOLO las neuronas trainables (frozen=False) en un bloque.
    Las neuronas congeladas permanecen fijas.
    """
    
    # Extraer genomas trainables
    trainable_indices = [i for i, n in enumerate(brain.neurons) if not n.frozen]
    
    if len(trainable_indices) == 0:
        print("⚠️  No hay neuronas trainables")
        return brain, threshold
    
    best_fitness = -np.inf
    best_trainable_genomes = [brain.neurons[i].params.copy() for i in trainable_indices]
    best_threshold = threshold
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.002
    FEE = 0.0004
    
    # Metabolismo escalado con cantidad de neuronas
    base_metabolic = 0.01
    METABOLIC_COST_FIXED = base_metabolic * (len(brain.neurons) / 10.0)
    ENERGY_FROM_PROFIT = 0.1
    
    initial_mut = 0.90
    final_mut = 0.001
    
    for gen in range(generations):
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        gen_best_fitness = -np.inf
        gen_best_trainable_genomes = None
        gen_best_threshold = None
        
        for i in range(population_size):
            # Clonar brain
            test_brain = QuaternionBrain(n_neurons=0)
            test_brain.neurons = [QuaternionNeuron3D(n.params.copy(), frozen=n.frozen) for n in brain.neurons]
            test_brain.decision_threshold = best_threshold
            test_brain.frozen_gain = brain.frozen_gain
            
            # Elitismo en primera iteración
            if i == 0 and gen > 0:
                for idx, ti in enumerate(trainable_indices):
                    test_brain.neurons[ti].params = best_trainable_genomes[idx].copy()
            else:
                # Mutar solo trainables
                for idx, ti in enumerate(trainable_indices):
                    mutant = best_trainable_genomes[idx].copy()
                    mask = np.random.rand(*mutant.shape) < 0.1
                    mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
                    mutant[3] = np.abs(mutant[3])  # R_base
                    mutant[5:8] = np.abs(mutant[5:8])  # Sx, Sy, Sz
                    
                    # Normalizar cuaternión
                    quat = mutant[8:12]
                    mutant[8:12] = quat / (np.linalg.norm(quat) + 1e-8)
                    
                    # Clip worm_length
                    mutant[12] = np.clip(mutant[12], 3, 50)
                    
                    test_brain.neurons[ti].params = mutant
                
                # Mutar threshold
                test_brain.decision_threshold = best_threshold + np.random.normal(0, mut_rate * 0.5)
                test_brain.decision_threshold = np.clip(test_brain.decision_threshold, 0.1, 0.9)
            
            # Simular trading en el bloque
            balance = INITIAL_BALANCE
            position = 0
            entry_price = 0
            current_bet = 0
            trades = 0
            alive = True
            
            test_brain.energy = 100.0
            test_brain.worm_history = []
            
            for t in range(len(block_inputs)):
                test_brain.energy -= METABOLIC_COST_FIXED
                if test_brain.energy <= 0:
                    alive = False
                    break
                
                if balance <= 0:
                    alive = False
                    break
                
                signal = test_brain.predict(block_inputs[t], block_temp[t])
                current_price = block_prices[t]
                
                # Cierre
                if (signal > test_brain.decision_threshold and position == -1) or \
                   (signal < -test_brain.decision_threshold and position == 1):
                    pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                              else (entry_price - current_price) / entry_price * LEVERAGE
                    profit = current_bet * pnl_pct
                    cost = (current_bet * LEVERAGE) * FEE
                    balance += current_bet + (profit - cost)
                    
                    test_brain.feedback_all(profit)
                    
                    if profit > 0:
                        energy_boost = profit * ENERGY_FROM_PROFIT
                        test_brain.energy = min(test_brain.energy + energy_boost, test_brain.MAX_ENERGY)
                    
                    position = 0
                    current_bet = 0
                
                # Apertura
                if position == 0:
                    if signal > test_brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]:
                        position = 1
                        entry_price = current_price * (1 + SLIPPAGE)
                        current_bet = balance * BET_PERCENTAGE
                        balance -= current_bet
                        balance -= (current_bet * LEVERAGE) * FEE
                        trades += 1
                    elif signal < -test_brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]:
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
                        # FIX: No restar balance (ya fue descontado al abrir)
                        position = 0
                        current_bet = 0
            
            # Cierre final
            if alive and position != 0:
                final_price = block_prices[-1]
                pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                          else (entry_price - final_price) / entry_price * LEVERAGE
                profit = current_bet * pnl_pct
                cost = (current_bet * LEVERAGE) * FEE
                balance += current_bet + (profit - cost)
            
            # Fitness
            if not alive:
                fitness = 0
            elif trades == 0:
                fitness = 10
            elif trades < 5:
                fitness = balance * 0.5
            else:
                fitness = 2000 + balance
            
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_trainable_genomes = [test_brain.neurons[ti].params.copy() for ti in trainable_indices]
                gen_best_threshold = test_brain.decision_threshold
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_trainable_genomes = [g.copy() for g in gen_best_trainable_genomes]
            best_threshold = gen_best_threshold
    
    # Actualizar brain con los mejores trainables
    for idx, ti in enumerate(trainable_indices):
        brain.neurons[ti].params = best_trainable_genomes[idx].copy()
    brain.decision_threshold = best_threshold
    
    return brain, best_threshold

def run_school_training(total_candles=10000, block_size=2000, add_per_block=10,
                        generations_per_block=10, population_size=50):
    """
    Entrenamiento incremental por bloques (escuelita).
    
    - Descarga total_candles velas UNA VEZ
    - Divide en bloques de block_size
    - Por cada bloque:
      1. Congela neuronas existentes
      2. Agrega add_per_block neuronas nuevas (KMeans)
      3. Entrena solo las nuevas con frozen_gain=0.3
      4. Restaura frozen_gain=1.0
    """
    
    print(f"\n🏫 --- ESCUELITA: ENTRENAMIENTO INCREMENTAL ---")
    print(f"   Total velas: {total_candles}")
    print(f"   Bloques: {total_candles // block_size} × {block_size} velas")
    print(f"   Neuronas por bloque: {add_per_block}")
    print(f"   Generaciones por bloque: {generations_per_block}")
    print(f"   Población: {population_size}\n")
    
    prices, vols, volat = fetch_extended_data("PEPEUSDT", "1m", total_candles)
    inputs, prices_aligned, rolling_vol = prepare_inputs_3d(prices, vols, volat)
    
    if inputs is None:
        return None, None, None
    
    n_blocks = total_candles // block_size
    brain = QuaternionBrain(n_neurons=0)
    best_threshold = np.random.uniform(0.1, 0.3)
    
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size
        
        block_inputs = inputs[start:end]
        block_prices = prices_aligned[start:end]
        block_temp = rolling_vol[start:end]
        avg_vol = np.mean(block_temp)
        
        print(f"\n📚 BLOQUE {b+1}/{n_blocks} (velas {start}-{end})")
        
        # 1) Congelar neuronas existentes
        if len(brain.neurons) > 0:
            brain.freeze_all()
            print(f"   ❄️  Congeladas: {sum(1 for n in brain.neurons if n.frozen)} neuronas")
        
        # 2) Agregar nuevas neuronas (KMeans)
        new_neurons = spawn_neurons_kmeans(block_inputs, k=add_per_block, avg_volatility=avg_vol)
        brain.add_neurons(new_neurons)
        print(f"   ➕ Agregadas: {len(new_neurons)} neuronas nuevas (KMeans)")
        print(f"   🧠 Total neuronas: {len(brain.neurons)}")
        
        # 3) Durante entrenamiento, reducir aporte de congeladas
        brain.frozen_gain = 0.3
        
        # 4) Evolucionar solo trainables
        brain, best_threshold = evolve_trainables_on_block(
            brain, best_threshold,
            block_inputs, block_prices, block_temp,
            generations=generations_per_block,
            population_size=population_size
        )
        
        # 5) Restaurar gain para siguiente bloque
        brain.frozen_gain = 1.0
        
        print(f"   ✅ Bloque completado | Umbral: {best_threshold:.3f}")
    
    # Congelar todo para el test final
    brain.freeze_all()
    brain.frozen_gain = 1.0
    
    print(f"\n🎓 ESCUELITA COMPLETADA:")
    print(f"   Neuronas totales: {len(brain.neurons)}")
    print(f"   Umbral final: {best_threshold:.3f}")
    
    return brain, best_threshold, (inputs, prices_aligned, rolling_vol)

# ==============================================================================
# 🧪 TEST
# ==============================================================================

def run_test_simulation(brain, threshold, inputs, prices, rolling_volatility):
    """Test con brain completo"""
    brain.decision_threshold = threshold
    brain.frozen_gain = 1.0  # Todas las neuronas cuentan igual en test
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.002
    FEE = 0.0004
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    trades = 0
    actions = []
    equity_curve = []
    
    brain.worm_history = []
    
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
                trades += 1
                act = 1
            elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]:
                position = -1
                entry_price = current_price * (1 - SLIPPAGE)
                current_bet = balance * BET_PERCENTAGE
                balance -= current_bet
                balance -= (current_bet * LEVERAGE) * FEE
                trades += 1
                act = -1
        
        # Liquidación
        if position != 0:
            unrealized = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                         else (entry_price - current_price) / entry_price * LEVERAGE
            if unrealized <= -0.9:
                position = 0
                current_bet = 0
        
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
    
    return balance, actions, equity_curve, trades

# ==============================================================================
# 🚀 MAIN - ESCUELITA + TESTS
# ==============================================================================

if __name__ == "__main__":
    # ESCUELITA: 10k velas, 5 bloques de 2k, 10 neuronas por bloque
    brain, threshold, (inputs, prices, rolling_vol) = run_school_training(
        total_candles=10000,
        block_size=2000,
        add_per_block=10,
        generations_per_block=10,
        population_size=50
    )
    
    if brain is not None:
        # Guardar cerebro completo
        genome = brain.extract_genome()
        frozen_flags = np.array([n.frozen for n in brain.neurons])
        
        np.save("best_genome_school_FINAL.npy", genome)
        np.save("best_threshold_school_FINAL.npy", threshold)
        np.save("best_frozen_flags_school_FINAL.npy", frozen_flags)
        
        print("\n💾 Cerebro guardado:")
        print(f"   - best_genome_school_FINAL.npy ({len(genome)} neuronas)")
        print(f"   - best_threshold_school_FINAL.npy")
        print(f"   - best_frozen_flags_school_FINAL.npy")
        
        # TESTS CIEGOS
        print("\n" + "="*60)
        print("🎲 --- FASE 2: TEST CIEGO (10 PERIODOS ALEATORIOS) ---")
        print("="*60)
        
        start_2023_ms = int(time.mktime(time.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
        now_ms = int(time.time() * 1000)
        
        test_results = []
        
        for test_num in range(1, 11):
            print(f"\n🧪 TEST {test_num}/10:")
            
            random_test_time = random.randint(start_2023_ms, now_ms)
            
            test_prices, test_vols, test_volatilities = fetch_extended_data(
                "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
            )
            test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs_3d(test_prices, test_vols, test_volatilities)
            
            if test_inputs is not None:
                final_bal, acts, equity, test_trades = run_test_simulation(
                    brain, threshold, test_inputs, test_prices_aligned, test_rolling_vol
                )
                
                profit_pct = ((final_bal - 10000) / 10000) * 100
                test_results.append({
                    'balance': final_bal,
                    'profit_pct': profit_pct,
                    'trades': test_trades
                })
                
                print(f"   Balance Final: ${final_bal:,.2f} | P/L: {profit_pct:+.2f}% | Trades: {test_trades}")
        
        # Resumen final
        if test_results:
            avg_balance = np.mean([r['balance'] for r in test_results])
            avg_profit = np.mean([r['profit_pct'] for r in test_results])
            avg_trades = np.mean([r['trades'] for r in test_results])
            
            wins = sum(1 for r in test_results if r['profit_pct'] > 0)
            losses = len(test_results) - wins
            
            print("\n" + "="*60)
            print("📊 RESUMEN DE 10 TESTS:")
            print("="*60)
            print(f"   Balance Promedio:  ${avg_balance:,.2f}")
            print(f"   P/L Promedio:      {avg_profit:+.2f}%")
            print(f"   Trades Promedio:   {avg_trades:.1f}")
            print(f"   Tests Ganadores:   {wins}/10")
            print(f"   Tests Perdedores:  {losses}/10")
