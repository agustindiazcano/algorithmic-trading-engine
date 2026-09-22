import numpy as np
import matplotlib.pyplot as plt
import time
import requests 
import random
import sys
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

# Forzar UTF-8 en consola para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURACIÓN DE TRADING
TRADING_MODE = "LONG"  # Opciones: "LONG", "SHORT", "BOTH"

# ==============================================================================
# 🧬 NEURONA VOLUMÉTRICA 3D CON CUATERNIONES (S³)
# ==============================================================================

class QuaternionNeuron3D:
    """
    Neurona volumétrica avanzada con:
    - Rotaciones en S³ (cuaterniones)
    - Longitud de gusano variable
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
        """Escaneo volumétrico con cuaterniones"""
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
# 🧠 CEREBRO
# ==============================================================================

class QuaternionBrain:
    def __init__(self, n_neurons=10):
        self.neurons = []
        self.worm_history = []
        
        # Sistema de energía
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        
        # Umbral de decisión
        self.decision_threshold = np.random.uniform(0.1, 0.3)
        
        if n_neurons > 0:
            self.reset_genome(n_neurons)
    
    def reset_genome(self, n_neurons):
        self.neurons = []
        for _ in range(n_neurons):
            genome = make_random_genome()
            self.neurons.append(QuaternionNeuron3D(genome, frozen=False))

    def load_genome(self, genome_array):
        self.neurons = []
        for genome_slice in genome_array:
            self.neurons.append(QuaternionNeuron3D(genome_slice, frozen=True))
    
    def extract_genome(self):
        return np.array([neuron.params for neuron in self.neurons])

    def predict(self, market_point, current_volatility):
        """Predicción simple"""
        self.worm_history.append(market_point)
        if len(self.worm_history) > 100:
            self.worm_history.pop(0)
        
        max_activation = 0.0
        winner_activation = 0.0
        
        for neuron in self.neurons:
            activation = neuron.activate_3d_scan(self.worm_history, current_volatility)
            if abs(activation) > abs(max_activation):
                max_activation = activation
                winner_activation = activation
        
        return np.tanh(winner_activation)
    
    def feedback_all(self, pnl):
        for neuron in self.neurons:
            neuron.learn_from_result(pnl)

# ==============================================================================
# 📦 EXPERTO Y FINGERPRINTING
# ==============================================================================

@dataclass
class Expert:
    brain: QuaternionBrain
    threshold: float
    fingerprint: dict
    block_index: int

def compute_fingerprint(block_inputs, block_vol):
    """Calcula estadísticas del régimen de mercado"""
    mu = np.mean(block_inputs, axis=0)
    sd = np.std(block_inputs, axis=0) + 1e-8
    vmu = float(np.mean(block_vol))
    vsd = float(np.std(block_vol) + 1e-8)
    return {"mu": mu, "sd": sd, "vmu": vmu, "vsd": vsd}

def compute_context(inputs, rolling_vol, t, ctx_window=200):
    """Calcula contexto actual del mercado"""
    a = max(0, t-ctx_window+1)
    ctx_in = inputs[a:t+1]
    ctx_vol = rolling_vol[a:t+1]
    return compute_fingerprint(ctx_in, ctx_vol)

class ContextCache:
    """Cache optimizado para no recalcular contexto en cada tick"""
    def __init__(self, every=5, window=200):
        self.every = every
        self.window = window
        self.last_t = -1
        self.last_ctx = None

    def get(self, inputs, rolling_vol, t):
        if self.last_ctx is None or (t % self.every == 0 and t != self.last_t):
            self.last_ctx = compute_context(inputs, rolling_vol, t, self.window)
            self.last_t = t
        return self.last_ctx

# ==============================================================================
# 🧭 ROUTER: MIXTURE OF EXPERTS
# ==============================================================================

def expert_distance(fp, ctx, w_mu=1.0, w_sd=0.5, w_vol=0.5):
    """Distancia ponderada entre un experto y el contexto actual"""
    dmu = np.linalg.norm((fp["mu"] - ctx["mu"]))
    dsd = np.linalg.norm((fp["sd"] - ctx["sd"]))
    dvol = abs(fp["vmu"] - ctx["vmu"]) + 0.5*abs(fp["vsd"] - ctx["vsd"])
    return w_mu*dmu + w_sd*dsd + w_vol*dvol

def pick_top_k_experts(experts, ctx, k=3):
    """Selecciona los mejores k expertos para el contexto actual"""
    dists = np.array([expert_distance(e.fingerprint, ctx) for e in experts], dtype=float)
    idx = np.argsort(dists)[:min(k, len(experts))]
    
    # Pesos: inversa de la distancia (soft)
    eps = 1e-8
    w = 1.0 / (dists[idx] + eps)
    w = w / (np.sum(w) + eps)
    return idx, w

def moe_signal_winner(experts, inputs, rolling_vol, t, ctx, top_k=3):
    """Señal MoE tipo 'Winner Takes All' entre los top-k"""
    idx, _ = pick_top_k_experts(experts, ctx, k=top_k)
    best_s = 0.0
    
    for j in idx:
        expert = experts[j]
        s = expert.brain.predict(inputs[t], rolling_vol[t])
        
        # Solo considerar señales que superan el umbral del experto
        if abs(s) >= expert.threshold:
            if abs(s) > abs(best_s):
                best_s = s
    
    return float(np.tanh(best_s))

# ==============================================================================
# 🔧 HELPERS PARA GENOMAS Y KMEANS
# ==============================================================================

def random_unit_quat():
    q = np.random.randn(4)
    return q / (np.linalg.norm(q) + 1e-8)

def make_random_genome(center=None):
    q = random_unit_quat()
    
    if center is None:
        center = np.random.uniform(-3.0, 3.0, 3)
        
    return np.array([
        center[0], center[1], center[2],
        np.random.uniform(0.5, 1.5),      # R_base
        np.random.uniform(-1.0, 1.0),     # W
        np.random.uniform(0.5, 2.0),      # Sx
        np.random.uniform(0.5, 2.0),      # Sy
        np.random.uniform(0.5, 2.0),      # Sz
        q[0], q[1], q[2], q[3],
        np.random.uniform(3, 50)          # worm_length
    ], dtype=float)

def kmeans_centers(block_inputs, k=10, seed=42):
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        km.fit(block_inputs)
        return km.cluster_centers_
    except Exception:
        # Fallback
        idx = np.random.choice(len(block_inputs), size=min(k, len(block_inputs)), replace=False)
        return block_inputs[idx]

def spawn_neurons_kmeans(block_inputs, k=10, noise=0.05):
    """Crea genomas inicializados con KMeans"""
    centers = kmeans_centers(block_inputs, k=k)
    new_genomes = []
    for c in centers:
        c2 = c + np.random.normal(0, noise, size=3)
        g = make_random_genome(center=c2)
        new_genomes.append(g)
    return new_genomes

# ==============================================================================
# 🌍 MERCADO REAL
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=5000, custom_end_time=None):
    if custom_end_time is None:
        current_end_time = int(time.time() * 1000)
    else:
        current_end_time = int(custom_end_time)
    
    print(f"📡 Bajando {total_candles} velas de {symbol}...")
    
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
            
            if not isinstance(data, list):
                print(f"⚠️  Binance error: {data}")
                break
            
            if not data:
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
            print(f"❌ Error: {e}")
            break
            
    if len(all_closes) < 100:
        return None, None, None
        
    prices = np.array(all_closes)
    volumes = np.array(all_volumes)
    highs = np.array(all_highs)
    lows = np.array(all_lows)
    volatilities = (highs - lows) / prices
    
    return prices, volumes, volatilities

def prepare_inputs_3d(prices, volumes, volatilities):
    if prices is None: return None, None, None
    
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
    rolling_vol = np.maximum(rolling_vol, 0.001)
    
    return inputs, prices, rolling_vol

# ==============================================================================
# 🏫 ESCUELITA: ENTRENAMIENTO DE EXPERTOS (BLOQUES)
# ==============================================================================

def evolve_expert_on_block(brain, threshold, block_inputs, block_prices, block_temp, generations=10, population_size=50):
    """Evoluciona un solo experto en un bloque específico"""
    
    best_fitness = -np.inf
    best_genome = brain.extract_genome()
    best_threshold = threshold
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.002
    FEE = 0.0004
    
    METABOLIC_COST = 0.01  # Costo fijo bajo por experto
    ENERGY_FROM_PROFIT = 0.1
    
    initial_mut = 0.90
    final_mut = 0.001
    
    for gen in range(generations):
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        gen_best_fitness = -np.inf
        gen_best_genome = None
        gen_best_threshold = None
        
        for i in range(population_size):
            # Clonar experto
            test_brain = QuaternionBrain(n_neurons=0)
            test_brain.load_genome(best_genome)
            # Descongelar para entrenar
            for n in test_brain.neurons: n.frozen = False
                
            test_brain.decision_threshold = best_threshold
            
            # Mutación (salvo elitismo)
            if i > 0 or gen == 0:
                current_genome = test_brain.extract_genome()
                mask = np.random.rand(*current_genome.shape) < 0.1
                current_genome[mask] += np.random.normal(0, mut_rate, size=current_genome[mask].shape)
                
                # Constraints
                current_genome[:, 3] = np.abs(current_genome[:, 3])  # R_base
                current_genome[:, 5:8] = np.abs(current_genome[:, 5:8]) # Scale
                
                # Cuaterniones unitarios
                for k in range(len(current_genome)):
                    q = current_genome[k, 8:12]
                    current_genome[k, 8:12] = q / (np.linalg.norm(q) + 1e-8)
                    current_genome[k, 12] = np.clip(current_genome[k, 12], 3, 50)
                
                test_brain.load_genome(current_genome)
                # Volver a descongelar tras load
                for n in test_brain.neurons: n.frozen = False
                
                test_brain.decision_threshold += np.random.normal(0, mut_rate * 0.5)
                test_brain.decision_threshold = np.clip(test_brain.decision_threshold, 0.1, 0.9)

            # Simulación
            balance = INITIAL_BALANCE
            position = 0
            entry_price = 0
            current_bet = 0
            trades = 0
            alive = True
            
            test_brain.energy = 100.0
            test_brain.worm_history = []
            
            for t in range(len(block_inputs)):
                test_brain.energy -= METABOLIC_COST
                if test_brain.energy <= 0 or balance <= 0:
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
                    
                    if profit > 0:
                        test_brain.energy = min(test_brain.energy + profit * ENERGY_FROM_PROFIT, test_brain.MAX_ENERGY)
                    
                    # 🧠 BALDWIN: Aprendizaje en vida
                    test_brain.feedback_all(profit)
                    
                    position = 0
                    current_bet = 0

                # Apertura
                if position == 0:
                    if signal > test_brain.decision_threshold:
                        position = 1
                        entry_price = current_price * (1 + SLIPPAGE)
                        current_bet = balance * BET_PERCENTAGE
                        balance -= current_bet
                        balance -= (current_bet * LEVERAGE) * FEE
                        trades += 1
                    elif signal < -test_brain.decision_threshold:
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
                        # 🧠 BALDWIN: Castigo fuerte por liquidación
                        loss = current_bet  # Perdió todo el bet
                        test_brain.feedback_all(-loss)
                        
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
                
                # 🧠 BALDWIN: Aprendizaje de cierre final
                test_brain.feedback_all(profit)
            
            # Fitness
            if not alive: fitness = 0
            elif trades == 0: fitness = 10
            elif trades < 5: fitness = balance * 0.5
            else: fitness = 2000 + balance
            
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_genome = test_brain.extract_genome()
                gen_best_threshold = test_brain.decision_threshold
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = gen_best_genome
            best_threshold = gen_best_threshold
            
    return best_genome, best_threshold

def run_school_moe_training(total_candles=10000, block_size=2000,
                            neurons_per_expert=10, generations=10, population_size=50):
    
    print(f"\n🏫 --- ESCUELITA MoE: TRAINING EXPERTOS ---")
    prices, vols, volat = fetch_extended_data("PEPEUSDT", "1m", total_candles)
    inputs, prices_aligned, rolling_vol = prepare_inputs_3d(prices, vols, volat)
    
    if inputs is None: return None, None
    
    experts = []
    n_blocks = total_candles // block_size
    
    for b in range(n_blocks):
        s = b * block_size
        e = (b + 1) * block_size
        
        block_inputs = inputs[s:e]
        block_prices = prices_aligned[s:e]
        block_vol = rolling_vol[s:e]
        
        print(f"\n📚 EXPERTO {b+1}/{n_blocks} (velas {s}-{e})")
        
        # Crear nuevo experto
        brain = QuaternionBrain(n_neurons=0) # Init vacío
        
        # Inicializar neuronas con KMeans sobre el bloque actual
        init_genomes = spawn_neurons_kmeans(block_inputs, k=neurons_per_expert)
        brain.load_genome(np.array(init_genomes))
        
        # Entrenar experto SOLO en este bloque
        best_genome, best_thr = evolve_expert_on_block(
            brain, brain.decision_threshold, block_inputs, block_prices, block_vol,
            generations=generations, population_size=population_size
        )
        
        # Cargar mejor genoma y congelar
        brain.load_genome(best_genome) # Esto congela las neuronas por defecto en load_genome (frozen=True)
        brain.decision_threshold = best_thr
        
        # Calcular fingerprint
        fp = compute_fingerprint(block_inputs, block_vol)
        experts.append(Expert(brain=brain, threshold=best_thr, fingerprint=fp, block_index=b))
        
        print(f"   ✅ Experto entrenado | Umbral: {best_thr:.3f}")
        
    return experts, (inputs, prices_aligned, rolling_vol)

# ==============================================================================
# 🧪 TEST CON ROUTER (MoE)
# ==============================================================================

def run_test_simulation_moe(experts, inputs, prices, rolling_volatility):
    # Reset histories
    for ex in experts:
        ex.brain.worm_history = []
        
    cache = ContextCache(every=5, window=200)
    
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
    
    print("\n⏯️  Ejecutando simulación MoE...")
    
    for t in range(len(inputs)):
        if balance <= 0: break
        
        # Router
        ctx = cache.get(inputs, rolling_volatility, t)
        signal = moe_signal_winner(experts, inputs, rolling_volatility, t, ctx, top_k=3)
        
        current_price = prices[t]
        act = 0
        
        # Cierre
        if (signal > 0.3 and position == -1) or (signal < -0.3 and position == 1): # Hardcoded exit threshold for simplicity
             pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                       else (entry_price - current_price) / entry_price * LEVERAGE
             profit = current_bet * pnl_pct
             cost = (current_bet * LEVERAGE) * FEE
             balance += current_bet + (profit - cost)
             position = 0
             current_bet = 0
        
        # Apertura
        if position == 0:
            # Signal ya viene filtrada por threshold del experto en moe_signal_winner
            # Pero igual aplicamos un umbral global mínimo de seguridad
            if abs(signal) > 0.1: 
                if signal > 0:
                    position = 1
                    entry_price = current_price * (1 + SLIPPAGE)
                    current_bet = balance * BET_PERCENTAGE
                    balance -= current_bet
                    balance -= (current_bet * LEVERAGE) * FEE
                    trades += 1
                    act = 1
                else:
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
# 🚀 MAIN
# ==============================================================================

if __name__ == "__main__":
    experts, _ = run_school_moe_training(
        total_candles=10000,
        block_size=2000,
        neurons_per_expert=10,
        generations=10,      # 10 generaciones
        population_size=50   # 50 individuos
    )
    
    if experts:
        # TESTS CIEGOS
        print("\n" + "="*60)
        print("🎲 --- FASE 2: TEST CIEGO MoE (10 PERIODOS) ---")
        print("="*60)
        
        start_2023_ms = int(time.mktime(time.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
        now_ms = int(time.time() * 1000)
        
        test_results = []
        
        for test_num in range(1, 11):
            print(f"\n🧪 TEST {test_num}/10:")
            random_test_time = random.randint(start_2023_ms, now_ms)
            
            # Fetch data for test
            inputs = None
            test_prices, test_vols, test_volatilities = fetch_extended_data(
                "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
            )
            
            test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs_3d(test_prices, test_vols, test_volatilities)
            
            if test_inputs is not None:
                final_bal, acts, equity, test_trades = run_test_simulation_moe(
                    experts, test_inputs, test_prices_aligned, test_rolling_vol
                )
                
                profit_pct = ((final_bal - 10000) / 10000) * 100
                test_results.append({
                    'balance': final_bal,
                    'profit_pct': profit_pct,
                    'trades': test_trades
                })
                
                print(f"   Balance Final: ${final_bal:,.2f} | P/L: {profit_pct:+.2f}% | Trades: {test_trades}")

        # Resumen
        if test_results:
            avg_profit = np.mean([r['profit_pct'] for r in test_results])
            print(f"\n📊 P/L Promedio: {avg_profit:+.2f}%")
