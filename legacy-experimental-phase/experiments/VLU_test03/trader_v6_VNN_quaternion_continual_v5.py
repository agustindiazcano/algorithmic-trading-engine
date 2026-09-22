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
TRADING_MODE = "LONG"

# ==============================================================================
# HIBRIDO: PRIORS + DECAY PARA HERENCIA DE ESTADO
# ==============================================================================

ALPHA_PRIOR = 10.0
BETA_PRIOR  = 1.0
INHERIT_RATE = 0.30  # 30% de lo aprendido se hereda, 70% se resetea al prior
AB_NOISE_STD = 0.02  
AB_MIN = 0.1
AB_MAX_SUM = 50.0

def decay_inherit_ab(parent_ab, inherit_rate=INHERIT_RATE, alpha0=ALPHA_PRIOR, beta0=BETA_PRIOR):
    """
    Decae el estado aprendido (alpha, beta) hacia los priors.
    parent_ab: (n, 2) matriz de [alpha, beta]
    """
    if parent_ab is None:
        return None
        
    alpha_p = parent_ab[:, 0]
    beta_p  = parent_ab[:, 1]

    # Fórmula de mezcla: Child = Prior + Rate * (Parent - Prior)
    alpha_c = alpha0 + inherit_rate * (alpha_p - alpha0)
    beta_c  = beta0  + inherit_rate * (beta_p  - beta0)

    # Ruido opcional
    if AB_NOISE_STD > 0:
        alpha_c += np.random.normal(0, AB_NOISE_STD, size=alpha_c.shape)
        beta_c  += np.random.normal(0, AB_NOISE_STD, size=beta_c.shape)

    # Clamp
    alpha_c = np.maximum(alpha_c, AB_MIN)
    beta_c  = np.maximum(beta_c,  AB_MIN)

    # Límite superior suave
    s = alpha_c + beta_c
    over = s > AB_MAX_SUM
    if np.any(over):
        scale = AB_MAX_SUM / (s[over] + 1e-8)
        alpha_c[over] *= scale
        beta_c[over]  *= scale

    return np.column_stack([alpha_c, beta_c])

# ==============================================================================
# 🧬 NEURONA VOLUMÉTRICA 3D (HIBRIDA)
# ==============================================================================

class QuaternionNeuron3D:
    def __init__(self, genome_slice, alpha=ALPHA_PRIOR, beta=BETA_PRIOR, frozen=False):
        self.params = genome_slice
        self.center = self.params[0:3]
        self.frozen = frozen
        
        # Estado Bayesiano (puede venir heredado)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.confidence = 0.5
    
    def get_ab(self):
        return self.alpha, self.beta
        
    def activate_3d_scan(self, market_trajectory, current_temp):
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
        if self.frozen: return
            
        learning_rate = 0.1
        if pnl > 0:
            self.alpha += learning_rate
        else:
            self.beta += learning_rate
        
        # Decay en vida si crece mucho (para mantener plasticidad)
        if self.alpha + self.beta > 50:
            self.alpha *= 0.95
            self.beta *= 0.95

# ==============================================================================
# 🧠 CEREBRO
# ==============================================================================

class QuaternionBrain:
    def __init__(self, n_neurons=10):
        self.neurons = []
        self.worm_history = []
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        self.decision_threshold = 0.2
        
        if n_neurons > 0:
            self.reset_genome(n_neurons)
    
    def reset_genome(self, n_neurons):
        self.neurons = []
        for _ in range(n_neurons):
            genome = make_random_genome()
            self.neurons.append(QuaternionNeuron3D(genome, alpha=ALPHA_PRIOR, beta=BETA_PRIOR))

    def load_params_ab(self, params_array, ab_array=None):
        self.neurons = []
        for i, genome_slice in enumerate(params_array):
            a, b = (ALPHA_PRIOR, BETA_PRIOR)
            if ab_array is not None:
                a, b = ab_array[i]
            # Por defecto al cargar, asumimos frozen=True si es un experto ya entrenado,
            # pero aquí se usa para cargar candidatos en evolución, así que frozen=False
            self.neurons.append(QuaternionNeuron3D(genome_slice, alpha=a, beta=b, frozen=False))
            
    def load_expert_state(self, params_array, ab_array, threshold):
        """Carga un estado completo de experto FINAL (y lo congela)"""
        self.neurons = []
        for i, genome_slice in enumerate(params_array):
            a, b = ab_array[i]
            self.neurons.append(QuaternionNeuron3D(genome_slice, alpha=a, beta=b, frozen=True))
        self.decision_threshold = threshold
    
    def extract_params(self):
        return np.array([n.params for n in self.neurons])
        
    def extract_ab(self):
        return np.array([[n.alpha, n.beta] for n in self.neurons])

    def predict(self, market_point, current_volatility):
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
    mu = np.mean(block_inputs, axis=0)
    sd = np.std(block_inputs, axis=0) + 1e-8
    vmu = float(np.mean(block_vol))
    vsd = float(np.std(block_vol) + 1e-8)
    return {"mu": mu, "sd": sd, "vmu": vmu, "vsd": vsd}

def compute_context(inputs, rolling_vol, t, ctx_window=200):
    a = max(0, t-ctx_window+1)
    ctx_in = inputs[a:t+1]
    ctx_vol = rolling_vol[a:t+1]
    return compute_fingerprint(ctx_in, ctx_vol)

class ContextCache:
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
# 🧭 ROUTER
# ==============================================================================

def expert_distance(fp, ctx, w_mu=1.0, w_sd=0.5, w_vol=0.5):
    dmu = np.linalg.norm((fp["mu"] - ctx["mu"]))
    dsd = np.linalg.norm((fp["sd"] - ctx["sd"]))
    dvol = abs(fp["vmu"] - ctx["vmu"]) + 0.5*abs(fp["vsd"] - ctx["vsd"])
    return w_mu*dmu + w_sd*dsd + w_vol*dvol

def pick_top_k_experts(experts, ctx, k=3):
    dists = np.array([expert_distance(e.fingerprint, ctx) for e in experts], dtype=float)
    idx = np.argsort(dists)[:min(k, len(experts))]
    eps = 1e-8
    w = 1.0 / (dists[idx] + eps)
    w = w / (np.sum(w) + eps)
    return idx, w

def moe_signal_winner(experts, inputs, rolling_vol, t, ctx, top_k=3):
    if not experts: return 0.0
    
    idx, _ = pick_top_k_experts(experts, ctx, k=top_k)
    best_s = 0.0
    
    for j in idx:
        expert = experts[j]
        s = expert.brain.predict(inputs[t], rolling_vol[t])
        if abs(s) >= expert.threshold:
            if abs(s) > abs(best_s):
                best_s = s
    return float(np.tanh(best_s))

# ==============================================================================
# 🔧 HELPERS
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
    except:
        idx = np.random.choice(len(block_inputs), size=min(k, len(block_inputs)), replace=False)
        return block_inputs[idx]

def spawn_neurons_kmeans(block_inputs, k=10, noise=0.05):
    centers = kmeans_centers(block_inputs, k=k)
    new_genomes = []
    for c in centers:
        c2 = c + np.random.normal(0, noise, size=3)
        g = make_random_genome(center=c2)
        new_genomes.append(g)
    return new_genomes

def mutate_params(genome, mut_rate):
    """Mutación lenta de parámetros geométricos"""
    mask = np.random.rand(*genome.shape) < 0.1
    noise = np.random.normal(0, mut_rate, size=genome[mask].shape)
    genome[mask] += noise
    
    # Constraints
    genome[:, 3] = np.abs(genome[:, 3])   # R_base
    genome[:, 5:8] = np.abs(genome[:, 5:8]) # Stretch
    
    # Normalize quats
    for k in range(len(genome)):
        q = genome[k, 8:12]
        genome[k, 8:12] = q / (np.linalg.norm(q) + 1e-8)
        genome[k, 12] = np.clip(genome[k, 12], 3, 50)
    
    return genome

# ==============================================================================
# ⏱️ EVALUACIÓN MULTI-SLICE (ANTI-OVERFIT)
# ==============================================================================

def simulate_trading_segment(brain, threshold, inputs, prices, temps):
    """Simula trading en un segmento, manteniendo estado del brain"""
    INITIAL_BALANCE = 10000.0 # Virtual balance reset per segment for simplistic calc
    # OR we can pass initial balance. Let's return PnL per segment.
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    trades = 0
    
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.002
    FEE = 0.0004
    ENERGY_FROM_PROFIT = 0.1
    METABOLIC_COST = 0.01

    for t in range(len(inputs)):
        brain.energy -= METABOLIC_COST
        if brain.energy <= 0 or balance <= 0:
            break
        
        signal = brain.predict(inputs[t], temps[t])
        current_price = prices[t]
        
        # Cierre
        if (signal > threshold and position == -1) or \
           (signal < -threshold and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                      else (entry_price - current_price) / entry_price * LEVERAGE
            profit = current_bet * pnl_pct
            cost = (current_bet * LEVERAGE) * FEE
            balance += current_bet + (profit - cost)
            
            if profit > 0:
                brain.energy = min(brain.energy + profit * ENERGY_FROM_PROFIT, brain.MAX_ENERGY)
            
            # 🧠 BALDWIN: Aprendizaje en vida
            brain.feedback_all(profit)
            
            position = 0
            current_bet = 0

        # Apertura
        if position == 0:
            if signal > threshold:
                position = 1
                entry_price = current_price * (1 + SLIPPAGE)
                current_bet = balance * BET_PERCENTAGE
                balance -= current_bet
                balance -= (current_bet * LEVERAGE) * FEE
                trades += 1
            elif signal < -threshold:
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
                loss = current_bet
                brain.feedback_all(-loss) # Castigo Baldwin
                position = 0
                current_bet = 0
    
    # Cierre final de segmento
    if position != 0:
        final_price = prices[-1]
        pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                  else (entry_price - final_price) / entry_price * LEVERAGE
        profit = current_bet * pnl_pct
        cost = (current_bet * LEVERAGE) * FEE
        # Final feedback
        brain.feedback_all(profit) 
        balance += current_bet + (profit - cost)
        
    return balance, trades

def eval_candidate_multislice(brain, threshold, inputs, prices, temps, n_slices=3):
    brain.worm_history = []
    brain.energy = 100.0
    
    L = len(inputs)
    cuts = np.linspace(0, L, n_slices+1, dtype=int)
    
    balances = []
    total_trades = 0
    
    for si in range(n_slices):
        a, b = cuts[si], cuts[si+1]
        
        # El brain MANTIENE su estado alpha/beta entre slices (aprendizaje continuo)
        bal, tr = simulate_trading_segment(brain, threshold, inputs[a:b], prices[a:b], temps[a:b])
        balances.append(bal)
        total_trades += tr
        
        if bal <= 0: # Muerte prematura
            break
            
    avg_bal = np.mean(balances) if balances else 0.0
    
    # Penalizar si no opera
    if total_trades == 0:
        fitness = 10.0
    elif total_trades < 3:
        fitness = avg_bal * 0.5
    else:
        # Fitness robusto: promedio - penalización por varianza entre slices (opcional)
        # Por ahora simple promedio
        fitness = avg_bal
        
    return fitness, total_trades

# ==============================================================================
# EVOLUCIÓN HÍBRIDA POR BLOQUE
# ==============================================================================

def evolve_expert_hybrid(brain, block_inputs, block_prices, block_temp, generations=10, population_size=50):
    
    best_fitness = -np.inf
    
    # Estado inicial "mejor" (genes + estado neutro o previo)
    best_params = brain.extract_params()
    best_ab_learned = brain.extract_ab() # Inicialmente priors
    best_threshold = brain.decision_threshold
    
    initial_mut = 0.5 # Menor mutación estructural
    final_mut = 0.01
    
    for gen in range(generations):
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        gen_best_fitness = -np.inf
        gen_best_params = None
        gen_best_ab = None
        gen_best_thr = None
        
        for i in range(population_size):
            # 1. Crear candidato (Hijo)
            test_brain = QuaternionBrain(n_neurons=0)
            
            # Herencia híbrida
            if i == 0: # Elitismo puro (re-evaluar el mejor para asegurar estabilidad)
                child_params = best_params.copy()
                child_ab = decay_inherit_ab(best_ab_learned, inherit_rate=1.0) # Mantener todo
                curr_thr = best_threshold
            else:
                # Mutación genética LENTA
                child_params = mutate_params(best_params.copy(), mut_rate)
                # Herencia de estado DECAY
                child_ab = decay_inherit_ab(best_ab_learned)
                curr_thr = np.clip(best_threshold + np.random.normal(0, mut_rate*0.5), 0.1, 0.9)
            
            test_brain.load_params_ab(child_params, child_ab)
            test_brain.decision_threshold = curr_thr
            
            # 2. Evaluación Multi-Slice
            # El brain aprenderá en vida durante esta eval
            fitness, trades = eval_candidate_multislice(test_brain, curr_thr, block_inputs, block_prices, block_temp, n_slices=3)
            
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_params = test_brain.extract_params() # Params (inmutables)
                gen_best_ab = test_brain.extract_ab()         # Estado APRENDIDO al final de la vida
                gen_best_thr = curr_thr

        # Selección generacional
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_params = gen_best_params
            best_ab_learned = gen_best_ab
            best_threshold = gen_best_thr
            print(f"  Gen {gen+1}: New Best Fitness={best_fitness:.0f} (AB sum={np.sum(best_ab_learned):.1f})")
            
    return best_params, best_ab_learned, best_threshold

def run_school_moe_training_v5(total_candles=10000, block_size=2000,
                               neurons_per_expert=10, generations=10, population_size=50):
    
    print(f"\n🏫 --- ESCUELITA v5 HIBRIDA: GENES + ESTADO ---")
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
        
        # Init con Kmeans
        init_genomes = spawn_neurons_kmeans(block_inputs, k=neurons_per_expert)
        
        # Brain temporal para arrancar
        brain = QuaternionBrain(n_neurons=0)
        brain.load_params_ab(np.array(init_genomes), None) # AB defaults
        
        # Evolución Híbrida
        best_params, best_ab, best_thr = evolve_expert_hybrid(
            brain, block_inputs, block_prices, block_vol,
            generations=generations, population_size=population_size
        )
        
        # Crear experto final congelado
        final_brain = QuaternionBrain(n_neurons=0)
        final_brain.load_expert_state(best_params, best_ab, best_thr)
        
        fp = compute_fingerprint(block_inputs, block_vol)
        experts.append(Expert(brain=final_brain, threshold=best_thr, fingerprint=fp, block_index=b))
        
        print(f"   ✅ Experto {b+1} finalizado. AB avg={np.mean(best_ab):.2f}")
        
    return experts, (inputs, prices_aligned, rolling_vol)

# ==============================================================================
# DATA LOADERS (COPY FROM V3)
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
            r = requests.get(url, timeout=10).json()
            if not isinstance(r, list): break
            if not r: break
            
            closes = [float(k[4]) for k in r]
            volumes = [float(k[5]) for k in r]
            highs = [float(k[2]) for k in r]
            lows = [float(k[3]) for k in r]
            
            all_closes = closes + all_closes
            all_volumes = volumes + all_volumes
            all_highs = highs + all_highs
            all_lows = lows + all_lows
            current_end_time = int(r[0][0]) - 1
        except: break
            
    if len(all_closes) < 100: return None, None, None
    
    prices = np.array(all_closes)
    volumes = np.array(all_volumes)
    volat = (np.array(all_highs) - np.array(all_lows)) / prices
    return prices, volumes, volat

def prepare_inputs_3d(prices, volumes, volatilities):
    if prices is None: return None, None, None
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    acceleration = np.diff(returns)
    acceleration = np.concatenate([[0], acceleration])
    
    # Simple Norm
    def normalize(x, w=100):
        norm = np.zeros_like(x)
        for i in range(len(x)):
            s = max(0, i-w)
            chunk = x[s:i+1]
            mu = np.mean(chunk)
            std = np.std(chunk) + 1e-8
            norm[i] = (x[i] - mu)/std
        return norm

    inputs = np.column_stack([normalize(returns), normalize(volumes), normalize(acceleration)])
    
    rolling_vol = np.array([np.std(returns[max(0, i-20):i+1]) for i in range(len(returns))])
    return inputs, prices, rolling_vol

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    experts, _ = run_school_moe_training_v5(
        total_candles=10000,
        block_size=2000,
        neurons_per_expert=10,
        generations=10,
        population_size=20 # Test rápido
    )
    
    if experts:
        print("\n" + "="*60)
        print("🎲 --- FASE TEST CIEGO V5 ---")
        print("="*60)
        # (Aquí iría el loop de tests ciegos idéntico a v3, omitido por brevedad pero funcional)
        start_2023_ms = int(time.mktime(time.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
        now_ms = int(time.time() * 1000)
        
        test_results = []
        
        for test_num in range(1, 11):
            print(f"\n🧪 TEST {test_num}/10:")
            random_test_time = random.randint(start_2023_ms, now_ms)
            
            # Fetch data for test
            test_prices, test_vols, test_volatilities = fetch_extended_data(
                "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
            )
            
            test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs_3d(test_prices, test_vols, test_volatilities)
            
            if test_inputs is not None:
                # Reutilizamos función de test MoE (copiada de v3 si estuviera, pero necesitamos definirla aquí)
                # Como no copie run_test_simulation_moe en v5, la defino ahora antes del main o la agrego.
                # Espera, olvidé copiar run_test_simulation_moe en v5! 
                pass

def run_test_simulation_moe(experts, inputs, prices, rolling_volatility):
    # Reset histories
    for ex in experts:
        ex.brain.worm_history = []
        ex.brain.energy = 100.0
        
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
        if (signal > 0.3 and position == -1) or (signal < -0.3 and position == 1):
             pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                       else (entry_price - current_price) / entry_price * LEVERAGE
             profit = current_bet * pnl_pct
             cost = (current_bet * LEVERAGE) * FEE
             balance += current_bet + (profit - cost)
             position = 0
             current_bet = 0
        
        # Apertura
        if position == 0:
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
        
    if position != 0:
        final_price = prices[-1]
        pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                  else (entry_price - final_price) / entry_price * LEVERAGE
        profit = current_bet * pnl_pct
        cost = (current_bet * LEVERAGE) * FEE
        balance += current_bet + (profit - cost)
        
    return balance, actions, equity_curve, trades

if __name__ == "__main__":
    experts, _ = run_school_moe_training_v5(
        total_candles=10000,
        block_size=2000,
        neurons_per_expert=10,
        generations=10,
        population_size=20
    )
    
    if experts:
        print("\n" + "="*60)
        print("🎲 --- FASE TEST CIEGO V5 (10 PERIODOS) ---")
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

        if test_results:
            avg_profit = np.mean([r['profit_pct'] for r in test_results])
            print(f"\n📊 P/L Promedio: {avg_profit:+.2f}%")
