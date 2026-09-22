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
# 🧬 NEURONA HÍBRIDA (Molecular + Bayesian + Markov)
# ==============================================================================

class HybridNeuron:
    """
    Neurona que combina:
    - MOLECULAR: Respiración termodinámica + Geometría elipsoidal + SE(2) search
    - BAYESIAN: Aprendizaje online de confianza (alpha/beta)
    - MARKOV: Estados discretos para filtro de ruido
    """
    def __init__(self, genome_slice):
        # GENOMA (8 parámetros heredables)
        # [t, p, vol, R_base, W, Sx, Sy, Theta]
        self.params = genome_slice
        
        # ESTADO MARKOVIANO (NO heredable)
        self.state = 0  # 0 = Dormant, 1 = Active
        self.transition_matrix = np.array([
            [0.95, 0.05],  # Dormant: difícil despertar (filtro ruido)
            [0.20, 0.80]   # Active: fácil mantenerse (momentum)
        ])
        
        # MEMORIA BAYESIANA (NO heredable)
        self.alpha = 1.0  # Aciertos
        self.beta = 1.0   # Fallos
        self.confidence = 0.5
    
    def activate_hybrid(self, market_window, current_volatility):
        """
        Activación híbrida completa.
        
        Returns:
            float: Activación ponderada por peso y confianza
        """
        # --- FASE 1: RESPIRACIÓN TERMODINÁMICA (Molecular) ---
        r_base = self.params[3]
        breathing_factor = 1.0 + (current_volatility * 5.0)
        r_effective = r_base * breathing_factor
        
        # --- FASE 2: GEOMETRÍA ELIPSOIDAL (Molecular) ---
        sx, sy = self.params[5], self.params[6]
        theta = self.params[7]
        c, s = np.cos(theta), np.sin(theta)
        
        R_matrix = np.array([[c, -s], [s, c]])
        S_matrix = np.array([[1/(sx+1e-6), 0], [0, 1/(sy+1e-6)]])
        Transformation = S_matrix @ R_matrix.T
        
        center = self.params[1:3]  # [precio, volumen]
        
        # --- FASE 3: SE(2) SEARCH (Molecular) ---
        max_activation = 0.0
        search_window = market_window[-5:] if len(market_window) >= 5 else market_window
        
        for point in search_window:
            diff = point[0:2] - center
            diff_transformed = Transformation @ diff
            dist = np.linalg.norm(diff_transformed)
            activation = np.maximum(0, 1 - dist / (r_effective + 1e-6))
            
            if activation > max_activation:
                max_activation = activation
        
        # --- FASE 4: FILTRO DE MARKOV (Bayesian) ---
        # Modular transición con señal
        p_wake = self.transition_matrix[0][1] + (max_activation * 0.3)
        p_sleep = self.transition_matrix[1][0] - (max_activation * 0.15)
        
        if self.state == 0:
            if np.random.rand() < p_wake:
                self.state = 1
        else:
            if np.random.rand() < p_sleep:
                self.state = 0
        
        # Si dormida, señal = 0
        if self.state == 0:
            return 0.0
        
        # --- FASE 5: PONDERACIÓN BAYESIANA (Bayesian) ---
        self.confidence = self.alpha / (self.alpha + self.beta)
        
        # Output final: Señal * Peso * Confianza
        weight = self.params[4]
        return max_activation * weight * self.confidence
    
    def feedback(self, profit_made):
        """Aprendizaje Bayesiano online"""
        learning_rate = 0.1
        
        if self.state == 1:  # Solo si participó
            if profit_made > 0:
                self.alpha += learning_rate
            else:
                self.beta += learning_rate
            
            # Normalización (olvido exponencial)
            if self.alpha + self.beta > 20:
                self.alpha *= 0.9
                self.beta *= 0.9

# ==============================================================================
# 🧠 CEREBRO HÍBRIDO
# ==============================================================================

class HybridBrain:
    def __init__(self, n_neurons=10):
        self.n_neurons = n_neurons
        self.neurons = []
        
        # Sistema de energía metabólica
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        
        # Gen de personalidad
        self.decision_threshold = np.random.uniform(0.3, 0.9)
        
        # Ventana de mercado
        self.market_window = []
        
        self.reset_genome()
    
    def reset_genome(self):
        """Crea población de neuronas híbridas"""
        self.neurons = []
        for _ in range(self.n_neurons):
            genome = np.array([
                np.random.uniform(-1.0, 1.0),     # t
                np.random.uniform(-1.0, 1.0),     # p
                np.random.uniform(-1.0, 1.0),     # vol
                np.random.uniform(0.5, 1.5),      # R_base
                np.random.uniform(-1.0, 1.0),     # W
                np.random.uniform(0.5, 2.0),      # Sx
                np.random.uniform(0.5, 2.0),      # Sy
                np.random.uniform(-np.pi/2, np.pi/2)  # Theta
            ])
            self.neurons.append(HybridNeuron(genome))
    
    def extract_genome(self):
        return np.array([neuron.params for neuron in self.neurons])
    
    def load_genome(self, genome_array):
        self.neurons = []
        for genome_slice in genome_array:
            self.neurons.append(HybridNeuron(genome_slice))
    
    def predict(self, market_point, current_volatility):
        """Predicción con todas las neuronas"""
        # Actualizar ventana
        self.market_window.append(market_point)
        if len(self.market_window) > 10:
            self.market_window.pop(0)
        
        # Activación total
        total_activation = 0
        for neuron in self.neurons:
            total_activation += neuron.activate_hybrid(self.market_window, current_volatility)
        
        return np.tanh(total_activation)
    
    def feedback_all(self, profit):
        """Propagar feedback a todas las neuronas"""
        for neuron in self.neurons:
            neuron.feedback(profit)

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

def prepare_inputs(prices, volumes, volatilities):
    if prices is None:
        return None, None, None
    
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    vol_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
    ret_norm = returns / (np.std(returns) + 1e-8)
    volatility_norm = volatilities / (np.std(volatilities) + 1e-8)
    
    inputs = np.column_stack([ret_norm, vol_norm, volatility_norm])
    
    # Volatilidad rolling para temperatura
    window = 20
    rolling_vol = np.array([np.std(returns[max(0, i-window):i+1]) for i in range(len(returns))])
    
    return inputs, prices, rolling_vol

# ==============================================================================
# ⚔️ ENTRENAMIENTO
# ==============================================================================

def run_evolution(generations=10, population_size=50):
    print(f"\n💀 --- FASE 1: ENTRENAMIENTO (DATOS RECIENTES) ---")
    
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 5000)
    inputs, prices_aligned, rolling_volatility = prepare_inputs(market_prices, market_vols, market_volatilities)
    
    if inputs is None: return None, None
    
    brain = HybridBrain(n_neurons=10)
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
            np.save(f"best_genome_hybrid_gen_{gen}.npy", best_genome)
            print(f"💾 [AUTO-SAVE] Gen {gen} guardada.")
        
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        print(f"\n--- GENERACIÓN {gen+1}/{generations} (Mutación: {mut_rate:.4f}) ---")
        
        gen_best_fitness = -np.inf
        gen_best_genome = None
        gen_best_threshold = None
        
        for i in range(population_size):
            brain = HybridBrain(n_neurons=10)
            
            # ELITISMO
            if i == 0 and gen > 0:
                brain.load_genome(best_genome)
                brain.decision_threshold = best_threshold
            else:
                mutant = np.copy(best_genome)
                mask = np.random.rand(*mutant.shape) < 0.1
                mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
                mutant[:, 3] = np.abs(mutant[:, 3])
                mutant[:, 5:7] = np.abs(mutant[:, 5:7])
                
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
                    
                    # FEEDBACK BAYESIANO
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
    
    np.save("best_genome_hybrid_FINAL.npy", best_genome)
    np.save("best_threshold_hybrid_FINAL.npy", best_threshold)
    return best_genome, best_threshold

# ==============================================================================
# 🧪 TEST
# ==============================================================================

def run_test_simulation(genome, threshold, inputs, prices, rolling_volatility):
    brain = HybridBrain(n_neurons=10)
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
            
            # Feedback en test también (online learning)
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
        test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs(test_prices, test_vols, test_volatilities)
        
        if test_inputs is not None:
            final_bal, acts, equity = run_test_simulation(winner_genome, winner_threshold, test_inputs, test_prices_aligned, test_rolling_vol)
            
            profit_pct = ((final_bal - 10000) / 10000) * 100
            print(f"\n📊 RESULTADO DEL TEST:")
            print(f"   Balance Inicial: $10,000")
            print(f"   Balance Final:   ${final_bal:,.2f}")
            print(f"   Profit/Loss:     {profit_pct:+.2f}%")
