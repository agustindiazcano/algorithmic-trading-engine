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
# 🧠 NEURONA BAYESIAN-MARKOV (Nueva Arquitectura)
# ==============================================================================

class BayesianMarkovNeuron:
    """
    Neurona híbrida que integra:
    - Geometría: Detección espacial (herencia del sistema anterior)
    - Markov: Estados discretos para filtrar ruido
    - Bayes: Aprendizaje online de confianza
    - Lagrange: Predicción de trayectorias
    """
    def __init__(self, genome_slice, A_dc=1.618):
        # 1. GENOMA GEOMÉTRICO (9 parámetros heredables)
        # [x, y, z, R, W, Sx, Sy, Sz, Theta]
        self.params = np.array(genome_slice)
        self.A_dc = A_dc
        
        # 2. ESTADO MARKOVIANO (NO heredable - se resetea cada lifetime)
        self.state = 0  # 0 = Dormant (filtro), 1 = Active (dispara)
        
        # Matriz de Transición: [[P(0->0), P(0->1)], [P(1->0), P(1->1)]]
        # Difícil despertar (filtro de ruido), fácil mantenerse despierta (momentum)
        self.transition_matrix = np.array([
            [0.95, 0.05],  # Dormant: 95% sigue dormida, 5% se activa
            [0.20, 0.80]   # Active: 20% se duerme, 80% sigue activa
        ])
        
        # 3. MEMORIA BAYESIANA (NO heredable - aprende durante lifetime)
        # Distribución Beta: P(éxito) ~ Beta(alpha, beta)
        self.alpha = 1.0  # Prior: 1 acierto
        self.beta = 1.0   # Prior: 1 fallo
        self.confidence = 0.5  # Probabilidad inicial de tener razón
        
    def lagrange_prediction(self, price_history):
        """
        Interpolación de Lagrange para predecir el próximo precio.
        Usa los últimos 3 puntos para extrapolar.
        """
        if len(price_history) < 3:
            return None
        
        # Tomar últimos 3 precios
        y0, y1, y2 = price_history[-3:]
        
        # Extrapolación cuadrática (Lagrange para t=0,1,2 prediciendo t=3)
        # Fórmula simplificada: P(3) = 3*y2 - 3*y1 + y0
        predicted = 3*y2 - 3*y1 + y0
        return predicted
    
    def activate(self, market_point, price_history):
        """
        Calcula la activación de la neurona combinando:
        1. Geometría (distancia al centro)
        2. Lagrange (predicción de trayectoria)
        3. Markov (filtro de estado)
        4. Bayes (confianza histórica)
        
        Returns: float (señal de activación)
        """
        # --- PASO 1: DETECCIÓN GEOMÉTRICA ---
        center = self.params[:3]
        radius = self.params[3] * self.A_dc
        weight = self.params[4]
        stretch = self.params[5:8]
        theta = self.params[8]
        
        # Rotación y distancia elipsoidal
        diff = market_point - center
        c, s = np.cos(theta), np.sin(theta)
        diff_rot = np.array([
            diff[0]*c - diff[1]*s,
            diff[0]*s + diff[1]*c,
            diff[2]
        ])
        
        dist = np.linalg.norm(diff_rot / (stretch + 1e-6))
        geometric_signal = max(0, 1 - dist / (radius + 1e-6))
        
        # --- PASO 2: PREDICCIÓN DE LAGRANGE ---
        trajectory_match = 1.0  # Default si no hay suficiente historia
        if len(price_history) >= 3:
            predicted_price = self.lagrange_prediction(price_history)
            actual_price = market_point[0]  # Asumimos que x es el precio
            
            # Error de predicción
            error = abs(predicted_price - actual_price)
            # Convertir error a señal (Gaussiana)
            trajectory_match = np.exp(-error * 10)  # Factor 10 para sensibilidad
        
        # Combinar geometría y trayectoria
        raw_signal = geometric_signal * 0.5 + trajectory_match * 0.5
        
        # --- PASO 3: FILTRO DE MARKOV ---
        # Probabilidad de transición modulada por la señal
        p_wake = self.transition_matrix[0][1] + (raw_signal * 0.3)
        p_sleep = self.transition_matrix[1][0] - (raw_signal * 0.15)
        
        # Transición estocástica
        if self.state == 0:  # Dormant
            if np.random.rand() < p_wake:
                self.state = 1
        else:  # Active
            if np.random.rand() < p_sleep:
                self.state = 0
        
        # Si está dormida, la señal se anula
        if self.state == 0:
            return 0.0
        
        # --- PASO 4: PONDERACIÓN BAYESIANA ---
        # Confianza = media de la distribución Beta
        self.confidence = self.alpha / (self.alpha + self.beta)
        
        # Output final: Señal * Peso genético * Confianza aprendida
        final_output = raw_signal * weight * self.confidence
        
        return final_output
    
    def feedback(self, profit_made):
        """
        Aprendizaje online (Bayesian update).
        Llamar después de cerrar un trade.
        """
        learning_rate = 0.1
        
        # Solo aprendemos si la neurona participó (estaba activa)
        if self.state == 1:
            if profit_made > 0:
                # Refuerzo positivo
                self.alpha += learning_rate
            else:
                # Refuerzo negativo
                self.beta += learning_rate
            
            # Normalización para evitar crecimiento infinito
            # (Olvido exponencial para adaptarse a nuevos regímenes)
            if self.alpha + self.beta > 20:
                self.alpha *= 0.9
                self.beta *= 0.9

# ==============================================================================
# 🧠 CEREBRO BAYESIAN (Orquestador de Neuronas)
# ==============================================================================

class BayesianBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.neurons = []
        
        # Sistema de energía metabólica
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        
        # Gen de personalidad: Umbral de decisión
        self.decision_threshold = np.random.uniform(0.3, 0.9)
        
        # Historial de precios para Lagrange
        self.price_history = []
        
        self.reset_genome()
    
    def reset_genome(self):
        """Crea población de neuronas con genomas aleatorios"""
        self.neurons = []
        for _ in range(self.n_neurons):
            genome = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3),  # Centro (x, y, z)
                [np.random.uniform(0.5, 1.5)],    # Radio
                [np.random.uniform(-1.0, 1.0)],   # Peso
                np.random.uniform(0.5, 1.5, 3),   # Stretch (sx, sy, sz)
                [np.random.uniform(-np.pi/2, np.pi/2)]  # Theta
            ])
            self.neurons.append(BayesianMarkovNeuron(genome, self.A_dc))
    
    def extract_genome(self):
        """Extrae genomas de todas las neuronas para evolución"""
        return np.array([neuron.params for neuron in self.neurons])
    
    def load_genome(self, genome_array):
        """Carga genomas en las neuronas (para elitismo/mutación)"""
        self.neurons = []
        for genome_slice in genome_array:
            self.neurons.append(BayesianMarkovNeuron(genome_slice, self.A_dc))
    
    def predict(self, market_point):
        """
        Predicción del cerebro completo.
        Agrega el punto al historial y consulta todas las neuronas.
        """
        # Actualizar historial de precios (solo últimos 10 para Lagrange)
        self.price_history.append(market_point[0])  # Asumimos x = precio
        if len(self.price_history) > 10:
            self.price_history.pop(0)
        
        # Activación de todas las neuronas
        total_activation = 0
        for neuron in self.neurons:
            total_activation += neuron.activate(market_point, self.price_history)
        
        # Tanh para normalizar a [-1, 1]
        return np.tanh(total_activation)
    
    def feedback_all(self, profit):
        """Propaga feedback a todas las neuronas"""
        for neuron in self.neurons:
            neuron.feedback(profit)

# ==============================================================================
# 🌍 MERCADO REAL (Igual que v1)
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
    all_opens = []
    
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
            opens = [float(k[1]) for k in data]
            
            all_closes = closes + all_closes
            all_volumes = volumes + all_volumes
            all_highs = highs + all_highs
            all_lows = lows + all_lows
            all_opens = opens + all_opens
            
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
        return None, None
    
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    vol_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
    ret_norm = returns / (np.std(returns) + 1e-8)
    volatility_norm = volatilities / (np.std(volatilities) + 1e-8)
    
    inputs = np.column_stack([ret_norm, vol_norm, volatility_norm])
    
    return inputs, prices

# ==============================================================================
# ⚔️ ENTRENAMIENTO (EVOLUCIÓN CON FEEDBACK)
# ==============================================================================

def run_evolution(generations=10, population_size=50):
    print(f"\n💀 --- FASE 1: ENTRENAMIENTO (DATOS RECIENTES) ---")
    
    # 1. Datos de entrenamiento
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 5000)
    inputs, prices_aligned = prepare_inputs(market_prices, market_vols, market_volatilities)
    
    if inputs is None: return None, None
    
    brain = BayesianBrain(n_neurons=10, A_dc=1.618)
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
        # Guardado periódico
        if gen % 1000 == 0 and gen > 0:
            np.save(f"best_genome_bayesian_gen_{gen}.npy", best_genome)
            print(f"💾 [AUTO-SAVE] Gen {gen} guardada.")
        
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        print(f"\n--- GENERACIÓN {gen+1}/{generations} (Mutación: {mut_rate:.4f}) ---")
        
        gen_best_fitness = -np.inf
        gen_best_genome = None
        gen_best_threshold = None
        
        # Evaluar población
        for i in range(population_size):
            # Crear cerebro
            brain = BayesianBrain(n_neurons=10, A_dc=1.618)
            
            # ELITISMO: El campeón pasa directo
            if i == 0 and gen > 0:
                brain.load_genome(best_genome)
                brain.decision_threshold = best_threshold
            else:
                # Mutar al mejor
                mutant = np.copy(best_genome)
                mask = np.random.rand(*mutant.shape) < 0.1
                mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
                mutant[:, 3] = np.abs(mutant[:, 3])
                mutant[:, 5:8] = np.abs(mutant[:, 5:8])
                
                brain.load_genome(mutant)
                
                # Mutar umbral
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
                
                signal = brain.predict(inputs[t])
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
    
    np.save("best_genome_bayesian_FINAL.npy", best_genome)
    np.save("best_threshold_bayesian_FINAL.npy", best_threshold)
    return best_genome, best_threshold

# ==============================================================================
# 🧪 TEST (OUT OF SAMPLE)
# ==============================================================================

def run_test_simulation(genome, threshold, inputs, prices):
    brain = BayesianBrain(n_neurons=10, A_dc=1.618)
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
        
        signal = brain.predict(inputs[t])
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
        
        # 2. Elegir fecha aleatoria
        now_ms = int(time.time() * 1000)
        six_months_ago_ms = now_ms - (180 * 24 * 60 * 60 * 1000)
        random_test_time = random.randint(six_months_ago_ms, now_ms)
        
        # 3. Bajar datos de test
        test_prices, test_vols, test_volatilities = fetch_extended_data(
            "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
        )
        test_inputs, test_prices_aligned = prepare_inputs(test_prices, test_vols, test_volatilities)
        
        if test_inputs is not None:
            # 4. EJECUTAR TEST
            final_bal, acts, equity = run_test_simulation(winner_genome, winner_threshold, test_inputs, test_prices_aligned)
            
            profit_pct = ((final_bal - 10000) / 10000) * 100
            print(f"\n📊 RESULTADO DEL TEST:")
            print(f"   Balance Inicial: $10,000")
            print(f"   Balance Final:   ${final_bal:,.2f}")
            print(f"   Profit/Loss:     {profit_pct:+.2f}%")
