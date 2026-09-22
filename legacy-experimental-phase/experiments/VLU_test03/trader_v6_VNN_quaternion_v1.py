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
# 🧬 NEURONA VOLUMÉTRICA 3D CON CUATERNIONES (S³)
# ==============================================================================

class QuaternionNeuron3D:
    """
    Neurona volumétrica avanzada con:
    - Rotaciones en S³ (cuaterniones) para evitar gimbal lock
    - Longitud de gusano variable (gen de escala temporal)
    - Respiración termodinámica
    - Aprendizaje Bayesiano
    """
    def __init__(self, genome_slice):
        # ADN GEOMÉTRICO (12 parámetros)
        # [x, y, z, R_base, W, Sx, Sy, Sz, Q0, Q1, Q2, Q3, worm_length]
        # Q0, Q1, Q2, Q3 = Cuaternión (rotación en S³)
        # worm_length = Longitud de trayectoria a escanear (3-50)
        self.params = genome_slice
        self.center = self.params[0:3]
        
        # Memoria Bayesiana
        # FIX B: Confianza ciega inicial (10.0) para que se atrevan a operar
        self.alpha = 10.0
        self.beta = 1.0
        self.confidence = 0.5
    
    def activate_3d_scan(self, market_trajectory, current_temp):
        """
        Escaneo volumétrico con cuaterniones y ventana adaptativa.
        
        Args:
            market_trajectory: Lista completa de puntos 3D disponibles
            current_temp: Volatilidad actual (temperatura)
        
        Returns:
            float: wi * phi * confidence (Ecuación 8)
        """
        # --- FASE 1: RESPIRACIÓN TERMODINÁMICA ---
        r_base = self.params[3]
        r_eff = r_base * (1.0 + current_temp * 3.0)
        
        # --- FASE 2: ROTACIÓN CON CUATERNIONES (S³) ---
        # Extraer cuaternión del genoma
        quat = self.params[8:12]
        # Normalizar para asegurar que sea unitario
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        
        # Convertir cuaternión a matriz de rotación
        rotation_matrix = R.from_quat(quat_norm).as_matrix()
        
        # --- FASE 3: ANISOTROPÍA ---
        stretch = self.params[5:8]
        S_inv = np.diag(1.0 / (stretch + 1e-6))
        
        # Transformación completa
        M = S_inv @ rotation_matrix.T
        
        # --- FASE 4: VENTANA ADAPTATIVA (Gen de Escala Temporal) ---
        # Longitud del gusano es un gen (3-50 puntos)
        worm_length = int(np.clip(self.params[12], 3, 50))
        
        # Tomar últimos N puntos según el gen
        adaptive_trajectory = market_trajectory[-worm_length:] if len(market_trajectory) >= worm_length else market_trajectory
        
        # --- FASE 5: ESCANEO DE TRAYECTORIA ---
        max_overlap = 0.0
        
        for point in adaptive_trajectory:
            diff = point - self.center
            local_diff = M @ diff
            dist = np.linalg.norm(local_diff)
            overlap = np.maximum(0, 1 - dist / (r_eff + 1e-6))
            
            if overlap > max_overlap:
                max_overlap = overlap
        
        # --- FASE 6: ECUACIÓN 8 (wi * phi * confidence) ---
        self.confidence = self.alpha / (self.alpha + self.beta)
        weight = self.params[4]
        
        return weight * max_overlap * self.confidence
    
    def learn_from_result(self, pnl):
        """Aprendizaje Bayesiano"""
        learning_rate = 0.1
        
        if pnl > 0:
            self.alpha += learning_rate
        else:
            self.beta += learning_rate
        
        if self.alpha + self.beta > 20:
            self.alpha *= 0.9
            self.beta *= 0.9

# ==============================================================================
# 🧠 CEREBRO CUATERNIÓNICO
# ==============================================================================

class QuaternionBrain:
    def __init__(self, n_neurons=10):
        self.n_neurons = n_neurons
        self.neurons = []
        
        # Sistema de energía
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        
        # Gen de personalidad
        # FIX A: Umbral muy bajo para arrancar (0.1 - 0.3)
        self.decision_threshold = np.random.uniform(0.1, 0.3)
        
        # Historial de trayectoria (máximo 100 puntos)
        self.worm_history = []
        
        self.reset_genome()
    
    def reset_genome(self, market_sample=None, avg_volatility=None):
        """Crea población de neuronas con cuaterniones
        
        Args:
            market_sample: Array de puntos 3D recientes del mercado para inicializar centros
            avg_volatility: Volatilidad promedio para escalar radios
        """
        self.neurons = []
        for _ in range(self.n_neurons):
            # Generar cuaternión aleatorio unitario
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)
            
            # Inicialización inteligente de centros
            if market_sample is not None and len(market_sample) > 0:
                # Samplear un punto aleatorio del mercado reciente
                idx = np.random.randint(0, len(market_sample))
                center = market_sample[idx] + np.random.randn(3) * 0.1  # Pequeño ruido
            else:
                # Fallback a random puro
                center = np.random.uniform(-3.0, 3.0, 3)
            
            # Radio base adaptado a volatilidad
            if avg_volatility is not None:
                R_base = np.random.uniform(0.5, 1.5) * (1.0 + avg_volatility * 2.0)
            else:
                R_base = np.random.uniform(0.5, 1.5)
            
            genome = np.array([
                center[0],                            # x (del mercado)
                center[1],                            # y (del mercado)
                center[2],                            # z (del mercado)
                R_base,                               # R_base (adaptado a volatilidad)
                np.random.uniform(-1.0, 1.0),         # W
                np.random.uniform(0.5, 2.0),          # Sx
                np.random.uniform(0.5, 2.0),          # Sy
                np.random.uniform(0.5, 2.0),          # Sz
                quat[0], quat[1], quat[2], quat[3],   # Cuaternión
                np.random.uniform(3, 50)              # worm_length
            ])
            self.neurons.append(QuaternionNeuron3D(genome))
    
    def extract_genome(self):
        return np.array([neuron.params for neuron in self.neurons])
    
    def load_genome(self, genome_array):
        self.neurons = []
        for genome_slice in genome_array:
            self.neurons.append(QuaternionNeuron3D(genome_slice))
    
    def predict(self, market_point, current_volatility):
        """
        Predicción con MAX-AGGREGATION (Ecuación 8).
        
        Args:
            market_point: Punto 3D actual
            current_volatility: Temperatura
        
        Returns:
            float: Señal del ganador [-1, 1]
        """
        # Actualizar historial (máximo 100 puntos)
        self.worm_history.append(market_point)
        if len(self.worm_history) > 100:
            self.worm_history.pop(0)
        
        # MAX-AGGREGATION
        max_activation = 0.0
        winner_activation = 0.0
        
        for neuron in self.neurons:
            activation = neuron.activate_3d_scan(self.worm_history, current_volatility)
            
            if abs(activation) > abs(max_activation):
                max_activation = activation
                winner_activation = activation
        
        return np.tanh(winner_activation)
    
    def feedback_all(self, pnl):
        """Propagar feedback"""
        for neuron in self.neurons:
            neuron.learn_from_result(pnl)

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
    
    # Normalización rolling (100 períodos)
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
    
    # Volatilidad rolling (temperatura)
    # CRÍTICO: Añadir epsilon para evitar mercado "congelado" (temp=0)
    vol_window = 20
    rolling_vol = np.array([np.std(returns[max(0, i-vol_window):i+1]) for i in range(len(returns))])
    
    # Temperatura mínima vital (evita r_eff estático)
    MIN_TEMP = 0.001
    rolling_vol = np.maximum(rolling_vol, MIN_TEMP)
    
    return inputs, prices, rolling_vol

# ==============================================================================
# ⚔️ ENTRENAMIENTO
# ==============================================================================

def run_evolution(generations=1000, population_size=10):
    print(f"\n💀 --- FASE 1: ENTRENAMIENTO (DATOS RECIENTES) ---")
    
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 5000)
    inputs, prices_aligned, rolling_volatility = prepare_inputs_3d(market_prices, market_vols, market_volatilities)
    
    if inputs is None: return None, None
    
    # Preparar sample del mercado para inicialización inteligente
    # Usar últimas 500 velas como referencia
    market_sample = inputs[-500:] if len(inputs) >= 500 else inputs
    avg_volatility = np.mean(rolling_volatility)
    
    brain = QuaternionBrain(n_neurons=10)
    brain.reset_genome(market_sample=market_sample, avg_volatility=avg_volatility)
    best_genome = brain.extract_genome()
    best_threshold = brain.decision_threshold
    best_fitness = -np.inf
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.002  # 0.2% (más conservador para memecoin)
    FEE = 0.0004      # 0.04% (taker fee sin descuentos)
    
    METABOLIC_COST_FIXED = 0.01
    ENERGY_FROM_PROFIT = 0.1
    
    initial_mut = 0.90
    final_mut = 0.001
    
    for gen in range(generations):
        if gen % 1000 == 0 and gen > 0:
            np.save(f"best_genome_quaternion_gen_{gen}.npy", best_genome)
            print(f"💾 [AUTO-SAVE] Gen {gen} guardada.")
        
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        print(f"\n--- GENERACIÓN {gen+1}/{generations} (Mutación: {mut_rate:.4f}) ---")
        
        gen_best_fitness = -np.inf
        gen_best_genome = None
        gen_best_threshold = None
        
        # Variables para trackear stats del mejor
        gen_best_trades = 0
        gen_best_balance = 0.0
        gen_best_energy = 0.0
        
        for i in range(population_size):
            brain = QuaternionBrain(n_neurons=10)
            
            # ELITISMO
            if i == 0 and gen > 0:
                brain.load_genome(best_genome)
                brain.decision_threshold = best_threshold
            else:
                mutant = np.copy(best_genome)
                mask = np.random.rand(*mutant.shape) < 0.1
                mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
                mutant[:, 3] = np.abs(mutant[:, 3])  # R_base
                mutant[:, 5:8] = np.abs(mutant[:, 5:8])  # Sx, Sy, Sz
                
                # Normalizar cuaterniones después de mutación
                for j in range(len(mutant)):
                    quat = mutant[j, 8:12]
                    mutant[j, 8:12] = quat / (np.linalg.norm(quat) + 1e-8)
                
                # Clip worm_length
                mutant[:, 12] = np.clip(mutant[:, 12], 3, 50)
                
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
                        # FIX: No restar balance (ya fue descontado al abrir)
                        # El margen ya se perdió, solo cerrar posición
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
            
            if not alive: 
                fitness = 0
            # FIX C: Hambre Extrema (Castigo por inactividad)
            elif trades == 0:
                fitness = 10  # Castigo total por no operar (muerte segura)
            elif trades < 5:
                fitness = balance * 0.5  # Si opera poco, pierde la mitad del mérito
            else: 
                fitness = 2000 + balance
            
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_genome = brain.extract_genome()
                gen_best_threshold = brain.decision_threshold
                
                # Guardar stats del mejor
                gen_best_trades = trades
                gen_best_balance = balance
                gen_best_energy = brain.energy
        
        # Mostrar stats del mejor de esta generación
        print(f"   Mejor: Balance ${gen_best_balance:.2f} | Trades: {gen_best_trades} | Energía: {gen_best_energy:.1f}")
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = np.copy(gen_best_genome)
            best_threshold = gen_best_threshold
            print(f"🏆 NUEVO RÉCORD: Balance ${gen_best_balance:.2f} | Trades: {gen_best_trades} | Umbral: {best_threshold:.2f}")
    
    np.save("best_genome_quaternion_FINAL.npy", best_genome)
    np.save("best_threshold_quaternion_FINAL.npy", best_threshold)
    return best_genome, best_threshold

# ==============================================================================
# 🧪 TEST
# ==============================================================================

def run_test_simulation(genome, threshold, inputs, prices, rolling_volatility):
    brain = QuaternionBrain(n_neurons=10)
    brain.load_genome(genome)
    brain.decision_threshold = threshold
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.002  # 0.2% (más conservador para memecoin)
    FEE = 0.0004      # 0.04% (taker fee sin descuentos)
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    trades = 0
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
    winner_genome, winner_threshold = run_evolution(generations=10, population_size=10)
    
    if winner_genome is not None:
        print("\n" + "="*60)
        print("🎲 --- FASE 2: TEST CIEGO (10 PERIODOS ALEATORIOS) ---")
        print("="*60)
        
        # Timestamp de inicio de 2023
        start_2023_ms = int(time.mktime(time.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
        now_ms = int(time.time() * 1000)
        
        test_results = []
        
        for test_num in range(1, 11):
            print(f"\n🧪 TEST {test_num}/10:")
            
            # Generar tiempo aleatorio desde 2023
            random_test_time = random.randint(start_2023_ms, now_ms)
            
            test_prices, test_vols, test_volatilities = fetch_extended_data(
                "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
            )
            test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs_3d(test_prices, test_vols, test_volatilities)
            
            if test_inputs is not None:
                final_bal, acts, equity, test_trades = run_test_simulation(winner_genome, winner_threshold, test_inputs, test_prices_aligned, test_rolling_vol)
                
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
