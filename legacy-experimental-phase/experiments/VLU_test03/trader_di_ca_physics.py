import numpy as np
import matplotlib.pyplot as plt
import time
import random
import requests
import pandas as pd

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL + FÍSICA DE FLUIDOS (DI-CA V2 - PHYSICS SURGERY)
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        # ELIMINADO: Estado Bayesiano (Alpha/Beta) - Ahora es Determinista
        
        # --- FÍSICA DE FLUIDOS (CONSTANTES UNIVERSALES) ---
        # [K_viscosidad, Alpha_geométrico, Peso_Físico]
        # Inicializamos cerca de las constantes universales descubiertas
        # K = 0.2816, Alpha = 0.2639
        self.physics_genome = np.array([0.2816, 0.2639, 1.0]) 
        
        self.reset_genome()

    def reset_genome(self):
        # 1. Genoma Neuronal (13 genes por neurona)
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3),     # x, y, z
                [np.random.uniform(0.5, 1.5)],       # R (radio - Píxel de Planck base)
                [np.random.uniform(-1.0, 1.0)],      # W (peso interacción)
                np.random.uniform(0.5, 1.5, 3),      # Sx, Sy, Sz (stretch)
                [np.random.uniform(-np.pi/2, np.pi/2)], # Theta (ángulo)
                [np.random.randint(5, 60)],          # L (longitud memoria)
                [np.random.uniform(0.1, 10.0)],      # Omega_0
                [np.random.uniform(0.1, 5.0)],       # Masa
                [np.random.choice([-1.0, 1.0])]      # Sigma (Spin)
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)
        
        # 2. Genoma Físico (Di-Ca) - Determinista con variación leve
        # K -> 0.2816 (Universal Viscosity)
        # Alpha -> 0.2639 (Universal Geometry)
        self.physics_genome = np.array([
            np.random.normal(0.2816, 0.05),  # K
            np.random.normal(0.2639, 0.05),  # Alpha
            np.random.uniform(0.5, 2.0)      # Weight (Influencia relativa)
        ])
        
        # ELIMINADO: self.neuron_states

    def predict(self, history_points, full_df_window=None):
        # Inferencia Física Determinista
        if len(self.genome) == 0: return 0.0
        
        n = self.genome.shape[0]
        
        # Extraer constantes físicas globales
        k_phys = abs(self.physics_genome[0])
        alpha_phys = abs(self.physics_genome[1])
        w_phys = abs(self.physics_genome[2])
        
        # --- A. CÁLCULO NEURONAL (V9 -> V2 Physics) ---
        centers = self.genome[:, :3]
        # radii = self.genome[:, 3] * self.A_dc # Ya no se usa como radio gaussiano, sino base del pixel
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8]
        thetas = self.genome[:, 8]
        lengths = self.genome[:, 9].astype(int)
        
        total_activation = 0.0
        total_weight_abs = 0.0
        
        # Constante de Estructura Fina (Coupling)
        ALPHA_FINE = 1.0 / 137.036
        
        # Píxel de Planck (Límite de Hardware)
        PIXEL_LIMIT = 0.25
        
        for i in range(n):
            worm_len = lengths[i]
            neuron_history = history_points[-worm_len:] if len(history_points) >= worm_len else history_points
            
            if len(neuron_history) == 0: continue

            # Geometría Neuronal
            c, s = np.cos(thetas[i]), np.sin(thetas[i])
            R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            center = centers[i]
            if neuron_history.ndim == 1:
                t = np.linspace(0, 1, len(neuron_history))
                neuron_history = np.column_stack((t, neuron_history, np.zeros_like(t)))
            
            diffs = neuron_history - center
            rotated_diffs = diffs @ R_mat.T
            norm_diffs = rotated_diffs / stretch[i]
            dists_sq = np.sum(norm_diffs**2, axis=1)
            min_dist_sq = np.min(dists_sq)
            dist_to_center = np.sqrt(min_dist_sq)
            
            # --- 1. FUNCIÓN DE ACTIVACIÓN (PÍXEL DE PLANCK) ---
            # Reemplaza la exponencial gaussiana
            # activation = 1.0 / (1.0 + (min_dist_sq / PIXEL_LIMIT)**alpha_phys)
            # Nota: Usamos alpha_phys global para consistencia dimensional
            activation = 1.0 / (1.0 + (min_dist_sq / PIXEL_LIMIT) ** alpha_phys)
            
            # --- 2. POTENCIAL DE HIGGS (REEMPLAZA MOMENTUM) ---
            # V = K * Vol^Alpha / Distancia
            # Necesitamos Volumen promedio de la historia de la neurona
            # neuron_history columns: [0]=price, [1]=volume_rel, [2]=volatility (según inputs en run_evolution)
            # En run_evolution inputs: col 1 es volumen relativo.
            if neuron_history.shape[1] > 1:
                avg_volume = np.mean(np.abs(neuron_history[:, 1])) + 1e-9
            else:
                avg_volume = 1.0
                
            # Potencial Hidrostático de Higgs
            # Si el volumen es alto y la distancia es corta -> Potencial Alto (Presión)
            higgs_potential = k_phys * (avg_volume ** alpha_phys) / (dist_to_center + 1e-8)
            
            # Saturación del Potencial (Hardware Limit)
            higgs_potential = np.clip(higgs_potential, 0, 10.0)
            
            # --- SEÑAL DE NEURONA ---
            # Signal = Activation * Weight * Higgs_Potential
            # Ya no hay 'Confidence' bayesiana.
            signal = activation * weights[i] * higgs_potential
            
            # Acoplamiento Fino (Suma ponderada por 1/137)
            # En realidad, ALPHA_FINE es muy pequeño (0.007). 
            # Si multiplicamos cada una, la señal será muy débil.
            # Tal vez la idea es que la SUMA total se escale por ALPHA_FINE?
            # El prompt dice: "Úsala como factor de acoplamiento al integrar neural_output y physics_output."
            # "final_signal = (neural_output * ALPHA_FINE) + (physics_output * (1 - ALPHA_FINE))"
            # Ah, OK. Entonces aquí sumamos normal.
            
            total_activation += signal
            total_weight_abs += abs(weights[i])
            
        # Neural Output Normalizado (-1 a 1)
        neural_output = total_activation / (total_weight_abs + 1e-9)
        neural_output = np.tanh(neural_output)
        
        # --- B. CÁLCULO FÍSICO GLOBAL (DCPI de Di-Ca) ---
        physics_output = 0.0
        
        if full_df_window is not None and not full_df_window.empty:
            # Usamos las mismas constantes K y Alpha
            
            window_len = len(full_df_window)
            if window_len > 25:
                df_slice = full_df_window.iloc[-25:].copy()
                
                delta_p = df_slice['close'].diff().abs()
                flujo_bruto = delta_p * (df_slice['volume'] ** alpha_phys)
                flujo_promedio = flujo_bruto.rolling(window=24).mean()
                
                current_flujo = flujo_bruto.iloc[-1]
                current_avg = flujo_promedio.iloc[-1]
                
                # DCPI Normalizado
                dcpi = (current_flujo / (current_avg + 1e-9)) / k_phys
                
                last_close = df_slice['close'].iloc[-1]
                sma_short = df_slice['close'].iloc[-5:].mean()
                trend_up = last_close > sma_short
                
                # Reglas de Saturación (Deterministas)
                if dcpi > 3.0: # Saturación Extrema -> Reversión
                    physics_signal = -1.0 if trend_up else 1.0 
                elif dcpi > 0.25: # Flujo Fuerte -> Momentum (Seguir Tendencia)
                    # OJO: Prompt dice "El objetivo es detectar saturación".
                    # Si momentum tradicional heuristic fue eliminado, aquí la física manda.
                    physics_signal = 1.0 if trend_up else -1.0
                else:
                    physics_signal = 0.0 # Laminar / Ruido
                
                physics_output = physics_signal # * w_phys (ya no multiplicamos por w_phys aquí, sino en la mezcla final?)
                # El prompt dice: "final_signal = (neural_output * ALPHA_FINE) + (physics_output * (1 - ALPHA_FINE))"
                # Eso le da MUCHO peso a la física (99.3%) y poco a la neuronal (0.7%).
                # Es lo que pidió el usuario ("Do Definitiva").
        
        # --- C. INTEGRACIÓN DE ESTRUCTURA FINA ---
        # Matrix Coupling
        # Neural = Materia (0.7%)
        # Physics = Campo (99.3%)
        # Si ALPHA_FINE = 1/137 ~ 0.007
        
        # Ajuste: Tal vez el usuario quiso decir al revés?
        # "Equilibrando la Materia (neuronas) con el Campo (física)"
        # "Ratio que usa el universo... para que el volumen no destruya tu cuenta".
        # Si usamos ALPHA_FINE para neural, la neurona es casi irrelevante.
        # Pero seguiremos la instrucción LITERAL:
        # "final_signal = (neural_output * ALPHA_FINE) + (physics_output * (1 - ALPHA_FINE))"
        
        final_signal = (neural_output * ALPHA_FINE) + (physics_output * (1.0 - ALPHA_FINE))
        
        return np.tanh(final_signal)

    def mutate(self, rate=0.1, power=0.1):
        # Mutar Neuronas
        if random.random() < rate:
            idx = random.randint(0, self.n_neurons - 1)
            gene_idx = random.randint(0, 12)
            self.genome[idx, gene_idx] += np.random.normal(0, power)
            
        # Mutar Física (K, Alpha) -> BIASED hacia Universal Constants
        # K=0.2816, Alpha=0.2639
        if random.random() < rate:
            p_idx = random.randint(0, 2)
            
            # Gaussian Blur centrado en Universal Constants
            target = 0.0
            if p_idx == 0: target = 0.2816 # K
            elif p_idx == 1: target = 0.2639 # Alpha
            else: target = 1.0 # Weight
            
            current = self.physics_genome[p_idx]
            
            # Tendencia hacia el target (Mean Reversion Mutation)
            # Nuevo valor = (Scout + Target) / 2 + Ruido
            bias_strength = 0.1
            noise = np.random.normal(0, power * 0.5)
            
            new_val = current + (target - current) * bias_strength + noise
            self.physics_genome[p_idx] = new_val
            
            # Constraints
            self.physics_genome[0] = max(0.01, self.physics_genome[0])
            self.physics_genome[1] = max(0.01, self.physics_genome[1])
            self.physics_genome[2] = max(0.0, self.physics_genome[2])


# ==============================================================================
# 🌍 MERCADO REAL EXTENDIDO (BAJA 5000 VELAS)
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1h", total_candles=2000, custom_end_time=None):
    # ... (Misma lógica de descarga robusta)
    limit_per_call = 1000
    remaining = int(total_candles)
    all_closes, all_volumes, all_highs, all_lows, all_opens, all_times = [], [], [], [], [], []

    if custom_end_time is None: current_end_time = int(time.time() * 1000)
    else: current_end_time = int(custom_end_time)

    print(f"[>] {symbol}: Bajando {total_candles} velas (EndTime={current_end_time})...")

    while remaining > 0:
        limit = min(limit_per_call, remaining)
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={current_end_time}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if not isinstance(data, list) or len(data) == 0: break
            
            closes  = [float(x[4]) for x in data]
            opens   = [float(x[1]) for x in data]
            highs   = [float(x[2]) for x in data]
            lows    = [float(x[3]) for x in data]
            volumes = [float(x[5]) for x in data]
            times   = [int(x[0]) for x in data]
            
            all_closes  = closes + all_closes
            all_opens   = opens + all_opens
            all_highs   = highs + all_highs
            all_lows    = lows + all_lows
            all_volumes = volumes + all_volumes
            all_times   = times + all_times
            
            current_end_time = int(data[0][0]) - 1
            remaining -= len(data)
            print(f"   ... Bajados {len(data)}. Restante: {remaining}")
            time.sleep(0.1)
        except Exception as e:
            print(f" [X] Error: {e}")
            break

    closes  = np.array(all_closes)
    opens   = np.array(all_opens)
    highs   = np.array(all_highs)
    lows    = np.array(all_lows)
    volumes = np.array(all_volumes)
    times   = np.array(all_times, dtype=np.int64)
    
    if len(closes) == 0: return np.array([]), np.array([]), np.array([]), np.array([]), pd.DataFrame()
    
    df_full = pd.DataFrame({'ts': times, 'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes})
    volatilities = (highs - lows) / (opens + 1e-12)
    
    print(f" [OK] DATA FINAL: {len(closes)} velas listas.")
    return closes, volumes, volatilities, times, df_full

# ==============================================================================
# ⚔️ EL COLISEO (ENTRENAMIENTO FÍSICO DETERMINISTA)
# ==============================================================================

def run_evolution(generations=50, population_size=20): 
    print(f" [SIM] INICIANDO SIMULACION FISICA (DI-CA V2)")
    print(f" [POP] POBLACION: {population_size} | ESTRATEGIA: Determinista")
    
    market_prices, market_vols, market_volatilities, market_times, df_full = fetch_extended_data("PEPEUSDT", "1h", 2000)
    
    if len(market_prices) < 100: return
        
    prices_aligned = market_prices[1:]
    vols_aligned = market_vols[1:]
    volatilities_aligned = market_volatilities[1:]
    
    p_changes = np.diff(market_prices) / market_prices[:-1]
    inputs = np.column_stack([
        p_changes / 0.01,
        vols_aligned / (np.mean(vols_aligned) + 1e-6),
        volatilities_aligned / 0.01
    ])
    
    brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_physics_genome = np.copy(brain.physics_genome)
    best_fitness = -np.inf
    
    INITIAL_BALANCE = 1000.0
    LEVERAGE = 10.0
    BET_PCT = 0.10
    FEE = 0.001
    
    mut_rate = 0.5
    
    for gen in range(generations):
        progress = gen / generations
        current_mut_rate = mut_rate * (1 - progress) + 0.01
        
        best_of_batch_genome = None
        best_of_batch_phys = None
        best_of_batch_fitness = -np.inf
        
        for _ in range(population_size):
            mutant_genome = np.copy(best_genome)
            mutant_phys = np.copy(best_physics_genome)
            
            # Mutar (biased for physics)
            brain.genome = mutant_genome
            brain.physics_genome = mutant_phys
            brain.mutate(rate=0.3, power=current_mut_rate) # Usamos el método interno
            
            mutant_genome = brain.genome
            mutant_phys = brain.physics_genome
            
            # --- EVALUAR ---
            balance = INITIAL_BALANCE
            position = 0
            entry_price = 0
            current_bet = 0
            
            for t in range(50, len(inputs)):
                window_start = max(0, t - 40)
                df_window = df_full.iloc[window_start:t+1]
                hist_start = max(0, t - 60)
                neural_history = inputs[hist_start:t+1]
                
                signal = brain.predict(neural_history, full_df_window=df_window)
                current_price = prices_aligned[t]
                
                THR = 0.6
                
                # Close
                if position != 0:
                    should_close = (signal > THR and position == -1) or (signal < -THR and position == 1)
                    pnl_unrealized = (current_price - entry_price)/entry_price * LEVERAGE * position
                    if pnl_unrealized < -0.2: should_close = True 
                    
                    if should_close:
                        profit = current_bet * pnl_unrealized
                        cost = current_bet * LEVERAGE * FEE
                        balance += profit - cost
                        position = 0
                
                # Open
                elif position == 0:
                    if signal > THR:
                        position = 1
                        entry_price = current_price
                        current_bet = balance * BET_PCT
                        balance -= (current_bet * LEVERAGE * FEE)
                    elif signal < -THR:
                        position = -1
                        entry_price = current_price
                        current_bet = balance * BET_PCT
                        balance -= (current_bet * LEVERAGE * FEE)
                
                if balance < 10.0: break
            
            fitness = balance - INITIAL_BALANCE
            if fitness > best_of_batch_fitness:
                best_of_batch_fitness = fitness
                best_of_batch_genome = np.copy(mutant_genome)
                best_of_batch_phys = np.copy(mutant_phys)
        
        if best_of_batch_fitness > best_fitness:
            best_fitness = best_of_batch_fitness
            best_genome = best_of_batch_genome
            best_physics_genome = best_of_batch_phys
            print(f" [WIN] [Gen {gen}] NEW RECORD! Profit: ${best_fitness:.2f} | K={best_physics_genome[0]:.3f} Alpha={best_physics_genome[1]:.3f}")
        elif gen % 5 == 0:
             print(f"   [Gen {gen}] Best Fit: ${best_fitness:.2f} (Batch: ${best_of_batch_fitness:.2f})")

    print("\n [FIN] RESULTADO FINAL (CONSTANTES UNIVERSALES AJUSTADAS):")
    print(f"   K (Viscosidad): {best_physics_genome[0]:.4f}")
    print(f"   Alpha (Geometría): {best_physics_genome[1]:.4f}")
    return best_genome, best_physics_genome

if __name__ == "__main__":
    run_evolution()
