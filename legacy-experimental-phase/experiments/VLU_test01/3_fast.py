import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. MOTOR DE MERCADO (Vectorizado) ---
def generate_market_data(n=3000):
    # Fondo
    price = np.cumsum(np.random.normal(0, 1.0, n)) + 1000
    volume = np.random.lognormal(0, 0.5, n)
    
    # Patrones
    patterns = []
    # Patrón: "The Dip" (Caída en V antes de subir)
    shape = np.array([0, -2, -5, -2, 0]) * 1.5
    
    for _ in range(30): # 30 Oportunidades
        idx = np.random.randint(50, n-50)
        price[idx:idx+5] += shape
        patterns.append(idx)
        
    return price, volume

# --- 2. CEREBRO VECTORIZADO (Nucleo Numpy) ---
# En lugar de objetos, usamos matrices para toda la población
# Pop Shape: (Pop_Size, 5) -> ADN de formas
# Thresholds: (Pop_Size, 1)

def run_generation_vectorized(pop_shapes, pop_params, price, volume):
    """
    Ejecuta toda la población sobre todo el mercado en OPERACIONES MATRICIALES.
    pop_shapes: (Pop, 5)
    pop_params: (Pop, 4) -> [Vol_Thresh, Trigger, TP, SL]
    """
    pop_size = len(pop_shapes)
    n_ticks = len(price)
    window = 5
    
    # 1. Pre-calcular TODAS las ventanas del mercado (Rolling Window)
    # Shape: (N_Ticks, 5)
    # Truco de stride_tricks para hacerlo sin copiar memoria
    shape_windows = np.lib.stride_tricks.sliding_window_view(price, window)
    vol_windows = np.lib.stride_tricks.sliding_window_view(volume, window)
    
    # Alineamos indices (sliding window recorta el final)
    # Windows[i] corresponde a price[i:i+5]
    
    # 2. Normalización Z-Score Masiva (Axis=1)
    means = np.mean(shape_windows, axis=1, keepdims=True)
    stds = np.std(shape_windows, axis=1, keepdims=True) + 1e-6
    if np.any(stds == 0): stds[:] = 1.0 # Parche seguridad
    
    norm_windows = (shape_windows - means) / stds
    # Norm_Windows Shape: (N_Windows, 5)
    
    # 3. Señal de Volumen
    # Media de volumen en la ventana
    vol_means = np.mean(vol_windows, axis=1) # (N_Windows,)
    # Contexto (Media movil simple aproximada para velocidad)
    # Usaremos una media global del volumen para simplificar calculo masivo
    vol_global_mean = np.mean(volume)
    vol_ratios = vol_means / vol_global_mean # (N_Windows,)

    # --- SIMULACIÓN DE TRADING (Iterativa por Ticks, Paralela por Población) ---
    # Esto es semi-vectorizado. No podemos vectorizar el tiempo (path dependence),
    # pero sí podemos vectorizar la población en cada tick.
    
    cash = np.full(pop_size, 10000.0)
    shares = np.zeros(pop_size)
    entry_prices = np.zeros(pop_size)
    entry_ticks = np.zeros(pop_size, dtype=int)
    n_trades = np.zeros(pop_size, dtype=int)
    
    # Extraemos parámetros para acceso rápido
    p_vol = pop_params[:, 0]
    p_trig = pop_params[:, 1]
    p_tp = pop_params[:, 2]
    p_sl = pop_params[:, 3]
    
    # Pre-calcular distancias es costoso (N_Windows * Pop_Size).
    # Lo hacemos on-the-fly o por bloques.
    
    for t in range(len(norm_windows) - 1): # t es el índice de la ventana
        # tick actual del precio (final de la ventana)
        curr_p = price[t + window - 1] 
        
        # Estado actual
        in_market = shares > 0
        
        # --- LÓGICA DE SALIDA (Para los que están dentro) ---
        if np.any(in_market):
            # Vectorized Exit Check
            # TP Check
            hit_tp = curr_p >= entry_prices * p_tp
            # SL Check
            hit_sl = curr_p <= entry_prices * p_sl
            # Time Stop (15 ticks)
            hit_time = (t - entry_ticks) > 15
            
            must_sell = in_market & (hit_tp | hit_sl | hit_time)
            
            # Ejecutar Ventas
            indices_sell = np.where(must_sell)[0]
            if len(indices_sell) > 0:
                cash[indices_sell] += shares[indices_sell] * curr_p
                shares[indices_sell] = 0
                
        # --- LÓGICA DE ENTRADA (Para los que están fuera) ---
        # Solo calculamos matching para quienes tienen cash > 0 (optimización)
        can_buy = (shares == 0) & (cash > 0)
        indices_can_buy = np.where(can_buy)[0]
        
        if len(indices_can_buy) > 0:
            # Ventana actual del mercado
            current_window = norm_windows[t] # (5,)
            current_vol = vol_ratios[t]
            
            # Subconjunto de la población activa
            active_shapes = pop_shapes[indices_can_buy] # (N_Active, 5)
            
            # Distancia Euclidiana Vectorizada
            # (P_i - M)^2
            dists = np.linalg.norm(active_shapes - current_window, axis=1)
            similarities = np.exp(-0.5 * dists)
            
            # Check Triggers
            triggers = (similarities > p_trig[indices_can_buy]) & \
                       (current_vol > p_vol[indices_can_buy])
            
            # Ejecutar Compras
            buyers = indices_can_buy[triggers]
            if len(buyers) > 0:
                shares[buyers] = cash[buyers] / curr_p
                cash[buyers] = 0
                entry_prices[buyers] = curr_p
                entry_ticks[buyers] = t
                n_trades[buyers] += 1
                
    # Valor final
    final_equity = cash + shares * price[-1]
    
    # Penalización por inactividad
    fitness = np.where(n_trades > 0, final_equity, 0)
    
    return fitness, n_trades

# --- 3. BUCLE EVOLUTIVO ---
def run_evolution_fast(generations=100, pop_size=100):
    # 1. Datos de Entrenamiento
    price_train, vol_train = generate_market_data(3000)
    
    # 2. Inicialización
    # Shapes: Normalizados N(0,1)
    pop_shapes = np.random.normal(0, 1, (pop_size, 5))
    # Params: [Vol(0.5-3), Trig(0.6-0.95), TP(1.01-1.1), SL(0.95-0.99)]
    pop_params = np.column_stack([
        np.random.uniform(0.5, 3.0, pop_size),
        np.random.uniform(0.6, 0.95, pop_size),
        np.random.uniform(1.01, 1.10, pop_size),
        np.random.uniform(0.90, 0.99, pop_size)
    ])
    
    start_time = time.time()
    
    best_dna_history = []
    
    print(f"Iniciando Evolución Acelerada ({generations} gens)...")
    print(f"{'GEN':<5} | {'MAX EQUITY':<12} | {'AVG TRADES':<10}")
    
    for gen in range(generations):
        # A. Evaluación Vectorizada
        fitness, trades = run_generation_vectorized(pop_shapes, pop_params, price_train, vol_train)
        
        # Stats
        best_idx = np.argmax(fitness)
        best_fit = fitness[best_idx]
        avg_trades = np.mean(trades)
        
        best_dna_history.append(pop_shapes[best_idx].copy())
        
        if gen % 10 == 0 or gen == generations-1:
            print(f"{gen:<5} | ${best_fit:<11.2f} | {avg_trades:<10.1f}")
            
        # B. Selección (Torneo Vectorizado o Top Elite)
        # Usaremos Top 20% Elitismo simple
        sorted_indices = np.argsort(fitness)[::-1]
        n_survivors = int(pop_size * 0.2)
        survivors_idx = sorted_indices[:n_survivors]
        
        survivor_shapes = pop_shapes[survivors_idx]
        survivor_params = pop_params[survivors_idx]
        
        # C. Reproducción
        new_shapes = []
        new_params = []
        
        # Elitismo puro (Pasan los mejores tal cual)
        new_shapes.append(survivor_shapes)
        new_params.append(survivor_params)
        
        n_missing = pop_size - n_survivors
        
        # Padres aleatorios de los supervivientes
        p1_idx = np.random.randint(0, n_survivors, n_missing)
        p2_idx = np.random.randint(0, n_survivors, n_missing)
        
        p1_s = survivor_shapes[p1_idx]
        p2_s = survivor_shapes[p2_idx]
        
        # Crossover (Promedio simple para formas continuas)
        child_s = (p1_s + p2_s) / 2.0
        # Mutación
        mutation_rate = 0.5 # Alta mutación inicial
        mutation_mask = np.random.rand(*child_s.shape) < mutation_rate
        child_s[mutation_mask] += np.random.normal(0, 0.3, np.sum(mutation_mask))
        
        # Params Crossover
        p1_p = survivor_params[p1_idx]
        p2_p = survivor_params[p2_idx]
        child_p = (p1_p + p2_p) / 2.0
        # Mutación Params
        mutation_mask_p = np.random.rand(*child_p.shape) < 0.3
        child_p[mutation_mask_p] += np.random.normal(0, 0.05, np.sum(mutation_mask_p))
        
        # Clipping params para que no se rompan
        child_p[:, 1] = np.clip(child_p[:, 1], 0.1, 0.99) # Trigger
        child_p[:, 2] = np.clip(child_p[:, 2], 1.001, 1.5) # TP
        child_p[:, 3] = np.clip(child_p[:, 3], 0.5, 0.999) # SL
        
        new_shapes.append(child_s)
        new_params.append(child_p)
        
        pop_shapes = np.vstack(new_shapes)
        pop_params = np.vstack(new_params)
        
    print(f"Evolución completada en {time.time() - start_time:.2f}s")
    
    # Extraer Ganador Final
    final_fitness, _ = run_generation_vectorized(pop_shapes, pop_params, price_train, vol_train)
    winner_idx = np.argmax(final_fitness)
    
    return pop_shapes[winner_idx], pop_params[winner_idx], best_dna_history

# --- 4. VALIDACIÓN OUT-OF-SAMPLE ---
def validate_winner(shape, params):
    print("\n>>> VALIDACIÓN OUT-OF-SAMPLE (Graduación) <<<")
    # Generar NUEVO mercado (Datos que nunca vio)
    price_test, vol_test = generate_market_data(2000) # 2000 ticks nuevos
    
    # Correr UN solo agente (vectorización de tamaño 1)
    shapes = shape.reshape(1, 5)
    parameters = params.reshape(1, 4)
    
    equity, trades = run_generation_vectorized(shapes, parameters, price_test, vol_test)
    
    final_score = equity[0]
    final_trades = trades[0]
    
    print(f"Equity Final en Test: ${final_score:.2f}")
    print(f"Trades Ejecutados:    {final_trades}")
    
    roi = (final_score - 10000) / 100
    if roi > 0 and final_trades > 5:
        print(f"RESULTADO: ✅ GRADUADO (+{roi:.2f}%)")
        return True
    else:
        print(f"RESULTADO: ❌ REPROBADO (ROI: {roi:.2f}%, Trades: {final_trades})")
        return False

# EXECUTE
best_shape, best_params, history = run_evolution_fast(generations=100)
validate_winner(best_shape, best_params)

# Visualizar Forma Aprendida
plt.figure()
plt.plot(best_shape, 'r-o', linewidth=3, label="Forma Ganadora (Z-Score)")
plt.title(f"Concepto de 'Oportunidad' Aprendido\nTP: {best_params[2]:.2f} | SL: {best_params[3]:.2f}")
plt.grid(True)
plt.legend()
plt.savefig('winner_dna.png')
