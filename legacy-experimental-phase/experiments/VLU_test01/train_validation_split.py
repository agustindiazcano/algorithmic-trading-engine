"""
RIGOROUS TRAIN/TEST VALIDATION PROTOCOL + MEMETIC EVOLUTION
-----------------------------------------------------------
1. Fetch Data
2. Split Train/Test
3. Train with HILL CLIMBING & LAGRANGE DIVERSITY
4. Validate BEST genome
"""
import numpy as np
import requests
import copy # Necesario para copiar genomas
import matplotlib.pyplot as plt
from datetime import datetime
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
from optimizers.pso_optimizer import PSOOptimizer
from optimizers.cma_es_optimizer import CMAESOptimizer

# --- CONFIGURATION ---
SYMBOL = "XRPUSDT"
INTERVAL = "1h"
TOTAL_CANDLES = 8000 
TRAIN_RATIO = 0.7
GENERATIONS = 1000 # Le damos tiempo para explorar
POP_SIZE = 40

# 🔥 NUEVOS PARAMETROS AGRESIVOS
HILL_CLIMBING_STEPS = 3  # Intentos de mejora individual por bot
HILL_CLIMBING_SIGMA = 0.05 # Cuánto puede variar para intentar mejorar
LAGRANGE_PENALTY = 5.0   # Cuánto % de profit le quitamos si es un copión
SIMILARITY_THRESHOLD = 0.3 # Distancia mínima para no ser considerado "clon"

def fetch_data(symbol, interval, limit):
    print(f"📡 Downloading {limit} candles for {symbol} {interval}...")
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_time = None
    batches = (limit // 1000) + 1
    for i in range(batches):
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time: params["endTime"] = end_time
        try:
            r = requests.get(url, params=params); data = r.json()
            if not data: break
            all_klines = data + all_klines
            end_time = data[0][0] - 1
        except Exception as e: break
    if len(all_klines) > limit: all_klines = all_klines[-limit:]
    prices = np.array([float(k[4]) for k in all_klines])
    volumes = np.array([float(k[5]) for k in all_klines])
    timestamps = [datetime.fromtimestamp(k[0]/1000) for k in all_klines]
    return prices, volumes, timestamps

def array_to_genome(arr):
    g = BounceHunterGenome(random_init=False)
    # Clip parameters to safe bounds
    g.radius_base = np.clip(arr[0], 0.01, 0.3) # Aumenté rango
    g.volatility_multiplier = np.clip(arr[1], 1.0, 6.0)
    g.inertia_up = np.clip(arr[2], 0.7, 0.99)
    g.inertia_down = np.clip(arr[3], 0.4, 0.99)
    g.volume_exhaustion_threshold = np.clip(arr[4], 0.4, 0.99)
    g.take_profit_tolerance = np.clip(arr[5], 0.001, 0.10) # TP más agresivo
    g.stop_loss_pct = np.clip(arr[6], 0.005, 0.15) # SL más amplio
    g.trailing_stop_trigger = np.clip(arr[7], 0.005, 0.15)
    g.trailing_stop_floor = np.clip(arr[8], 0.001, 0.10)
    g.rsi_overbought_threshold = np.clip(arr[9], 40, 95)
    g.volume_spike_multiplier = np.clip(arr[10], 1.2, 6.0)
    return g

# 🔥 FUNCIÓN DE HILL CLIMBING (Inteligencia Individual)
def hill_climbing_improve(base_genome_arr, prices, volumes):
    """
    Intenta mejorar el genoma moviendo un poco sus valores.
    Si encuentra una configuración mejor cerca, devuelve esa.
    """
    best_arr = np.copy(base_genome_arr)
    
    # Evaluar base
    g_base = array_to_genome(best_arr)
    vnn = BounceHunterVNN(g_base, prices, volumes)
    vnn.run()
    best_score = vnn.calculate_fitness()
    
    # Intentos de mejora (Micro-evolución)
    for _ in range(HILL_CLIMBING_STEPS):
        # Mutación pequeña (Exploración local)
        noise = np.random.normal(0, HILL_CLIMBING_SIGMA, size=len(base_genome_arr))
        test_arr = base_genome_arr + noise
        
        g_test = array_to_genome(test_arr)
        vnn_test = BounceHunterVNN(g_test, prices, volumes)
        vnn_test.run()
        score = vnn_test.calculate_fitness()
        
        if score > best_score:
            best_score = score
            best_arr = test_arr # ¡Encontró algo mejor!
            
    return best_arr, best_score

def main():
    # 1. DATA PREP
    prices, volumes, timestamps = fetch_data(SYMBOL, INTERVAL, TOTAL_CANDLES)
    split_idx = int(len(prices) * TRAIN_RATIO)
    train_prices = prices[:split_idx]; train_volumes = volumes[:split_idx]
    test_prices = prices[split_idx:]; test_volumes = volumes[split_idx:]
    test_dates = timestamps[split_idx:]
    
    print(f"\n✂️ SPLIT: Train {len(train_prices)} | Test {len(test_prices)}")
    
    # 2. TRAINING PHASE (La Escuelita AVANZADA)
    print("\n🏫 FASE 1: ESCUELITA DE ALTO RENDIMIENTO (Lagrange + Hill Climbing)...")
    
    pso = PSOOptimizer(dim=11, n_particles=POP_SIZE//2)
    cma = CMAESOptimizer(dim=11, population_size=POP_SIZE//2)
    
    best_global_genome = None
    best_global_fitness = -np.inf
    best_genome_arr = None # Para calcular distancias
    
    fitness_history = []

    for gen in range(GENERATIONS):
        # Ask optimizers
        pop_pso = pso.ask()
        pop_cma = cma.ask()
        
        raw_population = pop_pso + pop_cma
        final_population_arr = []
        fitnesses = []
        
        # --- 🔥 EVALUACIÓN CON HILL CLIMBING ---
        for i, ind_arr in enumerate(raw_population):
            # Aplicamos Hill Climbing a cada individuo
            # (El individuo intenta aprender por sí mismo antes del examen)
            improved_arr, score = hill_climbing_improve(ind_arr, train_prices, train_volumes)
            final_population_arr.append(improved_arr)
            fitnesses.append(score)
            
        # Convertir listas a numpy para facilitar matemáticas
        final_pop_matrix = np.array(final_population_arr)
        fitnesses = np.array(fitnesses)
        
        # Identificar al líder actual (antes de penalizar)
        current_best_idx = np.argmax(fitnesses)
        current_best_arr = final_pop_matrix[current_best_idx]
        
        # --- 🔥 PENALIZACIÓN LAGRANGE (Diversidad) ---
        # Si un bot es muy parecido al líder pero gana menos, lo castigamos duro.
        # Esto fuerza al optimizador a buscar picos DISTINTOS.
        if gen > 5: # Empezamos a castigar después de la gen 5
            for i in range(len(fitnesses)):
                if i == current_best_idx: continue # Al líder no se le castiga
                
                # Distancia Euclidiana (Similitud genética)
                dist = np.linalg.norm(final_pop_matrix[i] - current_best_arr)
                
                if dist < SIMILARITY_THRESHOLD:
                    # Penalización Lagrange: Restamos fitness por falta de originalidad
                    penalty = LAGRANGE_PENALTY * (1.0 - (dist / SIMILARITY_THRESHOLD))
                    fitnesses[i] -= penalty
        
        # Actualizar optimizadores
        # Nota: CMA-ES a veces se queja si le devolvemos vectores modificados,
        # así que le devolvemos las notas basadas en los vectores originales para estabilidad,
        # pero usamos los modificados para guardar el "Best".
        fit_pso = fitnesses[:len(pop_pso)]
        fit_cma = fitnesses[len(pop_pso):]
        
        pso.tell(fit_pso)
        cma.tell(pop_cma, fit_cma) # CMA recibe los originales pero con fitness penalizado/mejorado
        
        # Track Best Global
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        
        if gen_best_fit > best_global_fitness:
            best_global_fitness = gen_best_fit
            best_genome_arr = final_pop_matrix[gen_best_idx]
            best_global_genome = array_to_genome(best_genome_arr)
            print(f"  🚀 ¡NUEVO RÉCORD! {best_global_fitness:.2f}%")
            
        fitness_history.append(gen_best_fit)
        
        # Reporte inteligente
        if gen % 5 == 0:
            std_dev = np.std(fitnesses) # ¿Cuánta variedad hay?
            print(f" Gen {gen:<4} | Best: {best_global_fitness:.2f}% | Var(Div): {std_dev:.2f}")

    # 3. TESTING PHASE
    print("\n🔥 FASE 2: EL EXAMEN FINAL...")
    vnn_test = BounceHunterVNN(best_global_genome, test_prices, test_volumes)
    vnn_test.run()
    
    test_return = ((vnn_test.equity_curve[-1] - 10000) / 10000) * 100
    
    print(f" RESULTADO FINAL TEST: {test_return:.2f}%")
    if test_return > 5: print(" ✅ ESTRATEGIA ROBUSTA (Aprobado)")
    else: print(" ❌ ESTRATEGIA SOBREAJUSTADA (Reprobado)")

    # Plot
    plt.plot(test_dates, vnn_test.equity_curve)
    plt.title(f"Test Result: {test_return:.2f}%")
    plt.savefig("validation_advanced.png")
    print(" Gráfico guardado.")

if __name__ == "__main__":
    main()