"""
RIGOROUS TRAIN/TEST VALIDATION PROTOCOL
---------------------------------------
1. Fetch 1 year of data.
2. Split 70% Train / 30% Test.
3. Train 100 generations on Training Data.
4. Validate BEST genome on unseen Test Data.
"""
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
from optimizers.pso_optimizer import PSOOptimizer
from optimizers.cma_es_optimizer import CMAESOptimizer

# --- CONFIGURATION ---
SYMBOL = "XRPUSDT"
INTERVAL = "1h"
TOTAL_CANDLES = 8000 # ~1 year
TRAIN_RATIO = 0.7
GENERATIONS = 1000
POP_SIZE = 40

def fetch_data(symbol, interval, limit):
    print(f"📡 Downloading {limit} candles for {symbol} {interval}...")
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_time = None
    
    # Batch fetching (Binance limit 1000)
    batches = (limit // 1000) + 1
    for i in range(batches):
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time:
            params["endTime"] = end_time
            
        try:
            r = requests.get(url, params=params)
            data = r.json()
            if not data: break
            
            all_klines = data + all_klines
            end_time = data[0][0] - 1
            print(f"  Batch {i+1}/{batches}: Got {len(data)} candles")
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    # Crop to exact limit
    if len(all_klines) > limit:
        all_klines = all_klines[-limit:]
        
    prices = np.array([float(k[4]) for k in all_klines])
    volumes = np.array([float(k[5]) for k in all_klines])
    timestamps = [datetime.fromtimestamp(k[0]/1000) for k in all_klines]
    
    return prices, volumes, timestamps

def evaluate_population(genomes, prices, volumes):
    fitnesses = []
    for g in genomes:
        vnn = BounceHunterVNN(g, prices, volumes)
        vnn.run()
        fitnesses.append(vnn.calculate_fitness())
    return fitnesses

def array_to_genome(arr):
    # Mapping array to genome (Boundaries handled by optimizer, but clipping here for safety)
    g = BounceHunterGenome(random_init=False)
    # 0: Radius, 1: VolMult, 2: InertiaUp, 3: InertiaDown, 4: VolExhaust
    # 5: TP, 6: SL, 7: TrailTrig, 8: TrailFloor, 9: RSI, 10: Spike
    g.radius_base = np.clip(arr[0], 0.01, 0.2)
    g.volatility_multiplier = np.clip(arr[1], 1.0, 5.0)
    g.inertia_up = np.clip(arr[2], 0.8, 0.99)
    g.inertia_down = np.clip(arr[3], 0.5, 0.95)
    g.volume_exhaustion_threshold = np.clip(arr[4], 0.5, 0.95)
    g.take_profit_tolerance = np.clip(arr[5], 0.001, 0.05)
    g.stop_loss_pct = np.clip(arr[6], 0.005, 0.10)
    g.trailing_stop_trigger = np.clip(arr[7], 0.005, 0.10)
    g.trailing_stop_floor = np.clip(arr[8], 0.001, 0.05)
    g.rsi_overbought_threshold = np.clip(arr[9], 50, 90)
    g.volume_spike_multiplier = np.clip(arr[10], 1.5, 5.0)
    return g

def main():
    # 1. DATA PREP
    prices, volumes, timestamps = fetch_data(SYMBOL, INTERVAL, TOTAL_CANDLES)
    
    split_idx = int(len(prices) * TRAIN_RATIO)
    
    train_prices = prices[:split_idx]
    train_volumes = volumes[:split_idx]
    train_dates = timestamps[:split_idx]
    
    test_prices = prices[split_idx:]
    test_volumes = volumes[split_idx:]
    test_dates = timestamps[split_idx:]
    
    print(f"\n✂️ DATA SPLIT ({TRAIN_RATIO*100:.0f}/{100-TRAIN_RATIO*100:.0f})")
    print(f"  Training: {len(train_prices)} candles ({train_dates[0].date()} -> {train_dates[-1].date()})")
    print(f"  Testing:  {len(test_prices)} candles ({test_dates[0].date()} -> {test_dates[-1].date()})")
    print("  (Test data is COMPLETELY HIDDEN from evolution)")
    
    # 2. TRAINING PHASE (The School)
    print("\n🏫 FASE 1: LA ESCUELITA (Training)...")
    
    pso = PSOOptimizer(dim=11, n_particles=POP_SIZE//2)
    cma = CMAESOptimizer(dim=11, population_size=POP_SIZE//2)
    
    best_genome = None
    best_fitness = -np.inf
    
    print(f"  Running {GENERATIONS} generations...")
    
    for gen in range(GENERATIONS):
        # Ask
        pop_pso = pso.ask()
        pop_cma = cma.ask()
        
        # Convert to genomes
        genomes_pso = [array_to_genome(x) for x in pop_pso]
        genomes_cma = [array_to_genome(x) for x in pop_cma]
        all_genomes = genomes_pso + genomes_cma
        
        # Evaluate on TRAINING DATA only
        fitnesses = evaluate_population(all_genomes, train_prices, train_volumes)
        
        # Tell
        fit_pso = fitnesses[:len(pop_pso)]
        fit_cma = fitnesses[len(pop_pso):]
        
        pso.tell(fit_pso)
        cma.tell(pop_cma, fit_cma)
        
        # Track Best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_genome = all_genomes[gen_best_idx]
            
        if gen % 10 == 0:
            print(f"  Gen {gen:<3} | Best Train Fitness: {best_fitness:.2f}% | Avg: {np.mean(fitnesses):.2f}%")
            
    print(f"\n🎓 MEJOR ALUMNO (Escuelita):")
    print(f"  Fitness en Train: {best_fitness:.2f}%")
    
    # 3. TESTING PHASE (The Exam)
    print("\n🔥 FASE 2: EL EXAMEN (Test en Mundo Real)...")
    print("  Ejecutando mejor genoma en datos nunca vistos...")
    
    vnn_test = BounceHunterVNN(best_genome, test_prices, test_volumes)
    vnn_test.run()
    
    test_equity = vnn_test.equity_curve[-1]
    test_return = ((test_equity - 10000) / 10000) * 100
    test_trades = len([t for t in vnn_test.trades if t['type'] == 'sell'])
    
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"  TRAIN Return (Escuelita): {best_fitness:.2f}%")
    print(f"  TEST Return (Examen):     {test_return:.2f}%")
    print(f"  Trades en Test:           {test_trades}")
    
    # Diagnosis
    print("\n🩺 DIAGNÓSTICO:")
    if test_return > 5.0:
        print("  ✅ SANTO GRIAL (Robustez Confirmada)")
        print("  El bot generalizó correctamente y es rentable en datos nuevos.")
    elif test_return > 0:
        print("  ⚠️ APROBADO RASPOSO (Suerte/Marginal)")
        print("  Es rentable pero marginal. Podría ser ruido.")
    else:
        print("  ❌ FRAUDE (Overfitting)")
        print("  El bot memorizó el pasado pero falló en el futuro.")

    # Save Results
    best_genome.save('validated_best_genome.json')
    
    # Plot Test
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, vnn_test.equity_curve, label='Test Equity')
    plt.title(f'Test Set Performance: {test_return:.2f}% (Trades: {test_trades})')
    plt.axhline(10000, color='r', linestyle='--')
    plt.legend()
    plt.savefig('validation_test_chart.png')
    print("\nGráfico guardado: validation_test_chart.png")

if __name__ == "__main__":
    main()
