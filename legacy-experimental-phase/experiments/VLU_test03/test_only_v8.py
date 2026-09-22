import numpy as np
import time
import random
import sys
from trader_v8_spin_continuum import DynamicCausalBrain, fetch_extended_data, run_simulation_test

# Configurar UTF-8 para Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')

def prepare_data(prices, vols, volatilities):
    """
    Replica EXACTA de la normalización usada en el entrenamiento v8.
    """
    # 1. Diferencia porcentual de precio
    p_changes = np.diff(prices) / prices[:-1]
    
    # 2. Alinear arrays (perdemos el primer dato por el diff)
    prices_aligned = prices[1:]
    vols_aligned = vols[1:]
    volats_aligned = volatilities[1:]
    
    # 3. Normalizar
    # OJO: Usamos constantes hardcodeadas del entreno para consistencia
    # p_changes / 0.005
    # vols / (mean * 3)
    # volats / 0.005
    
    inputs = np.column_stack([
        p_changes / 0.005,                                      # Precio
        vols_aligned / (np.mean(vols_aligned) * 3 + 1e-6),      # Volumen
        volats_aligned / 0.005                                  # Volatilidad
    ])
    
    return inputs, prices_aligned

def main():
    print("\n🔮 CARGANDO CEREBRO v8 (Spin Continuum)...")
    
    try:
        winner = np.load("best_genome_spin_hard.npy")
        winner_thr = float(np.load("best_threshold_spin_hard.npy"))
        
        # ⚠️ --- MANUAL OVERRIDE (Opcional) ---
        # Si querés forzar un umbral, descomenta y poné valor (0.0 a 1.0)
        # Por ej: 0.2 es muy arriesgado, 0.9 es muy seguro.
        FORCE_THRESHOLD = 0.848

 # 0.65 
        
        if FORCE_THRESHOLD is not None:
            print(f"   ⚠️ OVERRIDE: Usando Umbral Manual {FORCE_THRESHOLD} (Original: {winner_thr:.3f})")
            winner_thr = FORCE_THRESHOLD
            
        print(f"   ✅ Genoma Cargado: {winner.shape[0]} neuronas")
        print(f"   ✅ Umbral Operativo: {winner_thr:.3f}")
    except Exception as e:
        print(f"❌ Error cargando archivos: {e}")
        print("   ¿Corriste el entrenamiento primero?")
        return

    # Inicializar Cerebro (IMPORTANTE: Usar len(winner) para evitar IndexError)
    brain = DynamicCausalBrain(n_neurons=len(winner), A_dc=1.618)
    brain.genome = winner
    # Reset estado Bayesiano para iniciar limpio (o se podría guardar/cargar también)
    brain.neuron_states = np.array([[10.0, 1.0] for _ in range(len(winner))])

    print("\n" + "="*60)
    print("🎲 TEST RAPIDO DE VALIDACION (10 ESCENARIOS)")
    print("="*60)

    # Configurar fechas para sampling
    start_time_ms = int(time.mktime(time.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
    now_ms = int(time.time() * 1000)
    
    results = []

    for i in range(1, 11):
        print(f"\n🧪 TEST {i}/10:")
        try:
            # Random time window
            random_time = random.randint(start_time_ms, now_ms)
            
            # Bajamos 5000 velas (igual que entreno)
            prices, vols, volats, times = fetch_extended_data("PEPEUSDT", "1m", 1000, custom_end_time=random_time)
            
            if len(prices) < 1000:
                print("⚠️  Poca data, saltando...")
                continue

            # Preparar inputs
            inputs, prices_aligned = prepare_data(prices, vols, volats)
            
            # Correr Simulación
            # Correr Simulación
            bal, trades, actions, trade_log = run_simulation_test(winner, winner_thr, inputs, prices_aligned, TRADING_MODE="BOTH")
            
            # Métricas
            profit_pct = ((bal - 10000) / 10000) * 100
            end_date = time.strftime('%Y-%m-%d', time.localtime(times[-1]/1000))
            
            print(f"   📅 Fin: {end_date} | 💰 Bal: ${bal:,.0f} ({profit_pct:+.1f}%) | 🤝 Trades: {trades}")
            
            # Detalle de Operaciones
            if trade_log:
                print("   📜 Detalle de Operaciones:")
                print("   " + "-"*90)
                print(f"   {'TIPO':<6} | {'ENTRADA':<10} | {'SALIDA':<10} | {'PNL %':<8} | {'PNL $':<10} | {'CONF %':<6} | {'COMP %':<6}")
                print("   " + "-"*90)
                for t in trade_log:
                    pnl_color = "+" if t['pnl_pct'] >= 0 else ""
                    print(f"   {t['type']:<6} | {t['entry_price']:<10.6f} | {t['exit_price']:<10.6f} | {pnl_color}{t['pnl_pct']:<7.2f}% | {pnl_color}${t['profit_usd']:<9.2f} | {t['confidence']:<6.1f}% | {t['compatibility']:<6.1f}%")
                print("   " + "-"*90)
            
            results.append(profit_pct)
            
            # Resetear estado del cerebro para el siguiente test
            brain.neuron_states = np.array([[10.0, 1.0] for _ in range(len(winner))])

        except Exception as e:
            print(f"❌ Error en test {i}: {e}")

    if results:
        wins = [r for r in results if r > 0]
        losses = [r for r in results if r < 0]
        zeros = [r for r in results if r == 0]
        
        n = len(results)
        
        print("\n" + "-"*60)
        print(f"📊 ESTADISTICAS FINAL ({n} Tests)")
        print(f"   🏆 Ganadores: {len(wins)} ({len(wins)/n*100:.1f}%)")
        print(f"   💀 Perdedores: {len(losses)} ({len(losses)/n*100:.1f}%)")
        print(f"   😐 Neutros:   {len(zeros)} ({len(zeros)/n*100:.1f}%)")
        print("-" * 60)
        print(f"📈 MEJOR: {np.max(results):+.2f}%")
        print(f"📉 PEOR:  {np.min(results):+.2f}%")
        print(f"⚖️ PROMEDIO: {np.mean(results):+.2f}%")
        print("-" * 60)
    else:
        print("\n❌ No se pudieron completar los tests.")

if __name__ == "__main__":
    main()
