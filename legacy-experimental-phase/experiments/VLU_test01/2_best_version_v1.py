import numpy as np
import matplotlib.pyplot as plt
import copy

# --- 1. EL ECOSISTEMA (Mercado con "Comida") ---
def generate_market_ecosystem(n=2000):
    # Fondo de ruido (Plancton)
    price = np.cumsum(np.random.normal(0, 0.5, n)) + 100
    volume = np.random.lognormal(0, 0.5, n)
    
    # Inyectamos "Comida" (Patrones repetitivos pero ocultos)
    # Patrón A: "Pump & Dump" (Subida rápida, bajada rápida)
    # Patrón B: "Acumulación" (Baja volatilidad, luego explosión)
    
    real_patterns = []
    
    # Inyectamos 15 Patrones A (Pump & Dump)
    shape_A = np.array([0, 1, 3, 1, 0]) # Forma de montaña
    for _ in range(15):
        idx = np.random.randint(50, n-50)
        price[idx:idx+5] += shape_A * 2.0
        volume[idx:idx+5] += 10.0
        real_patterns.append(idx)

    # Inyectamos 15 Patrones B (Caída en V - The Dip)
    shape_B = np.array([0, -1, -3, -1, 0]) # Forma de V
    for _ in range(15):
        idx = np.random.randint(50, n-50)
        price[idx:idx+5] += shape_B * 2.0
        volume[idx+1:idx+4] += 15.0 # Volumen solo en el fondo
        real_patterns.append(idx)
        
    return price, volume

# --- 2. EL ORGANISMO (VNN Genética) ---
class GeneticDrone:
    def __init__(self, genome=None):
        if genome is None:
            # ADN ALEATORIO
            # Gen 1: Forma del Precio (5 puntos) - Normalizada
            raw_shape = np.random.uniform(-1, 1, 5)
            self.shape_dna = (raw_shape - np.mean(raw_shape)) / (np.std(raw_shape) + 1e-6)
            
            # Gen 2: Umbral de Volumen (Bajamos un poco para que sobrevivan al inicio)
            self.vol_threshold = np.random.uniform(2, 8) 
            
            # Gen 3: Umbral de Disparo (Más permisivo al nacer)
            self.trigger_threshold = np.random.uniform(0.5, 0.8)
            
            # Gen 4: Take Profit (1% a 10%)
            self.take_profit = np.random.uniform(1.01, 1.10)
            # Gen 5: Stop Loss
            self.stop_loss = np.random.uniform(0.90, 0.99)
        else:
            self.shape_dna, self.vol_threshold, self.trigger_threshold, self.take_profit, self.stop_loss = genome

    def trade(self, prices, volumes):
        cash = 10000.0
        shares = 0
        entry_price = 0
        entry_tick = 0
        n_trades = 0
        
        # Recorremos el mercado
        for i in range(5, len(prices)-1):
            curr_p = prices[i]
            
            # --- CEREBRO VNN (Inferencia Rápida) ---
            w_p = prices[i-5:i]
            w_v = volumes[i-5:i]
            
            # 2. Normalizar ventana (Z-SCORE: Invariante a Escala)
            # Ahora la IA ve "formas puras", no precios absolutos
            w_p_norm = (w_p - np.mean(w_p)) / (np.std(w_p) + 1e-6)
            
            # 3. Comparar con ADN (Distancia Euclidiana)
            dist = np.linalg.norm(w_p_norm - self.shape_dna)
            similarity = np.exp(-0.5 * dist) 
            
            # Volumen relativo al promedio reciente (para detectar spikes reales)
            avg_vol_context = np.mean(volumes[max(0, i-20):i]) + 1e-6
            curr_vol_signal = np.mean(w_v) / avg_vol_context
            
            # --- ACCIÓN ---
            # VENTA
            if shares > 0:
                # Take Profit OR Stop Loss
                if curr_p >= entry_price * self.take_profit:
                    cash += shares * curr_p; shares = 0
                elif curr_p <= entry_price * self.stop_loss:
                    cash += shares * curr_p; shares = 0
                # Time Stop
                elif i - entry_tick > 15: 
                    cash += shares * curr_p; shares = 0
            
            # COMPRA
            if shares == 0:
                # Usamos logica difusa simple
                if similarity > self.trigger_threshold and curr_vol_signal > self.vol_threshold:
                    shares = cash / curr_p
                    cash = 0
                    entry_price = curr_p
                    entry_tick = i
                    n_trades += 1
                    
        equity = cash + (shares * prices[-1] if shares > 0 else 0)
        return equity, n_trades

# --- 3. LA EVOLUCIÓN (Algoritmo Genético) ---
def run_evolution(generations=20, pop_size=50):
    # 1. Generamos el Mercado (La Arena)
    prices, volumes = generate_market_ecosystem()
    
    # 2. Población Inicial (Caldo Primordial)
    population = [GeneticDrone() for _ in range(pop_size)]
    
    best_dna_history = []
    
    print(f"{'GEN':<4} | {'BEST EQUITY':<12} | {'TRADES':<6} | {'FORMA APRENDIDA (ADN)'}")
    print("-" * 70)
    
    for gen in range(generations):
        # A. EVALUACIÓN (Sobrevivencia)
        scores = []
        for drone in population:
            equity, trades = drone.trade(prices, volumes)
            # Fitness: Equity, pero penalizamos si no opera (trades=0)
            fitness = equity if trades > 0 else 0 
            scores.append((fitness, drone, trades))
            
        # Ordenar por ganancia (De mayor a menor)
        scores.sort(key=lambda x: x[0], reverse=True)
        
        best_equity, best_drone, best_trades = scores[0]
        
        # Log del Mejor de la Generación
        dna_str = "[" + ", ".join([f"{x:.1f}" for x in best_drone.shape_dna]) + "]"
        print(f"{gen:<4} | ${best_equity:<11.2f} | {best_trades:<6} | {dna_str}")
        
        best_dna_history.append(best_drone.shape_dna)
        
        # B. SELECCIÓN (Elitismo)
        # Nos quedamos con el Top 20% (Los Alphas)
        survivors = [s[1] for s in scores[:int(pop_size*0.2)]]
        
        # C. REPRODUCCIÓN (Crossover + Mutación)
        new_pop = []
        
        # El Top 1 pasa directo (Inmortalidad del Rey)
        new_pop.append(survivors[0]) 
        
        while len(new_pop) < pop_size:
            # Elegir 2 padres al azar de los sobrevivientes
            p1, p2 = np.random.choice(survivors, 2)
            
            # Crossover (Mezcla de genes)
            child_dna = np.where(np.random.rand(5) > 0.5, p1.shape_dna, p2.shape_dna)
            child_vol = np.mean([p1.vol_threshold, p2.vol_threshold])
            child_trig = np.mean([p1.trigger_threshold, p2.trigger_threshold])
            child_tp = np.mean([p1.take_profit, p2.take_profit])
            child_sl = np.mean([p1.stop_loss, p2.stop_loss])
            
            # Mutación (Radiación Cósmica - Cambio aleatorio)
            if np.random.rand() < 0.2: # 20% chance de mutar forma
                child_dna += np.random.normal(0, 0.5, 5)
            if np.random.rand() < 0.2: # Mutar thresholds
                child_tp += np.random.normal(0, 0.01)
                
            new_genome = (child_dna, child_vol, child_trig, child_tp, child_sl)
            new_pop.append(GeneticDrone(new_genome))
            
        population = new_pop
        
    return survivors[0], best_dna_history

# --- 4. EJECUCIÓN Y VISUALIZACIÓN ---
np.random.seed(42)
winner, history = run_evolution(generations=1000)

# Ver qué forma descubrió el ganador
plt.figure(figsize=(10, 6))

# Plotear evolución de la forma
for i, dna in enumerate(history):
    alpha = (i+1)/len(history)
    color = 'green' if i == len(history)-1 else 'gray'
    lw = 3 if i == len(history)-1 else 1
    label = "Forma Final (Ganadora)" if i == len(history)-1 else None
    plt.plot(dna, color=color, alpha=alpha, linewidth=lw, label=label)

plt.title("Evolución de la Memoria Geométrica (Lo que aprendió a ver)")
plt.xlabel("Tiempo Relativo (Ventana)")
plt.ylabel("Precio Relativo")
plt.grid(True)
plt.legend()
plt.show()

print(f"\nANÁLISIS DEL GANADOR:")
print(f"Estrategia: Entrar cuando vea la forma {np.round(winner.shape_dna, 2)}")
print(f"Salida: Take Profit al {winner.take_profit:.1%}, Stop Loss al {winner.stop_loss:.1%}")