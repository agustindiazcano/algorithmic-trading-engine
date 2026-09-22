import numpy as np
import matplotlib.pyplot as plt
import time
import random
import requests
import pandas as pd
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich import box
import math

MODERN_SPINNERS = {
    "grok_snake": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "pulse_block": ["░", "▒", "▓", "█", "▓", "▒", "░"],
}

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL + FÍSICA DE FLUIDOS (DIAZ-CANO V1)
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        # Estado Bayesiano (Alpha, Beta) por neurona
        self.neuron_states = np.array([[10.0, 1.0] for _ in range(n_neurons)])
        
        # --- FÍSICA DE FLUIDOS (GENOMA ADICIONAL) ---
        # [K_viscosidad, Alpha_geométrico, Peso_Físico]
        self.physics_genome = np.array([1.0, 0.2639, 1.0]) 
        
        self.reset_genome()

    def reset_genome(self):
        # 1. Genoma Neuronal (13 genes por neurona)
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3),     # x, y, z
                [np.random.uniform(0.5, 1.5)],       # R (radio)
                [np.random.uniform(-1.0, 1.0)],      # W (peso)
                np.random.uniform(0.5, 1.5, 3),      # Sx, Sy, Sz (stretch)
                [np.random.uniform(-np.pi/2, np.pi/2)], # Theta (ángulo)
                [np.random.randint(5, 60)],          # L (longitud memoria)
                [np.random.uniform(0.1, 10.0)],      # Omega_0
                [np.random.uniform(0.1, 5.0)],       # Masa
                [np.random.choice([-1.0, 1.0])]      # Sigma (Spin)
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)
        
        # 2. Genoma Físico (Diaz-Cano)
        # K: 0.1 a 5.0 (Viscosidad)
        # Alpha: 0.1 a 0.8 (Geometría)
        # Weight: 0.0 a 5.0 (Influencia en la decisión final)
        self.physics_genome = np.array([
            np.random.uniform(0.1, 5.0),  # K
            np.random.uniform(0.1, 0.8),  # Alpha
            np.random.uniform(0.0, 5.0)   # Weight
        ])
        
        self.neuron_states = np.array([[10.0, 1.0] for _ in range(self.n_neurons)])

    def predict(self, history_points, full_df_window=None):
        # Inferencia Híbrida: Neuronas + Fluidos
        if len(self.genome) == 0: return 0.0
        
        n = self.genome.shape[0]
        
        # --- A. CÁLCULO NEURONAL (V9) ---
        centers = self.genome[:, :3]
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8]
        thetas = self.genome[:, 8]
        lengths = self.genome[:, 9].astype(int)
        
        # Confianza Bayesiana
        alphas = self.neuron_states[:n, 0]
        betas = self.neuron_states[:n, 1]
        confidence = alphas / (alphas + betas)
        
        total_activation = 0.0
        total_weight_abs = 0.0
        
        for i in range(n):
            worm_len = lengths[i]
            # Usamos 'close' prices del history_points (asumiendo que es numpy array de cierre o similar)
            # En V9 history_points es el array de precios normalizados.
            neuron_history = history_points[-worm_len:] if len(history_points) >= worm_len else history_points
            
            if len(neuron_history) == 0: continue

            # Geometría Neuronal (Simplificada para brevedad)
            c, s = np.cos(thetas[i]), np.sin(thetas[i])
            R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            center = centers[i]
            # Proyectar precio 1D a 3D (Time, Price, Volatility) - Aquí asumimos input simple
            # Para compatibilidad con V9, history_points debe ser (N, 3).
            # Si es 1D, lo expandimos.
            if neuron_history.ndim == 1:
                # [t, p, 0]
                t = np.linspace(0, 1, len(neuron_history))
                neuron_history = np.column_stack((t, neuron_history, np.zeros_like(t)))
            
            diffs = neuron_history - center
            rotated_diffs = diffs @ R_mat.T
            norm_diffs = rotated_diffs / stretch[i]
            dists_sq = np.sum(norm_diffs**2, axis=1)
            min_dist_sq = np.min(dists_sq)
            activation = np.exp(-min_dist_sq / (radii[i]**2 + 1e-8))
            
            # W * Activation * Confidence
            signal = weights[i] * activation * confidence[i]
            total_activation += signal
            total_weight_abs += abs(weights[i] * confidence[i])
            
        neural_output = total_activation / (total_weight_abs + 1e-9)
        
        # --- B. CÁLCULO FÍSICO (DIAZ-CANO) ---
        physics_output = 0.0
        
        if full_df_window is not None and not full_df_window.empty:
            k_phys = self.physics_genome[0]
            alpha_phys = self.physics_genome[1]
            w_phys = self.physics_genome[2]
            
            # Usamos las últimas 24 velas para la normalización (como en diaz_cano_bot.py)
            window_len = len(full_df_window)
            if window_len > 25:
                # Extraemos slice reciente
                df_slice = full_df_window.iloc[-25:].copy()
                
                # Cálculo DCPI Normalizado
                delta_p = df_slice['close'].diff().abs()
                flujo_bruto = delta_p * (df_slice['volume'] ** alpha_phys)
                flujo_promedio = flujo_bruto.rolling(window=24).mean()
                
                # DCPI del último punto
                current_flujo = flujo_bruto.iloc[-1]
                current_avg = flujo_promedio.iloc[-1]
                
                dcpi = (current_flujo / (current_avg + 1e-9)) / k_phys
                
                # Interpretación de Señal Física:
                # - Saturación Extrema (>3.0): Reversión Fuerte
                # - Flujo Fuerte (>1.0): Momentum
                
                # Si DCPI > 3.0 (Saturación Extrema): 
                # - Tendencia Alcista -> Venta (Signal -1)
                # - Tendencia Bajista -> Compra (Signal +1)
                
                last_close = df_slice['close'].iloc[-1]
                sma_short = df_slice['close'].iloc[-5:].mean()
                trend_up = last_close > sma_short
                
                if dcpi > 3.0:
                    physics_signal = -1.0 if trend_up else 1.0 # Reversión
                elif dcpi > 1.0:
                    physics_signal = 1.0 if trend_up else -1.0 # Momentum (Follow Trend)
                else:
                    physics_signal = 0.0 # Ruido laminar
                
                physics_output = physics_signal * w_phys
        
        # --- C. INTEGRACIÓN (Hybrid Signal) ---
        final_signal = neural_output + physics_output
        
        # Tanh para normalizar a [-1, 1]
        return np.tanh(final_signal)

    def mutate(self, rate=0.1, power=0.1):
        # Mutar Neuronas
        if random.random() < rate:
            idx = random.randint(0, self.n_neurons - 1)
            gene_idx = random.randint(0, 12)
            self.genome[idx, gene_idx] += np.random.normal(0, power)
            
        # Mutar Física (K, Alpha, Weight)
        if random.random() < rate:
            p_idx = random.randint(0, 2)
            self.physics_genome[p_idx] += np.random.normal(0, power * 0.5)
            # Asegurar límites positivos para K y Alpha
            self.physics_genome[0] = max(0.01, self.physics_genome[0]) # K
            self.physics_genome[1] = max(0.01, self.physics_genome[1]) # Alpha
            self.physics_genome[2] = max(0.0, self.physics_genome[2])  # Weight

# --- SIMULACIÓN Y ENTRENAMIENTO (Skeleton) ---
# Copiamos la lógica de trader_v9_spin_dynamics.py y la adaptamos

# ==============================================================================
# 🌍 MERCADO REAL EXTENDIDO (BAJA 5000 VELAS)
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1h", total_candles=2000, custom_end_time=None):
    """
    Baja MUCHA data haciendo múltiples llamadas a la API. Version robusta 2.0
    """
    limit_per_call = 1000
    remaining = int(total_candles)

    all_closes, all_volumes, all_highs, all_lows, all_opens, all_times = [], [], [], [], [], []

    # Binance necesita un 'endTime' para ir hacia atrás. Empezamos desde AHORA o custom.
    if custom_end_time is None:
        current_end_time = int(time.time() * 1000)
    else:
        current_end_time = int(custom_end_time)

    print(f"[>] {symbol}: Bajando {total_candles} velas (EndTime={current_end_time})...")

    while remaining > 0:
        limit = min(limit_per_call, remaining)
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={current_end_time}"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            if not isinstance(data, list):
                print(f"⚠️  Binance error: {data}")
                break
            
            if len(data) == 0:
                print(" [!] Sin mas datos.")
                break

            # Binance devuelve [Open Time, Open, High, Low, Close, Volume, ...]
            # OJO: Los datos vienen del más viejo al más nuevo dentro del lote.
            # Pero como vamos hacia atrás (endTime), cada lote es "más viejo" que el anterior.
            
            closes  = [float(x[4]) for x in data]
            opens   = [float(x[1]) for x in data]
            highs   = [float(x[2]) for x in data]
            lows    = [float(x[3]) for x in data]
            volumes = [float(x[5]) for x in data]
            times   = [int(x[0]) for x in data]  # Open time in ms

            # Concatenamos AL PRINCIPIO porque vamos hacia atrás en el tiempo
            all_closes  = closes  + all_closes
            all_opens   = opens   + all_opens
            all_highs   = highs   + all_highs
            all_lows    = lows    + all_lows
            all_volumes = volumes + all_volumes
            all_times   = times   + all_times

            # El siguiente endTime debe ser el OpenTime del PRIMER elemento de este lote - 1ms
            current_end_time = int(data[0][0]) - 1
            remaining -= len(data)
            
            print(f"   ... Bajados {len(data)}. Restante: {remaining}")
            time.sleep(0.1) # Respetar rate limit
            
        except Exception as e:
            print(f" [X] Error bajando data: {e}")
            break

    # Convertir a numpy
    closes  = np.array(all_closes)
    opens   = np.array(all_opens)
    highs   = np.array(all_highs)
    lows    = np.array(all_lows)
    volumes = np.array(all_volumes)
    times   = np.array(all_times, dtype=np.int64)
    
    if len(closes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Crear DataFrame completo para la física (necesitamos Close, Volume, High, Low)
    df_full = pd.DataFrame({
        'ts': times,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    # Calcular volatilidad simple (High - Low) / Open
    volatilities = (highs - lows) / (opens + 1e-12)
    
    print(f" [OK] DATA FINAL: {len(closes)} velas listas.")
    return closes, volumes, volatilities, times, df_full

# ==============================================================================
# ⚔️ EL COLISEO RICH (VISUALIZACION)
# ==============================================================================

def create_header(gen, max_gens, step, best_fitness):
    spinner = MODERN_SPINNERS["grok_snake"][step % len(MODERN_SPINNERS["grok_snake"])]
    progress = (gen / max_gens) * 100
    bar_width = 30
    filled = int((progress / 100) * bar_width)
    bar = f"[{'━' * filled}{' ' * (bar_width - filled)}]"
    
    color = "bright_cyan" if (math.sin(step * 0.2) > 0) else "cyan"
    
    content = f"[{color}]🧬 DIAZ-CANO V1 - HYBRID EVOLUTION[/{color}]\n"
    content += f"{spinner} Generación {gen}/{max_gens} {bar} {progress:.1f}%\n"
    content += f"🏆 Mejor Fitness Global: ${best_fitness:.2f}"
    
    return Panel(Text.from_markup(content), title="🤖 HYBRID NEURAL FLOW V1", border_style="blue")

def create_table(gen, fitness, k, alpha, weight, trades, wins, losses):
    table = Table(box=box.SIMPLE_HEAVY, expand=True, border_style="bright_black")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right", style="yellow")
    
    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    table.add_row("Generación Actual", str(gen))
    table.add_row("Beneficio (Batch Best)", f"${fitness:.2f}")
    table.add_row("Trades Total", str(trades))
    table.add_row("Win / Loss", f"[green]{wins}[/green] / [red]{losses}[/red]")
    table.add_row("Win Rate", f"{win_rate:.1f}%")
    table.add_row("K (Viscosidad)", f"{k:.4f}")
    table.add_row("Alpha (Geometría)", f"{alpha:.4f}")
    table.add_row("Peso Físico", f"{weight:.4f}")
    
    return table

def run_evolution(generations=50, population_size=20): 
    console = Console()
    console.clear()
    console.print(f" [SIM] INICIANDO SIMULACION HIBRIDA (NEURAL + FLUIDOS)", style="bold green")
    
    market_prices, market_vols, market_volatilities, market_times, df_full = fetch_extended_data("PEPEUSDT", "1h", 2000)
    
    if len(market_prices) < 100:
        print("Pocos datos, abortando.")
        return None
        
    prices_aligned = market_prices[1:]
    vols_aligned = market_vols[1:]
    volatilities_aligned = market_volatilities[1:]
    
    # Inputs para la red neuronal (Normalizados)
    p_changes = np.diff(market_prices) / market_prices[:-1]
    inputs = np.column_stack([
        p_changes / 0.01,
        vols_aligned / (np.mean(vols_aligned) + 1e-6),
        volatilities_aligned / 0.01
    ])
    
    # Inicializar Brain y Genoma
    brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_physics_genome = np.copy(brain.physics_genome) # [K, Alpha, Weight]
    best_fitness = -np.inf
    
    INITIAL_BALANCE = 1000.0
    LEVERAGE = 10.0
    BET_PCT = 0.10 # 10% del capital por operación (Dynamic)
    FEE = 0.001
    
    mut_rate = 0.5
    
    # Layout Rich
    layout = Layout()
    layout.split(
        Layout(name="header", size=5),
        Layout(name="main", ratio=1)
    )
    
    step_anim = 0
    
    with Live(layout, refresh_per_second=4, screen=False) as live:
        for gen in range(generations):
            progress = gen / generations
            current_mut_rate = mut_rate * (1 - progress) + 0.01
            
            best_of_batch_genome = None
            best_of_batch_phys = None
            best_of_batch_fitness = -np.inf
            
            best_batch_trades = 0
            best_batch_wins = 0
            best_batch_losses = 0
            
            # --- EVOLUTION LOOP ---
            for _ in range(population_size):
                step_anim += 1
                layout["header"].update(create_header(gen, generations, step_anim, best_fitness))
                
                # Clonar padres
                mutant_genome = np.copy(best_genome)
                mutant_phys = np.copy(best_physics_genome)
                
                # 1. Mutar Genoma Neuronal
                mask = np.random.rand(*mutant_genome.shape) < 0.1
                mutant_genome[mask] += np.random.normal(0, current_mut_rate, size=mutant_genome[mask].shape)
                # Constraints
                mutant_genome[:, 3] = np.abs(mutant_genome[:, 3]) # Radios
                mutant_genome[:, 9] = np.clip(np.abs(mutant_genome[:, 9]), 5, 60).astype(int) # Length
                
                # 2. Mutar Genoma Físico (K, Alpha, Weight)
                if np.random.rand() < 0.3: # 30% chance de mutar física
                    idx = np.random.randint(0, 3)
                    mutant_phys[idx] += np.random.normal(0, current_mut_rate * 0.5)
                    # Constraints Físicos
                    mutant_phys[0] = max(0.01, mutant_phys[0])  # K
                    mutant_phys[1] = max(0.01, mutant_phys[1])  # Alpha
                    mutant_phys[2] = max(0.0, mutant_phys[2])   # Weight
                
                # Cargar en el cerebro
                brain.genome = mutant_genome
                brain.physics_genome = mutant_phys
                
                # --- EVALUAR ---
                balance = INITIAL_BALANCE
                position = 0
                entry_price = 0
                current_bet = 0
                
                ind_trades = 0
                ind_wins = 0
                ind_losses = 0
                
                # Loop de Trading
                for t in range(50, len(inputs)): # Start at 50 to have history
                    # Pasamos la ventana de DF real para la física (necesita precios sin normalizar y volumen real)
                    window_start = max(0, t - 40)
                    df_window = df_full.iloc[window_start:t+1]
                    
                    # Input Neuronal
                    hist_start = max(0, t - 60)
                    neural_history = inputs[hist_start:t+1]
                    
                    signal = brain.predict(neural_history, full_df_window=df_window)
                    current_price = prices_aligned[t]
                    
                    # Trading Logic (Threshold = 0.6)
                    THR = 0.6
                    
                    # Close
                    if position != 0:
                        should_close = (signal > THR and position == -1) or (signal < -THR and position == 1)
                        # Stop Loss 5%
                        pnl_unrealized = (current_price - entry_price)/entry_price * LEVERAGE * position
                        if pnl_unrealized < -0.2: should_close = True 
                        
                        if should_close:
                            profit = (current_bet * pnl_unrealized) - (current_bet * LEVERAGE * FEE)
                            balance += profit
                            position = 0
                            
                            ind_trades += 1
                            if profit > 0: ind_wins += 1
                            else: ind_losses += 1
                    
                    # Open
                    elif position == 0:
                        if signal > THR:
                            position = 1
                            entry_price = current_price
                            current_bet = balance * BET_PCT # Dynamic bet size
                            balance -= (current_bet * LEVERAGE * FEE)
                        elif signal < -THR:
                            position = -1
                            entry_price = current_price
                            current_bet = balance * BET_PCT # Dynamic bet size
                            balance -= (current_bet * LEVERAGE * FEE)
                    
                    if balance < 10.0: break # Ruined
                
                fitness = balance - INITIAL_BALANCE
                if fitness > best_of_batch_fitness:
                    best_of_batch_fitness = fitness
                    best_of_batch_genome = np.copy(mutant_genome)
                    best_of_batch_phys = np.copy(mutant_phys)
                    
                    best_batch_trades = ind_trades
                    best_batch_wins = ind_wins
                    best_batch_losses = ind_losses
            
            # Selección Generacional
            if best_of_batch_fitness > best_fitness:
                best_fitness = best_of_batch_fitness
                best_genome = best_of_batch_genome
                best_physics_genome = best_of_batch_phys
                
                console.print(f"[bold green]🔥 NEW RECORD Gen {gen}: ${best_fitness:.2f}[/bold green]")
                
                # Guardar el mejor cerebro en caliente
                np.savez("best_brain_v1_hybrid.npz", genome=best_genome, physics_genome=best_physics_genome)
                console.print(f"[dim]💾 Cerebro guardado en best_brain_v1_hybrid.npz[/dim]")
            
            # Update Table
            layout["main"].update(create_table(
                gen, best_of_batch_fitness, 
                best_of_batch_phys[0], best_of_batch_phys[1], best_of_batch_phys[2],
                best_batch_trades, best_batch_wins, best_batch_losses
            ))

    print("\n [FIN] RESULTADO FINAL:")
    print(f"   K (Viscosidad): {best_physics_genome[0]:.4f}")
    print(f"   Alpha (Geometría): {best_physics_genome[1]:.4f}")
    print(f"   Peso Físico: {best_physics_genome[2]:.4f}")
    
    # Guardado Final
    np.savez("best_brain_v1_hybrid_final.npz", genome=best_genome, physics_genome=best_physics_genome)
    
    return best_genome, best_physics_genome

if __name__ == "__main__":
    run_evolution()
