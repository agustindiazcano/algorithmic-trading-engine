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
# 🧠 CEREBRO VOLUMETRIC LOGIC (DI-CA V2 - VL OPTIMIZED)
# ==============================================================================

class VolumetricLogicBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        
        # --- FÍSICA DE FLUIDOS (CONSTANTES UNIVERSALES) ---
        # K_universal = 0.2816
        # Alpha_universal = 0.2639
        self.physics_genome = np.array([0.2816, 0.2639, 1.0]) 
        
        self.reset_genome()

    def reset_genome(self):
        # 1. Genoma Neuronal (13 genes por neurona - Centros de Decisión)
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3),     # x, y, z (Centro de la Bola)
                [0.25],                              # R (Planck Radius - Fijo en 0.25 por definición)
                [np.random.uniform(-1.0, 1.0)],      # W (Peso / Verdad de la Bola)
                np.random.uniform(0.5, 1.5, 3),      # Sx, Sy, Sz (Deformación de la Bola)
                [np.random.uniform(-np.pi/2, np.pi/2)], # Theta
                [np.random.randint(5, 60)],          # L (Memoria)
                [np.random.uniform(0.1, 10.0)],      # Omega_0
                [np.random.uniform(0.1, 5.0)],       # Masa
                [np.random.choice([-1.0, 1.0])]      # Sigma
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)
        
        # 2. Genoma Físico - Micro-ajustes alrededor de universales
        self.physics_genome = np.array([
            np.random.normal(0.2816, 0.01),  # K (Muy poca variacion)
            np.random.normal(0.2639, 0.01),  # Alpha
            np.random.uniform(0.8, 1.2)      # Weight
        ])

    def calculate_dcpi_series(self, closes, volumes):
        """
        Pre-calcula la serie DCPI completa usando Numpy para eficiencia O(n).
        """
        k_phys = abs(self.physics_genome[0])
        alpha_phys = abs(self.physics_genome[1])
        
        # Numpy diff (shift -1)
        delta_p = np.abs(np.diff(closes, prepend=closes[0]))
        
        # Flujo Bruto = DeltaP * Volume^Alpha
        flujo_bruto = delta_p * (volumes ** alpha_phys)
        
        # Rolling Mean (Window 24) - Numpy pure implementation
        window = 24
        pad = np.zeros(window-1)
        padded_flujo = np.concatenate((pad, flujo_bruto))
        
        # Convolution for moving average
        kernel = np.ones(window) / window
        flujo_promedio = np.convolve(flujo_bruto, kernel, mode='full')[:len(flujo_bruto)]
        
        # DCPI = (FlujoBruto / FlujoPromedio) / K
        dcpi_series = (flujo_bruto / (flujo_promedio + 1e-9)) / k_phys
        
        return dcpi_series

    def predict(self, history_points, current_dcpi=0.0, trend_up=True):
        # Inferencia Geométrica (Volumetric Logic)
        if len(self.genome) == 0: return 0.0, 0.0
        
        n = self.genome.shape[0]
        
        PLANCK_R = 0.25
        ALPHA_FINE = 1.0 / 137.036
        
        # --- A. CÁLCULO NEURONAL (LATTICE MAX-AGGREGATION) ---
        centers = self.genome[:, :3]
        weights = self.genome[:, 4] # Valor de Verdad (-1 a 1)
        stretch = self.genome[:, 5:8]
        thetas = self.genome[:, 8]
        lengths = self.genome[:, 9].astype(int)
        
        max_activation_val = 0.0 # Valor absoluto maximo encontrado
        dominant_signal = 0.0    # Señal correspondiente a ese maximo
        
        for i in range(n):
            worm_len = lengths[i]
            neuron_history = history_points[-worm_len:] if len(history_points) >= worm_len else history_points
            if len(neuron_history) == 0: continue

            # Geometría
            c, s = np.cos(thetas[i]), np.sin(thetas[i])
            R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            center = centers[i]
            if neuron_history.ndim == 1:
                t = np.linspace(0, 1, len(neuron_history))
                neuron_history = np.column_stack((t, neuron_history, np.zeros_like(t)))
            
            diffs = neuron_history - center
            rotated_diffs = diffs @ R_mat.T
            norm_diffs = rotated_diffs / stretch[i] # Deformación métrica
            dists_sq = np.sum(norm_diffs**2, axis=1)
            min_dist = np.sqrt(np.min(dists_sq))
            
            # --- 1. CONE ACTIVATION (PHI) ---
            # phi = max(0, 1 - d/R)
            if min_dist > PLANCK_R:
                phi = 0.0
            else:
                phi = 1.0 - (min_dist / PLANCK_R)
            
            # --- 2. MAX AGGREGATION (VDM) ---
            activation_strength = abs(weights[i]) * phi
            
            if activation_strength > max_activation_val:
                max_activation_val = activation_strength
                dominant_signal = np.sign(weights[i]) * activation_strength
        
        neural_output = dominant_signal 
        
        # --- B. CÁLCULO FÍSICO (DCPI & THRESHOLD) ---
        # Usamos el DCPI pre-calculado
        
        # --- 3. THRESHOLD DECISION D_P(S; 4.0) ---
        if current_dcpi > 4.0:
            physics_signal = -1.0 if trend_up else 1.0
        elif current_dcpi > 0.25:
            physics_signal = 1.0 if trend_up else -1.0
        else:
            physics_signal = 0.0
        
        physics_output = physics_signal

        # --- C. INTERFERENCIA DE CAMPOS (FINE STRUCTURE) ---
        final_signal = (neural_output * ALPHA_FINE) + (physics_output * (1.0 - ALPHA_FINE))
        
        return np.tanh(final_signal), current_dcpi

    def mutate(self, rate=0.1, power=0.1):
        # Mutar Neuronas
        if random.random() < rate:
            idx = random.randint(0, self.n_neurons - 1)
            gene_idx = random.randint(0, 12)
            if gene_idx != 3: 
                self.genome[idx, gene_idx] += np.random.normal(0, power)
            
        # Mutar Física (Micro-ajustes)
        if random.random() < rate:
            p_idx = random.randint(0, 2)
            
            # Target Universal
            target = 0.0
            if p_idx == 0: target = 0.2816 
            elif p_idx == 1: target = 0.2639
            else: target = 1.0
            
            current = self.physics_genome[p_idx]
            bias = 0.2 
            noise = np.random.normal(0, power * 0.1)
            
            new_val = current + (target - current) * bias + noise
            self.physics_genome[p_idx] = new_val
            
            self.physics_genome[0] = max(0.01, self.physics_genome[0])
            self.physics_genome[1] = max(0.01, self.physics_genome[1])
            self.physics_genome[2] = max(0.0, self.physics_genome[2])

# ==============================================================================
# 🌍 MERCADO REAL (DATA)
# ==============================================================================
def fetch_extended_data(symbol="PEPEUSDT", interval="1h", total_candles=2000, custom_end_time=None):
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
    return closes, volumes, volatilities, times, df_full

# ==============================================================================
# ⚔️ EL COLISEO RICH (VISUALIZACION + SIMULACION)
# ==============================================================================

def create_header(gen, max_gens, step, best_fitness):
    spinner = MODERN_SPINNERS["grok_snake"][step % len(MODERN_SPINNERS["grok_snake"])]
    progress = (gen / max_gens) * 100
    bar_width = 30
    filled = int((progress / 100) * bar_width)
    bar = f"[{'━' * filled}{' ' * (bar_width - filled)}]"
    
    color = "bright_cyan" if (math.sin(step * 0.2) > 0) else "cyan"
    
    content = f"[{color}]🧬 VOLUMETRIC LOGIC EVOLUTION V2[/{color}]\n"
    content += f"{spinner} Generación {gen}/{max_gens} {bar} {progress:.1f}%\n"
    content += f"🏆 Mejor Fitness Global: ${best_fitness:.2f}"
    
    return Panel(Text.from_markup(content), title="🤖 DI-CA VL BRAIN", border_style="blue")

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
    
    return table

def run_evolution(generations=50, population_size=20): 
    console = Console()
    console.clear()
    console.print(f" [SIM] INICIANDO SIMULACION VOLUMETRIC LOGIC (DI-CA V2 VL)", style="bold green")
    
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
    
    s_closes = pd.Series(market_prices)
    sma_short = s_closes.rolling(window=5).mean().values[1:] 
    trend_up_array = prices_aligned > sma_short

    brain = VolumetricLogicBrain(n_neurons=10, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_physics_genome = np.copy(brain.physics_genome)
    best_fitness = -np.inf
    
    INITIAL_BALANCE = 1000.0
    LEVERAGE = 10.0
    BET_PCT = 0.10
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
            
            for _ in range(population_size):
                step_anim += 1
                layout["header"].update(create_header(gen, generations, step_anim, best_fitness))
                
                mutant_genome = np.copy(best_genome)
                mutant_phys = np.copy(best_physics_genome)
                
                brain.genome = mutant_genome
                brain.physics_genome = mutant_phys
                brain.mutate(rate=0.3, power=current_mut_rate)
                
                mutant_genome = brain.genome
                mutant_phys = brain.physics_genome

                dcpi_series = brain.calculate_dcpi_series(market_prices, market_vols)
                
                balance = INITIAL_BALANCE
                position = 0
                entry_price = 0
                current_bet = 0
                
                ind_trades = 0
                ind_wins = 0
                ind_losses = 0
                
                for t in range(50, len(inputs)):
                    current_dcpi = dcpi_series[t+1]
                    is_trend_up = trend_up_array[t]

                    hist_start = max(0, t - 60)
                    neural_history = inputs[hist_start:t+1]
                    
                    signal, _ = brain.predict(
                        neural_history, 
                        current_dcpi=current_dcpi, 
                        trend_up=is_trend_up
                    )
                    
                    current_price = prices_aligned[t]
                    
                    THR = 0.6
                    
                    if position != 0:
                        should_close = (signal > THR and position == -1) or (signal < -THR and position == 1)
                        pnl_unrealized = (current_price - entry_price)/entry_price * LEVERAGE * position
                        if pnl_unrealized < -0.2: should_close = True 
                        
                        if should_close:
                            profit = (current_bet * pnl_unrealized) - (current_bet * LEVERAGE * FEE)
                            balance += profit
                            position = 0
                            
                            ind_trades += 1
                            if profit > 0: ind_wins += 1
                            else: ind_losses += 1
                    
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
                    
                    best_batch_trades = ind_trades
                    best_batch_wins = ind_wins
                    best_batch_losses = ind_losses
            
            if best_of_batch_fitness > best_fitness:
                best_fitness = best_of_batch_fitness
                best_genome = best_of_batch_genome
                best_physics_genome = best_of_batch_phys
                
                console.print(f"[bold green]🔥 NEW RECORD Gen {gen}: ${best_fitness:.2f}[/bold green]")
                
                # Guardar el mejor cerebro en caliente
                np.savez("best_brain_v2_vl.npz", genome=best_genome, physics_genome=best_physics_genome)
                console.print(f"[dim]💾 Cerebro guardado en best_brain_v2_vl.npz[/dim]")
                
            # Update Table
            layout["main"].update(create_table(
                gen, best_of_batch_fitness, 
                best_of_batch_phys[0], best_of_batch_phys[1], best_of_batch_phys[2],
                best_batch_trades, best_batch_wins, best_batch_losses
            ))

    print("\n [FIN] VL RESULTADO FINAL:")
    print(f"   K Universal: {best_physics_genome[0]:.4f}")
    print(f"   Alpha Universal: {best_physics_genome[1]:.4f}")
    
    # Guardado Final
    np.savez("best_brain_v2_vl_final.npz", genome=best_genome, physics_genome=best_physics_genome)

if __name__ == "__main__":
    run_evolution()
