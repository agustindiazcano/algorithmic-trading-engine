import numpy as np
import matplotlib.pyplot as plt
import time
import requests 
import random # <--- Necesario para elegir la fecha aleatoria
import sys

# Forzar UTF-8 en consola para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURACIÓN DE TRADING
TRADING_MODE = "BOTH"  # Opciones: "LONG", "SHORT", "BOTH"

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        # Sistema de energía metabólica (separado del balance)
        self.energy = 100.0
        self.MAX_ENERGY = 200.0
        # GEN DE PERSONALIDAD: Umbral de decisión (valentía/cautela)
        # 0.3 = Muy agresivo, 0.9 = Muy conservador
        self.decision_threshold = np.random.uniform(0.3, 0.9)
        self.reset_genome()

    def reset_genome(self):
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3), 
                [np.random.uniform(0.5, 1.5)],   
                [np.random.uniform(-1.0, 1.0)],  
                np.random.uniform(0.5, 1.5, 3),
                [np.random.uniform(-np.pi/2, np.pi/2)] 
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def predict(self, points):
            if len(self.genome) == 0: return 0.0
            
            centers = self.genome[:, :3]      
            radii = self.genome[:, 3] * self.A_dc
            weights = self.genome[:, 4]
            stretch = self.genome[:, 5:8]     
            thetas = self.genome[:, 8]        
            
            point = points[-1].reshape(1, 3) 
            diff = point - centers 
            
            c, s = np.cos(thetas), np.sin(thetas)
            dx = diff[:, 0] * c - diff[:, 1] * s
            dy = diff[:, 0] * s + diff[:, 1] * c
            dz = diff[:, 2] 
            
            diff_rotated = np.stack([dx, dy, dz], axis=1)
            diff_stretched = diff_rotated / (stretch + 1e-6)
            dists = np.linalg.norm(diff_stretched, axis=1)
            
            safe_radii = np.maximum(radii, 1e-6)
            raw_overlap = np.maximum(0, 1 - dists / safe_radii)
            plateau = np.minimum(1.0, raw_overlap * 3.0)
            total_activation = np.sum(plateau * weights)
            
            return np.tanh(total_activation)

# ==============================================================================
# 🌍 MERCADO REAL (MODIFICADO PARA ACEPTAR FECHAS)
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=20000, custom_end_time=None):
    """
    Si custom_end_time es None, baja datos desde AHORA hacia atrás.
    Si se pone un timestamp, baja datos desde esa fecha hacia atrás.
    """
    # Si no hay fecha específica, usamos el tiempo actual
    if custom_end_time is None:
        current_end_time = int(time.time() * 1000)
        print(f"📡 Bajando {total_candles} velas recientes de {symbol}...")
    else:
        current_end_time = int(custom_end_time)
        # Convertimos a fecha legible para mostrar en consola
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
            response = requests.get(url)
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                print("   ⚠️ No hay más datos en Binance para esta fecha.")
                break
                
            closes = [float(x[4]) for x in data]
            opens = [float(x[1]) for x in data]
            highs = [float(x[2]) for x in data]
            lows = [float(x[3]) for x in data]
            volumes = [float(x[5]) for x in data]
            
            all_closes = closes + all_closes
            all_volumes = volumes + all_volumes
            all_highs = highs + all_highs
            all_lows = lows + all_lows
            all_opens = opens + all_opens
            
            # Actualizamos el tiempo para ir más atrás
            current_end_time = int(data[0][0]) - 1
            time.sleep(0.2) 
            
        except Exception as e:
            print(f"❌ Error bajando lote: {e}")
            break

    closes = np.array(all_closes)
    opens = np.array(all_opens)
    highs = np.array(all_highs)
    lows = np.array(all_lows)
    volumes = np.array(all_volumes)
    
    # Evitar división por cero en volatilidad
    with np.errstate(divide='ignore', invalid='ignore'):
        volatilities = (highs - lows) / opens
        volatilities = np.nan_to_num(volatilities)
    
    print(f"✅ DATA CARGADA: {len(closes)} velas.")
    return closes, volumes, volatilities

# ==============================================================================
# 🛠️ PREPARACIÓN DE DATOS
# ==============================================================================

def prepare_inputs(prices, vols, volatilities):
    if len(prices) < 2: return None, None
    
    p_changes = np.diff(prices) / prices[:-1]
    prices_aligned = prices[1:]
    vols_aligned = vols[1:]
    volatilities_aligned = volatilities[1:]
    
    # Normalización simple
    inputs = np.column_stack([
        p_changes / 0.005,    
        vols_aligned / (np.mean(vols_aligned) * 3 + 1e-6), 
        volatilities_aligned / 0.005 
    ])
    return inputs, prices_aligned

# ==============================================================================
# ⚔️ ENTRENAMIENTO (EVOLUCIÓN)
# ==============================================================================

def run_evolution(generations=10, population_size=50): 
    print(f"\n💀 --- FASE 1: ENTRENAMIENTO (DATOS RECIENTES) ---")
    
    # 1. Bajamos datos recientes para entrenar
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 5000)
    inputs, prices_aligned = prepare_inputs(market_prices, market_vols, market_volatilities)
    
    if inputs is None: return None

    brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_threshold = brain.decision_threshold  # Guardar umbral inicial
    best_fitness = -np.inf
    
    # Parámetros
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10     # 10% del capital por trade
    SLIPPAGE = 0.001     
    FEE = 0.001
    
    # Sistema de energía (separado del balance)
    METABOLIC_COST_FIXED = 0.01  # Energía perdida por vela
    ENERGY_FROM_PROFIT = 0.1     # % de profit que se convierte en energía 
    
    initial_mut = 0.90   
    final_mut = 0.001    
    
    for gen in range(generations):
        # Guardado periódico
        if gen % 1000 == 0 and gen > 0:
            np.save(f"best_genome_gen_{gen}.npy", best_genome)
            print(f"💾 [AUTO-SAVE] Gen {gen} guardada.")

        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        print(f"\n--- GENERACIÓN {gen+1}/{generations} (Mutación: {mut_rate:.4f}) ---")
        
        gen_best_fitness = -np.inf
        gen_best_genome = None

        # Evaluar población
        for i in range(population_size):
            # Crear cerebro
            brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
            
            # ELITISMO: El campeón pasa directo (sin mutación)
            if i == 0 and gen > 0:
                mutant = np.copy(best_genome)  # Sin mutación
                brain.decision_threshold = best_threshold  # Heredar umbral
            else:
                # Mutar al mejor
                mutant = np.copy(best_genome)
                mask = np.random.rand(*mutant.shape) < 0.1
                mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
                mutant[:, 3] = np.abs(mutant[:, 3]) 
                mutant[:, 5:8] = np.abs(mutant[:, 5:8])
                
                # Mutar umbral de decisión (gen de personalidad)
                brain.decision_threshold = best_threshold + np.random.normal(0, mut_rate * 0.5)
                brain.decision_threshold = np.clip(brain.decision_threshold, 0.3, 0.9) 
            
            brain.genome = mutant
        
            balance = INITIAL_BALANCE
            position = 0
            entry_price = 0
            current_bet = balance * 1.00
            trades = 0
            alive = True
        
            for t in range(len(inputs)):
                # Metabolismo: afecta energía, NO balance
                brain.energy -= METABOLIC_COST_FIXED
                if brain.energy <= 0:
                    alive = False
                    break
                
                # Muerte por bancarrota
                if balance <= 0:
                    alive = False
                    break
                    
                signal = brain.predict([inputs[t]]) 
                current_price = prices_aligned[t]
                
                # Cierre
                if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
                    pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 else (entry_price - current_price) / entry_price * LEVERAGE
                    profit = current_bet * pnl_pct
                    cost = (current_bet * LEVERAGE) * FEE 
                    balance += current_bet + (profit - cost)  # Devolver margen + P&L
                    
                    # Recuperar energía con ganancias
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
                        current_bet = balance * BET_PERCENTAGE  # 10% del capital
                        balance -= current_bet  # Bloquear margen
                        balance -= (current_bet * LEVERAGE) * FEE  # Fee de apertura
                        trades += 1
                    elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]: 
                        position = -1
                        entry_price = current_price * (1 - SLIPPAGE) 
                        current_bet = balance * BET_PERCENTAGE  # 10% del capital
                        balance -= current_bet  # Bloquear margen
                        balance -= (current_bet * LEVERAGE) * FEE  # Fee de apertura
                        trades += 1
                
                # Liquidación
                if position != 0:
                    unrealized = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 else (entry_price - current_price) / entry_price * LEVERAGE
                    if unrealized <= -0.9: 
                        balance -= current_bet
                        position = 0
                        current_bet = 0
        
            # Cierre final forzado para calcular fitness
            if alive and position != 0:
                final_price = prices_aligned[-1]
                pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE if position == 1 else (entry_price - final_price) / entry_price * LEVERAGE
                profit = current_bet * pnl_pct
                cost = (current_bet * LEVERAGE) * FEE  # Fee consistente con cierres normales
                balance += current_bet + (profit - cost)  # Devolver margen + P&L

            if not alive: fitness = 0
            elif trades < 3: fitness = 500 
            else: fitness = 2000 + balance
            
            # Mejor de esta generación
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_genome = np.copy(mutant)
                gen_best_threshold = brain.decision_threshold  # Guardar umbral ganador
        
        # Actualizar mejor global
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = np.copy(gen_best_genome)
            best_threshold = gen_best_threshold
            net_profit = balance - INITIAL_BALANCE
            print(f"🏆 MEJOR DE GEN {gen}: Balance ${balance:.2f} | Trades: {trades} | Energía: {brain.energy:.1f} | Umbral: {best_threshold:.2f}")

    np.save("best_genome_FINAL.npy", best_genome)
    np.save("best_threshold_FINAL.npy", best_threshold)  # Guardar umbral
    return best_genome, best_threshold  # Retornar ambos

# ==============================================================================
# 🧪 TEST DE BATALLA (OUT OF SAMPLE)
# ==============================================================================

def run_test_simulation(genome, threshold, inputs, prices):
    """
    Ejecuta el genoma SIN mutación sobre datos nuevos.
    """
    brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
    brain.genome = genome
    brain.decision_threshold = threshold  # Usar umbral evolucionado
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10     # 10% del capital por trade
    SLIPPAGE = 0.001
    FEE = 0.001
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    actions = [] # Para graficar: 1=Buy, -1=Sell, 0=Hold
    equity_curve = []
    
    print("\n⏯️  Ejecutando simulación...")
    
    for t in range(len(inputs)):
        # Muerte por bancarrota
        if balance <= 0:
            break
        
        signal = brain.predict([inputs[t]])
        current_price = prices[t]
        act = 0
        
        # Cierre
        if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 else (entry_price - current_price) / entry_price * LEVERAGE
            profit = current_bet * pnl_pct
            cost = (current_bet * LEVERAGE) * FEE 
            balance += current_bet + (profit - cost)  # Devolver margen + P&L
            position = 0
            current_bet = 0
        
        # Apertura
        if position == 0:
            if signal > brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]: 
                position = 1
                entry_price = current_price * (1 + SLIPPAGE) 
                current_bet = balance * BET_PERCENTAGE  # 10% del capital
                balance -= current_bet  # Bloquear margen
                balance -= (current_bet * LEVERAGE) * FEE  # Fee de apertura
                act = 1
            elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]: 
                position = -1
                entry_price = current_price * (1 - SLIPPAGE) 
                current_bet = balance * BET_PERCENTAGE  # 10% del capital
                balance -= current_bet  # Bloquear margen
                balance -= (current_bet * LEVERAGE) * FEE  # Fee de apertura
                act = -1
        
        # Liquidación
        if position != 0:
            unrealized = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 else (entry_price - current_price) / entry_price * LEVERAGE
            if unrealized <= -0.9: 
                balance -= current_bet
                position = 0
                current_bet = 0
        
        actions.append(act)
        equity_curve.append(balance)

    return balance, actions, equity_curve

# ==============================================================================
# 🚀 MAIN: ENTRENAMIENTO + TEST ALEATORIO
# ==============================================================================

if __name__ == "__main__":
    # 1. ENTRENAR (Datos recientes)
    # Ajusta generations a lo que necesites (ej. 5000)
    winner_genome, winner_threshold = run_evolution(generations=10)
    
    if winner_genome is not None:
        print("\n" + "="*60)
        print("🎲 --- FASE 2: TEST CIEGO (OUT OF SAMPLE) ---")
        print("="*60)
        
        # 2. CALCULAR FECHA ALEATORIA (Últimos 12 meses)
        # 1 año en milisegundos = 365 * 24 * 60 * 60 * 1000
        ms_per_year = 31536000000 
        now_ms = int(time.time() * 1000)
        
        # Definimos un rango: Entre hace 1 año y hace 20 días (para no solapar con el entrenamiento reciente)
        buffer_ms = 20 * 24 * 60 * 60 * 1000 # 20 días
        
        random_past_time = random.randint(now_ms - ms_per_year, now_ms - buffer_ms)
        
        # 3. BAJAR DATOS DEL PASADO
        test_prices, test_vols, test_volatilities = fetch_extended_data(
            "PEPEUSDT", "1m", 5000, custom_end_time=random_past_time
        )
        
        test_inputs, test_prices_aligned = prepare_inputs(test_prices, test_vols, test_volatilities)
        
        if test_inputs is not None:
            # 4. EJECUTAR TEST
            final_bal, acts, equity = run_test_simulation(winner_genome, winner_threshold, test_inputs, test_prices_aligned)
            
            profit_pct = ((final_bal - 10000) / 10000) * 100
            print(f"\n📊 RESULTADO DEL TEST:")
            print(f"   Balance Inicial: $10,000")
            print(f"   Balance Final:   ${final_bal:.2f}")
            print(f"   Profit/Loss:     {profit_pct:.2f}%")
            
            # 5. GRAFICAR
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Gráfico de Precio
            ax1.plot(test_prices_aligned, label='Precio PEPE (Test)', color='black', alpha=0.6)
            buys = [i for i, x in enumerate(acts) if x == 1]
            sells = [i for i, x in enumerate(acts) if x == -1]
            if buys: ax1.scatter(buys, test_prices_aligned[buys], color='green', marker='^', s=50, label='Buy')
            if sells: ax1.scatter(sells, test_prices_aligned[sells], color='red', marker='v', s=50, label='Sell')
            ax1.set_title(f"Test Aleatorio (Hace {(now_ms - random_past_time)/86400000:.1f} días) - Precio")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de Equity (Balance)
            ax2.plot(equity, label='Equity Curve', color='blue')
            ax2.axhline(y=10000, color='r', linestyle='--', alpha=0.5)
            ax2.set_title(f"Curva de Capital (Final: ${final_bal:.2f})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        else:
            print("❌ Error preparando datos de test.")