import numpy as np
import matplotlib.pyplot as plt
import time
import random # <--- FIXED: Agregado import random
import requests # <--- Necesario para hablar con Binance (pip install requests)

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL (VERSIÓN 2 - MINAS ESTÁTICAS)
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        # Estado Bayesiano (Alpha, Beta) por neurona
        # Inicializamos en [10.0, 1.0] (Bias positivo inicial, pero corregible)
        self.neuron_states = np.array([[10.0, 1.0] for _ in range(n_neurons)])
        self.reset_genome()

    def reset_genome(self):
        # Agregamos UN gen más al final: Longitud de Gusano (Length)
        # Genoma: [x, y, z, R, W, Sx, Sy, Sz, ÁNGULO, LONGITUD]
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3), 
                [np.random.uniform(0.5, 1.5)],   
                [np.random.uniform(-1.0, 1.0)],  
                np.random.uniform(0.5, 1.5, 3),
                [np.random.uniform(-np.pi/2, np.pi/2)], # SPIN
                [np.random.randint(3, 40)]              # LENGTH (Worm History)
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)
        # Reset estados al cambiar genoma
        self.neuron_states = np.array([[10.0, 1.0] for _ in range(self.n_neurons)])

    def predict(self, history_points):
        # Inferencia Volumétrica SECUENCIAL CON SPIN 2D 🌪️
        if len(self.genome) == 0: return 0.0
        
        # 1. Desempaquetar
        n = self.genome.shape[0] # Use actual genome size
        centers = self.genome[:, :3]      # x, y, z
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8]     # Sx, Sy, Sz
        thetas = self.genome[:, 8]        # Ángulo
        lengths = self.genome[:, 9].astype(int) # Longitud de memoria
        
        # Confianza Bayesiana: Alpha / (Alpha + Beta)
        # Aseguramos que los estados no excedan n
        alphas = self.neuron_states[:n, 0]
        betas = self.neuron_states[:n, 1]
        confidence = alphas / (alphas + betas)
        
        total_activation = 0.0
        
        # Estadísticas para el usuario
        max_activation = 0.0
        weighted_conf_sum = 0.0
        total_weight_abs = 0.0
        
        # Para cada neurona, escaneamos SU propia ventana de historia
        for i in range(n):
            worm_len = lengths[i]
            # Tomamos los últimos N puntos disponibles
            neuron_history = history_points[-worm_len:] if len(history_points) >= worm_len else history_points
            
            if len(neuron_history) == 0: continue

            # Vectorización local: Rotar toda la trayectoria
            # Matriz de rotación 2D para esta neurona
            c, s = np.cos(thetas[i]), np.sin(thetas[i])
            R_mat = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])
            
            # Centro de la neurona
            center = centers[i]
            
            # Diferencia de toda la trayectoria
            diffs = neuron_history - center
            
            # Rotar diffs
            # diffs es (N, 3). R_mat es (3, 3). Queremos (N, 3).
            # (R @ diffs.T).T  = diffs @ R.T
            rotated_diffs = diffs @ R_mat.T
            
            # Distancia euclidiana ponderada (stretch)
            # dist = sqrt(sum((x/sx)^2))
            norm_diffs = rotated_diffs / stretch[i]
            dists_sq = np.sum(norm_diffs**2, axis=1)
            
            # Encontrar el punto de "mínima distancia" (máxima coincidencia) en la trayectoria
            min_dist_sq = np.min(dists_sq)
            
            # Activación Gaussiana sobre la mejor coincidencia
            # exp(-d^2 / R^2)
            activation = np.exp(-min_dist_sq / (radii[i]**2 + 1e-8))
            
            # Ponderar por Peso y CONFIANZA
            total_activation += activation * weights[i] * confidence[i]
            
            # Tracking Stats
            if activation > max_activation: max_activation = activation
            weighted_conf_sum += confidence[i] * abs(weights[i])
            total_weight_abs += abs(weights[i])
            
        # Guardar stats de la última predicción
        self.last_stats = {
            "compatibility": max_activation,
            "confidence": weighted_conf_sum / (total_weight_abs + 1e-9)
        }
            
        return np.tanh(total_activation)

    def feedback(self, pnl):
        """Actualización Bayesiana del Estado"""
        # Si ganamos, reforzamos neuronas activas. Si perdemos, dudamos.
        # Simplificación: Asumimos que todas contribuyeron (Distributed Representation)
        # O podríamos guardar cuáles se activaron. Por ahora, actualización global suave.
        
        learning_rate = 0.1
        
        if pnl > 0:
            # WIN: Aumentar Alpha (creencia en éxito)
            self.neuron_states[:, 0] += learning_rate
        else:
            # LOSS: Aumentar Beta (creencia en fallo)
            self.neuron_states[:, 1] += learning_rate
            
        # Decay para mantener plasticidad (evitar Alpha -> infinito)
        mask_high = (self.neuron_states[:, 0] + self.neuron_states[:, 1]) > 50.0
        self.neuron_states[mask_high] *= 0.95

# ==============================================================================
# 🌍 EL MERCADO REAL (CONEXIÓN A BINANCE)
# ==============================================================================

# ==============================================================================
# 🌍 MERCADO REAL EXTENDIDO (BAJA 5000 VELAS)
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=5000, custom_end_time=None):
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

    print(f"📡 {symbol}: Bajando {total_candles} velas (EndTime={current_end_time})...")

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
                print("⚠️  Sin mas datos.")
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
            print(f"❌ Error bajando data: {e}")
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

    # Calcular volatilidad simple (High - Low) / Open
    volatilities = (highs - lows) / (opens + 1e-12)
    
    print(f"✅ DATA FINAL: {len(closes)} velas listas.")
    return closes, volumes, volatilities, times

# ==============================================================================
# 📜 TRADE JOURNAL SYSTEM
# ==============================================================================

from datetime import datetime, timezone

def ms_to_utc(ms: int) -> str:
    """Convert millisecond timestamp to UTC ISO string."""
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat(timespec="seconds")

def simulate_with_log(genome, threshold, inputs, prices, times_ms, neuron_states=None, TRADING_MODE="BOTH",
                      INITIAL_BALANCE=10000.0, MAX_BET=3333.0, SLIPPAGE=0.001,
                      FEE=0.0005, LEVERAGE=50.0):
    """
    Run simulation and log all trade events (OPEN/CLOSE/LIQUIDATION) with full context.
    
    Returns:
        balance (float): Final balance
        trades (int): Number of trades executed
        log (list): List of dicts, each representing a trade event
    """
    brain = DynamicCausalBrain(n_neurons=genome.shape[0], A_dc=1.618)
    brain.genome = genome
    
    # Si tenemos estados aprendidos (Alpha/Beta), los usamos
    if neuron_states is not None:
        brain.neuron_states = np.copy(neuron_states)
    else:
        # Si no, arranca de cero (pero aprenderá durante el replay)
        brain.neuron_states = np.array([[10.0, 1.0] for _ in range(brain.n_neurons)])

    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0.0
    entry_t = None
    current_bet = 0.0
    trades = 0
    log = []

    for t in range(len(inputs)):
        # Pasamos la HISTORIA de puntos (hasta t)
        history_start = max(0, t - 60) # Un poco más que el max length (40)
        history_points = inputs[history_start : t+1]
        
        signal = brain.predict(history_points)
        price = float(prices[t])
        ts = int(times_ms[t])

        # --- CLOSE (flip signal) ---
        if (signal > threshold and position == -1) or (signal < -threshold and position == 1):
            pnl_pct = ((price - entry_price) / entry_price * LEVERAGE) if position == 1 else ((entry_price - price) / entry_price * LEVERAGE)
            profit = current_bet * pnl_pct
            fee_exit = (current_bet * LEVERAGE) * FEE

            balance += (profit - fee_exit)
            
            # 🧠 FEEDBACK BAYESIANO
            brain.feedback(profit - fee_exit)

            log.append({
                "t": t, "time": ms_to_utc(ts),
                "event": "CLOSE",
                "pos": position,
                "entry_t": entry_t, "entry_price": entry_price,
                "exit_price": price,
                "signal": float(signal), "thr": float(threshold),
                "p_change": float(inputs[t][0]), "vol_rel": float(inputs[t][1]), "volat": float(inputs[t][2]),
                "bet": float(current_bet),
                "pnl_pct": float(pnl_pct),
                "fee_exit": float(fee_exit),
                "balance": float(balance),
            })

            position = 0
            current_bet = 0.0
            entry_price = 0.0
            entry_t = None

        # --- OPEN ---
        if position == 0:
            if signal > threshold and TRADING_MODE in ["LONG", "BOTH"]:
                position = 1
                entry_price = price * (1 + SLIPPAGE)
                entry_t = t
                current_bet = min(balance, MAX_BET)
                fee_entry = (current_bet * LEVERAGE) * FEE
                balance -= fee_entry
                trades += 1

                log.append({
                    "t": t, "time": ms_to_utc(ts),
                    "event": "OPEN_LONG",
                    "pos": position,
                    "entry_t": entry_t, "entry_price": entry_price,
                    "signal": float(signal), "thr": float(threshold),
                    "p_change": float(inputs[t][0]), "vol_rel": float(inputs[t][1]), "volat": float(inputs[t][2]),
                    "bet": float(current_bet),
                    "fee_entry": float(fee_entry),
                    "balance": float(balance),
                })

            elif signal < -threshold and TRADING_MODE in ["SHORT", "BOTH"]:
                position = -1
                entry_price = price * (1 - SLIPPAGE)
                entry_t = t
                current_bet = min(balance, MAX_BET)
                fee_entry = (current_bet * LEVERAGE) * FEE
                balance -= fee_entry
                trades += 1

                log.append({
                    "t": t, "time": ms_to_utc(ts),
                    "event": "OPEN_SHORT",
                    "pos": position,
                    "entry_t": entry_t, "entry_price": entry_price,
                    "signal": float(signal), "thr": float(threshold),
                    "p_change": float(inputs[t][0]), "vol_rel": float(inputs[t][1]), "volat": float(inputs[t][2]),
                    "bet": float(current_bet),
                    "fee_entry": float(fee_entry),
                    "balance": float(balance),
                })

            if balance <= 100:  # Game Over
                break

        # --- LIQUIDATION ---
        if position != 0:
            unrealized = ((price - entry_price) / entry_price * LEVERAGE) if position == 1 else ((entry_price - price) / entry_price * LEVERAGE)
            if unrealized <= -0.9:
                balance -= current_bet
                
                # 🧠 FEEDBACK BAYESIANO (Caída fuerte)
                brain.feedback(-current_bet)

                log.append({
                    "t": t, "time": ms_to_utc(ts),
                    "event": "LIQUIDATION",
                    "pos": position,
                    "entry_t": entry_t, "entry_price": entry_price,
                    "mark_price": price,
                    "unrealized_pnl": float(unrealized),
                    "bet": float(current_bet),
                    "balance": float(balance),
                })
                position = 0
                current_bet = 0.0
                entry_price = 0.0
                entry_t = None

    # Final close if still in position
    if position != 0:
        price = float(prices[-1])
        ts = int(times_ms[-1])
        pnl_pct = ((price - entry_price) / entry_price * LEVERAGE) if position == 1 else ((entry_price - price) / entry_price * LEVERAGE)
        profit = current_bet * pnl_pct
        fee_exit = (current_bet * LEVERAGE) * FEE
        balance += (profit - fee_exit)
        
        log.append({
            "t": len(inputs)-1, "time": ms_to_utc(ts),
            "event": "FINAL_CLOSE",
            "pos": position,
            "entry_t": entry_t, "entry_price": entry_price,
            "exit_price": price,
            "pnl_pct": float(pnl_pct),
            "fee_exit": float(fee_exit),
            "balance": float(balance),
        })

    return balance, trades, log

# ==============================================================================
# ⚔️ EL COLISEO (ENTRENAMIENTO REALISTA)
# ==============================================================================

def run_evolution(generations=100, population_size=20): # <-- Population Size
    print(f"💀 INICIANDO SIMULACIÓN CON DATOS REALES")
    print(f"👥 POBLACIÓN POR GENERACIÓN: {population_size}")
    
    # MODO DE TRADING: "LONG", "SHORT", "BOTH"
    TRADING_MODE = "BOTH" # <--- CAMBIAR ACÁ
    print(f"🎮 MODO: {TRADING_MODE}")
    
    # 1. Bajamos la realidad
    market_prices, market_vols, market_volatilities, market_times = fetch_extended_data("PEPEUSDT", "1m", 5000)
    
    if len(market_prices) < 100:
        print("Pocos datos, abortando.")
        return None, None, None, None

    # 2. Normalización Robusta (Crucial para que la IA entienda el precio real)
    # Convertimos precio absoluto en % de cambio
    p_changes = np.diff(market_prices) / market_prices[:-1]
    
    # Alineamos arrays (perdemos el primer dato por el diff)
    prices_aligned = market_prices[1:]
    vols_aligned = market_vols[1:]
    volatilities_aligned = market_volatilities[1:]
    times_aligned = market_times[1:]  # Align timestamps too
    
    # Normalizamos inputs entre -1 y 1 aprox
    inputs = np.column_stack([
        p_changes / 0.005,    # Sensibilidad normal (0.5% cambio = 1.0 input)
        vols_aligned / (np.mean(vols_aligned) * 3 + 1e-6), # Volumen relativo a la media
        volatilities_aligned / 0.005 # Volatilidad
    ])
    
    brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_threshold = 0.8  # Init default
    best_fitness = -np.inf
    
    # --- REGLAS DEL JUEGO REAL ---
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0       # Simulamos Spot/x1 primero. Si gana acá, gana en x50.
    MAX_BET = 3333.0      # Apuesta máxima fija
    SLIPPAGE = 0.001     # 0.1% de penalización por entrar a mercado
    FEE = 0.001          # 0.1% comisión Binance (Taker)
    METABOLIC_COST = 0.00005 # Costo de vida bajo
    
    # Mutación Rápida (para ver resultados hoy)
    initial_mut = 0.90   # 10%
    final_mut = 0.001    # 0.1%
    
    start_time = time.time()
    
    for gen in range(generations):
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        # Mutación y Selección en Lote (1+N Hill Climbing)
        best_of_batch_genome = None
        best_of_batch_thr = None
        best_of_batch_fitness = -np.inf
        best_of_batch_stats = {} # Para guardar profit, balance, etc.
        
        # Generar y evaluar N mutantes en paralelo (secuencial en código pero conceptualmente paralelo)
        for _ in range(population_size):
            mutant = np.copy(best_genome)
            mask = np.random.rand(*mutant.shape) < 0.1
            mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
            mutant[:, 3] = np.abs(mutant[:, 3]) # Radios positivos
            mutant[:, 5:8] = np.abs(mutant[:, 5:8]) # Stretch positivo
            
            # Mutación de Umbral
            mutant_threshold = best_threshold + np.random.normal(0, mut_rate * 0.5)
            mutant_threshold = np.clip(mutant_threshold, 0.1, 0.95)
            
            brain.genome = mutant
            
            # --- EVALUAR MUTANTE ---
            balance = INITIAL_BALANCE
            position = 0
            entry_price = 0
            current_bet = 0
            trades = 0
            alive = True
            
            # Bucle de Trading Rápido
            for t in range(len(inputs)):
                # Señal (CON HISTORIA)
                history_start = max(0, t - 60)
                history_points = inputs[history_start : t+1]
                signal = brain.predict(history_points)
                current_price = prices_aligned[t]
                
                # -- CIERRE --
                if (signal > mutant_threshold and position == -1) or \
                   (signal < -mutant_threshold and position == 1):
                    
                    pnl_pct = 0
                    if position == 1:
                        pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price * LEVERAGE
                    
                    notional_fee = (current_bet * LEVERAGE) * FEE 
                    balance += (current_bet * pnl_pct) - notional_fee
                    
                    # 🧠 FEEDBACK
                    brain.feedback((current_bet * pnl_pct) - notional_fee)
                    
                    position = 0
                    current_bet = 0
                
                # -- APERTURA --
                if position == 0:
                    if signal > mutant_threshold and TRADING_MODE in ["LONG", "BOTH"]: # LONG
                        position = 1
                        entry_price = current_price * (1 + SLIPPAGE) 
                        current_bet = min(balance, MAX_BET)
                        entry_fee = (current_bet * LEVERAGE) * FEE
                        balance -= entry_fee
                        trades += 1
                    elif signal < -mutant_threshold and TRADING_MODE in ["SHORT", "BOTH"]: # SHORT
                        position = -1
                        entry_price = current_price * (1 - SLIPPAGE) 
                        current_bet = min(balance, MAX_BET)
                        entry_fee = (current_bet * LEVERAGE) * FEE
                        balance -= entry_fee
                        trades += 1
                    
                    if balance <= 100: # Game Over
                        alive = False
                        break
                
                # -- LIQUIDATION CHECK --
                if position != 0:
                    unrealized_pnl = 0
                    if position == 1:
                        unrealized_pnl = (current_price - entry_price) / entry_price * LEVERAGE
                    else:
                        unrealized_pnl = (entry_price - current_price) / entry_price * LEVERAGE
                    
                    if unrealized_pnl <= -0.9: # Margin Call
                        balance -= current_bet
                        position = 0
                        current_bet = 0
            
            # Cierre final
            if position != 0:
                final_price = prices_aligned[-1]
                pnl_pct = 0
                if position == 1:
                    pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE
                else:
                    pnl_pct = (entry_price - final_price) / entry_price * LEVERAGE
                
                notional_fee = (current_bet * LEVERAGE) * FEE
                balance += (current_bet * pnl_pct) - notional_fee

            # Fitness
            if not alive: fitness = -999999
            else:
                net_profit = balance - INITIAL_BALANCE
                fitness = net_profit
            
            # ¿Es el mejor del lote?
            if fitness > best_of_batch_fitness:
                best_of_batch_fitness = fitness
                best_of_batch_genome = np.copy(mutant)
                best_of_batch_thr = mutant_threshold
                best_of_batch_stats = {
                    'balance': balance, 
                    'trades': trades, 
                    'profit': balance - INITIAL_BALANCE
                }

        # --- SELECCIÓN ---
        # Si el mejor del lote supera al campeón histórico
        if best_of_batch_fitness > best_fitness:
            best_fitness = best_of_batch_fitness
            best_genome = np.copy(best_of_batch_genome)
            best_threshold = best_of_batch_thr
            
            # 📜 Generar log del nuevo campeón
            best_genome_copy = np.copy(best_genome) # Copia para no alterar estado durante log
            # Necesitamos un brain nuevo para el log que arranque con el estado DE ESTE MOMENTO
            # Pero simulate_with_log crea un brain nuevo. 
            # OJO: El estado (Alpha/Beta) está en brain.neuron_states
            # Deberíamos pasar el estado también o dejar que aprenda en el log.
            # Por simplicidad, simulate_with_log usará un brain nuevo que aprenderá desde 0 en el replay.
            
            bal_dbg, tr_dbg, log_dbg = simulate_with_log(
                best_genome, best_threshold,
                inputs, prices_aligned, times_aligned,
                neuron_states=brain.neuron_states, # <-- PASAMOS ESTADO ACTUAL
                TRADING_MODE=TRADING_MODE,
                FEE=FEE, LEVERAGE=LEVERAGE, SLIPPAGE=SLIPPAGE, MAX_BET=MAX_BET
            )
            best_log = log_dbg
            
            print(f"🏆 [Gen {gen}] RÉCORD: Balance ${best_of_batch_stats['balance']:.2f} (Profit: ${best_of_batch_stats['profit']:.2f}) | Trades: {best_of_batch_stats['trades']} | Mut: {mut_rate:.3f} | Thr: {best_threshold:.3f}")

    return best_genome, best_threshold, prices_aligned, vols_aligned, inputs, times_aligned, best_log

# ==============================================================================
# 🧪 HELPERS DE TESTING
# ==============================================================================

def run_simulation_test(genome, threshold, inputs, prices, TRADING_MODE="BOTH"):
    """Simulación pura sin gráficos ni outputs, devuelve métricas."""
    brain = DynamicCausalBrain(n_neurons=len(genome), A_dc=1.618)
    brain.genome = genome
    
    INITIAL_BALANCE = 10000.0
    MAX_BET = 3333.0
    SLIPPAGE = 0.001
    FEE = 0.001
    LEVERAGE = 50.0
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    current_bet = 0
    trades = 0
    actions = []
    trade_log = [] # Lista de diccionarios con detalles de cada trade
    
    # Init entry stats
    entry_stats = {"compatibility": 0, "confidence": 0}
    entry_idx = 0
    
    for t in range(len(inputs)):
        history_start = max(0, t - 60)
        history_points = inputs[history_start : t+1]
        signal = brain.predict(history_points)
        stats = getattr(brain, "last_stats", {"compatibility": 0, "confidence": 0})
        
        current_price = prices[t]
        act = 0
        
        # -- CIERRE --
        if (signal > threshold and position == -1) or \
           (signal < -threshold and position == 1):
            
            pnl_pct = 0
            if position == 1:
                pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE
            else:
                pnl_pct = (entry_price - current_price) / entry_price * LEVERAGE
            
            profit = current_bet * pnl_pct
            cost = (current_bet * LEVERAGE) * FEE 
            balance += (profit - cost)
            
            # 🧠 FEEDBACK
            brain.feedback(profit - cost)
            
            # Registrar Trade Cerrado
            trade_log.append({
                "type": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": current_price,
                "pnl_pct": pnl_pct * 100,
                "profit_usd": profit - cost,
                "entry_time_idx": entry_idx,
                "exit_time_idx": t,
                "compatibility": entry_stats["compatibility"] * 100, # %
                "confidence": entry_stats["confidence"] * 100        # %
            })
            
            position = 0
            current_bet = 0
        
        # -- APERTURA --
        if position == 0:
            if signal > threshold and TRADING_MODE in ["LONG", "BOTH"]: # LONG
                position = 1
                entry_price = current_price * (1 + SLIPPAGE) 
                current_bet = min(balance, MAX_BET)
                entry_fee = (current_bet * LEVERAGE) * FEE
                entry_stats = stats.copy()
                entry_idx = t
                balance -= entry_fee
                trades += 1
                act = 1

            elif signal < -threshold and TRADING_MODE in ["SHORT", "BOTH"]: # SHORT
                position = -1
                entry_price = current_price * (1 - SLIPPAGE) 
                current_bet = min(balance, MAX_BET)
                entry_fee = (current_bet * LEVERAGE) * FEE
                entry_stats = stats.copy()
                entry_idx = t
                balance -= entry_fee
                trades += 1
                act = -1
        
        # -- LIQUIDATION CHECK --
        if position != 0:
            unrealized_pnl = 0
            if position == 1:
                unrealized_pnl = (current_price - entry_price) / entry_price * LEVERAGE
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price * LEVERAGE
            
            if unrealized_pnl <= -0.9: # Margin Call
                balance -= current_bet
                
                # Registrar Liquidación
                trade_log.append({
                    "type": "LIQUIDATION",
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": -100.0,
                    "profit_usd": -current_bet,
                    "entry_time_idx": entry_idx,
                    "exit_time_idx": t,
                    "compatibility": entry_stats["compatibility"] * 100,
                    "confidence": entry_stats["confidence"] * 100
                })

                # 🧠 FEEDBACK
                brain.feedback(-current_bet)
                
                position = 0
                current_bet = 0

        actions.append(act)

    # Cierre final al marcar precio de mercado
    if position != 0:
        final_price = prices[-1]
        pnl_pct = 0
        if position == 1:
            pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE
        else:
            pnl_pct = (entry_price - final_price) / entry_price * LEVERAGE
        
        notional_fee = (current_bet * LEVERAGE) * FEE # <-- FIXED: Consistent fee
        balance += (current_bet * pnl_pct) - notional_fee
        
        # Registrar Trade Final (Forzado por fin de simulación)
        trade_log.append({
            "type": "LONG (END)" if position == 1 else "SHORT (END)",
            "entry_price": entry_price,
            "exit_price": final_price,
            "pnl_pct": pnl_pct * 100,
            "profit_usd": (current_bet * pnl_pct) - notional_fee,
            "entry_time_idx": entry_idx,
            "exit_time_idx": len(inputs)-1,
            "compatibility": entry_stats["compatibility"] * 100,
            "confidence": entry_stats["confidence"] * 100
        })

    return balance, trades, actions, trade_log

if __name__ == "__main__":
    # Ajustá 'generations' según tu paciencia (mínimo 2000 para que aprenda algo)
    winner, winner_thr, prices, vols, inputs, times, best_log = run_evolution(generations=100)
    
    if winner is not None:
        print("\n[+] GUARDANDO GENOMA GANADOR...")
        
        # Timestamp sin año: MM-DD-HH-MM
        ts = time.strftime("%m-%d-%H-%M")
        
        # Guardar con timestamp
        np.save(f"best_genome_spin_hard_{ts}.npy", winner)
        np.save(f"best_threshold_spin_hard_{ts}.npy", winner_thr)
        
        # Guardar tambien como 'latest' para facilitar el test rápido
        np.save("best_genome_spin_hard.npy", winner)
        np.save("best_threshold_spin_hard.npy", winner_thr)
        
        print(f"   [+] Archivo guardado: best_genome_spin_hard_{ts}.npy (y latest)")
        print(f"   [+] Umbral guardado: {winner_thr:.3f}")
        
        # Trade Log
        if best_log:
            import json
            with open(f"best_trades_log_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(best_log, f, ensure_ascii=False, indent=2)
            # Guardar latest
            with open("best_trades_log.json", "w", encoding="utf-8") as f:
                json.dump(best_log, f, ensure_ascii=False, indent=2)
                
            print(f"   [+] Trade log guardado: best_trades_log_{ts}.json")

        # ----------------------------------------------------------------------
        # FASE 1: REPLAY (In-Sample)
        # ----------------------------------------------------------------------
        print("\n" + "="*60)
        print("🎥 FASE 1: REPLAY (IN-SAMPLE) - LO QUE YA VIO")
        print("="*60)
        
        # Usamos los datos 'prices', 'inputs' que retorna run_evolution (son los de entrenamiento)
        # OJO: run_evolution devuelve inputs normalizados, prices alineados.
        
        # Graficamos el Replay
        brain = DynamicCausalBrain(n_neurons=len(winner), A_dc=1.618)
        brain.genome = winner
        # Reset states for winner
        brain.neuron_states = np.array([[10.0, 1.0] for _ in range(brain.n_neurons)])
        
        actions = []
        TRADING_MODE = "BOTH" # Ajustar si se quiere visualización distinta
        
        for t in range(len(inputs)):
            history_start = max(0, t - 60)
            history_points = inputs[history_start : t+1]
            signal = brain.predict(history_points)
            act = 0
            if signal > winner_thr and TRADING_MODE in ["LONG", "BOTH"]: act = 1
            elif signal < -winner_thr and TRADING_MODE in ["SHORT", "BOTH"]: act = -1
            actions.append(act)
            
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prices, label='BTC Real', color='black', alpha=0.6)
        buys = [i for i, x in enumerate(actions) if x == 1]
        sells = [i for i, x in enumerate(actions) if x == -1]
        if buys: ax.scatter(buys, prices[buys], color='green', marker='^', s=30, label='Compra')
        if sells: ax.scatter(sells, prices[sells], color='red', marker='v', s=30, label='Venta')
        ax.set_title("FASE 1: Replay In-Sample")
        ax.legend()
        # plt.show() # Bloqueante <--- DESACTIVADO POR PEDIDO
        print("✅ Gráfico generado (pero no mostrado para no bloquear)")

        # ----------------------------------------------------------------------
        # FASE 2: TEST BLINDADO (10 Escenarios Random desde 2024)
        # ----------------------------------------------------------------------
        print("\n" + "="*60)
        print("🎲 FASE 2: TEST CIEGO (10 ESCENARIOS RANDOM 2024-PRESENTE)")
        print("="*60)
        
        # Configurar fechas
        start_2024_ms = int(time.mktime(time.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
        now_ms = int(time.time() * 1000)
        
        # Asumimos que inputs.shape[0] es la longitud del entrenamiento. Usamos esa misma para el test.
        TEST_SIZE = len(inputs) 
        
        blind_results = []
        
        for i in range(1, 11):
            print(f"\\n🧪 TEST CIEGO {i}/10:")
            try:
                # FIX: high is out of bounds for int32. Usamos int64 o standard python random.
                random_time = random.randint(start_2024_ms, now_ms)
                
                # 1. Bajar Data Random (mismo tamaño que el entreno)
                t_prices, t_vols, t_volats, t_times = fetch_extended_data("PEPEUSDT", "1m", total_candles=len(prices), custom_end_time=random_time)
                
                if t_prices is None or len(t_prices) < 100:
                    print("⚠️ Datos insuficientes para test ciego.")
                    continue
                    
                # 2. Preprocesar Inputs (Igual que en run_evolution)
                # OJO: Necesitamos replicar la lógica de normalización exacta.
                # Lo ideal sería sacar la lógica de normalización a una función 'prepare_inputs'.
                # Por ahora la copiamos para no romper todo el refactor.
                
                t_p_changes = np.diff(t_prices) / t_prices[:-1]
                t_prices_aligned = t_prices[1:]
                t_vols_aligned = t_vols[1:]
                t_volatilities_aligned = t_volats[1:]
                
                t_inputs = np.column_stack([
                    t_p_changes / 0.005,
                    t_vols_aligned / (np.mean(t_vols_aligned) * 3 + 1e-6),
                    t_volatilities_aligned / 0.005
                ])
                
                # 3. Correr Test
                bal, trades, acts = run_simulation_test(winner, winner_thr, t_inputs, t_prices_aligned, TRADING_MODE="BOTH")
                
                profit_pct = ((bal - 10000) / 10000) * 100
                blind_results.append(profit_pct)
                
                fecha_fin = time.strftime('%Y-%m-%d', time.localtime(random_time/1000))
                print(f"   📅 Fin: {fecha_fin} | 💰 Bal: ${bal:,.0f} ({profit_pct:+.1f}%) | 🤝 Trades: {trades}")
                
            except Exception as e:
                print(f"❌ Error en test {i}: {e}")

        if blind_results:
            print(f"\\n📊 PROMEDIO TEST CIEGOS: {np.mean(blind_results):+.2f}%")

        # ----------------------------------------------------------------------
        # FASE 3: REPLAY FINAL (Consistency Check)
        # ----------------------------------------------------------------------
        print("\\n" + "="*60)
        print("🔄 FASE 3: REPLAY FINAL (CONSISTENCY CHECK)")
        print("="*60)
        
        # Volver a correr run_simulation_test con los inputs originales del entrenamiento
        bal_final, tr_final, acts_final = run_simulation_test(winner, winner_thr, inputs, prices, TRADING_MODE="BOTH")
        prof_final = ((bal_final - 10000)/10000)*100
        print(f"✅ Resultado Replay Final: Balance ${bal_final:,.2f} ({prof_final:+.2f}%)")