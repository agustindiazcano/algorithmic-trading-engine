import numpy as np
import matplotlib.pyplot as plt
import time
import requests # <--- Necesario para hablar con Binance (pip install requests)

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL (VERSIÓN 2 - MINAS ESTÁTICAS)
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        # Agregamos UN gen más al final: El Ángulo (Theta)
        # Genoma: [x, y, z, R, W, Sx, Sy, Sz, ÁNGULO]
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3), 
                [np.random.uniform(0.5, 1.5)],   
                [np.random.uniform(-1.0, 1.0)],  
                np.random.uniform(0.5, 1.5, 3),
                [np.random.uniform(-np.pi/2, np.pi/2)] # <--- SPIN: Rotación +/- 0.5 radianes (aprox 30 grados)
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def predict(self, points):
            # Inferencia Volumétrica CON SPIN 🌪️
            if len(self.genome) == 0: return 0.0
            
            # 1. Desempaquetar el genoma (Ahora tiene 9 genes, el último es el ángulo)
            centers = self.genome[:, :3]      # x, y, z
            radii = self.genome[:, 3] * self.A_dc
            weights = self.genome[:, 4]
            stretch = self.genome[:, 5:8]     # Sx, Sy, Sz
            thetas = self.genome[:, 8]        # <--- Ángulo de Rotación (Spin)
            
            # 2. DEFINIR EL PUNTO (Esta es la línea que faltaba)
            # Tomamos el último input del mercado y le damos forma (1, 3)
            point = points[-1].reshape(1, 3) 
            
            # 3. Calcular la diferencia cruda (Distancia lineal)
            diff = point - centers 
            
            # 4. APLICAR MATRIZ DE ROTACIÓN 2D (Solo en ejes 0 y 1: Precio y Volumen)
            # Esto permite que la neurona se "incline" para encajar mejor
            c, s = np.cos(thetas), np.sin(thetas)
            
            # Fórmula de rotación vectorial:
            # x_new = x * cos - y * sin
            # y_new = x * sin + y * cos
            dx = diff[:, 0] * c - diff[:, 1] * s
            dy = diff[:, 0] * s + diff[:, 1] * c
            dz = diff[:, 2] # El tercer eje lo dejamos quieto (eje de pivote)
            
            # Reconstruimos el vector de diferencia, pero ahora ROTADO
            diff_rotated = np.stack([dx, dy, dz], axis=1)
            
            # 5. Medimos distancia sobre el vector rotado (Invariancia de Spin)
            diff_stretched = diff_rotated / (stretch + 1e-6)
            dists = np.linalg.norm(diff_stretched, axis=1)
            
            # 6. Activación (Igual que antes)
            safe_radii = np.maximum(radii, 1e-6)
            raw_overlap = np.maximum(0, 1 - dists / safe_radii)
            
            # Plateau
            plateau = np.minimum(1.0, raw_overlap * 3.0)
            
            # Suma ponderada
            total_activation = np.sum(plateau * weights)
            return np.tanh(total_activation)

# ==============================================================================
# 🌍 EL MERCADO REAL (CONEXIÓN A BINANCE)
# ==============================================================================

# ==============================================================================
# 🌍 MERCADO REAL EXTENDIDO (BAJA 5000 VELAS)
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=5000):
    """
    Baja MUCHA data haciendo múltiples llamadas a la API.
    """
    print(f"📡 Conectando a Binance: Bajando {total_candles} velas de {symbol} ({interval})...")
    
    limit_per_call = 1000
    all_closes = []
    all_volumes = []
    all_highs = []
    all_lows = []
    all_opens = []
    
    # Binance necesita un 'endTime' para ir hacia atrás. Empezamos desde AHORA.
    current_end_time = int(time.time() * 1000) 
    
    # Hacemos 5 llamadas de 1000 velas cada una hacia atrás
    calls = total_candles // limit_per_call
    
    for i in range(calls):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit_per_call}&endTime={current_end_time}"
        try:
            response = requests.get(url)
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                break
                
            # Ordenamos los datos (vienen del más viejo al más nuevo de ese bloque)
            # Pero como vamos hacia atrás, tenemos que insertar al principio o concatenar inteligentemente
            # Lo más fácil: Guardamos bloques y después ordenamos todo por tiempo.
            
            # Extraemos data
            closes = [float(x[4]) for x in data]
            opens = [float(x[1]) for x in data]
            highs = [float(x[2]) for x in data]
            lows = [float(x[3]) for x in data]
            volumes = [float(x[5]) for x in data]
            close_times = [int(x[6]) for x in data]
            
            # Agregamos al inicio de nuestra lista maestra (porque vamos hacia el pasado)
            all_closes = closes + all_closes
            all_volumes = volumes + all_volumes
            all_highs = highs + all_highs
            all_lows = lows + all_lows
            all_opens = opens + all_opens
            
            # Actualizamos el end_time para la próxima llamada (el tiempo de apertura de la primera vela de este lote - 1ms)
            current_end_time = int(data[0][0]) - 1
            
            print(f"   ... Lote {i+1}/{calls} descargado. Total actual: {len(all_closes)}")
            time.sleep(0.5) # Respetar límites de API
            
        except Exception as e:
            print(f"❌ Error bajando lote: {e}")
            break

    # Convertir a numpy
    closes = np.array(all_closes)
    opens = np.array(all_opens)
    highs = np.array(all_highs)
    lows = np.array(all_lows)
    volumes = np.array(all_volumes)
    
    # Volatilidad Real
    volatilities = (highs - lows) / opens
    
    print(f"✅ DATA FINAL: {len(closes)} velas listas para la guerra.")
    return closes, volumes, volatilities

# ==============================================================================
# ⚔️ EL COLISEO (ENTRENAMIENTO REALISTA)
# ==============================================================================

def run_evolution(generations=100000): # Poner 10000+ para serio
    print(f"💀 INICIANDO SIMULACIÓN CON DATOS REALES")
    
    # 1. Bajamos la realidad
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", 5000)
    
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
    
    # Normalizamos inputs entre -1 y 1 aprox
    inputs = np.column_stack([
        p_changes / 0.005,    # Sensibilidad normal (0.5% cambio = 1.0 input)
        vols_aligned / (np.mean(vols_aligned) * 3 + 1e-6), # Volumen relativo a la media
        volatilities_aligned / 0.005 # Volatilidad
    ])
    
    brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_fitness = -np.inf
    
    # --- REGLAS DEL JUEGO REAL ---
    INITIAL_BALANCE = 1000.0
    LEVERAGE = 100.0       # Simulamos Spot/x1 primero. Si gana acá, gana en x50.
    MAX_BET = 200.0      # Apuesta máxima fija
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
        
        # Mutación
        mutant = np.copy(best_genome)
        mask = np.random.rand(*mutant.shape) < 0.1
        mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
        mutant[:, 3] = np.abs(mutant[:, 3]) # Radios positivos
        mutant[:, 5:8] = np.abs(mutant[:, 5:8]) # Stretch positivo
        
        brain.genome = mutant
        
        # --- BUCLE DE TRADING BLINDADO ---
        balance = INITIAL_BALANCE
        position = 0
        entry_price = 0
        current_bet = balance * 1.00
        trades = 0
        alive = True
        
        for t in range(len(inputs)):
            # Costo de vida
            balance -= (balance * METABOLIC_COST)
            if balance < 10: 
                alive = False
                break
                
            # Predicción
            signal = brain.predict([inputs[t]]) 
            current_price = prices_aligned[t]
            
            # -- CIERRE --
            if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
                pnl_pct = 0
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * LEVERAGE
                
                profit = current_bet * pnl_pct
                cost = current_bet * FEE
                balance += (profit - cost)
                position = 0
                current_bet = 0
            
            # -- APERTURA --
            if position == 0:
                if signal > 0.8: # LONG
                    position = 1
                    entry_price = current_price * (1 + SLIPPAGE) # Entrás caro
                    current_bet = min(balance, MAX_BET)
                    balance -= (current_bet * FEE)
                    trades += 1
                elif signal < -0.8: # SHORT
                    position = -1
                    entry_price = current_price * (1 - SLIPPAGE) # Vendés barato
                    current_bet = min(balance, MAX_BET)
                    balance -= (current_bet * FEE)
                    trades += 1
            
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
        if alive and position != 0:
            final_price = prices_aligned[-1]
            pnl_pct = 0
            if position == 1:
                pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE
            else:
                pnl_pct = (entry_price - final_price) / entry_price * LEVERAGE
            balance += (current_bet * pnl_pct) - (current_bet * FEE)

        # Fitness
        if not alive: fitness = 0
        elif trades < 3: fitness = 500 # Castigo por inactividad
        else: fitness = 2000 + balance
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = np.copy(mutant)
            net_profit = balance - INITIAL_BALANCE
            print(f"🏆 [Gen {gen}] RÉCORD: Balance ${balance:.2f} (Profit: ${net_profit:.2f}) | Trades: {trades} | Mut: {mut_rate:.3f}")

    return best_genome, prices_aligned, vols_aligned, inputs

# ==============================================================================
# 📊 VEREDICTO VISUAL
# ==============================================================================

if __name__ == "__main__":
    # Ajustá 'generations' según tu paciencia (mínimo 2000 para que aprenda algo)
    winner, prices, vols, inputs = run_evolution(generations=10000)
    
    if winner is not None:
        print("\n🎥 REPLAY EN DATA REAL...")
        brain = DynamicCausalBrain(n_neurons=10, A_dc=1.618)
        brain.genome = winner
        
        actions = []
        for t in range(len(inputs)):
            signal = brain.predict([inputs[t]])
            act = 0
            if signal > 0.8: act = 1
            elif signal < -0.8: act = -1
            actions.append(act)
            
        # Graficamos
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prices, label='BTC Real', color='black', alpha=0.6)
        
        buys = [i for i, x in enumerate(actions) if x == 1]
        sells = [i for i, x in enumerate(actions) if x == -1]
        
        if buys: ax.scatter(buys, prices[buys], color='green', marker='^', s=30, label='Compra')
        if sells: ax.scatter(sells, prices[sells], color='red', marker='v', s=30, label='Venta')
        
        ax.set_title("Resultados en Bitcoin Real (1m) - Test de Batalla")
        ax.legend()
        plt.show()