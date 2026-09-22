import numpy as np
import matplotlib.pyplot as plt
import time
import random

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL (TU VERSIÓN GANADORA V2)
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=50, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        # Genoma: [x, y, z, Radio, Peso, Sx, Sy, Sz]
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.0, 1.0, 3), # Posición (x,y,z) normalizada
                [np.random.uniform(0.5, 1.5)],   # Radio (Inflado)
                [np.random.uniform(-1.0, 1.0)],  # Peso (Negativo=Short, Positivo=Long)
                np.random.uniform(0.5, 1.5, 3)   # Stretch (Forma)
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def predict(self, points):
        # Inferencia Volumétrica Metamórfica
        if len(self.genome) == 0: return 0.0
        
        centers = self.genome[:, :3]
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8]
        
        # Procesamos el último punto del mercado (Estado actual)
        # points shape: (1, 3) -> [PrecioNorm, Volatilidad, Volumen]
        point = points[-1].reshape(1, 3) 
        
        diff = point - centers # Broadcasting directo
        diff_stretched = diff / (stretch + 1e-6)
        dists = np.linalg.norm(diff_stretched, axis=1)
        
        safe_radii = np.maximum(radii, 1e-6)
        raw_overlap = np.maximum(0, 1 - dists / safe_radii)
        
        # Activación Plateau
        plateau = np.minimum(1.0, raw_overlap * 3.0)
        
        # Suma ponderada de decisiones (Neuronas Long vs Neuronas Short)
        # Si el resultado es > 0.5 -> COMPRA. Si es < -0.5 -> VENTA.
        total_activation = np.sum(plateau * weights)
        
        # Normalizamos la salida entre -1 (Venta Fuerte) y 1 (Compra Fuerte)
        return np.tanh(total_activation)

# ==============================================================================
# 🎰 EL GENERADOR DE MERCADO SINTÉTICO (FRACTAL)
# ==============================================================================

def generate_crypto_market(steps=1000):
    """
    Genera una serie de precios estilo Cripto:
    - Ruido Browniano
    - Saltos de volatilidad (Pumps/Dumps)
    - Tendencias falsas
    """
    price = 1000.0
    prices = [price]
    volumes = []
    volatilities = []
    
    trend = 0
    volatility = 0.002 # 0.2% por tick base
    
    for i in range(steps):
        # Cambio de régimen aleatorio (Paz vs Guerra)
        if np.random.rand() < 0.02:
            volatility = np.random.uniform(0.001, 0.02) # Explosión de volatilidad
            trend = np.random.uniform(-0.01, 0.01) # Cambio de tendencia
            
        # Ruido
        noise = np.random.normal(0, volatility)
        
        # Movimiento
        change = trend + noise
        price = price * (1 + change)
        
        # Volumen correlacionado con la volatilidad (Geometría del panico)
        volume = abs(change) * 10000 + np.random.uniform(0, 500)
        
        prices.append(price)
        volumes.append(volume)
        volatilities.append(volatility)
        
        # Decaimiento de la tendencia (Mean reversion suave)
        trend *= 0.95 
        
    return np.array(prices), np.array(volumes), np.array(volatilities)

# ==============================================================================
# ☠️ SIMULADOR DE SUPERVIVENCIA (ENTRENAMIENTO)
# ==============================================================================

def run_evolution(generations=10000): # Poner 1M si tenés tiempo
    print(f"💀 INICIANDO SIMULACIÓN KAMIKAZE x50 ({generations} Gens)")
    
    # 1. Creamos el mercado (El Escenario)
    market_prices, market_vols, market_volatilities = generate_crypto_market(1000)
    
    # Pre-calcular inputs geométricos para velocidad
    # Input Vector: [Precio_Relativo, Volumen_Norm, Volatilidad_Norm]
    # Normalizamos todo entre -1 y 1 aprox
    p_changes = np.diff(market_prices) / market_prices[:-1]
    inputs = np.column_stack([
        p_changes / 0.05, # Precio normalizado (1% = 1.0)
        market_vols / (np.max(market_vols) + 1e-6), # Tamaño 1000 (Agregué +1e-6 para evitar div por 0)
        market_volatilities / 0.02              # Tamaño 1000
    ])
    
    brain = DynamicCausalBrain(n_neurons=50, A_dc=1.618)
    best_genome = np.copy(brain.genome)
    best_fitness = -np.inf
    
    # CONFIGURACIÓN DEL CASINO
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    METABOLIC_COST = 0.0001 # Pierde 0.01% por tick (Hambre)
    FEE = 0.001 # 0.1% comisión por trade

    # --- AGREGÁ ESTAS DOS LÍNEAS ACÁ ---
    MAX_BET = 100.0     # Límite de apuesta (para que no apueste infinito)
    SLIPPAGE = 0.002    # Factor Realidad: 0.2% de precio peor al entrar
    
    # Mutación adaptativa
    initial_mut = 0.5
    final_mut = 0.01
    
    start_time = time.time()
    
    for gen in range(generations):
        # Tasa de mutación
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        # Mutar
        mutant = np.copy(best_genome)
        mask = np.random.rand(*mutant.shape) < 0.1
        mutant[mask] += np.random.normal(0, mut_rate, size=mutant[mask].shape)
        # Corregir valores físicos
        mutant[:, 3] = np.abs(mutant[:, 3]) # Radios +
        mutant[:, 5:8] = np.abs(mutant[:, 5:8]) # Stretch +
        
        brain.genome = mutant
        
        # --- SIMULACIÓN DE TRADING ---
        balance = INITIAL_BALANCE
        position = 0 # 0=Cash, 1=Long, -1=Short
        entry_price = 0
        trades = 0
        alive = True
        equity_curve = [balance]
        
# --- BUCLE DE TRADING BLINDADO (Max Bet + Slippage) ---
        for t in range(len(inputs)):
            # 1. Costo de vida
            balance -= (balance * METABOLIC_COST)
            
            # 2. Muerte por Hambre
            if balance < 10: 
                alive = False
                break
                
            # 3. Predicción
            signal = brain.predict([inputs[t]]) 
            current_price = market_prices[t]
            
            # 4. Trading
            
            # -- CIERRE DE POSICIONES (Take Profit / Stop Loss) --
            if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
                pnl_pct = 0
                if position == 1: # Cerrar Long
                    # Al salir también podés tener slippage, pero simplifiquemos:
                    pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE
                else: # Cerrar Short
                    pnl_pct = (entry_price - current_price) / entry_price * LEVERAGE
                
                profit = current_bet * pnl_pct
                cost = current_bet * FEE
                balance += (profit - cost)
                
                position = 0
                current_bet = 0
            
            # -- APERTURA DE POSICIONES (ACÁ APLICAMOS EL SLIPPAGE) --
            if position == 0:
                if signal > 0.8: # ABRIR LONG
                    position = 1
                    # REALIDAD: Pagás un poquito más caro de lo que dice la pantalla
                    entry_price = current_price * (1 + SLIPPAGE) 
                    
                    current_bet = min(balance, MAX_BET)
                    balance -= (current_bet * FEE)
                    trades += 1
                    
                elif signal < -0.8: # ABRIR SHORT
                    position = -1
                    # REALIDAD: Vendés un poquito más barato de lo que dice la pantalla
                    entry_price = current_price * (1 - SLIPPAGE)
                    
                    current_bet = min(balance, MAX_BET)
                    balance -= (current_bet * FEE)
                    trades += 1
            
            # 5. LIQUIDATION CHECK
            if position != 0:
                unrealized_pnl = 0
                if position == 1:
                    unrealized_pnl = (current_price - entry_price) / entry_price * LEVERAGE
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price * LEVERAGE
                
                if unrealized_pnl <= -0.9: # Margin Call al -90%
                    balance -= current_bet
                    position = 0
                    current_bet = 0

        # Cierre final de posiciones
        if alive and position != 0:
                    final_price = market_prices[-1]
                    pnl_pct = 0
                    if position == 1:
                        pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE
                    else:
                        pnl_pct = (entry_price - final_price) / entry_price * LEVERAGE
                    
                    # Calculamos ganancia solo sobre lo apostado (current_bet)
                    profit = current_bet * pnl_pct
                    cost = current_bet * FEE
                    balance += (profit - cost)
            
        # --- FITNESS CALCULATION ---
        # --- FITNESS CALCULATION (CORREGIDO) ---
        ticks_survived = t  # 't' es hasta donde llegó el bucle
        
        if not alive:
            # Si murió, su puntaje es cuánto duró. 
            # (Ej: Durar 500 ticks es mejor que durar 10)
            fitness = ticks_survived 
        elif trades < 3:
            # Si sobrevivió pero fue cobarde
            fitness = ticks_survived / 2 
        else:
            # EL PREMIO GORDO: Sobrevivió todo el tiempo + Ganancia
            # Le sumamos 2000 para que siempre sea mejor que cualquier muerto (que maximo saca 1000)
            fitness = 2000 + balance
            
        # Evolución
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = np.copy(mutant)
            print(f"🏆 [Gen {gen}] NUEVO RÉCORD: ${best_fitness:.2f} | Trades: {trades} | Mut: {mut_rate:.3f}")
            
        if gen % 1000 == 0:
            print(f"   ... Simulando Gen {gen} ... (Best: ${best_fitness:.2f})")

    return best_genome, market_prices, market_vols, inputs

# ==============================================================================
# 📊 VISUALIZACIÓN DEL DEPREDADOR
# ==============================================================================

if __name__ == "__main__":
    # Entrenar (Poné 100000 o 1000000 si tenés tiempo)
    winner_genome, prices, vols, inputs = run_evolution(generations=10000) 
    
    print("\n🎥 REPLAY DEL MEJOR BOT EN ACCIÓN...")
    
    # Reconstruimos el mejor cerebro
    brain = DynamicCausalBrain(n_neurons=50, A_dc=1.618)
    brain.genome = winner_genome
    
    # Ejecutamos una vuelta para graficar
    balance = 1000
    position = 0
    actions = [] # Guardar puntos de compra/venta
    equity = []
    
    for t in range(len(inputs)):
        signal = brain.predict([inputs[t]])
        
        # Lógica simplificada para graficar
        action = 0 # 0 nada, 1 buy, -1 sell
        if signal > 0.8: action = 1
        elif signal < -0.8: action = -1
        actions.append(action)
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(prices, label='Precio Mercado', color='black', alpha=0.6)
    
    # Pintar zonas de compra/venta
    buy_idx = [i for i, a in enumerate(actions) if a == 1]
    sell_idx = [i for i, a in enumerate(actions) if a == -1]
    
    ax1.scatter(buy_idx, prices[buy_idx], color='green', marker='^', s=50, label='Long Signal')
    ax1.scatter(sell_idx, prices[sell_idx], color='red', marker='v', s=50, label='Short Signal')
    ax1.set_title("Estrategia Evolucionada ($A_{dc}$ Trading)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graficar Inputs que vio la IA (Volumen y Volatilidad)
    ax2.plot(inputs[:, 1], color='blue', alpha=0.5, label='Volumen (Percibido)')
    ax2.plot(inputs[:, 2], color='orange', alpha=0.5, label='Volatilidad (Percibido)')
    ax2.set_title("Lo que ve la IA (Geometría del Mercado)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()