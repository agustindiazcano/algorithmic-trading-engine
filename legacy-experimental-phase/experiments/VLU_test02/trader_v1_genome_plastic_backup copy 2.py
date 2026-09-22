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
TRADING_MODE = "LONG"  # Opciones: "LONG", "SHORT", "BOTH"

# CONFIGURACIÓN DE APUESTAS
BET_MODE = "COMPOUND"      # "COMPOUND" = sin tope, "CAPPED" = con tope fijo
BET_PERCENTAGE = 0.10      # % del balance por trade (0.01 a 0.50)
MAX_BET = 5000.0           # Límite máximo (solo para modo CAPPED)

# CONFIGURACIÓN DE APRENDIZAJE ONLINE
ONLINE_LEARNING = True     # True = Cerebro sigue aprendiendo en live/test, False = Congelado

# CONFIGURACIÓN DE TEST
NUM_TEST_PERIODS = 10      # Cantidad de períodos random a testear
TEST_CANDLES = 5000        # Velas por período de test

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL
# ==============================================================================

class NeuroPlasticBrain:
    def __init__(self, initial_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        # Inicializamos genoma y array de energía paralelo
        self.genome = [] 
        self.neuron_energy = [] # Energía de cada neurona
        self.neuron_age = []    # Para saber cuan vieja es
        
        # GEN DE PERSONALIDAD: Umbral de decisión (valentía/cautela)
        # 0.3 = Muy agresivo, 0.9 = Muy conservador
        self.decision_threshold = np.random.uniform(0.3, 0.9)
        
        self.reset_genome(initial_neurons)

        # Configuración Biológica
        self.STARVATION_RATE = 0.0005  # Costo de vivir por vela (Bajo para mantener memoria)
        self.MITOSIS_THRESHOLD = 2.5   # Energía necesaria para dividirse
        self.REWARD_FACTOR = 0.1       # Cuanta energía gana por acierto
        self.PENALTY_FACTOR = 0.2      # Cuanta energía pierde por error (Castigo fuerte)
        self.MAX_NEURONS = 2000         # Límite para que no explote la PC
        self.MIN_NEURONS = 5           # Para no quedar lobotomizado

    def reset_genome(self, n_neurons):
        self.genome = []
        self.neuron_energy = []
        self.neuron_age = []
        
        for _ in range(n_neurons):
            self.add_random_neuron()
            
        self.genome = np.array(self.genome)
        self.neuron_energy = np.array(self.neuron_energy)
        self.neuron_age = np.array(self.neuron_age)

    def add_random_neuron(self):
        # Genoma: [x, y, z, R, W, Sx, Sy, Sz, ÁNGULO]
        gene = np.concatenate([
            np.random.uniform(-1.0, 1.0, 3), 
            [np.random.uniform(0.5, 1.5)],   
            [np.random.uniform(-1.0, 1.0)],  
            np.random.uniform(0.5, 1.5, 3),
            [np.random.uniform(-np.pi/2, np.pi/2)] 
        ])
        self.genome.append(gene)
        self.neuron_energy.append(1.0) # Energía inicial estándar
        self.neuron_age.append(0)

    def predict(self, points):
        """
        Retorna la señal y, crucialmente, las ACTIVACIONES individuales
        para saber a quién culpar o premiar después.
        """
        if len(self.genome) == 0: return 0.0, None
        
        # Desempaquetado Vectorizado
        centers = self.genome[:, :3]      
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8]     
        thetas = self.genome[:, 8]        
        
        point = points[-1].reshape(1, 3) 
        diff = point - centers 
        
        # Rotación 2D (Spin)
        c, s = np.cos(thetas), np.sin(thetas)
        dx = diff[:, 0] * c - diff[:, 1] * s
        dy = diff[:, 0] * s + diff[:, 1] * c
        dz = diff[:, 2] 
        
        # Distancia Volumétrica
        diff_rotated = np.stack([dx, dy, dz], axis=1)
        diff_stretched = diff_rotated / (stretch + 1e-6)
        dists = np.linalg.norm(diff_stretched, axis=1)
        
        # Activación
        safe_radii = np.maximum(radii, 1e-6)
        raw_overlap = np.maximum(0, 1 - dists / safe_radii)
        activations = np.minimum(1.0, raw_overlap * 3.0) # [0 a 1]
        
        # Señal final ponderada
        total_activation = np.sum(activations * weights)
        
        # Retornamos la señal final Y el array de quienes se activaron
        return np.tanh(total_activation), activations

    def learn_and_adapt(self, reward_signal, activations):
        """
        reward_signal: +1 si ganamos, -1 si perdimos (o magnitud del PnL)
        activations: Array de qué neuronas participaron en la decisión
        """
        if activations is None: return

        # 1. ACTUALIZACIÓN DE ENERGÍA (Por Recompensa/Castigo)
        # Solo las neuronas que se activaron (participaron) reciben premio o castigo
        
        impact = activations * reward_signal
        
        # Si la neurona se activó y el resultado fue bueno -> Gana energía
        # Si la neurona se activó y el resultado fue malo -> Pierde MUCHA energía
        energy_delta = np.where(impact > 0, 
                                impact * self.REWARD_FACTOR, 
                                impact * self.PENALTY_FACTOR) # Castigo más fuerte que premio
        
        self.neuron_energy += energy_delta

    def _structural_plasticity(self):
        new_genome = []
        new_energy = []
        new_age = []
        
        born_count = 0
        dead_count = 0
        
        for i in range(len(self.genome)):
            energy = self.neuron_energy[i]
            gene = self.genome[i]
            age = self.neuron_age[i]
            
            # --- APOPTOSIS (MUERTE) ---
            # Si la energía es <= 0, la neurona muere (no se agrega a la nueva lista)
            if energy <= 0 and len(self.genome) - dead_count > self.MIN_NEURONS:
                dead_count += 1
                continue # Skip (Borrar)
                
            # --- MITOSIS (REPRODUCCIÓN) ---
            # Si tiene mucha energía, se divide
            if energy > self.MITOSIS_THRESHOLD and len(self.genome) + born_count < self.MAX_NEURONS:
                # 1. La madre sobrevive pero gasta energía en el parto
                new_genome.append(gene)
                new_energy.append(energy * 0.5) # Divide energía
                new_age.append(age)
                
                # 2. Nace la hija (Copia con ligera mutación)
                child_gene = np.copy(gene)
                # Mutación pequeña (Geometría local)
                mutation_noise = np.random.normal(0, 0.05, size=child_gene.shape)
                child_gene += mutation_noise
                # Asegurar valores físicos válidos
                child_gene[3] = abs(child_gene[3]) # Radio
                child_gene[5:8] = abs(child_gene[5:8]) # Stretch
                
                new_genome.append(child_gene)
                new_energy.append(energy * 0.5) # La hija recibe la otra mitad
                new_age.append(0) # Recién nacida
                
                born_count += 1
            else:
                # Supervivencia normal
                new_genome.append(gene)
                new_energy.append(energy)
                new_age.append(age)
        
        # Convertir listas a numpy de nuevo
        self.genome = np.array(new_genome)
        self.neuron_energy = np.array(new_energy)
        self.neuron_age = np.array(new_age)
        
        # Opcional: Logs si hubo cambios estructurales
        # if born_count > 0 or dead_count > 0:
        #     print(f"   🧬 [NEUROPLASTICIDAD] Nacidas: {born_count} | Muertas: {dead_count} | Total: {len(self.genome)}")

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
# 💰 CÁLCULO DE APUESTA
# ==============================================================================

def calculate_bet(balance):
    """
    Calcula la apuesta según el modo configurado.
    COMPOUND: balance * BET_PERCENTAGE (sin límite, crece exponencialmente)
    CAPPED: min(balance * BET_PERCENTAGE, MAX_BET) (con tope fijo)
    """
    base_bet = balance * BET_PERCENTAGE
    
    if BET_MODE == "COMPOUND":
        return base_bet
    elif BET_MODE == "CAPPED":
        return min(base_bet, MAX_BET)
    else:
        raise ValueError(f"BET_MODE inválido: {BET_MODE}. Usa 'COMPOUND' o 'CAPPED'.")

# ==============================================================================
# 💾 PERSISTENCIA DE ESTADO COMPLETO
# ==============================================================================

def save_brain_state(brain, filename, metadata=None):
    """
    Guarda el estado COMPLETO del cerebro (no solo genoma).
    Incluye energía, edad, vault, y decision_threshold para preservar la experiencia.
    """
    brain_state = {
        'genome': brain.genome,
        'neuron_energy': brain.neuron_energy,
        'neuron_age': brain.neuron_age,
        'vault_genome': getattr(brain, 'vault_genome', np.empty((0, 9))),
        'vault_ages': getattr(brain, 'vault_ages', np.empty(0, dtype=int)),
        'decision_threshold': brain.decision_threshold,
        'metadata': metadata or {}
    }
    np.save(filename, brain_state, allow_pickle=True)
    print(f"   💾 Estado completo guardado: {filename}")

def load_brain_state(filename):
    """Carga el estado COMPLETO del cerebro."""
    state = np.load(filename, allow_pickle=True).item()
    
    brain = NeuroPlasticBrain(initial_neurons=10)
    brain.genome = state['genome']
    brain.neuron_energy = state['neuron_energy']
    brain.neuron_age = state['neuron_age']
    brain.vault_genome = state.get('vault_genome', np.empty((0, 9)))
    brain.vault_ages = state.get('vault_ages', np.empty(0, dtype=int))
    brain.decision_threshold = state.get('decision_threshold', 0.6)
    
    metadata = state.get('metadata', {})
    print(f"   📂 Estado cargado: {filename}")
    if 'generation' in metadata:
        print(f"      Gen: {metadata['generation']} | Balance: ${metadata.get('balance', 0):.2f}")
    print(f"      Neuronas: {len(brain.genome)} activas")
    print(f"      Umbral decisión: {brain.decision_threshold:.2f}")
    
    return brain, metadata

# ==============================================================================
# ⚔️ ENTRENAMIENTO (EVOLUCIÓN)
# ==============================================================================

#     return brain.genome, balance

def run_evolution(generations=5, population_size=10, candles_limit=20000): 
    print(f"\n🧬 INICIANDO EVOLUCIÓN HÍBRIDA (LAMARCKIANA)...")
    print(f"   Población: {population_size} | Generaciones: {generations}")
    
    # 1. Carga de datos
    market_prices, market_vols, market_volatilities = fetch_extended_data("PEPEUSDT", "1m", candles_limit)
    inputs, prices_aligned = prepare_inputs(market_prices, market_vols, market_volatilities)
    
    if inputs is None: return None

    # Inicializar población aleatoria
    # Usamos una lista de cerebros iniciales
    pass  # Placeholder para lógica explicada abajo

    # Mejor Genoma Global
    best_global_genome = None
    best_global_fitness = -np.inf
    best_global_brain = None  # Guardar cerebro completo
    best_global_threshold = 0.6
    
    # Genoma base para mutar (El "Padre" de la generación)
    # Al principio es aleatorio
    parent_brain = NeuroPlasticBrain(initial_neurons=500) # Más neuronas iniciales
    current_best_genome = np.copy(parent_brain.genome)
    current_best_threshold = parent_brain.decision_threshold  # Inicializar umbral
    
    # Mutación Rápida (para ver resultados hoy)
    initial_mut = 0.90   # 90%
    final_mut = 0.001    # 0.1%

    for gen in range(generations):
        # Cálculo de tasa de mutación dinámica
        progress = gen / generations
        mut_rate = initial_mut * (1 - progress) + final_mut
        
        print(f"\n--- GENERACIÓN {gen+1}/{generations} (Mutación: {mut_rate:.4f}) ---")
        
        gen_best_fitness = -np.inf
        gen_best_genome = None
        
        # Evaluar Población
        for i in range(population_size):
            # 1. Crear individuo (Mutación del mejor actual)
            brain = NeuroPlasticBrain(initial_neurons=10) # Dummy init, lo sobreescribimos
            
            if i == 0 and gen > 0:
                 # El campeón pasa directo (Elitismo) pero VA A SEGUIR APRENDIENDO
                 brain.genome = np.array([np.copy(g) for g in current_best_genome]) # Numpy Array
                 brain.neuron_energy = np.ones(len(brain.genome)) # Reset energía para justicia
                 brain.neuron_age = np.zeros(len(brain.genome))
                 brain.decision_threshold = current_best_threshold  # Heredar personalidad
            else:
                # Mutar al padre
                # Estrategia: Tomamos el genoma aprendido del padre y lo mutamos un poco
                temp_genome = [np.copy(g) for g in current_best_genome]
                
                # Mutación Dinámica
                if np.random.rand() < 0.5:
                    # Perturbar valores
                    idx = np.random.randint(0, len(temp_genome))
                    mutation = np.random.normal(0, mut_rate, size=temp_genome[idx].shape)
                    temp_genome[idx] += mutation
                
                # Inyectar neurona nueva a veces
                if np.random.rand() < 0.3:
                    new_gene = np.concatenate([
                        np.random.uniform(-1.0, 1.0, 3), 
                        [np.random.uniform(0.5, 1.5)],   
                        [np.random.uniform(-1.0, 1.0)],  
                        np.random.uniform(0.5, 1.5, 3),
                        [np.random.uniform(-np.pi/2, np.pi/2)] 
                    ])
                    temp_genome.append(new_gene)
                
                brain.genome = np.array(temp_genome) # Ensure Numpy Array
                brain.neuron_energy = np.ones(len(brain.genome))
                brain.neuron_age = np.zeros(len(brain.genome))
                
                # Mutar umbral de decisión (gen de personalidad)
                brain.decision_threshold = current_best_threshold + np.random.normal(0, mut_rate * 0.5)
                brain.decision_threshold = np.clip(brain.decision_threshold, 0.3, 0.9)  # Mantener en rango válido

            # 2. VIVIR Y APRENDER (Plasticidad)
            # Ejecutamos la simulación de vida. El cerebro cambiará su estructura.
            # Pasamos inputs y prices a una función auxiliar para no repetir código
            learned_genome, final_balance = run_lifetime_simulation(brain, inputs, prices_aligned)
            
            fitness = final_balance
            
            # Log simple
            # print(f"   Ind {i}: Bal ${final_balance:.2f} | Neuronas Fin: {len(learned_genome)}")
            
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_genome = [np.copy(g) for g in learned_genome] # Guardamos su versión MODIFICADA (Lamarck)
                gen_best_threshold = brain.decision_threshold  # Guardar personalidad ganadora
                gen_best_brain = brain  # Guardar cerebro completo

        umbral_display = gen_best_threshold if 'gen_best_threshold' in locals() else 0.6
        print(f"🏆 MEJOR DE GEN {gen}: Balance ${gen_best_fitness:.2f} | Neuronas: {len(gen_best_genome)} | Umbral: {umbral_display:.2f}")
        
        # Selección para siguiente generación
        current_best_genome = gen_best_genome
        current_best_threshold = gen_best_threshold if 'gen_best_threshold' in locals() else 0.6
        
        if gen_best_fitness > best_global_fitness:
            best_global_fitness = gen_best_fitness
            best_global_genome = gen_best_genome
            best_global_threshold = gen_best_threshold if 'gen_best_threshold' in locals() else 0.6
            best_global_brain = gen_best_brain if 'gen_best_brain' in locals() else None
            
            # Guardar estado completo (no solo genoma)
            if best_global_brain is not None:
                metadata = {
                    'generation': gen,
                    'balance': best_global_fitness,
                    'timestamp': time.time()
                }
                save_brain_state(best_global_brain, "best_brain_COMPLETE.npy", metadata)
                print("   🌟 ¡NUEVO RÉCORD GLOBAL! Estado completo guardado.")

    # Retornar cerebro completo (no solo genoma)
    if best_global_brain is not None:
        return best_global_brain
    else:
        # Fallback: Si no hay cerebro completo, crear uno con el genoma
        brain = NeuroPlasticBrain(initial_neurons=10)
        brain.genome = np.array(best_global_genome)
        brain.decision_threshold = best_global_threshold
        return brain

def run_lifetime_simulation(brain, inputs, prices_aligned):
    # Hereda la lógica de 'run_neuroplastic_backtest' pero encapsulada
    balance = 10000.0
    position = 0
    entry_price = 0
    current_bet = 0
    leverage = 50.0
    trade_memory_activations = []
    
    METABOLIC_COST = 0.00005 

    for t in range(len(inputs)):
        # A. PREDECIR
        signal, activations = brain.predict([inputs[t]])
        current_price = prices_aligned[t]
        
        # A.1 Costo Metabólico Financiero (Si no genera, muere)
        balance -= (balance * METABOLIC_COST)
        
        # B. METABOLISMO NEURONAL
        brain.neuron_energy -= brain.STARVATION_RATE
        brain.neuron_age += 1
        
        # C. MEMORIA
        if position != 0:
            trade_memory_activations.append(activations)

        # D. LÓGICA TRADING
        # Umbral bajado a 0.6 para facilitar arranque
        if (signal > 0.6 and position == -1) or (signal < -0.6 and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price
            if position == -1: pnl_pct *= -1
            pnl_real = pnl_pct * leverage
            
            profit = current_bet * pnl_real
            fee = current_bet * 0.004 
            balance += (profit - fee)
            
            # E. OCURRE EL APRENDIZAJE
            if len(trade_memory_activations) > 0:
                mean_activations = np.mean(trade_memory_activations, axis=0)
                brain.learn_and_adapt(pnl_real, mean_activations)
            
            position = 0
            current_bet = 0
            trade_memory_activations = []

        if position == 0:
            # Respetar modo de trading configurado
            # Usar umbral de decisión del cerebro (gen de personalidad)
            if signal > brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]: 
                position = 1
                entry_price = current_price
                current_bet = calculate_bet(balance)  # Usa configuración global
                trade_memory_activations = [activations]
            elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]: 
                position = -1
                entry_price = current_price
                current_bet = calculate_bet(balance)  # Usa configuración global
                trade_memory_activations = [activations]

        # F. PLASTICIDAD ESTRUCTURAL
        if t % 50 == 0: # Más frecuente para dinamismo
            old_len = len(brain.genome)
            brain._structural_plasticity()
            if len(brain.genome) != old_len:
                # Si cambió la estructura, la memoria del trade ya no es válida (cambian índices)
                trade_memory_activations = []
            
    return brain.genome, balance

# ==============================================================================
# 🧪 TEST DE BATALLA (OUT OF SAMPLE)
# ==============================================================================

def run_test_simulation(brain, inputs, prices):
    """
    Ejecuta el cerebro completo (con todo su estado) sobre datos nuevos.
    Si ONLINE_LEARNING=True, el cerebro sigue adaptándose durante el test.
    """
    # El cerebro ya viene cargado con todo: genome, energy, age, vault, threshold
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    SLIPPAGE = 0.001
    FEE = 0.004
    METABOLIC_COST = 0.00005
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    actions = [] # Para graficar: 1=Buy, -1=Sell, 0=Hold
    equity_curve = []
    trade_memory_activations = []  # Para aprendizaje online
    
    learning_status = "ACTIVO" if ONLINE_LEARNING else "CONGELADO"
    print(f"\n⏯️  Ejecutando simulación... (Aprendizaje: {learning_status})")
    
    for t in range(len(inputs)):
        signal, activations = brain.predict([inputs[t]])
        current_price = prices[t]
        act = 0
        
        # APRENDIZAJE ONLINE: Costo metabólico y envejecimiento
        if ONLINE_LEARNING:
            balance -= (balance * METABOLIC_COST)
            brain.neuron_energy -= brain.STARVATION_RATE
            brain.neuron_age += 1
        
        # Memoria del trade actual
        if ONLINE_LEARNING and position != 0:
            trade_memory_activations.append(activations)
        
        # Cierre
        if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 else (entry_price - current_price) / entry_price * LEVERAGE
            profit = current_bet * pnl_pct
            cost = (current_bet * LEVERAGE) * FEE 
            balance += (profit - cost)
            
            # APRENDIZAJE ONLINE: Actualizar energías según resultado
            if ONLINE_LEARNING and len(trade_memory_activations) > 0:
                mean_activations = np.mean(trade_memory_activations, axis=0)
                brain.learn_and_adapt(pnl_pct, mean_activations)
                trade_memory_activations = []
            
            position = 0
            current_bet = 0
        
        # Apertura
        if position == 0:
            # Respetar modo de trading configurado
            # Usar umbral del cerebro (default 0.8 para test más conservador)
            threshold = brain.decision_threshold if hasattr(brain, 'decision_threshold') else 0.8
            if signal > threshold and TRADING_MODE in ["LONG", "BOTH"]: 
                position = 1
                entry_price = current_price * (1 + SLIPPAGE) 
                current_bet = calculate_bet(balance)  # Usa configuración global
                balance -= (current_bet * LEVERAGE) * FEE
                act = 1
                if ONLINE_LEARNING:
                    trade_memory_activations = [activations]
            elif signal < -threshold and TRADING_MODE in ["SHORT", "BOTH"]: 
                position = -1
                entry_price = current_price * (1 - SLIPPAGE) 
                current_bet = calculate_bet(balance)  # Usa configuración global
                balance -= (current_bet * LEVERAGE) * FEE
                act = -1
                if ONLINE_LEARNING:
                    trade_memory_activations = [activations]
        
        # APRENDIZAJE ONLINE: Plasticidad estructural cada 50 velas
        if ONLINE_LEARNING and t % 50 == 0:
            brain._structural_plasticity()
            # Limpiar memoria de trade porque el cerebro cambió de tamaño
            trade_memory_activations = []
        
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
# 📊 ESTADÍSTICAS DE TESTS MÚLTIPLES
# ==============================================================================

def print_test_statistics(results):
    """Muestra estadísticas de múltiples tests."""
    balances = [r['balance'] for r in results]
    profits = [r['profit_pct'] for r in results]
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS FINALES (MÚLTIPLES PERÍODOS)")
    print("="*60)
    print(f"   Profit Promedio: {np.mean(profits):.2f}%")
    print(f"   Mejor Test: {max(profits):.2f}%")
    print(f"   Peor Test: {min(profits):.2f}%")
    print(f"   Desviación Estándar: {np.std(profits):.2f}%")
    print(f"   Win Rate: {sum(1 for p in profits if p > 0)}/{len(results)} ({100*sum(1 for p in profits if p > 0)/len(results):.1f}%)")
    print(f"   Balance Final Promedio: ${np.mean(balances):.2f}")
    print("="*60)

# ==============================================================================
# 🚀 MAIN: ENTRENAMIENTO + TEST ALEATORIO
# ==============================================================================

if __name__ == "__main__":
    # 1. EVOLUCIONAR (Genetic Algorithm + Plasticity)
    winner_brain = run_evolution(generations=5, population_size=50, candles_limit=20000)
    
    if winner_brain is not None:
        print("\n" + "="*60)
        print(f"🎲 --- FASE 2: TEST MULTI-PERÍODO ({NUM_TEST_PERIODS} períodos random) ---")
        print("="*60)
        
        # Calcular rango de fechas
        ms_per_year = 31536000000 
        now_ms = int(time.time() * 1000)
        buffer_ms = 20 * 24 * 60 * 60 * 1000 # 20 días
        
        test_results = []
        
        for period in range(NUM_TEST_PERIODS):
            # Fecha random diferente para cada test
            random_past_time = random.randint(now_ms - ms_per_year, now_ms - buffer_ms)
            readable_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(random_past_time/1000))
            
            # Bajar datos del período
            test_prices, test_vols, test_volatilities = fetch_extended_data(
                "PEPEUSDT", "1m", TEST_CANDLES, custom_end_time=random_past_time
            )
            
            test_inputs, test_prices_aligned = prepare_inputs(test_prices, test_vols, test_volatilities)
            
            if test_inputs is not None:
                # CRÍTICO: Recargar cerebro limpio para cada test (si ONLINE_LEARNING=True)
                brain_copy, _ = load_brain_state("best_brain_COMPLETE.npy")
                
                # Ejecutar test
                final_bal, acts, equity = run_test_simulation(brain_copy, test_inputs, test_prices_aligned)
                
                profit_pct = ((final_bal - 10000) / 10000) * 100
                
                # Guardar resultado
                test_results.append({
                    'period': period + 1,
                    'balance': final_bal,
                    'profit_pct': profit_pct,
                    'date': readable_date
                })
                
                # Mostrar resultado individual
                status = "✅" if profit_pct > 0 else "❌"
                print(f"{status} Test {period+1}/{NUM_TEST_PERIODS}: {readable_date} → ${final_bal:.2f} ({profit_pct:+.2f}%)")
        
        # Mostrar estadísticas agregadas
        if test_results:
            print_test_statistics(test_results)
        else:
            print("❌ Error: No se pudieron ejecutar los tests.")