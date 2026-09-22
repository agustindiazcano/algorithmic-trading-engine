import numpy as np
import matplotlib.pyplot as plt
import time
import requests 
import random # <--- Necesario para elegir la fecha aleatoria
import sys

# Forzar UTF-8 en consola para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURACIÓN EXPERIMENTAL
ENABLE_APOPTOSIS = True  # <--- Poner en False para que NUNCA mueran las neuronas
ENABLE_VAULT_SYSTEM = True  # <--- Sistema de Memoria a Largo Plazo (Hibernación)
VAULT_CHECK_INTERVAL = 10  # <--- Revisar vault cada N velas (optimización de performance)

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
        
        # --- MEMORIA A LARGO PLAZO (VAULT) ---
        self.vault_genome = np.empty((0, 9))  # Bóveda de veteranos hibernados
        self.vault_ages = np.empty(0, dtype=int)  # Preservar edad de veteranos
        
        self.reset_genome(initial_neurons)

        # Configuración Biológica
        self.STARVATION_RATE = 0.0005  # Costo de vivir por vela (Bajo para mantener memoria)
        self.MITOSIS_THRESHOLD = 2.5   # Energía necesaria para dividirse
        self.REWARD_FACTOR = 0.1       # Cuanta energía gana por acierto
        self.PENALTY_FACTOR = 0.2      # Cuanta energía pierde por error (Castigo fuerte)
        self.MAX_NEURONS = 2000         # Límite para que no explote la PC
        self.MIN_NEURONS = 5           # Para no quedar lobotomizado
        self.VETERAN_AGE_THRESHOLD = 2000  # Edad mínima para hibernar en lugar de morir

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
        self.neuron_energy.append(10.0) # Energía inicial ALTA para sobrevivir exploración
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
    
    def check_vault_reactivation(self, point):
        """
        Revisa si el mercado actual 'toca' alguna memoria del pasado.
        Si hay coincidencia geométrica, la neurona vuelve a la vida.
        Returns True si se reactivaron neuronas (cambió el tamaño del genoma).
        """
        if not ENABLE_VAULT_SYSTEM or len(self.vault_genome) == 0:
            return False
        
        # Calculamos distancia de las veteranas al punto actual
        centers = self.vault_genome[:, :3]
        radii = self.vault_genome[:, 3] * self.A_dc
        
        point_reshaped = point.reshape(1, 3)
        diff = point_reshaped - centers
        dists = np.linalg.norm(diff, axis=1)
        
        # Si el punto está dentro del radio de alguna veterana (coincidencia de arquetipo)
        matches = np.where(dists < radii)[0]
        
        if len(matches) > 0:
            for idx in matches[:3]:  # Limitar a 3 reactivaciones por vez para no saturar
                gene = self.vault_genome[idx]
                age = self.vault_ages[idx]
                
                # La neurona despierta
                self.genome = np.vstack([self.genome, gene])
                self.neuron_energy = np.append(self.neuron_energy, 1.5)  # Despierta con energía media
                self.neuron_age = np.append(self.neuron_age, age)  # Preserva su veteranía
            
            # Eliminar del vault (usando mask para evitar problemas de índice)
            mask = np.ones(len(self.vault_genome), dtype=bool)
            mask[matches[:3]] = False
            self.vault_genome = self.vault_genome[mask]
            self.vault_ages = self.vault_ages[mask]
            return True  # Indica que el genoma cambió
        
        return False

    def learn_and_adapt(self, reward_signal, activations):
        """
        reward_signal: +1 si ganamos, -1 si perdimos (o magnitud del PnL)
        activations: Array de qué neuronas participaron en la decisión
        """
        if activations is None: return
        
        # VALIDACIÓN: Si el genoma cambió de tamaño entre predict y learn, abortar
        if len(activations) != len(self.neuron_energy):
            return  # Ignorar este aprendizaje, data inconsistente

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
            
            # --- APOPTOSIS / HIBERNACIÓN ---
            # Si la energía es <= 0, la neurona muere o hiberna
            if energy <= 0:
                # SISTEMA DE VAULT: Guardar veteranos en lugar de eliminarlos
                if ENABLE_VAULT_SYSTEM and age > self.VETERAN_AGE_THRESHOLD and len(self.genome) - dead_count > self.MIN_NEURONS:
                    # Hibernar: Guardar en vault
                    self.vault_genome = np.vstack([self.vault_genome, gene])
                    self.vault_ages = np.append(self.vault_ages, age)
                    dead_count += 1
                    continue
                # APOPTOSIS CLÁSICA (solo si está habilitada Y no califica para vault)
                elif ENABLE_APOPTOSIS and len(self.genome) - dead_count > self.MIN_NEURONS:
                    dead_count += 1
                    continue
                # Si ambos están deshabilitados, la neurona sobrevive con energía 0
                
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

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=10000, custom_end_time=None):
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

    return brain.genome, balance

def run_evolution(generations=1000, population_size=10, candles_limit=10000): 
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
    
    # Genoma base para mutar (El "Padre" de la generación)
    # Al principio es aleatorio
    parent_brain = NeuroPlasticBrain(initial_neurons=500) # Más neuronas iniciales
    current_best_genome = np.copy(parent_brain.genome)
    
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
        
        # ESTADÍSTICAS DE LA GENERACIÓN
        profitable_count = 0
        gen_stats_best = {}
        
        # Evaluar Población
        for i in range(population_size):
            # ... (código existente de creación/mutación) ...
            # 1. Crear individuo (Mutación del mejor actual)
            brain = NeuroPlasticBrain(initial_neurons=10) # Dummy init, lo sobreescribimos
            
            if i == 0 and gen > 0:
                 # El campeón pasa directo (Elitismo) pero VA A SEGUIR APRENDIENDO
                 brain.genome = np.array([np.copy(g) for g in current_best_genome]) # Numpy Array
                 brain.neuron_energy = np.ones(len(brain.genome)) * 10.0 # Reset energía con boost
                 brain.neuron_age = np.zeros(len(brain.genome))
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
                brain.neuron_energy = np.ones(len(brain.genome)) * 10.0
                brain.neuron_age = np.zeros(len(brain.genome))

            # 2. VIVIR Y APRENDER (Plasticidad)
            learned_genome, final_balance, stats = run_lifetime_simulation(brain, inputs, prices_aligned)
            
            fitness = final_balance
            
            if fitness > 10000:
                profitable_count += 1
            
            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_genome = [np.copy(g) for g in learned_genome]
                gen_stats_best = stats

        # REPORTE DETALLADO
        profit_pct = ((gen_best_fitness - 10000) / 10000) * 100
        avg_profit_per_trade = 0
        if gen_stats_best.get('trades', 0) > 0:
            avg_profit_per_trade = gen_stats_best.get('total_pnl_pct', 0) / gen_stats_best.get('trades', 1) * 100
        
        win_rate = 0
        if gen_stats_best.get('trades', 0) > 0:
            win_rate = (gen_stats_best.get('wins', 0) / gen_stats_best.get('trades', 1)) * 100

        print(f"🏆 CAMPEÓN GEN {gen}:")
        print(f"   💰 Balance: ${gen_best_fitness:.2f} ({profit_pct:.2f}%)")
        print(f"   📊 Trades: {gen_stats_best.get('trades',0)} (W: {gen_stats_best.get('wins',0)} | L: {gen_stats_best.get('losses',0)})")
        print(f"   🎯 Win Rate: {win_rate:.1f}% | Avg Trade: {avg_profit_per_trade:.2f}%")
        print(f"   🧠 Neuronas: {len(gen_best_genome)} (Nac: {gen_stats_best.get('born',0)} | Mue: {gen_stats_best.get('dead',0)})")
        if ENABLE_VAULT_SYSTEM:
            print(f"   🏛️  Vault: {gen_stats_best.get('vault_size',0)} veteranos hibernados")
        print(f"   👥 Población Rentable: {profitable_count}/{population_size}")
        
        if gen_best_fitness > best_global_fitness:
            best_global_fitness = gen_best_fitness
            best_global_genome = gen_best_genome
            np.save("best_genome_EVO_PLASTIC1.npy", np.array(best_global_genome))
            print("   🌟 ¡NUEVO RÉCORD GLOBAL! Guardado.")

        # SELECCIÓN: RED DE SEGURIDAD (SAFETY NET)
        # Si la generación actual colapsó (perdió mucho dinero o neuronas respecto al mejor histórico)
        # Revertimos al "Save Game" (Best Global)
        # Criterio: Si el fitness actual es < 50% del mejor histórico, O si tiene < 10% de neuronas
        
        # NUEVO: Solo activar después de 50% del entrenamiento para permitir exploración
        is_disaster = False
        if best_global_genome is not None and gen > generations * 0.5:
             ratio_fitness = 0
             if best_global_fitness > 0:
                 ratio_fitness = gen_best_fitness / best_global_fitness
             
             ratio_neurons = 0
             if len(best_global_genome) > 0:
                 ratio_neurons = len(gen_best_genome) / len(best_global_genome)
                 
             if ratio_fitness < 0.5 or ratio_neurons < 0.1:
                 is_disaster = True
        
        if is_disaster:
            print("   🚨 COLAPSO DETECTADO: Revertiendo al Mejor Genoma Histórico (Safety Net)")
            current_best_genome = [np.copy(g) for g in best_global_genome]
        else:
            current_best_genome = gen_best_genome

    return np.array(best_global_genome)

def run_lifetime_simulation(brain, inputs, prices_aligned):
    # Hereda la lógica de 'run_neuroplastic_backtest' pero encapsulada
    balance = 10000.0
    position = 0
    entry_price = 0
    current_bet = 0
    leverage = 50.0
    trade_memory_activations = []
    expected_genome_size = len(brain.genome)  # Track genome size for consistency
    
    METABOLIC_COST = 0.00005 

    stats = {
        'trades': 0, 'wins': 0, 'losses': 0, 
        'born': 0, 'dead': 0, 'total_pnl_pct': 0.0,
        'vault_size': 0
    }

    for t in range(len(inputs)):
        # ... (PREDICCIÓN, METABOLISMO, MEMORIA igual) ...
        # A. PREDECIR
        signal, activations = brain.predict([inputs[t]])
        current_price = prices_aligned[t]
        
        # A.0 REACTIVACIÓN DE VAULT (con optimización de intervalo)
        vault_activated = False
        if ENABLE_VAULT_SYSTEM and t % VAULT_CHECK_INTERVAL == 0:
            vault_activated = brain.check_vault_reactivation(inputs[t])
            if vault_activated and position != 0:
                # Si despertaron neuronas durante un trade, descartamos la memoria
                # porque el vector de activaciones cambió de tamaño
                trade_memory_activations = []
        
        # A.1 Costo Metabólico Financiero
        balance -= (balance * METABOLIC_COST)
        
        # B. METABOLISMO NEURONAL (INERCIA BIOLÓGICA)
        # Las neuronas viejas consumen menos (Efficient Veterans)
        # Costo = Base / (1 + Edad * 0.001)
        metabolic_drag = brain.STARVATION_RATE / (1 + brain.neuron_age * 0.001)
        brain.neuron_energy -= metabolic_drag
        
        brain.neuron_age += 1
        
        # C. MEMORIA (Solo si el tamaño del genoma es consistente con el trade actual)
        if position != 0:
            current_genome_size = len(brain.genome)
            if current_genome_size != expected_genome_size:
                # El genoma cambió mid-trade (mitosis/vault), descartamos memoria antigua
                trade_memory_activations = []
                expected_genome_size = current_genome_size
            # Ahora sí, append (solo si el size es correcto)
            trade_memory_activations.append(activations)

        # D. LÓGICA TRADING
        if (signal > 0.6 and position == -1) or (signal < -0.6 and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price
            if position == -1: pnl_pct *= -1
            pnl_real = pnl_pct * leverage
            
            profit = current_bet * pnl_real
            fee = current_bet * 0.010 
            balance += (profit - fee)
            
            # Actualizar Stats
            stats['trades'] += 1
            stats['total_pnl_pct'] += pnl_pct # Guardamos sin lever para ratio real
            if profit > 0: stats['wins'] += 1
            else: stats['losses'] += 1
            
            # E. OCURRE EL APRENDIZAJE
            if len(trade_memory_activations) > 0:
                # Filtrar activaciones de tamaño consistente (por si mitosis ocurrió mid-trade)
                sizes = [len(act) for act in trade_memory_activations]
                most_common_size = max(set(sizes), key=sizes.count)
                consistent_activations = [act for act in trade_memory_activations if len(act) == most_common_size]
                
                if len(consistent_activations) > 0:
                    mean_activations = np.mean(consistent_activations, axis=0)
                    brain.learn_and_adapt(pnl_real, mean_activations)
            
            position = 0
            current_bet = 0
            trade_memory_activations = []

        if position == 0:
            if signal > 0.6: 
                position = 1
                entry_price = current_price
                current_bet = balance * 0.1 
                trade_memory_activations = [activations]
                expected_genome_size = len(brain.genome)  # Lock size for this trade
            elif signal < -0.6: 
                position = -1
                entry_price = current_price
                current_bet = balance * 0.1
                trade_memory_activations = [activations]
                expected_genome_size = len(brain.genome)  # Lock size for this trade

        # F. PLASTICIDAD ESTRUCTURAL
        if t % 50 == 0: 
            old_len = len(brain.genome)
            brain._structural_plasticity()
            diff = len(brain.genome) - old_len
            if diff > 0: stats['born'] += diff
            elif diff < 0: stats['dead'] += abs(diff)
            
            if len(brain.genome) != old_len:
                trade_memory_activations = []
    
    # Capturar tamaño final del vault
    stats['vault_size'] = len(brain.vault_genome)
            
    return brain.genome, balance, stats

# ==============================================================================
# 🧪 TEST DE BATALLA (OUT OF SAMPLE)
# ==============================================================================

def run_test_simulation(genome, inputs, prices):
    """
    Ejecuta el genoma SIN mutación sobre datos nuevos.
    """
    brain = NeuroPlasticBrain(initial_neurons=10, A_dc=1.618)
    brain.genome = genome
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    MAX_BET = 3333.0
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
        signal, _ = brain.predict([inputs[t]])
        current_price = prices[t]
        act = 0
        
        # Cierre
        if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 else (entry_price - current_price) / entry_price * LEVERAGE
            profit = current_bet * pnl_pct
            cost = (current_bet * LEVERAGE) * FEE 
            balance += (profit - cost)
            position = 0
            current_bet = 0
        
        # Apertura
        if position == 0:
            if signal > 0.8: 
                position = 1
                entry_price = current_price * (1 + SLIPPAGE) 
                current_bet = min(balance, MAX_BET)
                balance -= (current_bet * LEVERAGE) * FEE
                act = 1
            elif signal < -0.8: 
                position = -1
                entry_price = current_price * (1 - SLIPPAGE) 
                current_bet = min(balance, MAX_BET)
                balance -= (current_bet * LEVERAGE) * FEE
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
    # 1. EVOLUCIONAR (Genetic Algorithm + Plasticity)
    # 1000 generaciones x 50 individuos = 50,000 vidas simuladas
    winner_genome = run_evolution(generations=10, population_size=50, candles_limit=10000)
    
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
            final_bal, acts, equity = run_test_simulation(winner_genome, test_inputs, test_prices_aligned)
            
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