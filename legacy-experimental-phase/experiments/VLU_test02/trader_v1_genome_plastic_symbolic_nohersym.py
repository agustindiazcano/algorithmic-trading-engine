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

# CONFIGURACIÓN DE METABOLISMO
METABOLIC_COST_FIXED = 0.01  # Energía perdida por vela (no afecta balance)
ENERGY_FROM_PROFIT = 0.1     # % de profit que se convierte en energía

# CONFIGURACIÓN DE NOVELTY-DRIVEN GROWTH
NOVELTY_THRESHOLD = 0.1    # Umbral de activación para crear neurona (Growing Neural Gas)
YOUNG_AGE_LIMIT = 500      # Velas de inmunidad para neuronas jóvenes

# CONFIGURACIÓN DE TEST
NUM_TEST_PERIODS = 10      # Cantidad de períodos random a testear
TEST_CANDLES = 5000        # Velas por período de test

# CONFIGURACIÓN DE COMPRESIÓN SIMBÓLICA
SYMBOLIC_COMPRESSION = True      # Activar motor simbólico
CONSOLIDATION_INTERVAL = 1000    # Velas entre consolidaciones
MIN_CLUSTER_SIZE = 3             # Mínimo neuronas para simbolizar
SYMBOL_RESOLUTION = 0.2          # Granularidad del grid espacial (aumentado de 0.05)

# ==============================================================================
# 🧠 SYMBOLIC CORTEX - Compresión de Memoria
# ==============================================================================

class SymbolicCortex:
    """
    Motor de compresión simbólica: Convierte grupos de neuronas cercanas
    en símbolos reutilizables (memoria a largo plazo).
    """
    def __init__(self, resolution=0.05):
        self.resolution = resolution  # Tamaño de celda de cuantización
        self.symbol_registry = {}     # {Hash_Geométrico: ID_Símbolo}
        self.vocab_size = 0           # Cantidad de "conceptos" aprendidos
        self.known_patterns = {}      # {ID: Estructura_Relativa}
    
    def quantize(self, vector):
        """Convierte coordenadas continuas a discretas para el hash."""
        return tuple(np.round(vector / self.resolution).astype(int))

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
        
        # PERFORMANCE TRACKING (Layer 1: Active neurons)
        self.neuron_score = np.array([])           # EMA de utilidad
        self.neuron_best_score = np.array([])      # Máximo score histórico
        self.neuron_hits = np.array([], dtype=int) # Trades ganadores
        self.protected = np.array([], dtype=bool)  # Flag de protección
        
        # VAULT SYSTEM (Layer 2: Champion neurons)
        self.vault_genome = np.empty((0, 9))       # Neuronas campeonas
        self.vault_score = np.array([])            # Score de vault
        self.vault_hits = np.array([], dtype=int)  # Hits de vault
        
        # GEN DE PERSONALIDAD: Umbral de decisión (valentía/cautela)
        # 0.3 = Muy agresivo, 0.9 = Muy conservador
        self.decision_threshold = np.random.uniform(0.3, 0.9)
        
        # ENERGÍA METABÓLICA (Supervivencia)
        self.energy = 100.0  # Energía inicial
        self.MAX_ENERGY = 200.0  # Tope máximo
        
        # TRACKING DE NACIMIENTOS
        self.neurons_born_by_novelty = 0
        self.neurons_born_by_mitosis = 0
        
        # MOTOR SIMBÓLICO (Layer 3: Compressed symbols)
        self.cortex = SymbolicCortex(resolution=SYMBOL_RESOLUTION)
        self.symbol_genome = np.empty((0, 9))  # Símbolos consolidados (misma estructura que genome)
        
        self.reset_genome(initial_neurons)

        # Configuración Biológica (BALANCEADO para 20k velas)
        self.STARVATION_RATE = 0.00005  # Muy bajo: 20k velas = 1.0 energía (vs 10.0 antes)
        self.MITOSIS_THRESHOLD = 2.5   # Energía necesaria para dividirse
        self.REWARD_FACTOR = 0.15      # Recompensa aumentada (antes 0.1)
        self.PENALTY_FACTOR = 0.15     # Castigo = recompensa (antes 0.2)
        self.MAX_NEURONS = 2000        # Límite para que no explote la PC
        self.MIN_NEURONS = 5           # Para no quedar lobotomizado
        
        # VAULT CONFIG (RELAJADO para promoción más frecuente)
        self.PROMOTE_SCORE_THR = 0.1   # Score mínimo para vault (bajado de 0.5)
        self.PROMOTE_HITS_THR = 3      # Trades ganadores mínimos (bajado de 10)
        self.VAULT_WEIGHT = 0.3        # Peso de vault en predict

    def reset_genome(self, n_neurons):
        self.genome = []
        self.neuron_energy = []
        self.neuron_age = []
        
        for _ in range(n_neurons):
            self.add_random_neuron()
            
        self.genome = np.array(self.genome)
        self.neuron_energy = np.array(self.neuron_energy)
        self.neuron_age = np.array(self.neuron_age)
        
        # Inicializar tracking arrays
        self.neuron_score = np.zeros(len(self.genome))
        self.neuron_best_score = np.zeros(len(self.genome))
        self.neuron_hits = np.zeros(len(self.genome), dtype=int)
        self.protected = np.zeros(len(self.genome), dtype=bool)

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
        Predicción 3 capas: Active + Vault + Symbols.
        Retorna señal y activaciones (solo de neuronas activas para aprendizaje).
        """
        point = points[-1].reshape(1, 3)
        total_activation = 0
        activations = np.array([])
        
        # PASE 1: NEURONAS ACTIVAS (memoria a corto plazo, peso 1.0)
        if len(self.genome) > 0:
            centers = self.genome[:, :3]      
            radii = self.genome[:, 3] * self.A_dc
            weights = self.genome[:, 4]
            stretch = self.genome[:, 5:8]     
            thetas = self.genome[:, 8]        
            
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
            activations = np.minimum(1.0, raw_overlap * 3.0)  # [0 a 1]
            
            # Contribución de neuronas activas
            total_activation += np.sum(activations * weights)
        
        # PASE 2: VAULT NEURONS (campeonas protegidas, peso 0.3)
        if len(self.vault_genome) > 0:
            vault_centers = self.vault_genome[:, :3]
            vault_radii = self.vault_genome[:, 3] * self.A_dc
            vault_weights = self.vault_genome[:, 4]
            vault_stretch = self.vault_genome[:, 5:8]
            vault_thetas = self.vault_genome[:, 8]
            
            vault_diff = point - vault_centers
            
            # Rotación 2D (Spin)
            vault_c, vault_s = np.cos(vault_thetas), np.sin(vault_thetas)
            vault_dx = vault_diff[:, 0] * vault_c - vault_diff[:, 1] * vault_s
            vault_dy = vault_diff[:, 0] * vault_s + vault_diff[:, 1] * vault_c
            vault_dz = vault_diff[:, 2]
            
            # Distancia Volumétrica
            vault_diff_rotated = np.stack([vault_dx, vault_dy, vault_dz], axis=1)
            vault_diff_stretched = vault_diff_rotated / (vault_stretch + 1e-6)
            vault_dists = np.linalg.norm(vault_diff_stretched, axis=1)
            
            # Activación
            vault_safe_radii = np.maximum(vault_radii, 1e-6)
            vault_raw_overlap = np.maximum(0, 1 - vault_dists / vault_safe_radii)
            vault_activations = np.minimum(1.0, vault_raw_overlap * 3.0)
            
            # Contribución de vault (reducida: memoria histórica)
            total_activation += np.sum(vault_activations * vault_weights) * self.VAULT_WEIGHT
        
        # PASE 3: SÍMBOLOS CONSOLIDADOS (memoria comprimida, peso 0.5)
        if len(self.symbol_genome) > 0:
            sym_centers = self.symbol_genome[:, :3]
            sym_radii = self.symbol_genome[:, 3] * self.A_dc
            sym_weights = self.symbol_genome[:, 4]
            sym_stretch = self.symbol_genome[:, 5:8]
            sym_thetas = self.symbol_genome[:, 8]
            
            sym_diff = point - sym_centers
            
            # Rotación 2D (Spin)
            sym_c, sym_s = np.cos(sym_thetas), np.sin(sym_thetas)
            sym_dx = sym_diff[:, 0] * sym_c - sym_diff[:, 1] * sym_s
            sym_dy = sym_diff[:, 0] * sym_s + sym_diff[:, 1] * sym_c
            sym_dz = sym_diff[:, 2]
            
            # Distancia Volumétrica
            sym_diff_rotated = np.stack([sym_dx, sym_dy, sym_dz], axis=1)
            sym_diff_stretched = sym_diff_rotated / (sym_stretch + 1e-6)
            sym_dists = np.linalg.norm(sym_diff_stretched, axis=1)
            
            # Activación
            sym_safe_radii = np.maximum(sym_radii, 1e-6)
            sym_raw_overlap = np.maximum(0, 1 - sym_dists / sym_safe_radii)
            sym_activations = np.minimum(1.0, sym_raw_overlap * 3.0)
            
            # Contribución de símbolos
            total_activation += np.sum(sym_activations * sym_weights) * 0.5
        
        # Retornamos señal final Y activaciones de neuronas activas (para aprendizaje)
        return np.tanh(total_activation), activations
    
    def spawn_neuron_at(self, input_point):
        """Crea neurona específica en las coordenadas del mercado actual (Growing Neural Gas)."""
        if len(self.genome) >= self.MAX_NEURONS:
            return  # No crear si ya llegamos al límite
        
        cx, cy, cz = input_point[0], input_point[1], input_point[2]
        
        gene = np.array([
            cx, cy, cz,           # Centro = input actual
            0.3,                  # Radio pequeño (específico)
            np.random.uniform(-1, 1),  # Peso random
            0.8, 0.8, 0.8,        # Stretch conservador
            0.0                   # Sin rotación
        ])
        
        self.genome = np.vstack([self.genome, gene])
        self.neuron_energy = np.append(self.neuron_energy, 1.0)
        self.neuron_age = np.append(self.neuron_age, 0)
        
        # Actualizar tracking arrays
        self.neuron_score = np.append(self.neuron_score, 0.0)
        self.neuron_best_score = np.append(self.neuron_best_score, 0.0)
        self.neuron_hits = np.append(self.neuron_hits, 0)
        self.protected = np.append(self.protected, False)
        
        self.neurons_born_by_novelty += 1
    
    def check_novelty(self, input_point, activations, allow_spawn=True):
        """Detecta si el cerebro NO reconoce el patrón actual y crea neurona solo si allow_spawn=True."""
        max_activation = np.max(activations) if len(activations) > 0 else 0
        
        if max_activation < NOVELTY_THRESHOLD and allow_spawn:
            # Zona desconocida del espacio 3D
            self.spawn_neuron_at(input_point)
            return True
        return False
    
    def update_scores(self, pnl_net, activations):
        """
        Actualiza scores de neuronas después de cada trade.
        pnl_net: Retorno neto del trade (profit - cost) / bet
        activations: Array de activaciones de neuronas activas
        """
        if len(activations) == 0 or len(activations) != len(self.genome):
            return
        
        # Score delta: contribución ponderada por activación
        score_delta = activations * pnl_net
        
        # EMA del score (99% histórico, 1% nuevo)
        self.neuron_score = 0.99 * self.neuron_score + 0.01 * score_delta
        
        # Actualizar mejor score histórico
        self.neuron_best_score = np.maximum(self.neuron_best_score, self.neuron_score)
        
        # Contar hits (trades ganadores con activación significativa)
        if pnl_net > 0:
            hits = (activations > 0.2).astype(int)
            self.neuron_hits += hits
    
    def promote_to_vault(self, verbose=False):
        """Promociona neuronas campeonas al vault (protección permanente)."""
        if len(self.genome) == 0:
            return
        
        # Identificar elegibles: buen score + suficientes hits + no protegidas
        eligible = (self.neuron_best_score > self.PROMOTE_SCORE_THR) & \
                   (self.neuron_hits >= self.PROMOTE_HITS_THR) & \
                   (~self.protected)
        
        if not np.any(eligible):
            return
        
        indices = np.where(eligible)[0]
        
        # Copiar a vault
        promoted_genes = self.genome[indices]
        promoted_scores = self.neuron_best_score[indices]
        promoted_hits = self.neuron_hits[indices]
        
        if len(self.vault_genome) == 0:
            self.vault_genome = promoted_genes
            self.vault_score = promoted_scores
            self.vault_hits = promoted_hits
        else:
            self.vault_genome = np.vstack([self.vault_genome, promoted_genes])
            self.vault_score = np.append(self.vault_score, promoted_scores)
            self.vault_hits = np.append(self.vault_hits, promoted_hits)
        
        # Marcar como protegidas (no se eliminan de active, solo se protegen)
        self.protected[indices] = True
        
        if verbose:
            print(f"🏆 Promoción: {len(indices)} neuronas → Vault (total: {len(self.vault_genome)})")
    
    
    def consolidate_memory(self, verbose=False):
        """
        Comprime neuronas cercanas en símbolos (vectorizado, con promedio angular correcto).
        Mantiene top-2 prototipos por cluster (Fase 3: Protection Logic).
        """
        if len(self.genome) < MIN_CLUSTER_SIZE * 2:
            return  # Muy pocas neuronas
        
        # 1. Agrupamiento espacial (grid-based clustering)
        clusters = {}
        centers = self.genome[:, :3]  # cx, cy, cz
        
        for i, center in enumerate(centers):
            grid_key = self.cortex.quantize(center)
            if grid_key not in clusters:
                clusters[grid_key] = []
            clusters[grid_key].append(i)
        
        # 2. Simbolización con protección de prototipos
        new_symbols = []
        neurons_to_remove = set()
        
        for indices in clusters.values():
            if len(indices) < MIN_CLUSTER_SIZE:
                continue  # Mínimo para simbolizar
            
            # Extraer genes del cluster
            cluster_genes = self.genome[indices]
            
            # PROMEDIO VECTORIAL (correcto para ángulos)
            # Posición, radio, peso, stretch: promedio simple
            center_avg = np.mean(cluster_genes[:, :3], axis=0)
            radius_avg = np.mean(cluster_genes[:, 3])
            weight_avg = np.mean(cluster_genes[:, 4])
            stretch_avg = np.mean(cluster_genes[:, 5:8], axis=0)
            
            # ÁNGULO: Promedio vectorial (arctan2 de senos/cosenos)
            # Evita bug de promediar 350° y 10° = 180° (opuesto) en lugar de 0° (correcto)
            angles = cluster_genes[:, 8]
            angle_avg = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
            
            # Construir símbolo con MISMA estructura que genome
            symbol = np.concatenate([
                center_avg,      # [0:3]
                [radius_avg],    # [3]
                [weight_avg],    # [4]
                stretch_avg,     # [5:8]
                [angle_avg]      # [8]
            ])
            
            new_symbols.append(symbol)
            
            # FASE 3: Mantener top-2 prototipos por cluster
            cluster_scores = self.neuron_best_score[indices]
            top_k = min(2, len(indices))  # Mantener hasta 2 prototipos
            best_local_indices = np.argsort(-cluster_scores)[:top_k]  # Índices locales dentro del cluster
            best_global_indices = [indices[i] for i in best_local_indices]
            
            # Proteger prototipos (marcar como protected)
            for idx in best_global_indices:
                self.protected[idx] = True
            
            # Marcar para eliminación a todas EXCEPTO las prototipos
            for idx in indices:
                if idx not in best_global_indices:
                    neurons_to_remove.add(idx)
        
        # 3. Actualizar memoria simbólica
        if new_symbols:
            new_symbols_array = np.array(new_symbols)
            if len(self.symbol_genome) == 0:
                self.symbol_genome = new_symbols_array
            else:
                self.symbol_genome = np.vstack([self.symbol_genome, new_symbols_array])
        
        # 4. Eliminar neuronas comprimidas (mantener prototipos)
        keep_indices = [i for i in range(len(self.genome)) if i not in neurons_to_remove]
        if keep_indices:
            self.genome = self.genome[keep_indices]
            self.neuron_energy = self.neuron_energy[keep_indices]
            self.neuron_age = self.neuron_age[keep_indices]
            self.neuron_score = self.neuron_score[keep_indices]
            self.neuron_best_score = self.neuron_best_score[keep_indices]
            self.neuron_hits = self.neuron_hits[keep_indices]
            self.protected = self.protected[keep_indices]
        
        if verbose:
            protected_count = np.sum(self.protected)
            print(f"🧹 Consolidación: {len(self.symbol_genome)} símbolos | {len(self.genome)} neuronas activas | {protected_count} protegidas")

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
        """Maneja mitosis (nacimiento) y apoptosis (muerte) con inmunidad por edad y PROTECCIÓN VAULT."""
        new_genome = []
        new_energy = []
        new_age = []
        new_score = []
        new_best_score = []
        new_hits = []
        new_protected = []
        
        born_count = 0
        dead_count = 0
        
        # Inmunidad para jóvenes
        young_immunity = self.neuron_age < YOUNG_AGE_LIMIT
        
        for i in range(len(self.genome)):
            energy = self.neuron_energy[i]
            gene = self.genome[i]
            age = self.neuron_age[i]
            is_young = young_immunity[i]
            is_protected = self.protected[i]  # NUEVO: Vault protection
            
            # --- APOPTOSIS (MUERTE) con Inmunidad y Protección ---
            # Protegidas: NUNCA mueren (vault champions)
            # Jóvenes: inmunes (no mueren por baja energía)
            # Viejos: mueren si energy <= 0
            if energy <= 0 and not is_young and not is_protected and len(self.genome) - dead_count > self.MIN_NEURONS:
                dead_count += 1
                continue # Skip (Borrar)
                
            # --- MITOSIS (REPRODUCCIÓN) ---
            # Si tiene mucha energía, se divide
            if energy > self.MITOSIS_THRESHOLD and len(self.genome) + born_count < self.MAX_NEURONS:
                # 1. La madre sobrevive pero gasta energía en el parto
                new_genome.append(gene)
                new_energy.append(energy / 2)
                new_age.append(age)
                new_score.append(self.neuron_score[i])
                new_best_score.append(self.neuron_best_score[i])
                new_hits.append(self.neuron_hits[i])
                new_protected.append(self.protected[i])
                
                # 2. La hija nace con mutación leve
                mutation = np.random.normal(0, 0.1, size=len(gene))
                child_gene = gene + mutation
                child_gene[3] = np.clip(child_gene[3], 0.1, 2.0)  # Radio válido
                child_gene[4] = np.clip(child_gene[4], -1.0, 1.0) # Peso válido
                
                new_genome.append(child_gene)
                new_energy.append(energy / 2)
                new_age.append(0)  # Recién nacida
                new_score.append(0.0)  # Score inicial
                new_best_score.append(0.0)
                new_hits.append(0)
                new_protected.append(False)  # Hija no protegida
                
                born_count += 1
                self.neurons_born_by_mitosis += 1
            else:
                # Sobrevive normal
                new_genome.append(gene)
                new_energy.append(energy)
                new_age.append(age)
                new_score.append(self.neuron_score[i])
                new_best_score.append(self.neuron_best_score[i])
                new_hits.append(self.neuron_hits[i])
                new_protected.append(self.protected[i])
        
        # LÍMITE DURO: Forzar muerte de los más débiles si excedemos MAX_NEURONS
        # PERO: Protegidas siempre sobreviven
        if len(new_genome) > self.MAX_NEURONS:
            energies = np.array(new_energy)
            protected_flags = np.array(new_protected)
            
            # Separar protegidas de no protegidas
            protected_indices = np.where(protected_flags)[0]
            unprotected_indices = np.where(~protected_flags)[0]
            
            # Ordenar no protegidas por energía
            unprotected_energies = energies[unprotected_indices]
            sorted_unprotected = unprotected_indices[np.argsort(unprotected_energies)]
            
            # Calcular cuántas matar
            to_kill = len(new_genome) - self.MAX_NEURONS
            
            # Matar solo no protegidas (las más débiles)
            if len(sorted_unprotected) >= to_kill:
                kill_indices = set(sorted_unprotected[:to_kill])
                keep_indices = [i for i in range(len(new_genome)) if i not in kill_indices]
            else:
                # Si no hay suficientes no protegidas, mantener todas las protegidas + las mejores no protegidas
                keep_indices = list(protected_indices) + list(sorted_unprotected[-(self.MAX_NEURONS - len(protected_indices)):])
            
            new_genome = [new_genome[i] for i in keep_indices]
            new_energy = [new_energy[i] for i in keep_indices]
            new_age = [new_age[i] for i in keep_indices]
            new_score = [new_score[i] for i in keep_indices]
            new_best_score = [new_best_score[i] for i in keep_indices]
            new_hits = [new_hits[i] for i in keep_indices]
            new_protected = [new_protected[i] for i in keep_indices]
        
        self.genome = np.array(new_genome)
        self.neuron_energy = np.array(new_energy)
        self.neuron_age = np.array(new_age)
        self.neuron_score = np.array(new_score)
        self.neuron_best_score = np.array(new_best_score)
        self.neuron_hits = np.array(new_hits)
        self.protected = np.array(new_protected)
        
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
    Incluye energía, edad, vault, símbolos, tracking, y decision_threshold para preservar la experiencia.
    Fase 4: Vault + Performance Tracking.
    """
    brain_state = {
        # Active neurons
        'genome': brain.genome,
        'neuron_energy': brain.neuron_energy,
        'neuron_age': brain.neuron_age,
        
        # Performance tracking (Fase 2)
        'neuron_score': brain.neuron_score,
        'neuron_best_score': brain.neuron_best_score,
        'neuron_hits': brain.neuron_hits,
        'protected': brain.protected,
        
        # Vault system (Fase 2)
        'vault_genome': brain.vault_genome,
        'vault_score': brain.vault_score,
        'vault_hits': brain.vault_hits,
        
        # Symbols (Fase 1)
        'symbol_genome': brain.symbol_genome,
        
        # Brain state
        'decision_threshold': brain.decision_threshold,
        'energy': brain.energy,
        
        'metadata': metadata or {}
    }
    np.save(filename, brain_state, allow_pickle=True)
    print(f"   💾 Estado completo guardado: {filename}")

def load_brain_state(filename):
    """
    Carga el estado COMPLETO del cerebro.
    Fase 4: Restaura vault + tracking arrays.
    """
    state = np.load(filename, allow_pickle=True).item()
    
    brain = NeuroPlasticBrain(initial_neurons=10)
    
    # Active neurons
    brain.genome = state['genome']
    brain.neuron_energy = state['neuron_energy']
    brain.neuron_age = state['neuron_age']
    
    # Performance tracking (Fase 2) - con fallback para compatibilidad
    brain.neuron_score = state.get('neuron_score', np.zeros(len(brain.genome)))
    brain.neuron_best_score = state.get('neuron_best_score', np.zeros(len(brain.genome)))
    brain.neuron_hits = state.get('neuron_hits', np.zeros(len(brain.genome), dtype=int))
    brain.protected = state.get('protected', np.zeros(len(brain.genome), dtype=bool))
    
    # Vault system (Fase 2) - con fallback
    brain.vault_genome = state.get('vault_genome', np.empty((0, 9)))
    brain.vault_score = state.get('vault_score', np.array([]))
    brain.vault_hits = state.get('vault_hits', np.array([], dtype=int))
    
    # Symbols (Fase 1)
    brain.symbol_genome = state.get('symbol_genome', np.empty((0, 9)))
    
    # Brain state
    brain.decision_threshold = state.get('decision_threshold', 0.6)
    brain.energy = state.get('energy', 100.0)
    
    metadata = state.get('metadata', {})
    print(f"   📂 Estado cargado: {filename}")
    if 'generation' in metadata:
        print(f"      Gen: {metadata['generation']} | Balance: ${metadata.get('balance', 0):.2f}")
    print(f"      Neuronas: {len(brain.genome)} activas | Vault: {len(brain.vault_genome)} | Símbolos: {len(brain.symbol_genome)}")
    print(f"      Protegidas: {np.sum(brain.protected)} | Umbral: {brain.decision_threshold:.2f} | Energía: {brain.energy:.1f}")
    
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
                 # El campeón pasa directo (Elitismo) - HEREDAR CEREBRO COMPLETO
                 brain.genome = np.array([np.copy(g) for g in current_best_genome])
                 brain.neuron_energy = np.full(len(brain.genome), 5.0)
                 brain.neuron_age = np.zeros(len(brain.genome))
                 
                 # NUEVO: Heredar tracking arrays del mejor (no resetear)
                 if current_best_brain is not None:
                     brain.neuron_score = np.copy(current_best_brain.neuron_score)
                     brain.neuron_best_score = np.copy(current_best_brain.neuron_best_score)
                     brain.neuron_hits = np.copy(current_best_brain.neuron_hits)
                     brain.protected = np.copy(current_best_brain.protected)
                     
                     # Heredar vault completo
                     brain.vault_genome = np.copy(current_best_brain.vault_genome)
                     brain.vault_score = np.copy(current_best_brain.vault_score)
                     brain.vault_hits = np.copy(current_best_brain.vault_hits)
                     
                     # Heredar símbolos
                     brain.symbol_genome = np.copy(current_best_brain.symbol_genome)
                 else:
                     # Fallback: resetear si no hay cerebro previo
                     brain.neuron_score = np.zeros(len(brain.genome))
                     brain.neuron_best_score = np.zeros(len(brain.genome))
                     brain.neuron_hits = np.zeros(len(brain.genome), dtype=int)
                     brain.protected = np.zeros(len(brain.genome), dtype=bool)
                 
                 brain.decision_threshold = current_best_threshold
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
                
                brain.genome = np.array(temp_genome)
                brain.neuron_energy = np.full(len(brain.genome), 5.0)
                brain.neuron_age = np.zeros(len(brain.genome))
                
                # NUEVO: Heredar tracking arrays parcialmente (con ajuste de tamaño)
                if current_best_brain is not None and len(current_best_brain.genome) > 0:
                    old_size = len(current_best_brain.genome)
                    new_size = len(brain.genome)
                    
                    # Copiar datos existentes + extender con zeros si creció
                    if new_size > old_size:
                        brain.neuron_score = np.concatenate([current_best_brain.neuron_score, np.zeros(new_size - old_size)])
                        brain.neuron_best_score = np.concatenate([current_best_brain.neuron_best_score, np.zeros(new_size - old_size)])
                        brain.neuron_hits = np.concatenate([current_best_brain.neuron_hits, np.zeros(new_size - old_size, dtype=int)])
                        brain.protected = np.concatenate([current_best_brain.protected, np.zeros(new_size - old_size, dtype=bool)])
                    else:
                        # Si redujo, truncar
                        brain.neuron_score = np.copy(current_best_brain.neuron_score[:new_size])
                        brain.neuron_best_score = np.copy(current_best_brain.neuron_best_score[:new_size])
                        brain.neuron_hits = np.copy(current_best_brain.neuron_hits[:new_size])
                        brain.protected = np.copy(current_best_brain.protected[:new_size])
                    
                    # Heredar vault completo (no depende del tamaño del genome)
                    brain.vault_genome = np.copy(current_best_brain.vault_genome)
                    brain.vault_score = np.copy(current_best_brain.vault_score)
                    brain.vault_hits = np.copy(current_best_brain.vault_hits)
                    
                    # Heredar símbolos
                    brain.symbol_genome = np.copy(current_best_brain.symbol_genome)
                else:
                    # Fallback
                    brain.neuron_score = np.zeros(len(brain.genome))
                    brain.neuron_best_score = np.zeros(len(brain.genome))
                    brain.neuron_hits = np.zeros(len(brain.genome), dtype=int)
                    brain.protected = np.zeros(len(brain.genome), dtype=bool)
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
        
        # Print estadísticas de esta generación (Fase 5: Vault stats)
        if gen_best_brain is not None:
            novelty_births = gen_best_brain.neurons_born_by_novelty
            mitosis_births = gen_best_brain.neurons_born_by_mitosis
            total_neurons = len(gen_best_brain.genome)
            total_symbols = len(gen_best_brain.symbol_genome)
            total_vault = len(gen_best_brain.vault_genome)
            total_protected = np.sum(gen_best_brain.protected) if len(gen_best_brain.protected) > 0 else 0
            
            print(f"🏆 MEJOR DE GEN {gen}: Balance ${gen_best_fitness:.2f}")
            print(f"   📊 Memoria: {total_neurons} activas | {total_vault} vault | {total_symbols} símbolos | {total_protected} protegidas")
            print(f"   🌱 Nacimientos: {novelty_births} novedad + {mitosis_births} mitosis | Umbral: {umbral_display:.2f}")
        else:
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
        
        # A.1 NOVELTY DETECTION: Solo si NO hay trade abierto (evita shape mismatch)
        allow_spawn = (position == 0)
        brain.check_novelty(inputs[t], activations, allow_spawn=allow_spawn)
        
        current_price = prices_aligned[t]
        
        # A.2 Metabolismo: Afecta ENERGÍA, no balance
        brain.energy -= METABOLIC_COST_FIXED
        
        # A.3 Muerte por inanición
        if brain.energy <= 0:
            balance = 0  # Penalización extrema
            break  # Termina simulación
        
        # B. METABOLISMO NEURONAL
        brain.neuron_energy -= brain.STARVATION_RATE
        brain.neuron_age += 1
        
        # C. MEMORIA
        if position != 0:
            trade_memory_activations.append(activations)

        # D. LÓGICA TRADING (cierre de posición)
        if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price
            if position == -1: pnl_pct *= -1
            pnl_real = pnl_pct * leverage
            
            profit = current_bet * pnl_real
            cost = (current_bet * leverage) * 0.004  # Fee sobre valor nocional (realista)
            balance += current_bet + (profit - cost)  # Devolver capital + ganancia/pérdida
            
            # D.1 Recuperar energía con ganancias
            if profit > 0:
                energy_boost = profit * ENERGY_FROM_PROFIT
                brain.energy = min(brain.energy + energy_boost, brain.MAX_ENERGY)
            
            # E. OCURRE EL APRENDIZAJE
            if len(trade_memory_activations) > 0:
                # Filtrar activaciones válidas (mismo shape)
                valid_activations = [act for act in trade_memory_activations if len(act) == len(brain.genome)]
                
                if len(valid_activations) > 0:
                    mean_activations = np.mean(valid_activations, axis=0)
                    brain.learn_and_adapt(pnl_pct, mean_activations)  # Usar pnl_pct sin leverage
                    
                    # E.1 ACTUALIZAR SCORES (Fase 2: Performance Tracking)
                    pnl_net = (profit - cost) / current_bet  # Retorno neto real
                    brain.update_scores(pnl_net, mean_activations)
                
                trade_memory_activations = []
            
            position = 0
            current_bet = 0
            trade_memory_activations = []

        if position == 0:
            # Respetar modo de trading configurado
            # Usar umbral de decisión del cerebro (gen de personalidad)
            if signal > brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]: 
                position = 1
                entry_price = current_price
                current_bet = min(balance * BET_PERCENTAGE, MAX_BET if BET_MODE == "CAPPED" else balance)
                current_bet = min(current_bet, balance)
                balance -= current_bet  # Restar margen al abrir
                trade_memory_activations = [activations]
                
            elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]:
                position = -1
                entry_price = current_price
                current_bet = min(balance * BET_PERCENTAGE, MAX_BET if BET_MODE == "CAPPED" else balance)
                current_bet = min(current_bet, balance)
                balance -= current_bet  # Restar margen al abrir
                trade_memory_activations = [activations]

        # F. PLASTICIDAD ESTRUCTURAL
        if t % 50 == 0: # Más frecuente para dinamismo
            old_len = len(brain.genome)
            brain._structural_plasticity()
            if len(brain.genome) != old_len:
                # Si cambió la estructura, la memoria del trade ya no es válida (cambian índices)
                trade_memory_activations = []
        
        # F.1 PROMOCIÓN A VAULT (Fase 2: cada 500 velas, más frecuente)
        if t % 500 == 0 and t > 0:
            brain.promote_to_vault(verbose=(t % 5000 == 0))
            
        # F.2 CONSOLIDACIÓN SIMBÓLICA (cada 1000 velas)
        if SYMBOLIC_COMPRESSION and t % CONSOLIDATION_INTERVAL == 0 and t > 0:
            brain.consolidate_memory(verbose=(t % (CONSOLIDATION_INTERVAL * 5) == 0))
            trade_memory_activations = []  # Limpiar porque cambiaron índices
            
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
        
        # Novelty detection solo si NO hay trade abierto (online learning)
        if ONLINE_LEARNING:
            allow_spawn = (position == 0)
            brain.check_novelty(inputs[t], activations, allow_spawn=allow_spawn)
        
        current_price = prices[t]
        act = 0
        
        # APRENDIZAJE ONLINE: Metabolismo (afecta energía, no balance)
        if ONLINE_LEARNING:
            brain.energy -= METABOLIC_COST_FIXED
            brain.neuron_energy -= brain.STARVATION_RATE
            brain.neuron_age += 1
            
            # Muerte por inanición en test online
            if brain.energy <= 0:
                balance = 0
                break
        
        # Memoria del trade actual
        if ONLINE_LEARNING and position != 0:
            trade_memory_activations.append(activations)
        
        # Cierre (mismas reglas que training)
        if (signal > 0.8 and position == -1) or (signal < -0.8 and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price
            if position == -1: pnl_pct *= -1
            pnl_real = pnl_pct * LEVERAGE
            
            profit = current_bet * pnl_real
            cost = (current_bet * LEVERAGE) * FEE  # Fee sobre valor nocional (realista)
            balance += current_bet + (profit - cost)  # Devolver capital + ganancia/pérdida
            
            # Recuperar energía con ganancias (training + test online)
            if ONLINE_LEARNING and profit > 0:
                energy_boost = profit * ENERGY_FROM_PROFIT
                brain.energy = min(brain.energy + energy_boost, brain.MAX_ENERGY)
            
            # APRENDIZAJE ONLINE: Actualizar energías según resultado
            if ONLINE_LEARNING and len(trade_memory_activations) > 0:
                # Filtrar activaciones válidas (mismo shape)
                valid_activations = [act for act in trade_memory_activations if len(act) == len(brain.genome)]
                
                if len(valid_activations) > 0:
                    mean_activations = np.mean(valid_activations, axis=0)
                    brain.learn_and_adapt(pnl_pct, mean_activations)
                
                trade_memory_activations = []
            
            position = 0
            current_bet = 0
        
        # Apertura (usa umbral del cerebro, igual que training)
        if position == 0:
            if signal > brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]: 
                position = 1
                entry_price = current_price * (1 + SLIPPAGE) 
                current_bet = calculate_bet(balance)  # Usa configuración global
                balance -= current_bet  # Restar margen al abrir
                act = 1
                if ONLINE_LEARNING:
                    trade_memory_activations = [activations]
            elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]: 
                position = -1
                entry_price = current_price * (1 - SLIPPAGE) 
                current_bet = calculate_bet(balance)  # Usa configuración global
                balance -= current_bet  # Restar margen al abrir
                act = -1
                if ONLINE_LEARNING:
                    trade_memory_activations = [activations]
        
        # APRENDIZAJE ONLINE: Plasticidad estructural cada 50 velas
        # F. PLASTICIDAD ESTRUCTURAL (cada 50 velas)
        if ONLINE_LEARNING and t % 50 == 0:
            brain._structural_plasticity()
            trade_memory_activations = [] # Limpiar memoria
        
        # F.1 CONSOLIDACIÓN SIMBÓLICA (cada 1000 velas)
        if SYMBOLIC_COMPRESSION and t % CONSOLIDATION_INTERVAL == 0 and t > 0:
            brain.consolidate_memory(verbose=(t % (CONSOLIDATION_INTERVAL * 5) == 0))
            trade_memory_activations = [] # Limpiar memoria de trade porque el cerebro cambió de tamaño
        
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