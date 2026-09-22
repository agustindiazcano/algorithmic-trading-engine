import numpy as np
import matplotlib.pyplot as plt
import copy

# ==============================================================================
# ⚙️ CONFIGURACIÓN "GOD MODE"
# ==============================================================================
SYMBOL = "PEPEUSDT"
INTERVAL = "1m"
BACKTEST_CANDLES = 20000  # Prueba con 20k o 50k
START_DATE = None         # None = Datos recientes

# Configuración Evolutiva
POPULATION_SIZE = 15      # 15 Traders compitiendo (Individuos por generación)
EVALUATION_WINDOW = 200   # Evaluamos quién es el líder cada 200 velas (más tiempo para demostrar valía)

# Configuración Biológica (Por Cerebro)
INITIAL_NEURONS = 10      # Empiezan pequeños
MAX_NEURONS = 150         # Pueden crecer hasta 150 si son exitosos
MITOSIS_THRESHOLD = 3.0   # Energía necesaria para duplicarse (Alto = difícil crecer)

# ==============================================================================
# 🧠 CEREBRO NEUROPLÁSTICO (MITOSIS + APOPTOSIS)
# ==============================================================================

class NeuroPlasticBrain:
    def __init__(self, initial_neurons=10, A_dc=1.618):
        self.A_dc = A_dc
        self.genome = [] 
        self.neuron_energy = [] 
        self.neuron_age = []
        
        # Identidad
        self.name = "Genesis"
        self.virtual_balance = 10000.0
        self.position = 0
        self.entry_price = 0
        
        # Iniciar neuronas aleatorias
        for _ in range(initial_neurons):
            self._add_random_neuron()
            
        self.genome = np.array(self.genome)
        self.neuron_energy = np.array(self.neuron_energy)
        self.neuron_age = np.array(self.neuron_age)

    def _add_random_neuron(self):
        gene = np.concatenate([
            np.random.uniform(-1.0, 1.0, 3), 
            [np.random.uniform(0.5, 1.5)],   
            [np.random.uniform(-1.0, 1.0)],  
            np.random.uniform(0.5, 1.5, 3),
            [np.random.uniform(-np.pi/2, np.pi/2)] 
        ])
        self.genome.append(gene)
        self.neuron_energy.append(1.0) 
        self.neuron_age.append(0)

    def predict(self, points):
        if len(self.genome) == 0: return 0.0, None
        
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
        activations = np.minimum(1.0, raw_overlap * 3.0) 
        
        total_activation = np.sum(activations * weights)
        return np.tanh(total_activation), activations

    def biological_step(self, reward_signal, activations):
        """ Ciclo de vida: Comer energía, crecer o morir """
        if activations is None: return

        # 1. Alimentación
        impact = activations * reward_signal
        # Si acierta gana 0.1, si falla pierde 0.2 (Castigo duro)
        energy_delta = np.where(impact > 0, impact * 0.1, impact * 0.2)
        
        self.neuron_energy += energy_delta
        self.neuron_energy -= 0.001 # Costo metabólico por vivir
        self.neuron_age += 1
        
        # 2. Reestructuración (Mitosis/Apoptosis)
        self._structural_plasticity()

    def _structural_plasticity(self):
        new_genome = []
        new_energy = []
        new_age = []
        born = 0
        
        for i in range(len(self.genome)):
            e = self.neuron_energy[i]
            if e <= 0: continue # Muerte (Apoptosis)
            
            # Mitosis (Solo si hay espacio)
            if e > MITOSIS_THRESHOLD and len(self.genome) + born < MAX_NEURONS:
                # Madre
                new_genome.append(self.genome[i])
                new_energy.append(e * 0.5)
                new_age.append(self.neuron_age[i])
                
                # Hija
                child = np.copy(self.genome[i])
                child += np.random.normal(0, 0.05, size=child.shape) # Mutación leve
                child[3] = abs(child[3])
                child[5:8] = abs(child[5:8])
                
                new_genome.append(child)
                new_energy.append(e * 0.5)
                new_age.append(0)
                born += 1
            else:
                new_genome.append(self.genome[i])
                new_energy.append(e)
                new_age.append(self.neuron_age[i])

        if len(new_genome) > 0:
            self.genome = np.array(new_genome)
            self.neuron_energy = np.array(new_energy)
            self.neuron_age = np.array(new_age)
        else:
            # Si mueren todas, renace una aleatoria (Fénix)
            self.genome = []
            self.neuron_energy = []
            self.neuron_age = []
            self._add_random_neuron()

    def clone_with_mutation(self):
        """ Para la reproducción entre individuos de la colmena """
        clon = copy.deepcopy(self)
        # Mutación global leve
        mask = np.random.rand(*clon.genome.shape) < 0.1
        clon.genome[mask] += np.random.normal(0, 0.05, size=clon.genome[mask].shape)
        return clon

# ==============================================================================
# 🧬 COLMENA (HIVE MIND) - GESTOR DE POBLACIÓN
# ==============================================================================

class HiveMind:
    def __init__(self):
        self.population = []
        for i in range(POPULATION_SIZE):
            brain = NeuroPlasticBrain(initial_neurons=INITIAL_NEURONS)
            brain.name = f"Individuo_{i}"
            self.population.append(brain)
            
        self.leader_idx = 0
        self.candles_processed = 0
        self.generation = 0

    def process_candle(self, input_3d, price, price_change_last_candle):
        signals = []
        
        for brain in self.population:
            # 1. Predecir
            sig, activations = brain.predict(input_3d)
            signals.append(sig)
            
            # 2. Actualizar Balance Virtual (Trading simulado)
            self.update_virtual_trading(brain, price, sig)
            
            # 3. Aprendizaje Biológico (Mitosis interna)
            # Recompensa basada en lo que pasó en la vela ANTERIOR vs predicción anterior
            # (Simplificación: Usamos la dirección de la vela actual vs señal actual para feedback inmediato)
            # Un feedback más puro usaría t-1, pero para plasticidad rápida esto sirve:
            # Si predigo UP y la vela es VERDE -> Recompensa
            
            reward = 0
            if sig > 0.1: reward = price_change_last_candle * 100 
            elif sig < -0.1: reward = -price_change_last_candle * 100
            
            brain.biological_step(reward, activations)
            
        self.candles_processed += 1
        
        # 4. Evolución Externa (Torneo)
        if self.candles_processed % EVALUATION_WINDOW == 0:
            self.evolve_population()
            
        return signals[self.leader_idx]

    def update_virtual_trading(self, brain, price, signal):
        # Cierre
        if brain.position != 0:
            close = False
            if brain.position == 1 and signal < -0.5: close = True
            if brain.position == -1 and signal > 0.5: close = True
            
            if close:
                pnl = (price - brain.entry_price) / brain.entry_price
                if brain.position == -1: pnl *= -1
                brain.virtual_balance *= (1 + pnl * 50 - 0.001) # x50 Lev
                brain.position = 0

        # Apertura
        if brain.position == 0:
            if abs(signal) > 0.8:
                brain.entry_price = price
                brain.position = 1 if signal > 0 else -1
                brain.virtual_balance -= (brain.virtual_balance * 0.001)

    def evolve_population(self):
        self.generation += 1
        balances = [b.virtual_balance for b in self.population]
        best_idx = np.argmax(balances)
        leader = self.population[best_idx]
        
        if best_idx != self.leader_idx:
            print(f"♻️ [GEN {self.generation}] Nuevo Líder: {leader.name} (${leader.virtual_balance:.0f}) | Neuronas: {len(leader.genome)}")
            self.leader_idx = best_idx
        
        # Selección Natural: Matar al 50% más pobre
        sorted_indices = np.argsort(balances)[::-1] # Mejores primero
        survivors = []
        
        # Los mejores sobreviven y se reinician para la siguiente ronda
        for i in range(POPULATION_SIZE // 2):
            idx = sorted_indices[i]
            survivor = self.population[idx]
            survivor.virtual_balance = 10000.0 # Reset marcador
            survivors.append(survivor)
            
        # Rellenar con clones mutados de los mejores
        new_pop = survivors[:]
        i = 0
        while len(new_pop) < POPULATION_SIZE:
            parent = survivors[i % len(survivors)]
            child = parent.clone_with_mutation()
            child.name = f"Gen{self.generation}_HijoDe_{parent.name}"
            child.virtual_balance = 10000.0
            # IMPORTANTE: El hijo hereda las neuronas (memoria) del padre
            new_pop.append(child)
            i += 1
            
        self.population = new_pop
        self.leader_idx = 0 # El mejor anterior ahora es el 0
        print(f"   🧬 Población renovada. Tamaño actual de redes: {[len(p.genome) for p in self.population[:5]]}...")

# ==============================================================================
# 🛠️ CARGA DE DATOS Y EJECUCIÓN
# ==============================================================================

# (Aquí pegas las funciones get_data_and_transform y el main loop igual que antes)
# Solo recuerda que al llamar a hive.process_candle, ahora pide 3 argumentos:
# hive.process_candle(input_3d, current_price, price_change_pct)

import requests
import time

def get_data_and_transform(symbol, interval, limit, start_date=None):
    # ... (Tu código de descarga de siempre) ...
    # Voy a poner un dummy simple para que el ejemplo funcione si copias y pegas
    # PERO USA TU FUNCIÓN DE DESCARGA ORIGINAL, ES MEJOR.
    pass 

if __name__ == "__main__":
    # Simulación rápida del concepto
    # Asegúrate de tener la función get_data_and_transform definida o importada
    print("Para ejecutar, integra este código con la función de descarga de datos que ya tienes.")
    print(f"Sistema configurado para: {POPULATION_SIZE} Individuos x {MAX_NEURONS} Neuronas máx.")