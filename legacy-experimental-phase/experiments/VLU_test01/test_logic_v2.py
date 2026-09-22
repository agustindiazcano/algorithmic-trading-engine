import numpy as np

class DynamicCausalBrain:
    def __init__(self, n_neurons=5, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        self.genome = []
        for _ in range(self.n_neurons):
            # Neurona: [x, y, z, radius_base, weight]
            # Inicializamos cerca del 0,1 para XOR
            gene = np.concatenate([
                np.random.uniform(-0.5, 1.5, 3), # Posición (x,y,z)
                [np.random.uniform(0.1, 0.5)],   # Radio base pequeño
                [np.random.uniform(0.0, 1.0)]    # Peso
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def activation_vl(self, point, neuron):
        center = neuron[:3]
        radius = neuron[3] * self.A_dc
        weight = neuron[4]
        
        dist = np.linalg.norm(point - center)
        if radius <= 1e-6: return 0
        overlap = max(0, 1 - dist / radius)
        return weight * overlap

    def predict(self, points):
        results = []
        for p in points:
            activations = [self.activation_vl(p, n) for n in self.genome]
            results.append(max(activations) if activations else 0)
        return np.array(results)

def train_genetic_robust(brain, X, y, generations=1000, population_size=100):
    population = [np.copy(brain.genome) for _ in range(population_size)]
    best_genome = None
    best_fitness = -np.inf
    
    for gen in range(generations):
        scores = []
        for ind_genome in population:
            brain.genome = ind_genome
            preds = brain.predict(X)
            
            # MSE negativo
            mse = np.mean((y - preds)**2)
            # Penalización suave por complejidad (radios muy grandes)
            penalty = np.mean(ind_genome[:, 3]) * 0.01
            fitness = -mse - penalty
            scores.append(fitness)

        scores = np.array(scores)
        elite_idx = np.argmax(scores)
        current_best = scores[elite_idx]
        
        if current_best > best_fitness:
            best_fitness = current_best
            best_genome = np.copy(population[elite_idx])
        
        # Selección: Torneo o Elitismo simple
        # Acá usamos elitismo agresivo: el top 20% se reproduce
        sorted_indices = np.argsort(scores)[::-1]
        survivors = [population[i] for i in sorted_indices[:population_size//5]]
        
        new_population = []
        while len(new_population) < population_size:
            # Cruzamiento (opcional, aca hacemos mutacion directa de supervivientes)
            parent = survivors[np.random.randint(len(survivors))]
            child = np.copy(parent)
            
            # Mutación
            if np.random.rand() < 0.8: # Alta probabilidad de mutar
                mask = np.random.rand(*child.shape) < 0.1 # Mutar 10% de los genes
                noise = np.random.normal(0, 0.1, size=child.shape)
                child[mask] += noise[mask]
                child[:, 3] = np.abs(child[:, 3]) # Radios positivos
            
            new_population.append(child)
            
        population = new_population

    brain.genome = best_genome
    return best_fitness

# Configurar XOR
X_xor = np.array([
    [0,0,0], [1,1,0], # FALSE (0)
    [1,0,0], [0,1,0], # TRUE (1)
    # [0,0,1], [1,1,1]  # Ruido extra (opcional)
])
y_xor = np.array([0, 0, 1, 1])

# Ejecutar con esteroides
brain = DynamicCausalBrain(n_neurons=4, A_dc=1.618) # 4 neuronas para cubrir opciones
fitness = train_genetic_robust(brain, X_xor, y_xor, generations=2000, population_size=100)

print(f"Fitness final: {fitness}")
preds = brain.predict(X_xor)
for i, p in enumerate(preds):
    print(f"In: {X_xor[i]} -> Out: {p:.4f} (Target: {y_xor[i]})")