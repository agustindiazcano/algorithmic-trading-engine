import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- CEREBRO VECTORIZADO (MÁXIMA VELOCIDAD) ---
class DynamicCausalBrain:
    def __init__(self, n_neurons=20, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        # Genoma como matriz (N, 5) para vectorización
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(0, 1, 2),   # x, y
                np.random.uniform(0, 2.5, 1), # z (Contexto)
                [np.random.uniform(0.1, 0.5)],# Radio
                [np.random.uniform(0.0, 1.0)] # Peso
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def predict(self, points):
        # Vectorized Prediction (Cientos de veces más rápido)
        # points: (P, 3)
        # centers: (N, 3)
        if len(self.genome) == 0: return np.zeros(len(points))
        
        centers = self.genome[:, :3]
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        
        # Broadcasting: (P, 1, 3) - (1, N, 3) -> (P, N, 3)
        diff = points[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2) # (P, N)
        
        # Overlap = max(0, 1 - dist/radius)
        # Evitar división por cero
        safe_radii = np.maximum(radii, 1e-6)
        overlap = np.maximum(0, 1 - dists / safe_radii[None, :]) # (P, N)
        
        # Weighted activation
        activations = overlap * weights[None, :] # (P, N)
        
        # Max aggregation per point
        # Si no hay activacion, 0
        return np.max(activations, axis=1)

    def add_neurons(self, n_new):
        for _ in range(n_new):
            gene = np.concatenate([
                np.random.uniform(0, 1, 2),
                np.random.uniform(0, 2.5, 1),
                [np.random.uniform(0.1, 0.5)],
                [np.random.uniform(0.0, 1.0)]
            ])
            self.genome = np.vstack([self.genome, gene])

# --- ENTRENADOR INCREMENTAL ---
def train_task(brain, X, y, task_name, generations=1000, pop_size=50):
    print(f"\n🧠 Entrenando Tarea: {task_name}...")
    start_t = time.time()
    
    current_best_genome = np.copy(brain.genome)
    best_fitness = -np.inf
    history = []

    for gen in range(generations):
        # Mutación Vectorizada (Más limpia)
        # Creamos población (Pop, N, 5)
        # Pero es más fácil iterar para mantener la lógica de "selección"
        
        population = []
        # Elitismo: Mantenemos el mejor sin cambios
        population.append(current_best_genome)
        
        # Generar mutantes
        for _ in range(pop_size - 1):
            mutant = np.copy(current_best_genome)
            if np.random.rand() < 0.9: 
                mask = np.random.rand(*mutant.shape) < 0.1
                noise = np.random.normal(0, 0.1, size=mutant.shape)
                mutant[mask] += noise[mask]
                mutant[:, 3] = np.abs(mutant[:, 3])
            population.append(mutant)

        # Evaluar
        best_gen_fitness = -np.inf
        best_gen_genome = None
        
        # Aquí todavía loopeamos population (50 no es grave), 
        # pero predict() adentro ya es vectorizado y rápido.
        for ind_genome in population:
            brain.genome = ind_genome
            preds = brain.predict(X)
            mse = np.mean((y - preds)**2)
            fitness = -mse 
            
            if fitness > best_gen_fitness:
                best_gen_fitness = fitness
                best_gen_genome = ind_genome

        # Update Global Best
        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            current_best_genome = np.copy(best_gen_genome)
            
        history.append(-best_fitness)
        
        if gen % 200 == 0 or gen == generations-1:
            print(f"  [Gen {gen}] Error (MSE): {-best_fitness:.5f}")

    brain.genome = current_best_genome
    print(f"✅ {task_name} Completado en {time.time()-start_t:.1f}s. Error Final: {-best_fitness:.5f}")
    return history

def get_math_data(operation="SUM"):
    x = np.random.rand(100); y = np.random.rand(100)
    inputs = []; targets = []
    
    for i in range(100):
        if operation == "SUM":
            inputs.append([x[i], y[i], 0.0])
            targets.append((x[i] + y[i]) / 2.0)
        elif operation == "SUB":
            inputs.append([x[i], y[i], 1.0])
            targets.append((x[i] - y[i] + 1) / 2.0)
        elif operation == "COMPLEX":
            inputs.append([x[i], y[i], 2.0])
            targets.append(np.sin(x[i] * np.pi) * y[i])
    return np.array(inputs), np.array(targets)

def run_math_class():
    brain = DynamicCausalBrain(n_neurons=15, A_dc=1.618)
    
    X_sum, y_sum = get_math_data("SUM")
    train_task(brain, X_sum, y_sum, "SUMA (Contexto Z=0)", generations=1000)
    
    print("\n✨ Agregando nuevas neuronas...")
    brain.add_neurons(15) 
    
    X_sub, y_sub = get_math_data("SUB")
    X_mixed = np.vstack([X_sum, X_sub])
    y_mixed = np.concatenate([y_sum, y_sub])
    
    # MODIFICADO: 100 Gens
    train_task(brain, X_mixed, y_mixed, "RESTA + REPASO SUMA (Contexto Z=1)", generations=100)

    print("\n🎓 EXAMEN FINAL DE MATEMÁTICA VOLUMÉTRICA")
    test_cases = [
        ([0.2, 0.2, 0.0], "0.2 + 0.2", 0.2), # (0.4)/2 = 0.2
        ([0.8, 0.1, 0.0], "0.8 + 0.1", 0.45), # (0.9)/2 = 0.45
        ([1.0, 1.0, 0.0], "1.0 + 1.0", 1.0),
        
        ([0.5, 0.5, 1.0], "0.5 - 0.5", 0.5), # (0)/2 + 0.5 = 0.5
        ([0.9, 0.1, 1.0], "0.9 - 0.1", 0.9), # (0.8)/2 + 0.5 = 0.9
        
        # OOD
        ([100.0, 1000.0, 0.0], "100 + 1000 (OOD)", 550.0),
        ([39200.0, 5839.0, 1.0], "39200 - 5839 (OOD)", 16681.0)
    ]
    
    for inp, desc, expected in test_cases:
        pred = brain.predict(np.array([inp]))[0]
        print(f"Operación: {desc:<20} | Input Z={inp[2]} | Predicción IA: {pred:.4f} (Target: {expected:.4f})")
    
    # Visualization (simplified/safe)
    try:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        for neuron in brain.genome:
            x, y, z, r, w = neuron
            if z < 0.5: color = 'blue'
            elif z < 1.5: color = 'red'
            else: color = 'green'
            ax.scatter(x, y, z, s=r*100, c=color, alpha=0.3) # Scatter es mas rapido que wireframe
        plt.savefig('test_logic_v4_result.png')
        print("\nGráfico guardado: test_logic_v4_result.png")
    except Exception as e:
        print(f"Error plot: {e}")

if __name__ == "__main__":
    run_math_class()