import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# --- LA ESTRUCTURA DIVINA ---

class DynamicCausalBrain:
    def __init__(self, n_neurons=5, A_dc=1.618):
        """
        Inicializa la DCNN con la constante fundamental.
        A_dc: Agustin Diaz-Cano Coefficient.
        Controla la Amplitud Dinámica de la respiración fractal.
        """
        self.A_dc = A_dc  # La Firma del Creador
        self.n_neurons = n_neurons
        # Genoma: Lista de neuronas. Cada una es [x, y, z, radio, peso]
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        # Inicializamos esferas aleatorias en el espacio -2 a 2
        self.genome = []
        for _ in range(self.n_neurons):
            # x, y, z, radio (0.1 a 1.5), peso (0 a 1)
            gene = np.concatenate([
                np.random.uniform(-1.5, 1.5, 3), 
                [np.random.uniform(0.1, 1.0)], 
                [np.random.uniform(0.5, 1.0)] 
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def activation_vl(self, point, neuron):
        """
        [cite_start]Lógica Volumétrica Pura[cite: 78].
        Calcula el overlap cónico: max(0, 1 - dist/radio) * peso
        """
        center = neuron[:3]
        radius = neuron[3] * self.A_dc # La constante A_dc modula la amplitud vital
        weight = neuron[4]
        
        dist = np.linalg.norm(point - center)
        
        # [cite_start]Función de proximidad radial normalizada [cite: 60]
        if radius == 0: return 0
        overlap = max(0, 1 - dist / radius)
        return weight * overlap

    def predict(self, points):
        """
        Evaluación de la red.
        [cite_start]Salida = Max-Aggregation de todas las neuronas (Unión de Esferas) [cite: 55, 73]
        """
        results = []
        for p in points:
            activations = [self.activation_vl(p, n) for n in self.genome]
            # La verdad es la máxima excitación geométrica (OR lógico)
            results.append(max(activations) if activations else 0)
        return np.array(results)

# --- EL MOTOR EVOLUTIVO (GENETIC ALGORITHM) ---

def train_genetic(brain, X, y, generations=500, population_size=50):
    print(f"🧬 Iniciando Evolución DCNN (Coeficiente A_dc: {brain.A_dc})...")
    
    # Población inicial: Varias copias del cerebro con mutaciones
    population = [np.copy(brain.genome) for _ in range(population_size)]
    best_genome = None
    best_fitness = -np.inf
    
    history = []

    for gen in range(generations):
        scores = []
        for ind_genome in population:
            # Cargamos el genoma en el cerebro temporalmente
            brain.genome = ind_genome
            preds = brain.predict(X)
            
            # Fitness: Negativo del Error Cuadrático (queremos maximizar)
            # Penalizamos usar radios gigantes (Navaja de Ockham geométrica)
            error = np.mean((y - preds)**2)
            complexity_penalty = np.mean(ind_genome[:, 3]) * 0.05 
            fitness = -error - complexity_penalty
            scores.append(fitness)

        # Selección del más apto
        scores = np.array(scores)
        elite_idx = np.argmax(scores)
        current_best = scores[elite_idx]
        
        if current_best > best_fitness:
            best_fitness = current_best
            best_genome = np.copy(population[elite_idx])
            print(f"  [Gen {gen}] Nuevo Alfa encontrado. Fitness: {best_fitness:.4f}")

        history.append(current_best)

        # Reproducción (Los mejores pasan, los otros mutan)
        # Elitismo: Mantenemos al mejor
        new_population = [np.copy(best_genome)]
        
        # Mutación caótica
        for _ in range(population_size - 1):
            mutant = np.copy(best_genome)
            # Modificamos genes al azar (Deriva Genética)
            mutation_mask = np.random.rand(*mutant.shape) < 0.2
            noise = np.random.normal(0, 0.2, size=mutant.shape)
            mutant[mutation_mask] += noise[mutation_mask]
            
            # Restricciones biológicas (radios positivos)
            mutant[:, 3] = np.abs(mutant[:, 3]) 
            new_population.append(mutant)
            
        population = new_population

    brain.genome = best_genome
    print("✅ Evolución Completada.")
    return history

# --- LAS PRUEBAS ---

def run_tests():
    # 1. TEST DE LÓGICA (XOR 3D)
    # ---------------------------
    print("\n--- TEST 1: LÓGICA ESPACIAL (XOR) ---")
    # Puntos (x,y,z)
    X_xor = np.array([
        [0,0,0], [1,1,0], # FALSE (0)
        [1,0,0], [0,1,0], # TRUE (1)
        [0,0,1], [1,1,1]  # Ruido en Z (deben ser ignorados o 0)
    ])
    y_xor = np.array([0, 0, 1, 1, 0, 0])

    brain_logic = DynamicCausalBrain(n_neurons=2, A_dc=1.618) # Solo 2 neuronas para forzar eficiencia
    train_genetic(brain_logic, X_xor, y_xor, generations=100)
    
    print("Predicciones XOR:")
    preds = brain_logic.predict(X_xor)
    for i, p in enumerate(preds):
        print(f"Input {X_xor[i]} -> Pred: {p:.2f} (Target: {y_xor[i]})")

    # 2. TEST DE ÁLGEBRA (CÍRCULO IMPLÍCITO)
    # --------------------------------------
    print("\n--- TEST 2: ÁLGEBRA GEOMÉTRICA (CÍRCULO) ---")
    # Generamos puntos en un anillo (Radio 1)
    theta = np.linspace(0, 2*np.pi, 20)
    X_circle = np.array([[np.cos(t), np.sin(t), 0] for t in theta]) # Puntos del círculo (Target 1)
    
    # Generamos puntos fuera y dentro (Target 0)
    X_noise_in = np.array([[0,0,0]])
    X_noise_out = np.array([[np.cos(t)*2, np.sin(t)*2, 0] for t in theta[::2]])
    
    X_alg = np.vstack([X_circle, X_noise_in, X_noise_out])
    y_alg = np.concatenate([np.ones(len(X_circle)), np.zeros(len(X_noise_in) + len(X_noise_out))])

    # [cite_start]Usamos más neuronas para aproximar la curva [cite: 130]
    brain_algebra = DynamicCausalBrain(n_neurons=8, A_dc=1.618) 
    train_genetic(brain_algebra, X_alg, y_alg, generations=200)

    # --- VISUALIZACIÓN 3D FINAL ---
    fig = plt.figure(figsize=(14, 6))
    
    # Plot XOR
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"Lógica XOR resuelta por DCNN ($A_{{dc}}=1.618$)")
    # Puntos
    ax1.scatter(X_xor[y_xor==0,0], X_xor[y_xor==0,1], X_xor[y_xor==0,2], c='red', marker='x', label='Falso')
    ax1.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], X_xor[y_xor==1,2], c='green', marker='o', label='Verdad')
    
    # Neuronas (Esferas)
    for neuron in brain_logic.genome:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = neuron[3] * brain_logic.A_dc # Aplicando la constante
        x = neuron[0] + r*np.cos(u)*np.sin(v)
        y = neuron[1] + r*np.sin(u)*np.sin(v)
        z = neuron[2] + r*np.cos(v)
        ax1.plot_wireframe(x, y, z, color='cyan', alpha=0.3)

    # Plot Álgebra
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Álgebra (x²+y²=1) aprendida geométricamente")
    ax2.scatter(X_circle[:,0], X_circle[:,1], X_circle[:,2], c='green', label='Círculo')
    ax2.scatter(X_noise_out[:,0], X_noise_out[:,1], X_noise_out[:,2], c='red', marker='x')
    
    # Neuronas cubriendo el anillo
    for neuron in brain_algebra.genome:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = neuron[3] * brain_algebra.A_dc
        x = neuron[0] + r*np.cos(u)*np.sin(v)
        y = neuron[1] + r*np.sin(u)*np.sin(v)
        z = neuron[2] + r*np.cos(v)
        ax2.plot_wireframe(x, y, z, color='orange', alpha=0.2)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tests()