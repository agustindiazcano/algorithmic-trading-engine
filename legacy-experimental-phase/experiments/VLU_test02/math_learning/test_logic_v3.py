import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- EL CEREBRO DE AGUSTÍN (DCNN) ---
class DynamicCausalBrain:
    def __init__(self, n_neurons=20, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        # Genoma: [x, y, z, radio_base, peso]
        # x,y: Inputs numéricos
        # z: Contexto (0=Suma, 1=Resta, etc)
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        self.genome = []
        for _ in range(self.n_neurons):
            # Inicializamos en el espacio de trabajo (0 a 1 en xy, 0 a 2 en z)
            gene = np.concatenate([
                np.random.uniform(0, 1, 2),   # x, y
                np.random.uniform(0, 2.5, 1), # z (Contexto)
                [np.random.uniform(0.1, 0.5)],# Radio
                [np.random.uniform(0.0, 1.0)] # Peso (Output value)
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def activation_vl(self, point, neuron):
        # Punto: [x, y, context_z]
        center = neuron[:3]
        radius = neuron[3] * self.A_dc
        weight = neuron[4]
        
        dist = np.linalg.norm(point - center)
        
        if radius <= 1e-6: return 0
        # Función Cónica VL
        overlap = max(0, 1 - dist / radius)
        return weight * overlap

    def predict(self, points):
        results = []
        for p in points:
            # Max-Aggregation (La neurona que más "sabe" responde)
            activations = [self.activation_vl(p, n) for n in self.genome]
            results.append(max(activations) if activations else 0)
        return np.array(results)

    def add_neurons(self, n_new):
        """
        ¡MEMORIA ETERNA!
        Agrega nuevas neuronas virgenes sin tocar las viejas.
        """
        for _ in range(n_new):
            gene = np.concatenate([
                np.random.uniform(0, 1, 2),
                np.random.uniform(0, 2.5, 1),
                [np.random.uniform(0.1, 0.5)],
                [np.random.uniform(0.0, 1.0)]
            ])
            # Truco numpy para agregar filas
            self.genome = np.vstack([self.genome, gene])

# --- ENTRENADOR INCREMENTAL ---
def train_task(brain, X, y, task_name, generations=1000, pop_size=50):
    print(f"\n🧠 Entrenando Tarea: {task_name}...")
    
    # Congelamos (guardamos) el mejor cerebro actual para mutarlo
    current_best_genome = np.copy(brain.genome)
    best_fitness = -np.inf
    
    # Historial
    history = []

    for gen in range(generations):
        # Creamos población mutando el cerebro actual
        population = []
        for _ in range(pop_size):
            mutant = np.copy(current_best_genome)
            # Mutación
            if np.random.rand() < 0.9: # Alta tasa de mutación
                mask = np.random.rand(*mutant.shape) < 0.1
                noise = np.random.normal(0, 0.1, size=mutant.shape)
                mutant[mask] += noise[mask]
                mutant[:, 3] = np.abs(mutant[:, 3]) # Radios positivos
            population.append(mutant)

        # Evaluar
        scores = []
        for ind_genome in population:
            brain.genome = ind_genome
            preds = brain.predict(X)
            mse = np.mean((y - preds)**2)
            fitness = -mse 
            scores.append(fitness)

        # Selección
        scores = np.array(scores)
        elite_idx = np.argmax(scores)
        
        if scores[elite_idx] > best_fitness:
            best_fitness = scores[elite_idx]
            current_best_genome = np.copy(population[elite_idx])
            if gen % 200 == 0:
                print(f"  [Gen {gen}] Error (MSE): {-best_fitness:.5f}")
        
        history.append(-best_fitness)

    brain.genome = current_best_genome
    print(f"✅ {task_name} Aprendido. Error Final: {-best_fitness:.5f}")
    return history

# --- GENERADOR DE DATOS MATEMÁTICOS ---
def get_math_data(operation="SUM"):
    # Generamos 100 puntos aleatorios (x, y) entre 0 y 1
    x = np.random.rand(100)
    y = np.random.rand(100)
    
    inputs = []
    targets = []
    
    for i in range(100):
        if operation == "SUM":
            # Contexto Z = 0.0
            inputs.append([x[i], y[i], 0.0])
            # Target: (x+y)/2 para que quede entre 0 y 1
            targets.append((x[i] + y[i]) / 2.0)
            
        elif operation == "SUB":
            # Contexto Z = 1.0
            inputs.append([x[i], y[i], 1.0])
            # Target: (x-y+1)/2 para normalizar (0.5 es el 0)
            targets.append((x[i] - y[i] + 1) / 2.0)
            
        elif operation == "COMPLEX":
            # Contexto Z = 2.0
            # Función Loca: sin(x*pi) * y
            inputs.append([x[i], y[i], 2.0])
            val = np.sin(x[i] * np.pi) * y[i]
            targets.append(val) # Ya está entre 0 y 1 aprox

    return np.array(inputs), np.array(targets)

# --- EJECUCIÓN DEL EXPERIMENTO ---
def run_math_class():
    # 1. Nace el Cerebro
    brain = DynamicCausalBrain(n_neurons=15, A_dc=1.618) # Pocas neuronas al inicio
    
    # 2. Tarea 1: APRENDER A SUMAR
    X_sum, y_sum = get_math_data("SUM")
    train_task(brain, X_sum, y_sum, "SUMA (Contexto Z=0)", generations=1000)
    
    # 3. CRECIMIENTO NEURONAL (No borramos lo viejo, agregamos capacidad)
    print("\n✨ Agregando nuevas neuronas para la siguiente tarea...")
    brain.add_neurons(15) 
    
    # 4. Tarea 2: APRENDER A RESTAR (Entrenamos con DATOS MIXTOS para consolidar)
    #    Si entrenamos solo con resta, el GA podría optimizar olvidando la suma.
    #    Para simular memoria real, le recordamos "repasar" la suma.
    X_sub, y_sub = get_math_data("SUB")
    
    # Dataset Mixto (Vida Real)
    X_mixed = np.vstack([X_sum, X_sub])
    y_mixed = np.concatenate([y_sum, y_sub])
    
    train_task(brain, X_mixed, y_mixed, "RESTA + REPASO SUMA (Contexto Z=1)", generations=1500)

    # 5. PRUEBA DE FUEGO (VALIDACIÓN)
    print("\n🎓 EXAMEN FINAL DE MATEMÁTICA VOLUMÉTRICA")
    
    # Casos de prueba manuales
    test_cases = [
        # SUMA (Z=0) -> Esperado (a+b)/2
        ([0.2, 0.2, 0.0], "0.2 + 0.2", 0.2), # (0.4)/2 = 0.2
        ([0.8, 0.1, 0.0], "0.8 + 0.1", 0.45), # (0.9)/2 = 0.45
        
        # RESTA (Z=1) -> Esperado (a-b+1)/2
        ([0.5, 0.5, 1.0], "0.5 - 0.5", 0.5), # (0)/2 + 0.5 = 0.5
        ([0.9, 0.1, 1.0], "0.9 - 0.1", 0.9), # (0.8)/2 + 0.5 = 0.9
    ]
    
    for inp, desc, expected in test_cases:
        pred = brain.predict(np.array([inp]))[0]
        # Des-normalizamos para mostrar el numero real si quisieramos, 
        # pero mostramos el valor interno crudo vs esperado
        print(f"Operación: {desc} | Input Z={inp[2]} | Predicción IA: {pred:.4f} (Target: {expected:.4f})")

    # --- VISUALIZACIÓN DE LA MEMORIA ---
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Cerebro Matemático: Zonas de Suma (0) y Resta (1)")
    
    # Dibujamos las neuronas finales
    for neuron in brain.genome:
        x, y, z, r, w = neuron
        r = r * brain.A_dc * 0.5 # Reducimos visualmente para ver
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        xs = x + r * np.outer(np.cos(u), np.sin(v))
        ys = y + r * np.outer(np.sin(u), np.sin(v))
        zs = z + r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Color según especialidad (Z)
        color = 'blue' if z < 0.5 else ('red' if z < 1.5 else 'green')
        ax.plot_wireframe(xs, ys, zs, color=color, alpha=0.3)

    ax.set_xlabel('Num A')
    ax.set_ylabel('Num B')
    ax.set_zlabel('Contexto (Operación)')
    plt.show()

if __name__ == "__main__":
    run_math_class()