import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ==============================================================================
# 🧠 CEREBRO DYNAMIC CAUSAL (VERSIÓN FINAL: METAMÓRFICA & INFLADA)
# ==============================================================================

class DynamicCausalBrain:
    def __init__(self, n_neurons=100, A_dc=1.618):
        self.A_dc = A_dc
        self.n_neurons = n_neurons
        self.genome = [] 
        self.reset_genome()

    def reset_genome(self):
        # AHORA EL GENOMA ES MÁS LARGO PARA SOPORTAR FORMAS
        # Estructura: [x, y, z, Radio, Peso, Sx, Sy, Sz]
        self.genome = []
        for _ in range(self.n_neurons):
            gene = np.concatenate([
                np.random.uniform(-1.5, 1.5, 3), # Posición (x,y,z)
                
                # --- HIPER-INFLACIÓN ---
                # Radios grandes para asegurar solapamiento total
                [np.random.uniform(0.8, 1.5)],   # Radio
                
                [np.random.uniform(0.0, 1.0)],   # Peso
                
                # --- GENES DE FORMA (Estiramiento) ---
                # 1.0 = Esfera perfecta
                # >1.0 = Estirado (Elipsoide)
                np.random.uniform(0.5, 1.5, 3)   # Stretch X, Y, Z
            ])
            self.genome.append(gene)
        self.genome = np.array(self.genome)

    def predict(self, points):
        """
        Inferencia Volumétrica con Soporte Metamórfico.
        """
        if len(self.genome) == 0: return np.zeros(len(points))
        
        # Desempaquetamos genes
        centers = self.genome[:, :3]
        radii = self.genome[:, 3] * self.A_dc
        weights = self.genome[:, 4]
        stretch = self.genome[:, 5:8] # Genes de forma
        
        # 1. Diferencia Vectorial
        diff = points[:, None, :] - centers[None, :, :]
        
        # 2. APLICAR METAMORFOSIS (Estiramiento)
        # Deformamos el espacio alrededor de la neurona
        diff_stretched = diff / (stretch[None, :, :] + 1e-6)
        
        # 3. Distancia Euclidiana sobre el espacio deformado
        dists = np.linalg.norm(diff_stretched, axis=2) 
        
        # 4. Activación con MESETA (Plateau)
        # Usamos la técnica de cima plana para máxima estabilidad
        safe_radii = np.maximum(radii, 1e-6)
        raw_overlap = np.maximum(0, 1 - dists / safe_radii[None, :])
        
        # Multiplicador 3.0 para solidez (convierte cono en meseta)
        plateau_overlap = np.minimum(1.0, raw_overlap * 3.0) 
        
        activations = plateau_overlap * weights[None, :]
        return np.max(activations, axis=1)

# ==============================================================================
# 🧬 ENTRENAMIENTO EVOLUTIVO (SIMULATED ANNEALING)
# ==============================================================================

def train_geometric_concept(brain, X, y, generations=20000):
    print(f"🧠 Aprendiendo Concepto Geométrico ({len(X)} puntos)...")
    best_genome = np.copy(brain.genome)
    best_err = np.inf
    
    # PARAMETROS DE MICROCIRUGÍA (Mutación Dinámica)
    initial_mutation = 0.2  # Martillo (Exploración)
    final_mutation = 0.001  # Bisturí (Precisión)
    
    for gen in range(generations):
        # Decay lineal de la tasa de mutación
        progress = gen / generations
        mutation_rate = initial_mutation * (1 - progress) + final_mutation
        
        mutant = np.copy(best_genome)
        
        # Mutamos el 10% de las neuronas
        mask = np.random.rand(*mutant.shape) < 0.1
        
        # Aplicamos ruido
        noise = np.random.normal(0, mutation_rate, size=mutant[mask].shape)
        mutant[mask] += noise
        
        # CORRECCIONES FÍSICAS:
        # Radios (col 3) y Estiramientos (col 5,6,7) deben ser positivos
        mutant[:, 3] = np.abs(mutant[:, 3]) 
        mutant[:, 5:8] = np.abs(mutant[:, 5:8])
        
        brain.genome = mutant
        preds = brain.predict(X)
        err = np.mean((y - preds)**2)
        
        if err < best_err:
            best_err = err
            best_genome = np.copy(mutant)
            
        if gen % 1000 == 0 or gen == generations - 1:
            print(f"  [Gen {gen}] Error: {best_err:.6f} | Mutación: {mutation_rate:.5f}")
            
    brain.genome = best_genome
    print(f"✅ Concepto aprendido. Error final: {best_err:.8f}")

# ==============================================================================
# 🌪️ GENERADOR DE CAOS (LABERINTO DE ESPEJOS)
# ==============================================================================

def get_random_rotation_matrix():
    """Genera una matriz de rotación 3D aleatoria."""
    theta, phi, z = np.random.uniform(0, 2*np.pi, 3)
    
    # Rotación en X
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(theta), -np.sin(theta)], 
                   [0, np.sin(theta), np.cos(theta)]])
    # Rotación en Y
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)], 
                   [0, 1, 0], 
                   [-np.sin(phi), 0, np.cos(phi)]])
    # Rotación en Z
    Rz = np.array([[np.cos(z), -np.sin(z), 0], 
                   [np.sin(z), np.cos(z), 0], 
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def generate_mirror_maze(steps=200):
    """
    Trayectoria: Espiral en la esfera unitaria.
    Trampa: Rotación aleatoria del universo en cada tick.
    """
    t = np.linspace(0, 4*np.pi, steps)
    
    # Trayectoria Verdadera (Sobre la esfera de radio 1.0)
    true_x = np.sin(t) 
    true_y = np.cos(t)
    true_z = np.zeros_like(t) 
    
    base_path = np.column_stack([true_x, true_y, true_z])
    
    chaos_path = []
    
    for p in base_path:
        # APLICAMOS LA TRAMPA
        R = get_random_rotation_matrix()
        p_rotated = R @ p
        chaos_path.append(p_rotated)
        
    return np.array(chaos_path)

# ==============================================================================
# 🚀 EJECUCIÓN PRINCIPAL
# ==============================================================================

def run_stress_test():
    print("🛡️ INICIANDO PROTOCOLO Adc: ESTRÉS GEOMÉTRICO 🛡️")
    print("--------------------------------------------------")
    
    # 1. INSTANCIAMOS EL CEREBRO METAMÓRFICO
    brain = DynamicCausalBrain(n_neurons=100, A_dc=1.618)
    
    # 2. GENERAMOS DATOS DE ENTRENAMIENTO (Estáticos)
    # Enseñamos qué es una esfera (Radio 1.0 = BIEN, Otros = MAL)
    
    # Puntos positivos (Superficie, Target 1.0)
    theta = np.random.uniform(0, 2*np.pi, 300)
    phi = np.random.uniform(0, np.pi, 300)
    x = 1.0 * np.sin(phi) * np.cos(theta)
    y = 1.0 * np.sin(phi) * np.sin(theta)
    z = 1.0 * np.cos(phi)
    X_pos = np.column_stack([x, y, z])
    
    # Puntos negativos (Ruido, Target 0.0)
    X_neg_in = np.random.uniform(-0.6, 0.6, (150, 3)) # Adentro
    X_neg_out = np.random.uniform(-2, 2, (150, 3))    # Afuera
    X_neg_out = X_neg_out[np.linalg.norm(X_neg_out, axis=1) > 1.3] # Asegurar lejos
    
    X_train = np.vstack([X_pos, X_neg_in, X_neg_out])
    y_train = np.concatenate([np.ones(300), np.zeros(len(X_neg_in)+len(X_neg_out))])
    
    # 3. ENTRENAMIENTO
    # Usamos 200,000 gens como pediste para replicar el resultado perfecto
    train_geometric_concept(brain, X_train, y_train, generations=200000)
    
    # 4. EL TEST DEL CAOS
    print("\n🌪️ Generando Laberinto de Espejos Rotatorios...")
    chaos_data = generate_mirror_maze(steps=100)
    
    print(f"   Input Ejemplo (Tick 0): {chaos_data[0]}")
    print(f"   Input Ejemplo (Tick 1): {chaos_data[1]}")
    
    # 5. INFERENCIA
    start_time = time.time()
    predictions = brain.predict(chaos_data)
    end_time = time.time()
    
    # 6. RESULTADOS
    print(f"\n⚡ Tiempo de Reacción: {(end_time - start_time)*1000:.2f} ms")
    
    avg_activation = np.mean(predictions)
    std_activation = np.std(predictions)
    
    print("\n📊 REPORTE DE INVARIANZA:")
    print(f"   Activación Promedio (Esperado ~1.0): {avg_activation:.4f}")
    print(f"   Estabilidad (Desviación Estándar):   {std_activation:.4f}")
    
    # GRÁFICO FINAL
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(chaos_data[:, 0], label="X")
    plt.plot(chaos_data[:, 1], label="Y")
    plt.plot(chaos_data[:, 2], label="Z")
    plt.title("Visión IA Normal (CAOS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(predictions, color='green', linewidth=2, label="Percepción Adc")
    plt.axhline(y=1.0, color='r', linestyle='--', label="Target")
    plt.title(f"Visión VL ($A_{{dc}}$) - Prom: {avg_activation:.2f}")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_stress_test()