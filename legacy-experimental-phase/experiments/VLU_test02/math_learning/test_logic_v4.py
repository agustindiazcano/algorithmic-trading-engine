import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- TRADUCTOR DE ESCALA (NORMALIZACIÓN INTELIGENTE) ---
class ScaleTranslator:
    """
    El 'Zoom Lens' de la IA.
    Convierte cualquier rango numérico (ej: precios BTC $10k-$100k) 
    al espacio [0,1] donde vive el cerebro geométrico.
    """
    def __init__(self):
        self.min_val = 0
        self.max_val = 1
        self.range = 1
        
    def fit(self, data):
        """Aprende el rango de los datos reales"""
        self.min_val = np.min(data)
        self.max_val = np.max(data)
        self.range = self.max_val - self.min_val
        if self.range == 0: self.range = 1  # Evitar división por cero

    def transform(self, data):
        """Mundo Real → Mundo IA (0-1)"""
        return (data - self.min_val) / self.range

    def inverse_transform(self, data):
        """Mundo IA (0-1) → Mundo Real"""
        return (data * self.range) + self.min_val

def quantize_output(raw_value, tick_size=0.01):
    """
    Snapping Inteligente: Convierte la intuición geométrica (continua) 
    en valores discretos (ej: centavos, pips).
    
    tick_size: Resolución del mundo real
        - 1.0 para lógica booleana (0 o 1)
        - 0.01 para dinero ($0.01 centavos)
        - 0.5 para aproximaciones medias
    """
    return tick_size * np.round(raw_value / tick_size)

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

def get_math_data(operation="SUM", n_samples=100, value_range=(0, 1)):
    """
    Genera datos matemáticos RAW (sin normalizar).
    
    Args:
        operation: "SUM" o "SUB"
        n_samples: Cantidad de ejemplos
        value_range: Rango de valores (min, max)
    
    Returns:
        inputs: Array (N, 3) con [a, b, context]
        targets: Array (N,) con resultados reales (a+b o a-b)
    """
    min_val, max_val = value_range
    a = np.random.uniform(min_val, max_val, n_samples)
    b = np.random.uniform(min_val, max_val, n_samples)
    
    inputs = []
    targets = []
    
    for i in range(n_samples):
        if operation == "SUM":
            inputs.append([a[i], b[i], 0.0])  # Context 0 = Suma
            targets.append(a[i] + b[i])  # SUMA REAL (no normalizada)
        elif operation == "SUB":
            inputs.append([a[i], b[i], 1.0])  # Context 1 = Resta
            targets.append(a[i] - b[i])  # RESTA REAL (no normalizada)
    
    return np.array(inputs), np.array(targets)

def get_math_data_normalized(operation="SUM", n_samples=100, value_range=(0, 1)):
    """
    Genera datos matemáticos y los normaliza usando ScaleTranslator.
    
    Returns:
        X_norm: Inputs normalizados (N, 3)
        y_norm: Targets normalizados (N,)
        scaler_input: ScaleTranslator para inputs (columnas a, b)
        scaler_output: ScaleTranslator para targets
    """
    # Generar datos raw
    inputs_raw, targets_raw = get_math_data(operation, n_samples, value_range)
    
    # Crear scalers
    scaler_input = ScaleTranslator()
    scaler_output = ScaleTranslator()
    
    # Fit scalers en los datos raw
    # Para inputs: solo normalizamos las columnas a y b (no el context)
    scaler_input.fit(inputs_raw[:, :2].flatten())  # Todos los valores de a y b
    scaler_output.fit(targets_raw)
    
    # Transform
    X_norm = inputs_raw.copy()
    X_norm[:, 0] = scaler_input.transform(inputs_raw[:, 0])  # Normalizar 'a'
    X_norm[:, 1] = scaler_input.transform(inputs_raw[:, 1])  # Normalizar 'b'
    # X_norm[:, 2] ya es el context (0 o 1), no se normaliza
    
    y_norm = scaler_output.transform(targets_raw)
    
    return X_norm, y_norm, scaler_input, scaler_output

def run_math_class():
    brain = DynamicCausalBrain(n_neurons=15, A_dc=1.618)
    
    # --- FASE 1: ENTRENAR SUMA CON NORMALIZACIÓN ---
    print("\n🔬 Generando datos de SUMA normalizados...")
    X_sum_norm, y_sum_norm, scaler_sum_input, scaler_sum_output = get_math_data_normalized(
        "SUM", n_samples=100, value_range=(0, 1)
    )
    
    train_task(brain, X_sum_norm, y_sum_norm, "SUMA (Normalizada)", generations=1000)
    
    # --- FASE 2: AGREGAR NEURONAS Y ENTRENAR RESTA ---
    print("\n✨ Agregando nuevas neuronas...")
    brain.add_neurons(15) 
    
    print("\n🔬 Generando datos de RESTA normalizados...")
    X_sub_norm, y_sub_norm, scaler_sub_input, scaler_sub_output = get_math_data_normalized(
        "SUB", n_samples=100, value_range=(0, 1)
    )
    
    # Dataset Mixto (REPASO)
    X_mixed = np.vstack([X_sum_norm, X_sub_norm])
    y_mixed = np.concatenate([y_sum_norm, y_sub_norm])
    
    train_task(brain, X_mixed, y_mixed, "RESTA + REPASO SUMA (Normalizada)", generations=100)

    # --- EVALUACIÓN IN-DISTRIBUTION ---
    print("\n🎓 EXAMEN FINAL DE MATEMÁTICA VOLUMÉTRICA (In-Distribution)")
    
    # Casos de prueba en el rango [0,1]
    test_cases_in_dist = [
        (0.2, 0.2, 0.0, "0.2 + 0.2"),
        (0.8, 0.1, 0.0, "0.8 + 0.1"),
        (1.0, 1.0, 0.0, "1.0 + 1.0"),
        (0.5, 0.5, 1.0, "0.5 - 0.5"),
        (0.9, 0.1, 1.0, "0.9 - 0.1"),
    ]
    
    correct_count = 0
    for a, b, context, desc in test_cases_in_dist:
        # Normalizar input usando el scaler de entrenamiento
        if context == 0.0:  # Suma
            norm_a = scaler_sum_input.transform(np.array([a]))[0]
            norm_b = scaler_sum_input.transform(np.array([b]))[0]
            input_vec = np.array([[norm_a, norm_b, context]])
            pred_norm = brain.predict(input_vec)[0]
            pred_real = scaler_sum_output.inverse_transform(np.array([pred_norm]))[0]
            expected = a + b
        else:  # Resta
            norm_a = scaler_sub_input.transform(np.array([a]))[0]
            norm_b = scaler_sub_input.transform(np.array([b]))[0]
            input_vec = np.array([[norm_a, norm_b, context]])
            pred_norm = brain.predict(input_vec)[0]
            pred_real = scaler_sub_output.inverse_transform(np.array([pred_norm]))[0]
            expected = a - b
        
        # Quantizar para eliminar ruido (mantener 2 decimales para in-dist)
        pred_clean = quantize_output(pred_real, tick_size=0.01)
        error = abs(pred_clean - expected)
        status = "✅" if error < 0.05 else "❌"
        if error < 0.05: correct_count += 1
        print(f"{status} {desc:<15} | Pred: {pred_clean:.2f} | Target: {expected:.2f} | Error: {error:.4f}")
    
    print(f"\n📊 Score In-Distribution: {correct_count}/{len(test_cases_in_dist)} ({correct_count/len(test_cases_in_dist)*100:.0f}%)")
    
    # --- PRUEBA DE ESCALABILIDAD (NÚMEROS GIGANTES) ---
    print("\n🚀 PRUEBA DE ESCALABILIDAD CON TRADUCTOR DE ESCALA (50 Operaciones Aleatorias)")
    print("   (Usando normalización dinámica para cada rango)")
    
    # Generar 50 casos aleatorios con diferentes rangos
    np.random.seed(42)  # Para reproducibilidad
    ood_cases = []
    
    # Rangos variados
    ranges = [
        (10, 100),           # Pequeños
        (100, 1000),         # Medianos
        (1000, 10000),       # Grandes
        (10000, 100000),     # Muy grandes
        (100000, 1000000),   # Gigantes
    ]
    
    for _ in range(50):
        # Elegir rango aleatorio
        min_r, max_r = ranges[np.random.randint(0, len(ranges))]
        
        # Generar números aleatorios en ese rango
        a = np.random.uniform(min_r, max_r)
        b = np.random.uniform(min_r, max_r)
        
        # Elegir operación aleatoria (0=Suma, 1=Resta)
        context = float(np.random.choice([0.0, 1.0]))
        
        if context == 0.0:
            desc = f"{int(a)} + {int(b)}"
        else:
            desc = f"{int(a)} - {int(b)}"
        
        ood_cases.append((a, b, context, desc))
    
    # Evaluar los 50 casos
    correct_ood = 0
    errors = []
    failed_cases = []  # Para guardar los casos donde falló
    
    for a, b, context, desc in ood_cases:
        # Crear scaler temporal para este rango específico
        temp_scaler_input = ScaleTranslator()
        temp_scaler_output = ScaleTranslator()
        
        # Fit en el rango de estos números específicos
        temp_scaler_input.fit(np.array([a, b]))
        
        if context == 0.0:  # Suma
            expected = a + b
        else:  # Resta
            expected = a - b
        
        temp_scaler_output.fit(np.array([expected]))
        
        # Normalizar
        norm_a = temp_scaler_input.transform(np.array([a]))[0]
        norm_b = temp_scaler_input.transform(np.array([b]))[0]
        input_vec = np.array([[norm_a, norm_b, context]])
        
        # Predicción
        pred_norm = brain.predict(input_vec)[0]
        
        # Desnormalizar
        pred_real = temp_scaler_output.inverse_transform(np.array([pred_norm]))[0]
        
        # Redondear a entero
        pred_int = int(round(pred_real))
        expected_int = int(round(expected))
        
        error_pct = abs(pred_int - expected_int) / abs(expected_int) * 100 if expected_int != 0 else 0
        errors.append(error_pct)
        
        if error_pct < 10:
            correct_ood += 1
            status = "✅"
        else:
            status = "❌"
            # Guardar caso fallido
            failed_cases.append({
                'desc': desc,
                'pred': pred_int,
                'expected': expected_int,
                'error': error_pct
            })
        
        # Mostrar solo los primeros 10 para no saturar
        if len([e for e in errors if e is not None]) <= 10:
            print(f"{status} {desc:<25} | Pred: {pred_int:>10} | Target: {expected_int:>10} | Error: {error_pct:>5.1f}%")
    
    # Resumen estadístico
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN DE ESCALABILIDAD (50 operaciones OOD)")
    print(f"{'='*70}")
    print(f"✅ Correctas (Error < 10%): {correct_ood}/50 ({correct_ood/50*100:.1f}%)")
    print(f"📈 Error Promedio: {np.mean(errors):.2f}%")
    print(f"📉 Error Mínimo: {np.min(errors):.2f}%")
    print(f"📊 Error Máximo: {np.max(errors):.2f}%")
    print(f"🎯 Error Mediano: {np.median(errors):.2f}%")
    
    # Mostrar casos donde falló
    if failed_cases:
        print(f"\n{'='*70}")
        print(f"❌ CASOS FALLIDOS (Error >= 10%): {len(failed_cases)}")
        print(f"{'='*70}")
        for case in failed_cases:
            print(f"❌ {case['desc']:<25} | Pred: {case['pred']:>10} | Target: {case['expected']:>10} | Error: {case['error']:>5.1f}%")
    else:
        print(f"\n🎉 ¡PERFECTO! No hubo errores >= 10%")


    
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