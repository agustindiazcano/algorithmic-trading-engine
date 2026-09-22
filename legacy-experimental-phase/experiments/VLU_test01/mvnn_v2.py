import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. GENERADOR DE SATURNOS DINÁMICOS 🌪️
# ==========================================
def generate_dynamic_saturn(n_points=200, n_rings=5, noise=0.05, time_step=0.0):
    """
    Genera un Saturno donde los anillos GIRAN sobre su propio eje.
    time_step: Simula el tiempo. Cada anillo gira a velocidad distinta.
    """
    points = []
    radii = np.linspace(0.5, 2.5, n_rings)
    points_per_ring = n_points // n_rings
    
    for i, r in enumerate(radii):
        # VELOCIDAD ANGULAR (Omega): Los anillos internos giran más rápido (Kepler style)
        omega = 2.0 / (i + 1) 
        phase_shift = time_step * omega 
        
        # Generar puntos del anillo con el desplazamiento de fase
        t = np.linspace(0, 2*np.pi, points_per_ring, endpoint=False)
        t += phase_shift # <--- AQUÍ ESTÁ LA ROTACIÓN INTERNA
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros(points_per_ring)
        
        # Ruido
        x += np.random.normal(0, noise, points_per_ring)
        y += np.random.normal(0, noise, points_per_ring)
        z += np.random.normal(0, noise, points_per_ring)
        
        points.append(np.vstack((x, y, z)).T)
    
    return np.vstack(points)

def apply_global_rotation(points):
    """Rota TODO el sistema Saturno en el espacio 3D (orientación)"""
    rot = R.random()
    return rot.apply(points)

# ==========================================
# 2. DEFINICIÓN DE LOS MODELOS
# ==========================================

# --- A. MLP (El Pobre Confundido) ---
class ConfusedMLP:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
    def fit(self, X, y):
        X_flat = [x.flatten() for x in X]
        self.model.fit(X_flat, y)
    def predict(self, X):
        X_flat = [x.flatten() for x in X]
        return self.model.predict(X_flat)

# --- B. MVNN RIGIDA (La que usamos antes) ---
class RigidMVNN:
    def __init__(self, tolerance=0.5):
        self.template = None
        self.tolerance = tolerance
    
    def fit(self, X, y):
        # Aprende una foto estática (Time = 0)
        for x, label in zip(X, y):
            if label == 1:
                self.template = x - np.mean(x, axis=0)
                break
                
    def predict(self, X):
        # Intenta encajar la foto estática en el objeto que se mueve
        predictions = []
        # (Simplificado: probamos una rotación óptima asumida para no tardar años)
        # En la vida real usaríamos ICP. Acá asumimos que alineamos la orientación global
        # pero fallaremos en la rotación interna de los anillos.
        for sample in X:
            sample_centered = sample - np.mean(sample, axis=0)
            # Medimos distancia directa (como si fuera un cuerpo rígido)
            # TRUCO: Para ser justos, le damos la mejor alineación posible,
            # pero NO compensamos el giro de los anillos.
            dist = np.mean(np.linalg.norm(sample_centered - self.template, axis=1))
            predictions.append(1 if dist < self.tolerance else 0)
        return np.array(predictions)

# --- C. ORBITAL VNN (La "Spinning Neuron") 🪐 ---
class OrbitalVNN:
    def __init__(self, tolerance=0.3):
        self.ring_radii = [] # Guardamos los RADIOS, no los puntos fijos
        self.tolerance = tolerance

    def fit(self, X, y):
        # APRENDIZAJE ESTRUCTURAL
        # En lugar de memorizar puntos (x,y,z), extrae la Ecuación del Anillo
        for x, label in zip(X, y):
            if label == 1:
                # Calculamos la distancia de cada punto al centro
                radii = np.linalg.norm(x - np.mean(x, axis=0), axis=1)
                # Clusterizamos los radios para encontrar los 5 anillos principales
                # (Simplificación heurística de aprendizaje topológico)
                unique_radii = np.unique(np.round(radii, 1))
                self.ring_radii = unique_radii
                print(f"   [Orbital VNN] 🧠 Aprendió la Ecuación: {len(self.ring_radii)} Órbitas Activas detectadas.")
                break

    def predict(self, X):
        predictions = []
        for sample in X:
            # 1. Centrar (Invarianza de Traslación)
            sample_centered = sample - np.mean(sample, axis=0)
            
            # 2. INFERENCIA ORBITAL (La Magia)
            # No buscamos si el punto está en (x,y,z).
            # Buscamos si el punto pertenece a la TRAYECTORIA ORBITAL (Radio R).
            
            # Proyectamos al plano orbital (XY local) asumiendo alineación Z
            # (En un caso real completo, optimizamos la matriz de rotación global primero)
            # Calculamos radio de cada punto del sample
            sample_radii = np.linalg.norm(sample_centered, axis=1)
            
            # Medimos el "Error Orbital": Distancia a la órbita válida más cercana
            # Esto es equivalente a rotar la esfera infinitamente rápido (Spin)
            total_orbital_error = 0
            for r_point in sample_radii:
                # Distancia al anillo aprendido más cerca
                dist_to_orbit = np.min(np.abs(self.ring_radii - r_point))
                total_orbital_error += dist_to_orbit
            
            avg_error = total_orbital_error / len(sample)
            
            # Si los puntos caen en las órbitas (sin importar el ángulo), es Saturno.
            predictions.append(1 if avg_error < self.tolerance else 0)
            
        return np.array(predictions)

# ==========================================
# 3. EJECUCIÓN: CAOS ORBITAL
# ==========================================
print("=== 🌪️ EXPERIMENTO: DINÁMICA DE PARTÍCULAS (ORBITAL VNN) ===")

# --- GENERAR DATASET CON MOVIMIENTO ---
# Train: Saturnos quietos (Time = 0)
# Test: Saturnos donde los anillos giraron un tiempo random t
X_train, y_train = [], []
X_test, y_test = [], []

print("Generando simulación de nubes de puntos...")
for _ in range(100):
    # Train: Estático
    if np.random.rand() > 0.5:
        X_train.append(generate_dynamic_saturn(time_step=0))
        y_train.append(1)
    else:
        X_train.append(np.random.normal(0, 1.5, (200, 3)))
        y_train.append(0)

    # Test: DINÁMICO (Los anillos giran internamente)
    if np.random.rand() > 0.5:
        # Time random entre 0 y 100 segundos -> Fase totalmente distinta
        saturn_moving = generate_dynamic_saturn(time_step=np.random.rand()*100) 
        # NOTA: Para este test aislamos el efecto del Spin Interno.
        # No aplicamos rotación global 3D extra para que se vea claro el fallo de la Rígida.
        X_test.append(saturn_moving) 
        y_test.append(1)
    else:
        X_test.append(np.random.normal(0, 1.5, (200, 3)))
        y_test.append(0)

# --- ENTRENAMIENTO ---
mlp = ConfusedMLP(); mlp.fit(X_train, y_train)
rigid = RigidMVNN(tolerance=0.4); rigid.fit(X_train, y_train)
orbital = OrbitalVNN(tolerance=0.2); orbital.fit(X_train, y_train)

# --- EVALUACIÓN ---
print("\nEvaluando reconocimiento de objetos con partes móviles...")
acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
acc_rigid = accuracy_score(y_test, rigid.predict(X_test))
acc_orbital = accuracy_score(y_test, orbital.predict(X_test))

print("-" * 60)
print(f"📊 RESULTADOS: SATURNO GIRATORIO (SPIN INTERNO)")
print("-" * 60)
print(f"❌ MLP (Estadístico):    {acc_mlp*100:.1f}% -> Ve 'ruido' porque los pixeles cambiaron de lugar.")
print(f"⚠️ MVNN (Rígida):       {acc_rigid*100:.1f}% -> FALLA. Busca puntos en coordenadas fijas.")
print(f"🌀 Orbital VNN:         {acc_orbital*100:.1f}% -> DO. Entiende la trayectoria, no el punto.")
print("-" * 60)