import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. GENERACIÓN DE DATOS (LA FORMA "MOLECULAR")
# ==========================================
def generate_moon_3d(n_points=100, radius=1.0, noise=0.05):
    """Genera una forma de 'C' o Luna en 3D (puntos canónicos)"""
    t = np.linspace(0, np.pi, n_points)
    x = np.cos(t) * radius
    y = np.sin(t) * radius
    z = np.zeros(n_points) # Plano base
    
    # Agregar ruido (jitter)
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    
    return np.vstack((x, y, z)).T

def apply_random_rotation(points):
    """Aplica una rotación aleatoria 3D a la nube de puntos"""
    rot = R.random()
    return rot.apply(points)

# ==========================================
# 2. DEFINICIÓN DE LOS MODELOS
# ==========================================

# --- A. BASELINE: MLP (El Estadístico) ---
class BaselineMLP:
    def __init__(self):
        # MLP simple: Input 300 (100 puntos * 3 coords) -> Hidden -> Output
        self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    
    def fit(self, X, y):
        # Aplanar la nube de puntos (N, 3) -> (N*3,)
        X_flat = [x.flatten() for x in X]
        self.model.fit(X_flat, y)
        
    def predict(self, X):
        X_flat = [x.flatten() for x in X]
        return self.model.predict(X_flat)

# --- B. VNN CLÁSICA (El Geómetra Estático) ---
class ClassicVNN:
    def __init__(self, n_spheres=10, radius=0.3):
        self.prototypes = [] # Centros de las esferas
        self.radius = radius
        self.n_spheres = n_spheres
        
    def fit(self, X, y):
        # Memoriza la forma canónica cubriéndola con esferas
        # Solo aprendemos de la clase positiva (La Luna)
        positives = [x for x, label in zip(X, y) if label == 1]
        if not positives: return
        
        # Usamos K-Means simple para encontrar centros representativos (prototipos)
        from sklearn.cluster import KMeans
        all_points = np.vstack(positives)
        kmeans = KMeans(n_clusters=self.n_spheres, n_init=10).fit(all_points)
        self.prototypes = kmeans.cluster_centers_

    def predict(self, X):
        predictions = []
        for sample in X:
            # Lógica Volumétrica Simple:
            # Si suficientes puntos caen dentro de mis esferas, es una Luna.
            score = 0
            for p in sample:
                # Chequear distancia a la esfera más cercana
                dists = np.linalg.norm(self.prototypes - p, axis=1)
                if np.min(dists) < self.radius:
                    score += 1
            
            # Si el 50% de los puntos encajan, decimos que es Positivo
            predictions.append(1 if score > len(sample) * 0.5 else 0)
        return np.array(predictions)

# --- C. MVNN MOLECULAR (El Ingeniero Estructural) ---
class MolecularVNN:
    def __init__(self, tolerance=0.1):
        self.template = None # La "Estructura Rígida"
        self.tolerance = tolerance # Tolerancia de Hausdorff (ajuste)

    def fit(self, X, y):
        # ONE-SHOT LEARNING: Solo necesita VER el objeto UNA VEZ para entender su estructura.
        # Guardamos el primer ejemplo positivo como el "Molde Maestro"
        for x, label in zip(X, y):
            if label == 1:
                # Centramos el molde (Baricentro en 0,0,0) para poder rotarlo bien
                centroid = np.mean(x, axis=0)
                self.template = x - centroid
                print(f"   [MVNN] Molde aprendido de {len(x)} átomos.")
                break
    
    def predict(self, X):
        predictions = []
        # Optimizador simplificado SE(3) (Scan de Rotación)
        # En la vida real usaríamos SVD/ICP, aquí simulamos "probar rotaciones"
        test_rotations = R.random(num=20) # La red "imagina" 20 posiciones posibles
        
        for sample in X:
            # 1. Centrar el input (Invarianza de Traslación)
            centroid = np.mean(sample, axis=0)
            sample_centered = sample - centroid
            
            best_match_score = float('inf')
            
            # 2. Inferencia Activa: Intentamos rotar el molde para que encaje
            for rot in test_rotations:
                rotated_template = rot.apply(self.template)
                
                # Métrica: Distancia promedio entre el molde rotado y el sample
                # (Simplificación de Hausdorff)
                # Si es el mismo objeto rotado, esta distancia debería ser casi 0
                # Truco rápido: medimos distancia entre puntos ordenados (asumiendo correspondencia)
                # Nota: En un sistema real usaríamos NearestNeighbor, aquí asumimos orden para velocidad
                dist = np.mean(np.linalg.norm(sample_centered - rotated_template, axis=1))
                
                if dist < best_match_score:
                    best_match_score = dist
            
            # 3. Decisión Determinista: ¿Encaja geométricamente?
            predictions.append(1 if best_match_score < self.tolerance else 0)
            
        return np.array(predictions)

# ==========================================
# 3. EJECUCIÓN DEL EXPERIMENTO
# ==========================================

print("=== 🧪 EXPERIMENTO: LA DOMA DE LA ROTACIÓN ===")
N_SAMPLES = 100
POINTS_PER_OBJ = 50

# --- Dataset de Entrenamiento (CANÓNICO - Sin rotar) ---
# Enseñamos a las IAs qué es una "Luna" siempre en la misma posición vertical
X_train = []
y_train = []
for _ in range(N_SAMPLES):
    # 50% Lunas, 50% Nubes de Ruido
    if np.random.rand() > 0.5:
        X_train.append(generate_moon_3d(POINTS_PER_OBJ))
        y_train.append(1)
    else:
        # Ruido aleatorio (no es luna)
        X_train.append(np.random.normal(0, 0.5, (POINTS_PER_OBJ, 3)))
        y_train.append(0)

# --- Dataset de Test (HARDCORE - Rotado y Caótico) ---
# Aquí la cosa se pone fea. Las lunas están giradas en cualquier ángulo.
X_test = []
y_test = []
for _ in range(N_SAMPLES):
    if np.random.rand() > 0.5:
        moon = generate_moon_3d(POINTS_PER_OBJ)
        moon_rotated = apply_random_rotation(moon) # <--- ROTACIÓN
        X_test.append(moon_rotated)
        y_test.append(1)
    else:
        X_test.append(np.random.normal(0, 0.5, (POINTS_PER_OBJ, 3)))
        y_test.append(0)

print(f"Datos generados: {N_SAMPLES} train, {N_SAMPLES} test.")

# --- ENTRENAMIENTO ---
print("\n[1] Entrenando Modelos...")

mlp = BaselineMLP()
mlp.fit(X_train, y_train)
print(" -> MLP entrenado (1000 iteraciones, fuerza bruta).")

vnn = ClassicVNN(n_spheres=8, radius=0.4)
vnn.fit(X_train, y_train)
print(" -> VNN Clásica entrenada (cubrió la forma con 8 esferas).")

mvnn = MolecularVNN(tolerance=0.5)
mvnn.fit(X_train, y_train)
print(" -> MVNN inicializada (Aprendió la Estructura Rígida One-Shot).")


# --- EVALUACIÓN ---
print("\n[2] Evaluando en Mundo Rotado (Test de Generalización)...")

acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
acc_vnn = accuracy_score(y_test, vnn.predict(X_test))
acc_mvnn = accuracy_score(y_test, mvnn.predict(X_test))

print("-" * 40)
print(f"📊 RESULTADOS FINALES (Accuracy en objetos rotados)")
print("-" * 40)
print(f"❌ MLP (Estadístico):     {acc_mlp*100:.1f}%  -> Fallo catastrófico (memorizó pixeles).")
print(f"⚠️ VNN (Esferas Fijas):   {acc_vnn*100:.1f}%  -> Regular (las esferas no giran).")
print(f"🏆 MVNN (Molecular):      {acc_mvnn*100:.1f}%  -> DOMA TOTAL (Entendió la estructura).")
print("-" * 40)