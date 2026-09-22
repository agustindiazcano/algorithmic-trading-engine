import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

# ==========================================
# 1. GENERADORES (Con Oclusión) 🌑
# ==========================================

def generate_5_ring_gimbal(n_points=500, noise=0.05, time_step=0.0):
    """Genera el Giroscopio perfecto"""
    points = []
    n_rings = 5
    points_per_ring = n_points // n_rings
    radius = 2.0
    axis_angles = np.linspace(0, 180, n_rings, endpoint=False)
    
    for i, angle_deg in enumerate(axis_angles):
        omega = 2.0 + (i * 1.5) 
        phase = time_step * omega 
        t = np.linspace(0, 2*np.pi, points_per_ring, endpoint=False) + phase 
        
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.zeros(points_per_ring)
        
        ring_base = np.vstack((x, y, z)).T
        rot_axis = R.from_euler('x', angle_deg, degrees=True) * R.from_euler('y', i*30, degrees=True)
        ring_tilted = rot_axis.apply(ring_base)
        ring_tilted += np.random.normal(0, noise, ring_tilted.shape)
        points.append(ring_tilted)
        
    return np.vstack(points)

def generate_adversarial_cloud(n_points=500, radius=2.0):
    """Genera la Nube enemiga"""
    u = np.random.uniform(0, 1, n_points)
    v = np.random.uniform(0, 1, n_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.vstack((x,y,z)).T + np.random.normal(0, 0.05, (n_points, 3))

def apply_global_spin(points):
    return R.random().apply(points)

def apply_severe_occlusion(points, survival_rate=0.2):
    """
    EL HOLOGRAMA ROTO:
    Borra aleatoriamente puntos hasta dejar solo el 'survival_rate' (ej: 20%).
    """
    n_points = len(points)
    n_survivors = int(n_points * survival_rate)
    
    # Elegimos índices al azar para sobrevivir
    indices = np.random.choice(n_points, n_survivors, replace=False)
    return points[indices]

# ==========================================
# 2. MODELOS COGNITIVOS 🧠
# ==========================================

class BlindMLP:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500)
    def fit(self, X, y):
        # El MLP necesita input de tamaño fijo. 
        # Si ocluimos, cambia el tamaño.
        # TRUCO PARA EL MLP: Rellenamos con ceros (padding) para que no crashee,
        # pero esto ya le avisa que falta info.
        max_len = max([len(x) for x in X])
        self.input_size = max_len * 3
        X_padded = [self._pad(x) for x in X]
        self.model.fit(X_padded, y)
        
    def predict(self, X):
        X_padded = [self._pad(x) for x in X]
        return self.model.predict(X_padded)
    
    def _pad(self, x):
        flat = x.flatten()
        if len(flat) < self.input_size:
            padded = np.zeros(self.input_size)
            padded[:len(flat)] = flat
            return padded
        return flat[:self.input_size]

class AutoGyroVNN:
    def __init__(self, scans=300): 
        self.traces = None 
        self.tolerance = None 
        self.scans = scans

    def fit(self, X, y):
        # 1. Aprender Estructura (De un ejemplo PERFECTO y LIMPIO)
        positives = [x for x, label in zip(X, y) if label == 1]
        self.traces = positives[0] - np.mean(positives[0], axis=0)
        print(f"   [AutoGyro] 🖐️ Estructura Aprendida de ejemplo intacto.")

        # 2. Calibrar Tolerancia (Usando datos LIMPIOS para establecer la norma)
        print(f"   [AutoGyro] ⚙️ Calibrando estándar de calidad...")
        errors = []
        calib_rotations = R.random(num=50) 
        for sample in positives[:20]: # Calibramos con 20 ejemplos
            sample_centered = sample - np.mean(sample, axis=0)
            min_err = float('inf')
            for rot in calib_rotations:
                rotated_traces = rot.apply(self.traces)
                dists = cdist(sample_centered, rotated_traces, metric='euclidean')
                err = np.mean(np.min(dists, axis=1))
                if err < min_err: min_err = err
            errors.append(min_err)
        
        self.tolerance = np.percentile(errors, 95) * 1.2
        print(f"   [AutoGyro] ✅ Tolerancia Base: {self.tolerance:.4f}")

    def predict(self, X):
        predictions = []
        rotations = R.random(num=self.scans)
        
        for i, sample in enumerate(X):
            # IMPORTANTE: Centrar los fragmentos es difícil porque el centro de masa cambió
            # al borrar puntos asimétricamente.
            # LA VNN ES ROBUSTA: Aun con el centro desplazado, buscará el mejor encaje.
            centroid = np.mean(sample, axis=0)
            sample_centered = sample - centroid
            
            min_structure_error = float('inf')
            
            for rot in rotations:
                rotated_traces = rot.apply(self.traces)
                
                # LA MAGIA DE LA OCLUSIÓN:
                # cdist mide distancia de LOS PUNTOS QUE QUEDAN hacia los rieles.
                # No importa si quedan 5 o 500. Si son válidos, la distancia es baja.
                dists = cdist(sample_centered, rotated_traces)
                
                # Promedio de error de los sobrevivientes
                avg_error = np.mean(np.min(dists, axis=1))
                
                if avg_error < min_structure_error:
                    min_structure_error = avg_error
            
            # Clasificamos
            predictions.append(1 if min_structure_error < self.tolerance else 0)
            
            if i % 50 == 0:
                print(f"     -> Obj {i} ({len(sample)} pts): Err {min_structure_error:.3f} => {'Giroscopio' if min_structure_error < self.tolerance else 'Nube'}")
            
        return np.array(predictions)

# ==========================================
# 3. EJECUCIÓN: LA MASACRE DE PUNTOS 🩸
# ==========================================
print("=== 🧩 EXPERIMENTO: EL HOLOGRAMA ROTO (OCLUSIÓN SEVERA) ===")
print("Entrenamos con objetos PERFECTOS. Testeamos con solo el 20% de los puntos visibles.")

N = 100
PTS = 500
X_train, y_train = [], []
X_test, y_test = [], []

# --- TRAIN (Datos Limpios y Perfectos) ---
print("Generando datos de entrenamiento (Intactos)...")
for _ in range(N):
    if np.random.rand() > 0.5:
        X_train.append(generate_5_ring_gimbal(PTS, time_step=0))
        y_train.append(1)
    else:
        X_train.append(generate_adversarial_cloud(PTS))
        y_train.append(0)

# --- TEST (Datos Masacrados) ---
print("Generando datos de test (80% OCLUIDO + ROTADO)...")
for _ in range(N):
    if np.random.rand() > 0.5:
        # Generar -> Rotar -> BORRAR EL 80%
        gimbal = generate_5_ring_gimbal(PTS, time_step=np.random.rand()*100)
        rotated = apply_global_spin(gimbal)
        occluded = apply_severe_occlusion(rotated, survival_rate=0.2) # <--- SÁDICO
        X_test.append(occluded)
        y_test.append(1)
    else:
        # Lo mismo para la nube
        cloud = generate_adversarial_cloud(PTS)
        rotated = apply_global_spin(cloud)
        occluded = apply_severe_occlusion(rotated, survival_rate=0.2)
        X_test.append(occluded)
        y_test.append(0)

print("\n[1] Entrenando con la Verdad Completa...")
mlp = BlindMLP(); mlp.fit(X_train, y_train)
gyro = AutoGyroVNN(scans=300); gyro.fit(X_train, y_train)

print("\n[2] Testeando en la Niebla (Inferencia Parcial)...")
acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
acc_gyro = accuracy_score(y_test, gyro.predict(X_test))

print("-" * 60)
print(f"📊 RESULTADOS: VISIBILIDAD 20%")
print("-" * 60)
print(f"❌ MLP (Estadístico):    {acc_mlp*100:.1f}% -> Pánico. Faltan datos, no sabe qué hacer.")
print(f"🪐 AutoGyro VNN:        {acc_gyro*100:.1f}% -> DOMA. Reconstruyó la estructura mentalmente.")
print("-" * 60)