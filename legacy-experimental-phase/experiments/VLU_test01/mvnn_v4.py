import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

# ==========================================
# 1. GENERADORES DE MUNDOS (Física) 🌍
# ==========================================

def generate_5_ring_gimbal(n_points=500, noise=0.05, time_step=0.0):
    """
    Genera un Giroscopio de 5 Anillos con satélites orbitando.
    """
    points = []
    n_rings = 5
    points_per_ring = n_points // n_rings
    radius = 2.0
    
    # 5 ejes distribuidos
    axis_angles = np.linspace(0, 180, n_rings, endpoint=False)
    
    for i, angle_deg in enumerate(axis_angles):
        # Velocidad y fase orbital
        omega = 2.0 + (i * 1.5) 
        phase = time_step * omega 
        
        t = np.linspace(0, 2*np.pi, points_per_ring, endpoint=False)
        t += phase 
        
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.zeros(points_per_ring)
        
        ring_base = np.vstack((x, y, z)).T
        
        # Inclinar el anillo
        rot_axis = R.from_euler('x', angle_deg, degrees=True) * R.from_euler('y', i*30, degrees=True)
        ring_tilted = rot_axis.apply(ring_base)
        
        # Ruido
        ring_tilted += np.random.normal(0, noise, ring_tilted.shape)
        points.append(ring_tilted)
        
    return np.vstack(points)

def generate_adversarial_cloud(n_points=500, radius=2.0):
    """
    El Enemigo: Nube esférica del mismo tamaño.
    Mata al MLP porque ocupa el mismo volumen que el giroscopio.
    """
    u = np.random.uniform(0, 1, n_points)
    v = np.random.uniform(0, 1, n_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.vstack((x,y,z)).T + np.random.normal(0, 0.05, (n_points, 3))

def apply_global_spin(points):
    """Rota todo el sistema en 3D"""
    return R.random().apply(points)

# ==========================================
# 2. MODELOS COGNITIVOS 🧠
# ==========================================

# --- A. MLP (El Ciego) ---
class BlindMLP:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500)
    def fit(self, X, y):
        self.model.fit([x.flatten() for x in X], y)
    def predict(self, X):
        return self.model.predict([x.flatten() for x in X])

# --- B. AUTO-GYRO VNN (La Tuya, Calibrada) 🪐 ---
class AutoGyroVNN:
    def __init__(self, scans=300): 
        self.traces = None 
        self.tolerance = None # Se aprende en fit()
        self.scans = scans

    def fit(self, X, y):
        # 1. Aprender la Estructura (Trace)
        positives = [x for x, label in zip(X, y) if label == 1]
        # Usamos el primer ejemplo positivo como "Plano Maestro"
        self.traces = positives[0] - np.mean(positives[0], axis=0)
        print(f"   [AutoGyro] 🖐️ Estructura de 5 Anillos Aprendida.")

        # 2. AUTO-CALIBRACIÓN
        print(f"   [AutoGyro] ⚙️ Calibrando tolerancia con {len(positives)} ejemplos...")
        
        errors = []
        # Usamos 50 rotaciones para calibrar rápido
        calib_rotations = R.random(num=50) 
        
        for sample in positives:
            sample_centered = sample - np.mean(sample, axis=0)
            min_err = float('inf')
            
            for rot in calib_rotations:
                rotated_traces = rot.apply(self.traces)
                # Distance to Curve
                dists = cdist(sample_centered, rotated_traces, metric='euclidean')
                # Distancia promedio de cada satélite a su riel más cerca
                err = np.mean(np.min(dists, axis=1))
                if err < min_err: min_err = err
            
            errors.append(min_err)
        
        # DEFINIMOS LA TOLERANCIA
        # Aceptamos el 95% de la distribución de error de los casos buenos.
        # Multiplicamos por 1.15 para dar margen de seguridad ante ruido nuevo.
        self.tolerance = np.percentile(errors, 95) * 1.15
        print(f"   [AutoGyro] ✅ Tolerancia aprendida: {self.tolerance:.4f}")

    def predict(self, X):
        predictions = []
        rotations = R.random(num=self.scans) # Scans de test
        
        print(f"   [AutoGyro] Escaneando {len(X)} objetos...")
        for i, sample in enumerate(X):
            centroid = np.mean(sample, axis=0)
            sample_centered = sample - centroid
            
            min_structure_error = float('inf')
            
            for rot in rotations:
                rotated_traces = rot.apply(self.traces)
                dists = cdist(sample_centered, rotated_traces)
                avg_error = np.mean(np.min(dists, axis=1))
                
                if avg_error < min_structure_error:
                    min_structure_error = avg_error
            
            # Clasificación contra la tolerancia aprendida
            predictions.append(1 if min_structure_error < self.tolerance else 0)
            
            # Log de progreso para calmar la ansiedad
            if i % 50 == 0: 
                print(f"     -> Obj {i}: Err {min_structure_error:.3f} (Tol: {self.tolerance:.3f}) => {'Giroscopio' if min_structure_error < self.tolerance else 'Nube'}")
            
        return np.array(predictions)

# ==========================================
# 3. EJECUCIÓN FINAL 🚀
# ==========================================
print("=== 🖐️ EXPERIMENTO: GIROSCOPIO 5 ANILLOS (AUTO-CALIBRADO) ===")

N = 100
PTS = 400
X_train, y_train = [], []
X_test, y_test = [], []

# Train
print("Generando mundos de entrenamiento...")
for _ in range(N):
    if np.random.rand() > 0.5:
        X_train.append(generate_5_ring_gimbal(PTS, time_step=0))
        y_train.append(1)
    else:
        X_train.append(generate_adversarial_cloud(PTS))
        y_train.append(0)

# Test (Dinámico)
print("Generando mundos de test (Caos Dinámico)...")
for _ in range(N):
    if np.random.rand() > 0.5:
        # Giroscopio movido y rotado
        gimbal = generate_5_ring_gimbal(PTS, time_step=np.random.rand()*100)
        X_test.append(apply_global_spin(gimbal))
        y_test.append(1)
    else:
        # Nube rotada
        cloud = generate_adversarial_cloud(PTS)
        X_test.append(apply_global_spin(cloud))
        y_test.append(0)

print("\n[1] Entrenando Modelos...")
mlp = BlindMLP(); mlp.fit(X_train, y_train)
auto_gyro = AutoGyroVNN(scans=300); auto_gyro.fit(X_train, y_train)

print("\n[2] Testeando...")
acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
acc_gyro = accuracy_score(y_test, auto_gyro.predict(X_test))

print("-" * 60)
print(f"📊 RESULTADOS FINALES (CALIBRADOS)")
print("-" * 60)
print(f"❌ MLP:            {acc_mlp*100:.1f}%")
print(f"🪐 AutoGyro VNN:   {acc_gyro*100:.1f}%")
print("-" * 60)