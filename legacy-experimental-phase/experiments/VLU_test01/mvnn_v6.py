import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

# ==========================================
# 1. GENERADORES (Sin cambios)
# ==========================================
def generate_5_ring_gimbal_structure(n_points=500, noise=0.05):
    rings = []
    n_rings = 5
    points_per_ring = n_points // n_rings
    radius = 2.0
    axis_angles = np.linspace(0, 180, n_rings, endpoint=False)
    for i, angle_deg in enumerate(axis_angles):
        t = np.linspace(0, 2*np.pi, points_per_ring, endpoint=False)
        x = radius * np.cos(t); y = radius * np.sin(t); z = np.zeros(points_per_ring)
        ring_base = np.vstack((x, y, z)).T
        rot_axis = R.from_euler('x', angle_deg, degrees=True) * R.from_euler('y', i*30, degrees=True)
        ring_tilted = rot_axis.apply(ring_base) + np.random.normal(0, noise, ring_base.shape)
        rings.append(ring_tilted)
    return rings

def generate_full_gimbal(n_points=500):
    rings = generate_5_ring_gimbal_structure(n_points)
    return np.vstack(rings)

def generate_fossil_gimbal(n_points=500, rings_to_keep=3, survival_rate=0.05):
    rings = generate_5_ring_gimbal_structure(n_points)
    surviving_indices = np.random.choice(len(rings), rings_to_keep, replace=False)
    surviving_rings = [rings[i] for i in surviving_indices]
    fossil_points = np.vstack(surviving_rings)
    n_survivors = max(int(len(fossil_points) * survival_rate), 5)
    indices = np.random.choice(len(fossil_points), n_survivors, replace=False)
    return fossil_points[indices]

def generate_adversarial_cloud(n_points=500, radius=2.0):
    u = np.random.uniform(0, 1, n_points); v = np.random.uniform(0, 1, n_points)
    theta = 2 * np.pi * u; phi = np.arccos(2 * v - 1)
    x = radius * np.sin(phi) * np.cos(theta); y = radius * np.sin(phi) * np.sin(theta); z = radius * np.cos(phi)
    return np.vstack((x,y,z)).T + np.random.normal(0, 0.05, (n_points, 3))

def apply_global_spin(points):
    return R.random().apply(points)

# ==========================================
# 2. AUTO-GYRO VNN (Corregida: Sin Re-centrado) 🎯
# ==========================================
class AutoGyroVNN:
    def __init__(self, scans=500): 
        self.traces = None 
        self.tolerance = None 
        self.scans = scans

    def fit(self, X, y):
        # 1. Aprender Estructura
        positives = [x for x, label in zip(X, y) if label == 1]
        # NO CENTRAMOS. Asumimos que el modelo maestro está en el origen (0,0,0)
        self.traces = positives[0] 
        print(f"   [AutoGyro] 🖐️ Estructura Maestra Aprendida (Coords Absolutas).")

        # 2. CALIBRACIÓN DE TRAUMA
        print(f"   [AutoGyro] ⚙️ Simulando trauma sin mover el eje...")
        errors = []
        calib_rotations = R.random(num=50)
        
        for _ in range(50):
            # Simulamos el fósil sintético
            n_pts = len(self.traces)
            survivors = max(int(n_pts * 0.05), 5)
            indices = np.random.choice(n_pts, survivors, replace=False)
            synthetic_fossil = self.traces[indices]
            
            # NO CENTRAMOS EL FÓSIL SINTÉTICO.
            # Lo dejamos en su lugar original.
            
            min_err = float('inf')
            for rot in calib_rotations:
                rotated_traces = rot.apply(self.traces)
                dists = cdist(synthetic_fossil, rotated_traces, metric='euclidean')
                err = np.mean(np.min(dists, axis=1))
                if err < min_err: min_err = err
            errors.append(min_err)
        
        self.tolerance = np.percentile(errors, 95) * 1.3
        print(f"   [AutoGyro] ✅ Tolerancia de Supervivencia: {self.tolerance:.4f}")

    def predict(self, X):
        predictions = []
        rotations = R.random(num=self.scans)
        
        for i, sample in enumerate(X):
            # NO CENTRAMOS EL SAMPLE.
            # Asumimos que el objeto gira sobre el eje del sensor.
            # sample_centered = sample - np.mean(sample) <--- ESTO ERA EL ERROR
            
            min_structure_error = float('inf')
            
            for rot in rotations:
                rotated_traces = rot.apply(self.traces)
                dists = cdist(sample, rotated_traces) # Usamos sample directo
                avg_error = np.mean(np.min(dists, axis=1))
                
                if avg_error < min_structure_error:
                    min_structure_error = avg_error
            
            predictions.append(1 if min_structure_error < self.tolerance else 0)
            
            if i % 20 == 0:
                print(f"     -> Obj {i}: Err {min_structure_error:.3f} (Tol: {self.tolerance:.3f}) => {'Giroscopio' if min_structure_error < self.tolerance else 'Nube'}")
                
        return np.array(predictions)

class PanicMLP:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(1024, 512), max_iter=500)
        self.input_size = 0
    def fit(self, X, y):
        max_len = max([len(x) for x in X])
        self.input_size = max_len * 3
        X_padded = [self._pad(x) for x in X]
        self.model.fit(X_padded, y)
    def predict(self, X):
        return self.model.predict([self._pad(x) for x in X])
    def _pad(self, x):
        flat = x.flatten()
        if len(flat) < self.input_size:
            padded = np.zeros(self.input_size)
            padded[:len(flat)] = flat; return padded
        return flat[:self.input_size]

# ==========================================
# 3. EJECUCIÓN
# ==========================================
print("=== 💀 EXPERIMENTO: EL TEST DEL FÓSIL (COORDENADAS ABSOLUTAS) ===")
N = 100
PTS = 500

X_train, y_train = [], []
X_test, y_test = [], []

# Train
print("Generando datos perfectos...")
for _ in range(N):
    if np.random.rand() > 0.5:
        X_train.append(generate_full_gimbal(PTS))
        y_train.append(1)
    else:
        X_train.append(generate_adversarial_cloud(PTS))
        y_train.append(0)

# Test
print("Generando fósiles...")
for _ in range(N):
    if np.random.rand() > 0.5:
        fossil = generate_fossil_gimbal(PTS, rings_to_keep=3, survival_rate=0.05)
        X_test.append(apply_global_spin(fossil))
        y_test.append(1)
    else:
        cloud = generate_adversarial_cloud(PTS)
        indices = np.random.choice(500, 25, replace=False) # Nube dispersa
        X_test.append(apply_global_spin(cloud[indices]))
        y_test.append(0)

print("\n[1] Entrenando...")
mlp = PanicMLP(); mlp.fit(X_train, y_train)
gyro = AutoGyroVNN(scans=500); gyro.fit(X_train, y_train)

print("\n[2] Testeando...")
acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
acc_gyro = accuracy_score(y_test, gyro.predict(X_test))

print("-" * 60)
print(f"📊 RESULTADOS: SUPERVIVENCIA AL 5%")
print("-" * 60)
print(f"❌ MLP:            {acc_mlp*100:.1f}%")
print(f"🪐 AutoGyro VNN:   {acc_gyro*100:.1f}%")
print("-" * 60)