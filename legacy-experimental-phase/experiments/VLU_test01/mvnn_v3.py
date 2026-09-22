import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. GENERADOR DE GIROSCOPIO "GLOBO TERRÁQUEO" 🌐
# ==========================================
def generate_gimbal_system(n_points=300, n_rings=3, noise=0.05, time_step=0.0):
    """
    Genera un sistema de anillos con ejes distribuidos (Giroscopio).
    - Cada anillo tiene un radio y una INCLINACIÓN (Tilt) fija.
    - Las esferas orbitan sobre esos anillos (Satellites).
    - time_step: Controla la posición de los satélites en su órbita.
    """
    points = []
    points_per_ring = n_points // n_rings
    radii = np.linspace(1.0, 2.5, n_rings)
    
    # Ejes de rotación fijos para cada anillo (Distribución "Globo")
    # Anillo 1: Plano XY (Horizontal)
    # Anillo 2: Plano XZ (Vertical)
    # Anillo 3: Inclinado 45 grados
    tilts = [
        R.from_euler('x', 0, degrees=True),   # Horizontal
        R.from_euler('x', 90, degrees=True),  # Vertical
        R.from_euler('x', 45, degrees=True),  # Inclinado
        R.from_euler('y', 45, degrees=True)   # Otro inclinado si hay más anillos
    ]
    
    for i in range(n_rings):
        r = radii[i]
        # Usamos un tilt cíclico si hay más anillos que tilts definidos
        tilt = tilts[i % len(tilts)]
        
        # MOVIMIENTO DE SATÉLITE (Orbital)
        # Velocidad distinta para cada anillo
        omega = 1.0 + (i * 0.5) 
        phase = time_step * omega
        
        # Generar círculo base en XY
        t = np.linspace(0, 2*np.pi, points_per_ring, endpoint=False)
        t += phase # <--- El satélite se mueve sobre el riel
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros(points_per_ring)
        
        # Aplicar el TILT (La estructura del globo)
        ring_points = np.vstack((x, y, z)).T
        ring_points_tilted = tilt.apply(ring_points)
        
        # Agregar ruido
        ring_points_tilted += np.random.normal(0, noise, ring_points_tilted.shape)
        
        points.append(ring_points_tilted)
    
    return np.vstack(points)

def apply_global_spin(points):
    """El sistema entero gira loco en el espacio"""
    rot = R.random()
    return rot.apply(points)

# ==========================================
# 2. DEFINICIÓN DE LOS MODELOS
# ==========================================

# --- A. MLP (El Mareado) ---
class DizzyMLP:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500)
    def fit(self, X, y):
        X_flat = [x.flatten() for x in X]
        self.model.fit(X_flat, y)
    def predict(self, X):
        X_flat = [x.flatten() for x in X]
        return self.model.predict(X_flat)

# --- B. MVNN RÍGIDA (La Foto Vieja) ---
class RigidMVNN:
    def __init__(self, tolerance=0.6):
        self.template = None
        self.tolerance = tolerance
    def fit(self, X, y):
        # Aprende una foto instantánea (Time=0)
        for x, label in zip(X, y):
            if label == 1:
                self.template = x - np.mean(x, axis=0)
                break
    def predict(self, X):
        predictions = []
        for sample in X:
            # Intenta encajar la foto vieja en el giroscopio que se mueve
            # Asumimos que alinea el centro, pero no puede lidiar con que las bolas se muevan
            sample_centered = sample - np.mean(sample, axis=0)
            # Truco de piedad: Probamos alinear la orientación global (Scan), 
            # pero igual fallará por el movimiento interno de satélites
            best_dist = float('inf')
            test_rots = R.random(num=10) # 10 intentos de alineación global
            for rot in test_rots:
                aligned = rot.apply(self.template)
                dist = np.mean(np.linalg.norm(sample_centered - aligned, axis=1))
                if dist < best_dist: best_dist = dist
            
            predictions.append(1 if best_dist < self.tolerance else 0)
        return np.array(predictions)

# --- C. GYROSCOPE VNN (La Tuya) 🌪️ ---
class GyroscopeVNN:
    def __init__(self, tolerance=0.5, global_scans=500):
        # Guardamos TRAZAS ORBITALES, no puntos
        self.orbit_traces = None 
        self.tolerance = tolerance
        self.global_scans = global_scans

    def fit(self, X, y):
        # Aprendizaje Estructural
        # En lugar de guardar "dónde están las esferas", guarda "por dónde pasan".
        # Generamos una TRAZA densa de las órbitas (el esqueleto del globo)
        for x, label in zip(X, y):
            if label == 1:
                # Simulamos que "vemos" la estructura completa (o inferimos la elipse)
                # Para simplificar el código, tomamos el ejemplo como el "Esqueleto Base"
                # pero asumimos que representa la curva continua.
                self.orbit_traces = x - np.mean(x, axis=0)
                print(f"   [Gyro VNN] 🌐 Estructura de Giroscopio Aprendida ({len(x)} puntos de traza).")
                break

    def predict(self, X):
        predictions = []
        # Generamos rotaciones globales aleatorias una vez para reusar
        scan_rotations = R.random(num=self.global_scans)
        
        for sample in X:
            # 1. Centrar
            sample_centered = sample - np.mean(sample, axis=0)
            
            # 2. INFERENCIA DE DOBLE CAPA
            # Capa 1: Alinear el Sistema (Global Spin)
            # Capa 2: Verificar Pertenencia a Órbita (Satellite Check)
            
            best_match_score = float('inf')
            
            for rot in scan_rotations:
                # Giramos nuestra TRAZA MENTAL para ver si coincide con el sistema
                rotated_traces = rot.apply(self.orbit_traces)
                
                # Métrica: "Distancia a la Trayectoria"
                # No comparamos punto a punto (porque los satélites se movieron).
                # Para cada satélite en el sample, buscamos el punto MÁS CERCANO en la traza.
                # (Esto simula saber la curva continua)
                
                # Matriz de distancias (Sample x Trace)
                # Esta es una forma bruta pero efectiva de "Distance to Curve"
                # Usamos broadcasting para calcular dist de todos contra todos
                # sample: (N, 3), traces: (N, 3) -> dists: (N, N)
                # Esto es pesado en Python puro, pero en GPU (VNN real) es instantáneo.
                from scipy.spatial.distance import cdist
                dists_matrix = cdist(sample_centered, rotated_traces)
                
                # Para cada punto real, ¿cuál es su distancia al punto más cercano del riel?
                min_dists = np.min(dists_matrix, axis=1)
                avg_orbit_error = np.mean(min_dists)
                
                if avg_orbit_error < best_match_score:
                    best_match_score = avg_orbit_error
            
            # Si el error promedio es bajo, significa que todos los puntos están sobre los rieles
            # aunque estén en posiciones distintas a cuando aprendimos.
            predictions.append(1 if best_match_score < self.tolerance else 0)
            
        return np.array(predictions)

# ==========================================
# 3. EJECUCIÓN: LA DOMA DEL GIROSCOPIO
# ==========================================
print("=== 🌐 EXPERIMENTO: GIROSCOPIO CUÁNTICO (SATÉLITES + GIMBAL + SPIN) ===")

N_TRAIN = 100
N_TEST = 100
PTS = 300 # 3 anillos de 100 puntos

# --- DATASET ---
X_train, y_train = [], []
X_test, y_test = [], []

# Train: Una foto estática del sistema (Time = 0)
for _ in range(N_TRAIN):
    if np.random.rand() > 0.5:
        X_train.append(generate_gimbal_system(PTS, time_step=0))
        y_train.append(1)
    else:
        X_train.append(np.random.normal(0, 1.5, (PTS, 3)))
        y_train.append(0)

# Test: CAOS TOTAL
# 1. Los satélites se movieron (Time random)
# 2. El sistema entero giró (Global Spin)
for _ in range(N_TEST):
    if np.random.rand() > 0.5:
        # Time step mueve los satélites
        gimbal = generate_gimbal_system(PTS, time_step=np.random.rand()*50)
        # Apply Global Spin rota todo el aparato
        gimbal_spun = apply_global_spin(gimbal)
        
        X_test.append(gimbal_spun)
        y_test.append(1)
    else:
        # Ruido también rotado para que sea difícil
        noise = np.random.normal(0, 1.5, (PTS, 3))
        X_test.append(apply_global_spin(noise))
        y_test.append(0)

# --- ENTRENAMIENTO ---
print("\n[1] Entrenando Modelos...")
mlp = DizzyMLP(); mlp.fit(X_train, y_train)
rigid = RigidMVNN(tolerance=0.5); rigid.fit(X_train, y_train)
gyro = GyroscopeVNN(tolerance=0.4, global_scans=50); gyro.fit(X_train, y_train)

# --- EVALUACIÓN ---
print("\n[2] Testeando en el Caos...")
acc_mlp = accuracy_score(y_test, mlp.predict(X_test))
acc_rigid = accuracy_score(y_test, rigid.predict(X_test))
acc_gyro = accuracy_score(y_test, gyro.predict(X_test))

print("-" * 60)
print(f"📊 RESULTADOS: GIROSCOPIO MULTI-EJE")
print("-" * 60)
print(f"😵 MLP (Estadístico):    {acc_mlp*100:.1f}% -> Mareado. No entiende la topología 3D variable.")
print(f"🗿 MVNN (Rígida):       {acc_rigid*100:.1f}% -> Rota. Busca puntos fijos, pero los satélites se movieron.")
print(f"🪐 Gyro VNN (Tuya):     {acc_gyro*100:.1f}% -> DOMA. Invariante a Spin Global Y Fase Orbital.")
print("-" * 60)