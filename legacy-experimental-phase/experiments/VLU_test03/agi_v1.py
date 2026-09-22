import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

import numpy as np

class DiCaInertiaNeuron:
    """
    Motor de Inferencia de Carbono Computacional v4.1.
    """
    # 🔒 CONSTANTES UNIVERSALES DE LA RED
    K_VISCOSITY = 0.2816       # Viscosidad universal del fluido [cite: 1, 166, 281]
    R_PLANCK = 0.25            # Resolución mínima (Hardware de Planck) [cite: 1418]
    IMPEDANCE = 1/137          # Resistencia del vacío de Feynman
    
    def __init__(self, center, radius=0.4, label=0):
        self.c = np.array(center, dtype=np.float32) # Centroide (Posición) [cite: 19, 304]
        self.r = max(radius, self.R_PLANCK)          # Radio volumétrico [cite: 20, 305]
        self.votes = np.zeros(2)                     # Pesos de decisión (Markov) [cite: 21, 306]
        self.pain_accum = 0                          # Energía de dolor (Lagrange) [cite: 21, 307]
        self.n_samples = 0                           # Contador de masa (muestras) [cite: 21, 308]
        self.velocity = np.zeros_like(self.c)        # Twist cinemático (Velocidad ξ) 

    def match(self, x):
        """Verifica si el dato x cae dentro del radio de validez[cite: 22, 309]."""
        return np.sum((x - self.c)**2) <= self.r**2

    def get_proximity(self, x):
        """Función de Proximidad Radial Normalizada (Ecuación 8 del Paper)[cite: 122, 329]."""
        dist = np.linalg.norm(x - self.c)
        return max(0, 1 - dist / self.r) # Verdad determinista [cite: 126, 615]

    def update_physics(self, x, label, error):
        """
        Dinámica de Langevin: Actualiza la neurona como un fluido viscoso[cite: 878].
        """
        self.n_samples += 1
        
        # 1. Deriva Bayesiana modulada por Viscosidad (K) e Impedancia [cite: 921]
        # La inercia del fluido impide cambios erráticos por ruido térmico
        force = x - self.c
        lr = 0.1 / (np.sqrt(self.n_samples) * self.K_VISCOSITY)
        
        # Actualización de posición con 'fricción' universal
        self.velocity = (self.velocity * 0.9) + (force * lr)
        self.c += self.velocity # Movimiento por trayectoria, no solo salto [cite: 919]

        # 2. Actualización Markoviana de Votos (Política de Verdad) [cite: 31, 317]
        self.votes[int(label)] += 0.2 * (1 - self.votes[int(label)])

        # 3. Acumulación de Dolor para Mitosis [cite: 32, 318]
        if error > 0.4:
            self.pain_accum += 1

    def can_mitose(self):
        """
        Regla de Oro de Di-Ca: No hay división bajo el límite de Planck[cite: 55, 345].
        """
        # Solo se divide si el dolor es alto y el radio permite mayor resolución
        return self.pain_accum > 5 and self.r > (self.R_PLANCK * 2)

    def mitosis(self):
        """Genera dos neuronas hijas con el 60% del radio original[cite: 57, 348]."""
        new_radius = self.r * 0.6
        daughters = []
        for _ in range(2):
            # Desfase estocástico para exploración de fase [cite: 57, 347]
            jitter = np.random.normal(0, 0.05, self.c.shape)
            child = DiCaInertiaNeuron(self.c + jitter, new_radius)
            child.votes = self.votes.copy()
            daughters.append(child)
        return daughters

# --- 1. BASELINE: MLP (Mundo del Silicio) ---
class SilicioMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ) # [cite: 9-16]

    def forward(self, x): return self.net(x)

# --- 2. EL RETADOR: Cerebro de Carbono (Mundo de Di-Ca) ---
class VolumetricBrainCarbono:
    def __init__(self):
        self.neurons = []
        self.density_limit = 0.28 # Regla Vazza-Feletti (2020)
        self.volume_total = 4.0   # Espacio latente (-1 a 1 en 2D)

    def predict(self, x):
        if not self.neurons: return 0.5
        # Max-Aggregation: Inferencia por coincidencia geométrica [cite: 41, 128]
        activations = [n.get_proximity(x) * n.votes for n in self.neurons]
        agg = np.max(activations, axis=0)
        return agg[1] / (np.sum(agg) + 1e-6) if np.sum(agg) > 0 else 0.5

    def train_step(self, x, y):
        # 1. Encontrar neurona activa
        active = next((n for n in self.neurons if n.match(x)), None)
        
        if not active:
            # Creación de neurona (Nivel 1: Atómica) [cite: 1977]
            new_n = DiCaInertiaNeuron(x, radius=0.35)
            self.neurons.append(new_n)
            active = new_n

        # 2. Actualizar física con Inercia y Viscosidad K
        pred = self.predict(x)
        active.update_physics(x, y, abs(y - pred))

        # 3. Mitosis de Di-Ca (Respeta límite R=0.25)
        if active.can_mitose():
            daughters = active.mitosis()
            self.neurons.extend(daughters)
            self.neurons.remove(active)

        # 4. Gestión de Densidad Bio-Cósmica (Poda)
        self.manage_density()

    def manage_density(self):
        # Si los nodos activos superan el 28% del volumen, eliminamos las más débiles
        vol_actual = sum([np.pi * (n.r**2) for n in self.neurons])
        if (vol_actual / self.volume_total) > self.density_limit:
            # Ordenar por votos y eliminar la que menos 'verdad' aporta
            self.neurons.sort(key=lambda n: np.max(n.votes), reverse=True)
            self.neurons.pop()

# --- 3. EJECUCIÓN DEL EXPERIMENTO ---
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X = X.astype(np.float32)

# A. Entrenar Silicio (MLP)
mlp = SilicioMLP()
optimizer = optim.Adam(mlp.parameters(), lr=0.01)
# B. Entrenar Carbono (VNN Di-Ca)
brain = VolumetricBrainCarbono()
print(">>> Entrenando Carbono (VNN)...")

import time

# --- Medición Silicio (MLP) ---
start_mlp = time.time()
# Re-entrenamiento rápido para medición justa (o asumimos t=0 del anterior si queremos solo inferencia, pero el usuario pidió esto)
# El usuario pidió medir, pero el entrenamiento YA OCURRIÓ ARRIBA.
# Ajustaré el código para que mida lo que ya pasó si es posible, o envolveré.
# Como el código está lineal, lo mejor es insertar las mediciones ALREDEDOR de los loops de entrenamiento existentes.
# PERO, el usuario me dio un bloque para añadir "antes de la visualización".
# Si añado el bloque, necesito los tiempos.
# VOY A RE-ESTRUCTURAR LIGERAMENTE PARA CAPTURAR LOS TIEMPOS DURANTE EL ENTRENAMIENTO ARRIBA.

# MEJOR ESTRATEGIA: Modificar las secciones de entrenamiento para incluir start/end.

start_mlp_train = time.time()
for _ in range(200):
    loss = nn.BCELoss()(mlp(torch.tensor(X)).squeeze(), torch.tensor(y.astype(np.float32)))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
end_mlp_train = time.time()

# ... (VNN ya está definida)

start_vnn_train = time.time()
for i in range(len(X)):
    brain.train_step(X[i], y[i])
end_vnn_train = time.time()


# Bloque de Diagnóstico
params_mlp = sum(p.numel() for p in mlp.parameters()) 
masa_carbono = len(brain.neurons) 

print(f"\n📊 DIAGNÓSTICO DE EFICIENCIA DI-CA:")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"🔹 SILICIO (MLP):")
print(f"   - Masa (Parámetros): {params_mlp}")
print(f"   - Tiempo de ejecución: {end_mlp_train - start_mlp_train:.4f}s")
print(f"🔹 CARBONO (VNN):")
print(f"   - Masa (Neuronas): {masa_carbono}")
print(f"   - Tiempo de ejecución: {end_vnn_train - start_vnn_train:.4f}s")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"🏆 Veredicto: El Carbono es {params_mlp // max(1, masa_carbono)}x más ligero.")

# --- 4. VISUALIZACIÓN DE FRONTERAS ---
def plot_boundaries(model, ax, title, is_vnn=True):
    h = 0.05
    xx, yy = np.meshgrid(np.arange(-1.5, 2.5, h), np.arange(-1, 1.5, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if is_vnn: 
        Z = np.array([model.predict(p) for p in grid]).reshape(xx.shape)
    else: 
        Z = model(torch.tensor(grid, dtype=torch.float32)).detach().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# FIX: Agregamos is_vnn=False para el MLP
plot_boundaries(mlp, axes[0], "Silicio (MLP): Aproximación Estadística", is_vnn=False)

# Para el Cerebro de Carbono, is_vnn es True por defecto
plot_boundaries(brain, axes[1], f"Carbono (VNN): Certeza Geométrica\n({len(brain.neurons)} neuronas)")

plt.show()