import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Generación del Dataset XOR ---
def generate_xor_data(n_points=800):
    X = np.random.uniform(-1, 1, (n_points, 2))
    # Clase A: cuadrantes (+,+) y (-,-) | Clase B: (+,-) y (-,+)
    y = ((X[:, 0] * X[:, 1]) < 0).astype(np.float32)
    return X.astype(np.float32), y

# --- 2. Baseline: MLP (Multi-Layer Perceptron) ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- 3. Arquitectura VNN (Volumetric Neural Network) ---
class CognitiveNeuron:
    def __init__(self, center, radius):
        self.c = np.array(center)
        self.r = radius
        self.votes = np.zeros(2) # Markovian Policy (Local weights)
        self.pain_accum = 0      # Señal de Lagrange
        self.n_samples = 0       # Para Deriva Bayesiana

    def match(self, x):
        # Inferencia Cuadrática optimizada (L2^2) [cite: 2435, 3660]
        return np.sum((x - self.c)**2) <= self.r**2

    def update_vnn(self, x, label, error, advanced=True):
        if advanced:
            # DERIVA BAYESIANA: El centro persigue la densidad 
            self.n_samples += 1
            lr = 0.1 / np.sqrt(self.n_samples)
            self.c += lr * (x - self.c)
            
            # REGLA MARKOVIANA: Actualización local de la clase 
            self.votes[int(label)] += 0.2 * (1 - self.votes[int(label)])
            
            # UMBRAL DE LAGRANGE: Acumulación de dolor por error [cite: 2392, 2529]
            if error > 0.4: self.pain_accum += 1

class VolumetricBrain:
    def __init__(self, advanced=True):
        self.neurons = []
        self.advanced = advanced

    def predict(self, x):
        if not self.neurons: return 0.5
        # Max-aggregation de conos de verdad [cite: 2261, 2713]
        activations = [max(0, 1 - np.linalg.norm(x - n.c)/n.r) * n.votes for n in self.neurons]
        agg = np.max(activations, axis=0)
        return agg[1] / (np.sum(agg) + 1e-6) if np.sum(agg) > 0 else 0.5

    def train_step(self, x, y):
        active = next((n for n in self.neurons if n.match(x)), None)
        if not active:
            # Creación con transferencia (Generalización Zero-Shot) [cite: 2489, 3131]
            new_n = CognitiveNeuron(x, 0.4)
            self.neurons.append(new_n)
            active = new_n
        
        pred = self.predict(x)
        active.update_vnn(x, y, abs(y - pred), advanced=self.advanced)
        
        # MITOSIS FRACTAL: Especialización ante el dolor [cite: 2391, 2534]
        if self.advanced and active.pain_accum > 5 and active.r > 0.1:
            for _ in range(2):
                daughter = CognitiveNeuron(active.c + np.random.normal(0, 0.05, 2), active.r * 0.6)
                daughter.votes = active.votes.copy()
                self.neurons.append(daughter)
            self.neurons.remove(active)

# --- 4. Ejecución del Experimento ---
X, y = generate_xor_data()

# 1. Entrenar MLP con telemetría
print(">>> Entrenando MLP Baseline...")
mlp = MLP()
optimizer = optim.Adam(mlp.parameters(), lr=0.01)
criterion = nn.BCELoss()
for epoch in range(101):
    out = mlp(torch.tensor(X))
    loss = criterion(out.squeeze(), torch.tensor(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0: print(f"MLP Loss: {loss.item():.4f}")

# 2. Entrenar VNNs (Corrigiendo el bug de votos)
vnn_std = VolumetricBrain(advanced=False)
vnn_adv = VolumetricBrain(advanced=True)

print("\n>>> Entrenando VNNs...")
for i in range(len(X)):
    vnn_std.train_step(X[i], y[i])
    vnn_adv.train_step(X[i], y[i])
print(f"VNN Standard: {len(vnn_std.neurons)} neuronas")
print(f"VNN Avanzada: {len(vnn_adv.neurons)} neuronas")

# --- Visualización de Resultados ---
def plot_model(model, ax, title, is_vnn=True):
    h = 0.05
    xx, yy = np.meshgrid(np.arange(-1.1, 1.1, h), np.arange(-1.1, 1.1, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if is_vnn:
        Z = np.array([model.predict(p) for p in grid]).reshape(xx.shape)
    else:
        with torch.no_grad():
            inputs = torch.tensor(grid, dtype=torch.float32)
            Z = model(inputs).reshape(xx.shape).numpy()

    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # 3 Columnas ahora
plot_model(mlp, axes[0], "Baseline MLP", is_vnn=False)
plot_model(vnn_std, axes[1], f"VNN Standard ({len(vnn_std.neurons)} n)")
plot_model(vnn_adv, axes[2], f"VNN Avanzada ({len(vnn_adv.neurons)} n)")
plt.show()