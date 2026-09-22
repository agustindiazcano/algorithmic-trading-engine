import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Generación del Dataset XOR ---
def generate_xor_data(n_points=800):
    X = np.random.uniform(-1, 1, (n_points, 2))
    y = ((X[:, 0] * X[:, 1]) < 0).astype(np.float32)
    return X.astype(np.float32), y

# --- 2. Baseline: MLP ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# --- 3. Arquitectura VNN (Modular) ---
class CognitiveNeuron:
    def __init__(self, center, radius):
        self.c = np.array(center)
        self.r = radius
        self.votes = np.zeros(2)
        self.pain_accum = 0
        self.n_samples = 0

    def match(self, x):
        return np.sum((x - self.c)**2) <= self.r**2

    def update_vnn(self, x, label, error, use_bayes=True, use_markov=True, use_lagrange=True):
        if use_bayes:
            self.n_samples += 1
            lr = 0.1 / np.sqrt(self.n_samples)
            self.c += lr * (x - self.c)
        
        if use_markov:
            self.votes[int(label)] += 0.2 * (1 - self.votes[int(label)])
        
        if use_lagrange and error > 0.4:
            self.pain_accum += 1

class VolumetricBrain:
    def __init__(self, use_bayes=True, use_markov=True, use_mitosis=True):
        self.neurons = []
        self.use_bayes = use_bayes
        self.use_markov = use_markov
        self.use_mitosis = use_mitosis # Flag para controlar mitosis

    def predict(self, x):
        if not self.neurons: return 0.5
        activations = [max(0, 1 - np.linalg.norm(x - n.c)/n.r) * n.votes for n in self.neurons]
        agg = np.max(activations, axis=0)
        return agg[1] / (np.sum(agg) + 1e-6) if np.sum(agg) > 0 else 0.5

    def train_step(self, x, y):
        active = next((n for n in self.neurons if n.match(x)), None)
        if not active:
            new_n = CognitiveNeuron(x, 0.4)
            self.neurons.append(new_n)
            active = new_n
        
        pred = self.predict(x)
        active.update_vnn(x, y, abs(y - pred), 
                          use_bayes=self.use_bayes, 
                          use_markov=self.use_markov, 
                          use_lagrange=self.use_mitosis) # Solo acumula dolor si hay mitosis
        
        # MITOSIS CONDICIONAL
        if self.use_mitosis and active.pain_accum > 5 and active.r > 0.1:
            for _ in range(2):
                daughter = CognitiveNeuron(active.c + np.random.normal(0, 0.05, 2), active.r * 0.6)
                daughter.votes = active.votes.copy()
                self.neurons.append(daughter)
            self.neurons.remove(active)

# --- 4. Ejecución ---
X, y = generate_xor_data()

# Entrenar MLP
print(">>> Entrenando MLP...")
mlp = MLP()
opt = optim.Adam(mlp.parameters(), lr=0.01)
crit = nn.BCELoss()
for _ in range(101):
    loss = crit(mlp(torch.tensor(X)).squeeze(), torch.tensor(y))
    opt.zero_grad(); loss.backward(); opt.step()

# Entrenar Variantes VNN
vnn_std = VolumetricBrain(use_bayes=False, use_markov=False, use_mitosis=False) # Base tonta
vnn_fixed = VolumetricBrain(use_bayes=True, use_markov=True, use_mitosis=False) # Inteligente pero estéril
vnn_full = VolumetricBrain(use_bayes=True, use_markov=True, use_mitosis=True)   # Full Power

print("\n>>> Entrenando VNNs...")
for i in range(len(X)):
    vnn_std.train_step(X[i], y[i])
    vnn_fixed.train_step(X[i], y[i])
    vnn_full.train_step(X[i], y[i])

print(f"VNN Std: {len(vnn_std.neurons)} n")
print(f"VNN Fixed (No-Mitosis): {len(vnn_fixed.neurons)} n")
print(f"VNN Full (Mitosis): {len(vnn_full.neurons)} n")

# --- Visualización (4 Columnas) ---
def plot_model(model, ax, title, is_vnn=True):
    h = 0.05
    xx, yy = np.meshgrid(np.arange(-1.1, 1.1, h), np.arange(-1.1, 1.1, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if is_vnn: Z = np.array([model.predict(p) for p in grid]).reshape(xx.shape)
    else: Z = model(torch.tensor(grid, dtype=torch.float32)).detach().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    ax.set_title(title)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
plot_model(mlp, axes[0], "MLP", is_vnn=False)
plot_model(vnn_std, axes[1], "VNN Std (Vacía)")
plot_model(vnn_fixed, axes[2], f"VNN No-Mitosis ({len(vnn_fixed.neurons)} n)")
plot_model(vnn_full, axes[3], f"VNN Full ({len(vnn_full.neurons)} n)")
plt.show()