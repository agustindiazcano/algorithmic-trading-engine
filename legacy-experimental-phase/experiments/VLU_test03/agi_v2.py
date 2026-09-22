import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- 1. MEMORIA GEOMÉTRICA (Template Completo) ---
def get_full_template(n=100):
    # Creamos una 'Luna' perfecta como molde original [cite: 1312-1316]
    t = np.linspace(0, np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    return np.vstack([x, y]).T

# --- 2. MOTOR DE CARBONO (Inferencia por Integridad) ---
class CarbonIntegrityVNN:
    def __init__(self, template):
        self.template = template
        self.r = 0.25 # Resolución Di-Ca [cite: 1332]

    def check_identity(self, partial_obs):
        # Mide cuántos 'átomos' del molde encuentran apoyo en la realidad [cite: 1333-1340]
        matches = 0
        for t_point in self.template:
            dists = np.linalg.norm(partial_obs - t_point, axis=1)
            if np.min(dists) < self.r:
                matches += 1
        return matches / len(self.template)

# --- 3. ENTRENAMIENTO DEL SILICIO (MLP Overfitted) ---
template = get_full_template(100)
mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
opt = optim.Adam(mlp.parameters(), lr=0.01)
X_train = torch.tensor(template, dtype=torch.float32)
for _ in range(200):
    loss = nn.BCELoss()(mlp(X_train).squeeze(), torch.ones(len(template)))
    opt.zero_grad(); loss.backward(); opt.step()

# --- 4. EL TEST DE LA CEGUERA: BORRANDO EL 60% DEL OBJETO ---
# Cortamos el centro de la luna: solo quedan las puntas [cite: 1343-1345]
mask = (template[:, 0] < -0.4) | (template[:, 0] > 0.4)
broken_object = template[mask]

print(f"\n📡 INICIANDO TELEMETRÍA DE OCLUSIÓN...")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Puntos Originales: {len(template)} | Puntos Visibles: {len(broken_object)}")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Predicción Silicio (Cae porque 'no hay señal' en el centro) [cite: 1350]
with torch.no_grad():
    conf_s = mlp(torch.tensor(broken_object, dtype=torch.float32)).mean().item()

# Predicción Carbono (Reconoce la estructura por los puntos restantes) [cite: 1348, 1376]
brain_carbon = CarbonIntegrityVNN(template)
conf_c = brain_carbon.check_identity(broken_object)

print(f"🔹 CONFIDENCIA SILICIO: {conf_s:.4f} -> {'❌ COLAPSO (Ve manchas)' if conf_s < 0.5 else '✅ SOBREVIVE'}")
print(f"🔹 CONFIDENCIA CARBONO: {conf_c:.4f} -> {'✅ GESTALT (Ve el objeto completo)'}")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# --- 5. VISUALIZACIÓN DE LA 'DO TOTAL' ---
plt.figure(figsize=(10, 6))
plt.scatter(template[:, 0], template[:, 1], color='blue', alpha=0.1, label="Memoria (Forma Completa)")
plt.scatter(broken_object[:, 0], broken_object[:, 1], color='red', s=50, label="Realidad (Objeto Roto/Tapado)")
plt.plot(template[:, 0], template[:, 1], 'g--', alpha=0.3, label="Inferencia Molecular (Relleno Mental)")
plt.title(f"Do de Gestalt: Objeto Ocluido\nCarbono: {conf_c:.2f} | Silicio: {conf_s:.2f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()