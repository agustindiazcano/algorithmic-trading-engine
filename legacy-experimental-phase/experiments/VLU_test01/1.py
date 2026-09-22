import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. GENERADOR DE MERCADO (Stress Test) ---
def generate_chaotic_market(n=2000):
    returns = np.random.normal(0, 0.02, n)
    price = 100 * np.cumprod(1 + returns)
    volume = np.random.lognormal(0, 0.5, n)
    
    patterns = []
    # Definición del patrón base
    pat_p = np.array([0, -1.0, -2.5, -1.0, 0]) * 1.5 # Profundidad
    pat_v = np.array([1.0, 2.0, 4.0, 2.0, 1.0])      # Spike
    
    for _ in range(20): # 20 Oportunidades
        idx = np.random.randint(100, n-100)
        patterns.append(idx)
        current_p = price[idx]
        
        # Inyectar
        scale = current_p * 0.01
        for k in range(5):
            price[idx-2+k] = current_p + pat_p[k] * scale
            volume[idx-2+k] *= pat_v[k] # Multiplicativo para volumen
            
    return price, volume, patterns

# --- 2. VNN SWARM (Auto-Calibrada) ---
class VNN_Swarm_Calibrated(nn.Module):
    def __init__(self, n_neurons=50):
        super().__init__()
        
        # --- AUTO-CALIBRACIÓN ---
        # 1. Definimos el patrón IDEAL en crudo
        raw_t = np.linspace(-2, 2, 5)
        raw_p = np.array([0, -1.0, -2.5, -1.0, 0]) 
        raw_v = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
        
        # 2. Simulamos la normalización que hará el bot en vivo
        # (Z-Score de la ventana de 5 puntos)
        norm_p = (raw_p - np.mean(raw_p)) / (np.std(raw_p) + 1e-6)
        norm_v = (raw_v - np.mean(raw_v)) / (np.std(raw_v) + 1e-6)
        
        # 3. Ahora interpolamos ESTOS valores normalizados para las 50 neuronas
        t_dense = np.linspace(-2, 2, n_neurons)
        p_dense = np.interp(t_dense, raw_t, norm_p)
        v_dense = np.interp(t_dense, raw_t, norm_v)
        
        # 4. Inicializamos los centros con esta "Imagen Mental" perfecta
        init_centers = np.stack([t_dense, p_dense, v_dense], axis=1)
        # Pequeño jitter para robustez (Enjambre biológico)
        init_centers += np.random.normal(0, 0.05, init_centers.shape)
        
        self.centers = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32))
        
        # Ejes ajustados (Tolerancia)
        # [tol_t, tol_p, tol_v]
        init_axes = np.ones((n_neurons, 3)) * 0.4 # Bastante estricto
        init_axes[:, 0] = 0.2 # Tiempo muy estricto (secuencialidad)
        
        self.raw_axes = nn.Parameter(torch.tensor(init_axes, dtype=torch.float32))

    def get_axes(self): return torch.nn.functional.softplus(self.raw_axes) + 0.05

    def forward(self, x):
        # x: (Batch, 5, 3)
        n_neurons = self.centers.shape[0]
        axes = self.get_axes()
        
        x_exp = x.unsqueeze(2) 
        c_exp = self.centers.view(1, 1, n_neurons, 3)
        a_exp = axes.view(1, 1, n_neurons, 3)
        
        # Distancia (Swarm Matching)
        dist_sq = torch.sum(((x_exp - c_exp)/(a_exp/2))**2, dim=3)
        activations = torch.exp(-0.5 * dist_sq)
        
        # Lógica: Cada punto de la ventana debe encontrar SU neurona
        best_neuron_per_point, _ = torch.max(activations, dim=2)
        shape_score = torch.mean(best_neuron_per_point, dim=1)
        
        return shape_score

# --- 3. EXECUTION ---
print(">>> Iniciando Stress Test Corregido (Auto-Calibración)...")

n_universes = 50
results = []
all_curves = []

for universe_id in range(n_universes):
    price, volume, patterns = generate_chaotic_market(1500)
    agent = VNN_Swarm_Calibrated(50)
    
    cash = 10000
    shares = 0
    curve = [10000]
    entry_idx = -1
    window_size = 5
    
    for t in range(window_size, len(price)-1):
        w_p = price[t-window_size : t]
        w_v = volume[t-window_size : t]
        
        # Normalización IDÉNTICA a la inicialización
        w_p_norm = (w_p - np.mean(w_p)) / (np.std(w_p) + 1e-6)
        w_v_norm = (w_v - np.mean(w_v)) / (np.std(w_v) + 1e-6)
        
        t_steps = np.linspace(-2, 2, window_size)
        x_in = torch.tensor(np.stack([t_steps, w_p_norm, w_v_norm], axis=1), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            signal = agent(x_in).item()
        
        current_p = price[t]
        
        # Venta
        if shares > 0 and (t - entry_idx >= 5):
            cash += shares * current_p
            shares = 0
            
        # Compra (Ahora 0.7 debería ser fácil de alcanzar para patrones reales)
        if shares == 0 and signal > 0.65:
            shares = cash / current_p
            cash = 0
            entry_idx = t
            
        curve.append(cash + shares * current_p)
    
    final = curve[-1]
    ret = (final - 10000)/100
    results.append(ret)
    all_curves.append(curve)

# Stats
avg_ret = np.mean(results)
win_rate = np.sum(np.array(results) > 0) / n_universes * 100
print(f"\n=== RESULTADOS FINALES (50 Neuronas Calibradas) ===")
print(f"Retorno Promedio: {avg_ret:.2f}%")
print(f"Win Rate: {win_rate:.1f}%")

plt.figure(figsize=(12, 6))
for c in all_curves:
    plt.plot(c, alpha=0.3, color='green' if c[-1]>10000 else 'red')
plt.plot(np.mean(np.array([c[:len(all_curves[0])] for c in all_curves]), axis=0), color='k', linewidth=3)
plt.axhline(10000, linestyle=':', color='k')
plt.title(f"VNN Swarm - 50 Neuronas Auto-Calibradas\nWin Rate: {win_rate}%")
plt.show()