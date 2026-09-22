import numpy as np
import matplotlib.pyplot as plt
import requests
import json

# --- 1. CONFIGURACIÓN Y DATOS (Binance) ---
print("📡 Conectando a Binance (XRPUSDT 1h)...")
try:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000}
    response = requests.get(url, params=params)
    data = response.json()
    
    # Klines: [Time, Open, High, Low, Close, Volume, ...]
    # Close en index 4, Volumen en index 5
    prices = np.array([float(k[4]) for k in data])
    volumes = np.array([float(k[5]) for k in data])
    
    print(f"✅ Datos Binance: {len(prices)} velas. Precio final: ${prices[-1]:.4f}")
except Exception as e:
    print(f"❌ Error conectando a Binance: {e}")
    print("⚠️ Usando datos sintéticos...")
    ticks = 1000
    t = np.linspace(0, 1, ticks)
    prices = 1.6 + 0.3 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.02, ticks)
    volumes = 1000000 + np.random.normal(0, 100000, ticks)

# --- 2. CONFIGURACIÓN DEL ORGANISMO 3D (Carbono Puro) ---
class DarkMatterVNN_3D:
    def __init__(self, start_price, start_vol):
        # ESTADO 3D: [Precio, Volumen, Volatilidad]
        self.center_3d = np.array([float(start_price), float(start_vol), 0.0])
        # Radios iniciales proporcionales (5% precio, 100% volumen, 2% volatilidad)
        self.radius_3d = np.array([float(start_price) * 0.05, float(start_vol) * 1.0, 0.02])
        self.signals = np.zeros(len(prices))
        self.persistence = 0
        self.trades = 0
        self.history_centers = []
        self.history_radii_p = []
        
    def run(self, p_vec, v_vec):
        print(f"🧠 Inicializando VNN 3D (Precio+Volumen+Volatilidad)...")
        # Precalentamiento para estabilizar medias
        for i in range(25, len(p_vec)):
            # 1. PERCEPCIÓN: Calculamos la volatilidad local (Nerviosismo)
            volat = float(np.std(p_vec[i-20:i]))
            current_p = float(p_vec[i])
            current_v = float(v_vec[i])
            
            state = np.array([current_p, current_v, volat])
            
            # 2. RESPIRACIÓN ESTOICA (3D): 
            # El radio del volumen se expande para absorber el ruido de Binance
            target_r_p = (current_p * 0.045) + (volat * 2.0)
            target_r_v = np.mean(v_vec[i-20:i]) * 1.5 # Margen del 50% sobre el volumen medio
            
            # Ajuste de membrana con inercia (no colapsa instantáneamente)
            self.radius_3d[0] = 0.95 * self.radius_3d[0] + 0.05 * target_r_p
            self.radius_3d[1] = 0.95 * self.radius_3d[1] + 0.05 * target_r_v
            self.radius_3d[2] = 0.95 * self.radius_3d[2] + 0.05 * (volat * 1.2)
            
            # 3. EL CLIC GEOMÉTRICO (Distancia de Mahalanobis)
            # Usamos presión direccional simple para trading direccional
            p_press = (state[0] - self.center_3d[0]) / self.radius_3d[0]
            v_press = (state[1] - self.center_3d[1]) / self.radius_3d[1]
            
            # Lógica "Survivor": Ruptura de precio PERSISTENTE confirmada por VOLUMEN ALTO
            # Solo operamos si el volumen es mayor a la media (> -0.2 normalizado)
            if np.abs(p_press) > 1.2: # Rompió membrana precio
                # Confirmación de volumen: ¿Hay gasolina?
                if p_press < -1.0 and v_press > -0.2: # Caída con algo de volumen
                     self.persistence += 1 # Pánico real
                elif p_press > 1.0 and v_press > -0.2: # Subida con algo de volumen
                     self.persistence -= 1 # Euforia real
            else:
                self.persistence = int(self.persistence * 0.6) # Enfriamiento rápido
            
            # 4. DISPARO DEL AGENTE
            if self.persistence >= 4: # Confirmación de 4 ticks
                self.signals[i] = 1; self.persistence = 0; self.trades += 1
            elif self.persistence <= -4:
                self.signals[i] = -1; self.persistence = 0; self.trades += 1
            
            # 5. INERCIA MOLECULAR: El centro sigue al mercado lentamente
            self.center_3d = 0.98 * self.center_3d + 0.02 * state
            
            # Log
            self.history_centers.append(self.center_3d[0])
            self.history_radii_p.append(self.radius_3d[0])
            
        return self.signals

# --- 3. EJECUCIÓN CON DATOS REALES ---
vnn_logic = DarkMatterVNN_3D(prices[0], volumes[0])
vnn_signals = vnn_logic.run(prices, volumes)

# --- 4. RESULTADOS DE LA DO ---
def simulate_wealth(p, s, fee=0.001):
    cash, pos = 10000.0, 0.0
    hist = []
    # Usamos 100% equity como pidió el usuario (Timba mode)
    for i in range(len(p)):
        price = float(p[i])
        if s[i] == 1 and cash > 10: # Compra All-in
            shares = (cash * 0.99) / price # 1% buffer
            pos += shares; cash -= shares * price * (1 + fee)
        elif s[i] == -1 and pos > 0: # Venta All-out
            cash += pos * price * (1 - fee); pos = 0.0
        hist.append(float(cash + pos * price))
    return np.array(hist)

wealth_vnn = simulate_wealth(prices, vnn_signals)
wealth_bh = (prices / prices[0]) * 10000.0

# --- 5. VISUALIZACIÓN ---
print(f"\n📊 RESULTADOS VNN 3D:")
print(f"💰 Final VNN: ${wealth_vnn[-1]:,.2f}")
print(f"💎 Buy & Hold: ${wealth_bh[-1]:,.2f}")
print(f"🔄 Trades:     {vnn_logic.trades}")
print(f"🚀 Profit:     {((wealth_vnn[-1]-10000)/10000)*100:.2f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Precio y Respiración
ax1.plot(prices, 'k', alpha=0.5, label="Precio XRP")
# Reconstruir arrays de historia alineados
centers = np.array([prices[0]]*25 + vnn_logic.history_centers)
radii = np.array([0]*25 + vnn_logic.history_radii_p)
upper = centers + radii
lower = centers - radii

ax1.fill_between(range(len(prices)), upper, lower, color='orange', alpha=0.15, label="Membrana VNN")
ax1.plot(centers, color='orange', linestyle='--', alpha=0.8, label="Núcleo")

b_idx = np.where(vnn_signals == 1)[0]
s_idx = np.where(vnn_signals == -1)[0]
if len(b_idx)>0: ax1.scatter(b_idx, prices[b_idx], color='g', marker='^', s=100, label='Compra', zorder=5)
if len(s_idx)>0: ax1.scatter(s_idx, prices[s_idx], color='r', marker='v', s=100, label='Venta', zorder=5)
ax1.set_title("Do 3D: Precio + Volumen + Volatilidad")
ax1.legend()

# Panel 2: Equity
ax2.plot(wealth_vnn, color='purple', lw=2, label=f"VNN 3D")
ax2.plot(wealth_bh, color='gray', linestyle='--', label=f"Buy & Hold")
ax2.set_title("Curva de Supervivencia: Equity All-In")
ax2.grid(True, alpha=0.2)
ax2.legend()

plt.tight_layout()
plt.show()