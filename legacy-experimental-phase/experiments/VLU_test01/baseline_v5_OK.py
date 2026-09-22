import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generación del Mercado Nefasto ---
t = np.arange(2000)
# Precio: Un "Random Walk" con trampas de liquidez y ruido browniano pesado
np.random.seed(999)
price = 100 + np.cumsum(np.random.normal(0, 1.5, 2000))
# Inyectamos el "Latigazo": Caídas y subidas que rompen cualquier media móvil
price[800:850] += np.sin(np.linspace(0, 10, 50)) * 20 

# --- 2. Indicadores Clásicos (El Cebo para el Silicio) ---
def get_indicators(p):
    # RSI (Simple)
    delta = np.diff(p, prepend=p[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='same')
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    # Bollinger
    ma = np.convolve(p, np.ones(20)/20, mode='same')
    std = np.array([np.std(p[max(0, i-20):i+1]) for i in range(len(p))])
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return rsi, upper, lower

rsi, b_upper, b_lower = get_indicators(price)

# --- 3. Do de Silicio (MLP/Indicator Master) ---
# Este modelo opera si los indicadores "dicen que toca". Es carne de cañón.
silicon_signals = np.zeros(2000)
for i in range(1, 2000):
    if rsi[i] < 30 and price[i] < b_lower[i]: silicon_signals[i] = 1 # Compra "Libro"
    if rsi[i] > 70 and price[i] > b_upper[i]: silicon_signals[i] = -1 # Vende "Libro"

# --- 4. La VNN Molecular (El Depredador de Carbono Estoico) ---
class DarkMatterVNN:
    def __init__(self):
        self.center_3d = np.array([100.0, 50.0, 0.0]) # Precio, RSI, Volatilidad
        self.radius_3d = np.array([5.0, 20.0, 10.0]) # Elipsoide de control
        self.signals = np.zeros(2000)
        self.persistence = 0 # Filtro de Momento
        
    def run(self, p_vec, r_vec):
        for i in range(20, 2000):
            vol = np.std(p_vec[i-20:i])
            state = np.array([p_vec[i], r_vec[i], vol])
            
            # RESPIRACIÓN ESTOICA 3D (ORIGINAL):
            # El radio de precio (dim 0) tiene suelo de seguridad (4.0) + volatilidad
            target_r_price = 4.0 + (vol * 3.0)
            # Inercia en el radio también (respiración suave)
            self.radius_3d[0] = 0.95 * self.radius_3d[0] + 0.05 * target_r_price
            
            # Distancia de Mahalanobis simplificada
            dist = np.sqrt(np.sum(((state - self.center_3d) / self.radius_3d)**2))
            
            # --- FILTRO DE PERSISTENCIA (5 ticks) ---
            # Solo acumulamos energía si rompe la estructura (dist > 1.8)
            norm_diff_price = (state[0] - self.center_3d[0]) / self.radius_3d[0]
            
            if dist > 1.8:
                if norm_diff_price < -1.0: # Precio muy abajo -> Oportunidad Compra
                    self.persistence += 1
                elif norm_diff_price > 1.0: # Precio muy arriba -> Oportunidad Venta
                    self.persistence -= 1
            else:
                # Enfriamiento rápido si vuelve a la normalidad
                self.persistence = int(self.persistence * 0.5)

            # DISPARO
            if self.persistence >= 5:
                self.signals[i] = 1
                self.persistence = 2 # Reset parcial
            elif self.persistence <= -5:
                self.signals[i] = -1
                self.persistence = -2
            
            # Inercia Molecular PESADA (0.99)
            self.center_3d = 0.99 * self.center_3d + 0.01 * state
        return self.signals

vnn_signals = DarkMatterVNN().run(price, rsi)

# --- 5. Resultados de la Carnicería ---
# (Usando el motor de simulación realista con 0.3% de comisión/slippage)
def final_war(prices, signals):
    cash = 10000.0
    pos = 0
    fee = 0.003
    hist = []
    for p, s in zip(prices, signals):
        if s == 1 and cash > p:
            shares = cash // (p * (1+fee))
            pos += shares
            cash -= shares * p * (1+fee)
        elif s == -1 and pos > 0:
            cash += pos * p * (1-fee)
            pos = 0
        hist.append(cash + pos * p)
    return np.array(hist)

silicon_wealth = final_war(price, silicon_signals)
vnn_wealth = final_war(price, vnn_signals)

# --- 6. Visualización de la Victoria/Derrota ---
print(f"💰 Capital Inicial: $10,000")
print(f"📉 Silicon (RSI/Bollinger): ${silicon_wealth[-1]:,.2f}")
print(f"🧠 DarkMatter VNN:          ${vnn_wealth[-1]:,.2f}")

plt.figure(figsize=(12, 6))
plt.plot(silicon_wealth, label='Silicon (Indicadores)', color='gray', alpha=0.6)
plt.plot(vnn_wealth, label='DarkMatter VNN (Molecular)', color='purple', linewidth=2)
plt.title("Guerra de Trading: Carbono vs Silicio")
plt.xlabel("Tiempo (ticks)")
plt.ylabel("Capital ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()