import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generación de Mercado "Hijo de Puta" (Ultra-Volátil y Errático) ---
np.random.seed(420)
t = np.arange(1500)
# Tendencia base lenta
base = 100 + np.cumsum(np.random.normal(0, 0.5, 1500))
# Ruido de alta frecuencia masivo (el "temblor" que engaña al silicio)
noise = np.random.normal(0, 1.5, 1500)
# Saltos de volatilidad (Bull Traps y Flash Crashes)
traps = np.zeros(1500)
traps[300:350] += np.linspace(0, 15, 50) # Bull trap
traps[350:370] -= np.linspace(0, 20, 20) # Colapso tras la trampa
traps[800:820] -= 25 # Flash crash instantáneo
prices = base + noise + traps

# --- 2. Motor de Simulación Realista ---
def run_simulation(prices, signals, initial_cash=10000.0):
    cash = initial_cash
    position = 0
    comision_pct = 0.002 # 0.2% por operación
    slippage_pct = 0.001 # 0.1% de pérdida por ejecución tardía
    history = []
    trades = 0
    
    for i in range(len(prices)):
        p = prices[i]
        sig = signals[i]
        
        # Lógica de Ejecución
        if sig == 1 and cash > p: # COMPRA
            exec_price = p * (1 + slippage_pct)
            cost = exec_price * (1 + comision_pct)
            shares = cash // cost
            if shares > 0:
                position += shares
                cash -= shares * cost
                trades += 1
        elif sig == -1 and position > 0: # VENTA
            exec_price = p * (1 - slippage_pct)
            cash += position * exec_price * (1 - comision_pct)
            position = 0
            trades += 1
            
        history.append(cash + position * p)
    return np.array(history), trades

# --- 3. El Silicio (MLP/Linear Baseline) ---
# Simula un seguidor de tendencia clásico (Media Móvil) que se vuelve loco con el ruido
mlp_signals = np.zeros(1500)
fast_ma = np.array([np.mean(prices[max(0, i-5):i+1]) for i in range(1500)])
slow_ma = np.array([np.mean(prices[max(0, i-20):i+1]) for i in range(1500)])
for i in range(1, 1500):
    if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
        mlp_signals[i] = 1 # Buy
    elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
        mlp_signals[i] = -1 # Sell

# --- 4. El Carbono (VNN Molecular Respiratoria) ---
class CarbonVNN:
    def __init__(self):
        self.center = prices[0]
        self.r = 3.0
        self.signals = np.zeros(1500)
    
    def run(self, data):
        for i in range(1, 1500):
            p = data[i]
            # Cálculo de Volatilidad Local (Inhalación/Exhalación)
            local_vol = np.std(data[max(0, i-15):i+1])
            # La neurona RESPIRA: el radio crece con la incertidumbre
            # Factor de "Do": 3 sigmas de seguridad
            self.r = 0.9 * self.r + 0.1 * (local_vol * 3.0) 
            
            # Lógica de Membrana: Solo opera si la fuerza rompe la geometría
            if p > (self.center + self.r):
                self.signals[i] = 1
            elif p < (self.center - self.r):
                self.signals[i] = -1
            
            # El centro se desplaza con inercia molecular
            self.center = 0.92 * self.center + 0.08 * p
        return self.signals

vnn_logic = CarbonVNN()
vnn_signals = vnn_logic.run(prices)

# --- 5. Ejecución y Gráficos ---
mlp_history, mlp_trades = run_simulation(prices, mlp_signals)
vnn_history, vnn_trades = run_simulation(prices, vnn_signals)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

# Gráfico de Mercado
ax1.plot(prices, color='black', alpha=0.4, label="Mercado (Ruido + Trampas)")
ax1.set_title("Escenario de Mercado 'Hijo de Puta' (Volatilidad Extrema)")
ax1.legend()

# Gráfico de Patrimonio (Do)
ax2.plot(mlp_history, color='red', label=f"Silicio (MLP/MA): ${mlp_history[-1]:.0f} ({mlp_trades} trades)")
ax2.plot(vnn_history, color='green', linewidth=3, label=f"Carbono (VNN): ${vnn_history[-1]:.0f} ({vnn_trades} trades)")
ax2.fill_between(range(1500), 10000, vnn_history, color='green', alpha=0.1)

ax2.set_title("Patrimonio Neto: La Do de la Volatilidad")
ax2.set_ylabel("USD")
ax2.legend()
plt.tight_layout()
plt.savefig("do_volatil_final.png")

print(f"RESULTADO FINAL:")
print(f"MLP (Silicio): Final ${mlp_history[-1]:.2f} | Trades: {mlp_trades}")
print(f"VNN (Carbono): Final ${vnn_history[-1]:.2f} | Trades: {vnn_trades}")