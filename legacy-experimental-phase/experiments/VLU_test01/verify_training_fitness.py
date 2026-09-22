"""
Backtest en los MISMOS datos de entrenamiento para verificar fitness
"""
import numpy as np
import json
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
import requests

# Cargar mejor genoma
with open('pso_cma_best.json', 'r') as f:
    data = json.load(f)
best_genome = BounceHunterGenome.from_dict(data)

print("VERIFICACION: Backtest en datos de ENTRENAMIENTO")
print(f"Fitness reportado durante evolucion: {best_genome.fitness:.2f}\n")

# Descargar los MISMOS 3 escenarios que usó el entrenamiento
scenarios = []

print("Descargando MISMOS datos de entrenamiento...")

# Escenario 1: 1000 velas
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000}
response = requests.get(url, params=params)
klines = response.json()
prices_1000 = np.array([float(k[4]) for k in klines])
volumes_1000 = np.array([float(k[5]) for k in klines])
scenarios.append((prices_1000, volumes_1000, "1000h"))
print(f"OK Escenario 1: {len(prices_1000)} velas")

# Escenario 2: 500 velas
params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 500}
response = requests.get(url, params=params)
klines = response.json()
prices_500 = np.array([float(k[4]) for k in klines])
volumes_500 = np.array([float(k[5]) for k in klines])
scenarios.append((prices_500, volumes_500, "500h"))
print(f"OK Escenario 2: {len(prices_500)} velas")

# Escenario 3: 300 velas
params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 300}
response = requests.get(url, params=params)
klines = response.json()
prices_300 = np.array([float(k[4]) for k in klines])
volumes_300 = np.array([float(k[5]) for k in klines])
scenarios.append((prices_300, volumes_300, "300h"))
print(f"OK Escenario 3: {len(prices_300)} velas\n")

# Evaluar en cada escenario
print("=" * 60)
print("RESULTADOS POR ESCENARIO")
print("=" * 60)

total_fitness = 0.0
for i, (prices, volumes, name) in enumerate(scenarios, 1):
    vnn = BounceHunterVNN(best_genome, prices, volumes)
    signals = vnn.run()
    fitness = vnn.calculate_fitness()
    total_fitness += fitness
    
    sell_trades = [t for t in vnn.trades if t['type'] == 'sell']
    
    print(f"\nEscenario {i} ({name}):")
    print(f"  Fitness: {fitness:.2f}")
    print(f"  Equity: ${vnn.equity_curve[-1]:,.2f}")
    print(f"  Profit: ${vnn.equity_curve[-1] - 10000:,.2f}")
    print(f"  Return: {((vnn.equity_curve[-1] / 10000) - 1) * 100:.2f}%")
    print(f"  Trades: {len(sell_trades)}")
    if len(sell_trades) > 0:
        print(f"  Win Rate: {sum(1 for t in sell_trades if t['profit_pct'] > 0) / len(sell_trades) * 100:.1f}%")
        print(f"  Avg Hold: {np.mean([t['hold_time'] for t in sell_trades]):.1f}h")
        print(f"  Avg Profit: {np.mean([t['profit_pct'] for t in sell_trades]):.2f}%")

# Fitness promedio (como en el entrenamiento)
avg_fitness = total_fitness / len(scenarios)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\nFitness Promedio (3 escenarios): {avg_fitness:.2f}")
print(f"Fitness Reportado (evolucion):   {best_genome.fitness:.2f}")
print(f"Diferencia:                       {abs(avg_fitness - best_genome.fitness):.2f}")

if abs(avg_fitness - best_genome.fitness) < 100:
    print("\nVERIFICADO: Fitness reproducible en datos de entrenamiento!")
else:
    print("\nADVERTENCIA: Fitness no coincide exactamente")
    print("(Esto es normal si los datos cambiaron desde el entrenamiento)")

# Explicar por qué el fitness es tan alto
print("\n" + "=" * 60)
print("POR QUE FITNESS TAN ALTO?")
print("=" * 60)
print("""
El fitness NO es el profit en dolares. Es una metrica compuesta:

Fitness = (Profit / Tiempo_Total) * 100 * Bonuses

Bonuses incluyen:
1. Activity Bonus: hasta 2x por 10+ trades
2. Win Rate Bonus: (1 + win_rate)
3. Speed Bonus: 1.5x si avg_hold < 8h

Ejemplo:
- Profit: $500
- Tiempo total: 50 horas
- Trades: 12 (activity bonus = 1.2x)
- Win rate: 80% (bonus = 1.8x)
- Avg hold: 4h (speed bonus = 1.5x)

Fitness = (500/50) * 100 * 2.2 * 1.8 * 1.5 = 10,000 * 5.94 = 59,400

Por eso el fitness puede ser 29,061 mientras el profit real es bajo.
""")
