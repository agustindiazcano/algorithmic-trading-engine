"""
Backtest visual del mejor genoma con exit rules evolucionables
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
import requests

# Cargar mejor genoma
with open('pso_cma_best.json', 'r') as f:
    data = json.load(f)
best_genome = BounceHunterGenome.from_dict(data)

print("🧬 Mejor Genoma (11 parámetros evolucionables)")
print(f"  Fitness: {best_genome.fitness:.2f}")
print(f"\n📊 Parámetros Core:")
print(f"  Radius: {best_genome.radius_base:.3f}")
print(f"  Volatility Mult: {best_genome.volatility_multiplier:.2f}")
print(f"  Inertia Up: {best_genome.inertia_up:.3f}")
print(f"  Inertia Down: {best_genome.inertia_down:.3f}")
print(f"\n🚪 Exit Rules EVOLUCIONADAS:")
print(f"  Take Profit Tolerance: {best_genome.take_profit_tolerance*100:.2f}%")
print(f"  Stop Loss: {best_genome.stop_loss_pct*100:.1f}%")
print(f"  Trailing Stop Trigger: {best_genome.trailing_stop_trigger*100:.1f}%")
print(f"  Trailing Stop Floor: {best_genome.trailing_stop_floor*100:.1f}%")
print(f"  RSI Overbought: {best_genome.rsi_overbought_threshold:.1f}")
print(f"  Volume Spike Mult: {best_genome.volume_spike_multiplier:.2f}x")

# Descargar datos reales
print("\n📡 Descargando datos de XRP/USDT...")
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000}
response = requests.get(url, params=params)
klines = response.json()

prices = np.array([float(k[4]) for k in klines])  # Close price
volumes = np.array([float(k[5]) for k in klines])
print(f"✅ {len(prices)} velas descargadas")

# Ejecutar VNN
vnn = BounceHunterVNN(best_genome, prices, volumes)
signals = vnn.run()
fitness = vnn.calculate_fitness()

print(f"\n💰 Resultados del Backtest:")
print(f"  Equity Inicial: $10,000")
print(f"  Equity Final: ${vnn.equity_curve[-1]:,.2f}")
print(f"  Profit: ${vnn.equity_curve[-1] - 10000:,.2f}")
print(f"  Return: {((vnn.equity_curve[-1] / 10000) - 1) * 100:.2f}%")
print(f"  Fitness: {fitness:.2f}")

# Contar trades
buy_trades = [t for t in vnn.trades if t['type'] == 'buy']
sell_trades = [t for t in vnn.trades if t['type'] == 'sell']
print(f"\n📈 Trading Activity:")
print(f"  Total Trades: {len(sell_trades)} completos")
print(f"  Win Rate: {sum(1 for t in sell_trades if t['profit_pct'] > 0) / len(sell_trades) * 100:.1f}%")
print(f"  Avg Hold Time: {np.mean([t['hold_time'] for t in sell_trades]):.1f} horas")
print(f"  Avg Profit: {np.mean([t['profit_pct'] for t in sell_trades]):.2f}%")

# Analizar exit reasons
print(f"\n🚪 Exit Rule Analysis:")
exit_reasons = {
    'geometric_tp': 0,
    'stop_loss': 0,
    'trailing_stop': 0,
    'rsi_reversal': 0,
    'volume_spike': 0
}

for t in sell_trades:
    tick = t['tick']
    entry_tick = t['tick'] - t['hold_time']
    current_p = prices[tick]
    entry_p = prices[entry_tick]
    
    # Determinar razón de salida (simplificado)
    profit_pct = (current_p / entry_p) - 1
    
    if current_p >= vnn.center * (1 - best_genome.take_profit_tolerance):
        exit_reasons['geometric_tp'] += 1
    elif current_p < entry_p * (1 - best_genome.stop_loss_pct):
        exit_reasons['stop_loss'] += 1
    elif profit_pct > best_genome.trailing_stop_trigger:
        exit_reasons['trailing_stop'] += 1
    else:
        exit_reasons['rsi_reversal'] += 1  # Simplificado

for reason, count in exit_reasons.items():
    if count > 0:
        print(f"  {reason}: {count} ({count/len(sell_trades)*100:.1f}%)")

# Visualización
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# 1. Precio + Señales
ax1 = axes[0]
ax1.plot(prices, label='XRP/USDT', color='black', linewidth=1)
ax1.scatter([t['tick'] for t in buy_trades], 
           [prices[t['tick']] for t in buy_trades],
           color='green', marker='^', s=100, label='BUY', zorder=5)
ax1.scatter([t['tick'] for t in sell_trades], 
           [prices[t['tick']] for t in sell_trades],
           color='red', marker='v', s=100, label='SELL', zorder=5)
ax1.set_ylabel('Precio (USDT)')
ax1.set_title('Bounce Hunter VNN - Exit Rules Evolucionables')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Equity Curve
ax2 = axes[1]
ax2.plot(vnn.equity_curve, color='blue', linewidth=2)
ax2.axhline(10000, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('Equity ($)')
ax2.set_title(f'Equity Curve (Return: {((vnn.equity_curve[-1] / 10000) - 1) * 100:.2f}%)')
ax2.grid(alpha=0.3)

# 3. RSI
ax3 = axes[2]
rsi_values = []
for i in range(len(prices)):
    if i >= 20:
        rsi = vnn.calculate_rsi(prices[max(0, i-20):i+1])
        rsi_values.append(rsi)
    else:
        rsi_values.append(50)

ax3.plot(rsi_values, color='purple', linewidth=1)
ax3.axhline(best_genome.rsi_overbought_threshold, color='red', linestyle='--', 
           label=f'RSI Threshold: {best_genome.rsi_overbought_threshold:.1f}')
ax3.axhline(30, color='green', linestyle='--', alpha=0.3)
ax3.set_ylabel('RSI')
ax3.set_title('RSI (Evolvable Threshold)')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Volume
ax4 = axes[3]
ax4.bar(range(len(volumes)), volumes, color='gray', alpha=0.5)
# Marcar volume spikes
for i in range(10, len(volumes)):
    avg_vol = np.mean(volumes[i-10:i])
    if volumes[i] > avg_vol * best_genome.volume_spike_multiplier:
        ax4.bar(i, volumes[i], color='orange')

ax4.set_ylabel('Volume')
ax4.set_xlabel('Tick')
ax4.set_title(f'Volume (Spike Threshold: {best_genome.volume_spike_multiplier:.2f}x)')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('evolvable_exits_backtest.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Gráfico guardado: evolvable_exits_backtest.png")

# Guardar estadísticas
stats = {
    'genome': data,
    'backtest': {
        'initial_equity': 10000,
        'final_equity': float(vnn.equity_curve[-1]),
        'profit': float(vnn.equity_curve[-1] - 10000),
        'return_pct': float(((vnn.equity_curve[-1] / 10000) - 1) * 100),
        'fitness': float(fitness),
        'total_trades': len(sell_trades),
        'win_rate': float(sum(1 for t in sell_trades if t['profit_pct'] > 0) / len(sell_trades) * 100),
        'avg_hold_time': float(np.mean([t['hold_time'] for t in sell_trades])),
        'avg_profit_pct': float(np.mean([t['profit_pct'] for t in sell_trades]))
    },
    'exit_reasons': exit_reasons
}

with open('evolvable_exits_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"📊 Estadísticas guardadas: evolvable_exits_stats.json")
print("\n✅ Backtest completado!")
