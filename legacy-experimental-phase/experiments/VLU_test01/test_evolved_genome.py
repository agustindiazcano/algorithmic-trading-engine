import numpy as np
import matplotlib.pyplot as plt
import requests
from evolutionary_vnn import VNNGenome, DualNeuronVNN

print("🧬 Cargando genoma evolucionado...")
genome = VNNGenome.load("best_genome.json")

print(f"\n📊 Parámetros del genoma:")
print(f"  Anchor Inertia: {genome.anchor_inertia:.3f}")
print(f"  Hunter Inertia: {genome.hunter_inertia:.3f}")
print(f"  Trauma Sensitivity: {genome.trauma_sensitivity:.3f}")
print(f"  Trauma Multiplier: {genome.trauma_radius_multiplier:.2f}x")

# Descargar datos reales de XRP
print("\n📡 Descargando datos de mercado (XRPUSDT 1h)...")
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000}
response = requests.get(url, params=params)
data = response.json()

prices = np.array([float(k[4]) for k in data])
volumes = np.array([float(k[5]) for k in data]) + 1.0

print(f"✅ Datos cargados: {len(prices)} velas")
print(f"   Precio inicial: ${prices[0]:.4f}")
print(f"   Precio final: ${prices[-1]:.4f}")
print(f"   Cambio: {((prices[-1] / prices[0]) - 1) * 100:.2f}%")

# Ejecutar VNN
print("\n🧠 Ejecutando VNN evolucionada...")
vnn = DualNeuronVNN(genome, prices, volumes)
signals = vnn.run()

# Calcular Buy & Hold
buy_hold_equity = 10000 * (prices / prices[0])

# Estadísticas
print("\n" + "="*60)
print("📊 RESULTADOS DEL BACKTEST")
print("="*60)
print(f"💰 Equity Final VNN:    ${vnn.equity_curve[-1]:,.2f}")
print(f"💎 Buy & Hold:          ${buy_hold_equity[-1]:,.2f}")
print(f"🔄 Total Trades:        {len(vnn.trades)}")
print(f"🚫 Rogue Trades:        {vnn.rogue_trades}")
print(f"⚡ Trauma Activations:  {'Sí' if genome.used_trauma_response else 'No'}")

profit_vnn = ((vnn.equity_curve[-1] / 10000) - 1) * 100
profit_bh = ((buy_hold_equity[-1] / 10000) - 1) * 100
print(f"\n🚀 Rendimiento VNN:     {profit_vnn:+.2f}%")
print(f"📈 Rendimiento B&H:     {profit_bh:+.2f}%")
print(f"💪 Ventaja:             {profit_vnn - profit_bh:+.2f}%")

# Calcular drawdown
equity_array = np.array(vnn.equity_curve)
running_max = np.maximum.accumulate(equity_array)
drawdown = (running_max - equity_array) / running_max * 100
max_dd = np.max(drawdown)
print(f"📉 Max Drawdown:        {max_dd:.2f}%")

# Win rate
buy_trades = [t for t in vnn.trades if t['type'] == 'buy']
sell_trades = [t for t in vnn.trades if t['type'] == 'sell']
wins = sum(1 for i, sell in enumerate(sell_trades) if sell['price'] > buy_trades[i]['price'])
win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
print(f"🎯 Win Rate:            {win_rate:.1f}%")

print("\n" + "="*60)

# Gráficos
fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

# 1. Precio + Señales
ax1 = axes[0]
ax1.plot(prices, color='black', alpha=0.6, linewidth=1, label='Precio XRP')

# Marcar compras y ventas
buy_signals = [(i, prices[i]) for i, s in enumerate(signals) if s == 1]
sell_signals = [(i, prices[i]) for i, s in enumerate(signals) if s == -1]

if buy_signals:
    buy_x, buy_y = zip(*buy_signals)
    ax1.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Compra', zorder=5)

if sell_signals:
    sell_x, sell_y = zip(*sell_signals)
    ax1.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Venta', zorder=5)

ax1.set_title('VNN Evolucionada - Señales de Trading', fontsize=14, fontweight='bold')
ax1.set_ylabel('Precio (USD)', fontsize=11)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 2. Equity Curve
ax2 = axes[1]
ax2.plot(vnn.equity_curve, color='#00ff00', linewidth=2, label=f'VNN (${vnn.equity_curve[-1]:,.0f})')
ax2.plot(buy_hold_equity, color='#888888', linewidth=1.5, linestyle='--', label=f'Buy & Hold (${buy_hold_equity[-1]:,.0f})')
ax2.fill_between(range(len(vnn.equity_curve)), 10000, vnn.equity_curve, 
                  where=np.array(vnn.equity_curve) > 10000, color='green', alpha=0.1)
ax2.fill_between(range(len(vnn.equity_curve)), 10000, vnn.equity_curve, 
                  where=np.array(vnn.equity_curve) <= 10000, color='red', alpha=0.1)
ax2.axhline(10000, color='blue', linestyle=':', alpha=0.5, label='Capital Inicial')
ax2.set_title('Curva de Equity', fontsize=12, fontweight='bold')
ax2.set_ylabel('USD', fontsize=11)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# 3. Drawdown
ax3 = axes[2]
ax3.fill_between(range(len(drawdown)), 0, -drawdown, color='red', alpha=0.3)
ax3.plot(-drawdown, color='darkred', linewidth=1.5)
ax3.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Drawdown %', fontsize=11)
ax3.set_xlabel('Tiempo (horas)', fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evolved_vnn_backtest.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico guardado: evolved_vnn_backtest.png")

plt.show()

# Detalle de trades
if len(vnn.trades) > 0:
    print("\n📋 DETALLE DE TRADES:")
    print("-" * 60)
    for i in range(0, len(vnn.trades), 2):
        if i + 1 < len(vnn.trades):
            buy = vnn.trades[i]
            sell = vnn.trades[i + 1]
            profit = ((sell['price'] / buy['price']) - 1) * 100
            status = "✅ WIN" if profit > 0 else "❌ LOSS"
            print(f"Trade {i//2 + 1}: Buy ${buy['price']:.4f} → Sell ${sell['price']:.4f} | {profit:+.2f}% {status}")
