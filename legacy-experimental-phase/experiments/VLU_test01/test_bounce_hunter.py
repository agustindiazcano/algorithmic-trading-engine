import numpy as np
import matplotlib.pyplot as plt
import requests
from bounce_hunter_vnn import BounceHunterVNN, BounceHunterGenome

print("🧬 Cargando Bounce Hunter evolucionado...")
genome = BounceHunterGenome.load("bounce_hunter_best.json")

print(f"\n📊 Parámetros del genoma:")
print(f"  Radio Base: {genome.radius_base:.3f}")
print(f"  Inercia Up: {genome.inertia_up:.3f} (lenta para subir)")
print(f"  Inercia Down: {genome.inertia_down:.3f} (rápida para bajar)")
print(f"  Vol Exhaustion: {genome.volume_exhaustion_threshold:.3f}")
print(f"  Take Profit: {genome.take_profit_tolerance:.3f} ({genome.take_profit_tolerance*100:.1f}%)")
print(f"  Stop Loss: {genome.stop_loss_pct:.3f} ({genome.stop_loss_pct*100:.1f}%)")
print(f"  Max Hold: {genome.max_hold_time}h")

# Descargar datos reales
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
print(f"   Cambio mercado: {((prices[-1] / prices[0]) - 1) * 100:.2f}%")

# Ejecutar VNN
print("\n🧠 Ejecutando Bounce Hunter...")
vnn = BounceHunterVNN(genome, prices, volumes)
signals = vnn.run()
fitness = vnn.calculate_fitness()

# Buy & Hold
buy_hold_equity = 10000 * (prices / prices[0])

# Estadísticas
print("\n" + "="*60)
print("📊 RESULTADOS DEL BACKTEST - BOUNCE HUNTER")
print("="*60)
print(f"💰 Equity Final:        ${vnn.equity_curve[-1]:,.2f}")
print(f"💎 Buy & Hold:          ${buy_hold_equity[-1]:,.2f}")
print(f"🔄 Total Trades:        {len([t for t in vnn.trades if t['type'] == 'sell'])}")
print(f"🎯 Fitness:             {fitness:.2f}")

profit_vnn = ((vnn.equity_curve[-1] / 10000) - 1) * 100
profit_bh = ((buy_hold_equity[-1] / 10000) - 1) * 100
print(f"\n🚀 Rendimiento VNN:     {profit_vnn:+.2f}%")
print(f"📈 Rendimiento B&H:     {profit_bh:+.2f}%")
print(f"💪 Ventaja:             {profit_vnn - profit_bh:+.2f}%")

# Analizar trades
sell_trades = [t for t in vnn.trades if t['type'] == 'sell']
if sell_trades:
    wins = sum(1 for t in sell_trades if t['profit_pct'] > 0)
    win_rate = (wins / len(sell_trades)) * 100
    avg_profit = np.mean([t['profit_pct'] for t in sell_trades])
    avg_hold = np.mean([t['hold_time'] for t in sell_trades])
    
    print(f"\n📈 Win Rate:            {win_rate:.1f}%")
    print(f"⏱️  Avg Hold Time:       {avg_hold:.1f}h")
    print(f"💵 Avg Profit/Trade:    {avg_profit:+.2f}%")

print("\n" + "="*60)

# Gráficos
fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

# 1. Precio + Señales
ax1 = axes[0]
ax1.plot(prices, color='black', alpha=0.6, linewidth=1, label='Precio XRP')

buy_signals = [(i, prices[i]) for i, s in enumerate(signals) if s == 1]
sell_signals = [(i, prices[i]) for i, s in enumerate(signals) if s == -1]

if buy_signals:
    buy_x, buy_y = zip(*buy_signals)
    ax1.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Compra (Rebote)', zorder=5)

if sell_signals:
    sell_x, sell_y = zip(*sell_signals)
    ax1.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Venta (TP/SL)', zorder=5)

ax1.set_title('Bounce Hunter VNN - Pesca de Rebotes', fontsize=14, fontweight='bold')
ax1.set_ylabel('Precio (USD)', fontsize=11)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 2. Equity Curve
ax2 = axes[1]
ax2.plot(vnn.equity_curve, color='#00ff00', linewidth=2, label=f'Bounce Hunter (${vnn.equity_curve[-1]:,.0f})')
ax2.plot(buy_hold_equity, color='#888888', linewidth=1.5, linestyle='--', label=f'Buy & Hold (${buy_hold_equity[-1]:,.0f})')
ax2.axhline(10000, color='blue', linestyle=':', alpha=0.5, label='Capital Inicial')
ax2.set_title('Curva de Equity', fontsize=12, fontweight='bold')
ax2.set_ylabel('USD', fontsize=11)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# 3. Profit por Trade
ax3 = axes[2]
if sell_trades:
    trade_profits = [t['profit_pct'] for t in sell_trades]
    colors = ['green' if p > 0 else 'red' for p in trade_profits]
    ax3.bar(range(len(trade_profits)), trade_profits, color=colors, alpha=0.6)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title(f'Profit por Trade (Win Rate: {win_rate:.1f}%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Profit %', fontsize=11)
    ax3.set_xlabel('Trade #', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bounce_hunter_backtest.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico guardado: bounce_hunter_backtest.png")

plt.show()

# Top 5 mejores trades
if sell_trades:
    print("\n🏆 TOP 5 MEJORES TRADES:")
    print("-" * 60)
    sorted_trades = sorted(sell_trades, key=lambda t: t['profit_pct'], reverse=True)[:5]
    for i, t in enumerate(sorted_trades, 1):
        print(f"{i}. Profit: {t['profit_pct']:+.2f}% | Hold: {t['hold_time']}h | Precio: ${t['price']:.4f}")
