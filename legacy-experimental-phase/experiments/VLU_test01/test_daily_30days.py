"""
Backtest detallado en velas diarias (últimos 30 días)
"""
import numpy as np
import json
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Cargar mejor genoma
with open('pso_cma_best.json', 'r') as f:
    data = json.load(f)
best_genome = BounceHunterGenome.from_dict(data)

print("=" * 60)
print("BACKTEST DIARIO - ÚLTIMOS 30 DÍAS")
print("=" * 60)
print(f"\nGenoma: Fitness {best_genome.fitness:.2f}%")
print(f"Radius: {best_genome.radius_base*100:.1f}%")
print(f"TP: {best_genome.take_profit_tolerance*100:.2f}%")
print(f"Stop: {best_genome.stop_loss_pct*100:.1f}%")
print(f"Trailing Trigger: {best_genome.trailing_stop_trigger*100:.1f}%")
print(f"Trailing Floor: {best_genome.trailing_stop_floor*100:.1f}%")
print(f"RSI Threshold: {best_genome.rsi_overbought_threshold:.1f}")

# Descargar últimos 30 días (velas diarias)
print(f"\nDescargando velas diarias de XRP/USDT...")
url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "XRPUSDT", "interval": "1d", "limit": 30}
response = requests.get(url, params=params)
klines = response.json()

prices = np.array([float(k[4]) for k in klines])  # Close
volumes = np.array([float(k[5]) for k in klines])
timestamps = [datetime.fromtimestamp(k[0]/1000) for k in klines]

print(f"OK: {len(prices)} velas descargadas")
print(f"Periodo: {timestamps[0].strftime('%Y-%m-%d')} a {timestamps[-1].strftime('%Y-%m-%d')}")

# Ejecutar VNN
vnn = BounceHunterVNN(best_genome, prices, volumes)
signals = vnn.run()
fitness = vnn.calculate_fitness()

# Resultados
final_equity = vnn.equity_curve[-1]
return_pct = ((final_equity / 10000) - 1) * 100
profit = final_equity - 10000

print(f"\n{'='*60}")
print("RESULTADOS")
print(f"{'='*60}")
print(f"\nEquity Inicial: $10,000.00")
print(f"Equity Final:   ${final_equity:,.2f}")
print(f"Profit:         ${profit:,.2f}")
print(f"Return:         {return_pct:.2f}%")
print(f"Fitness:        {fitness:.2f}%")

# Analizar trades
buy_trades = [t for t in vnn.trades if t['type'] == 'buy']
sell_trades = [t for t in vnn.trades if t['type'] == 'sell']

print(f"\n{'='*60}")
print("TRADING ACTIVITY")
print(f"{'='*60}")
print(f"\nTotal Trades Completos: {len(sell_trades)}")

if len(sell_trades) > 0:
    wins = sum(1 for t in sell_trades if t['profit_pct'] > 0)
    losses = len(sell_trades) - wins
    win_rate = wins / len(sell_trades) * 100
    
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Hold Time: {np.mean([t['hold_time'] for t in sell_trades]):.1f} días")
    print(f"Avg Profit: {np.mean([t['profit_pct'] for t in sell_trades]):.2f}%")
    print(f"Best Trade: {max([t['profit_pct'] for t in sell_trades]):.2f}%")
    print(f"Worst Trade: {min([t['profit_pct'] for t in sell_trades]):.2f}%")
    
    # Detalles de cada trade
    print(f"\n{'='*60}")
    print("DETALLE DE TRADES")
    print(f"{'='*60}")
    print(f"\n{'#':<4} {'Entry Date':<12} {'Exit Date':<12} {'Hold':<6} {'Profit':<10} {'Type':<10}")
    print("-" * 60)
    
    for i, trade in enumerate(sell_trades, 1):
        entry_tick = trade['tick'] - trade['hold_time']
        entry_date = timestamps[entry_tick].strftime('%Y-%m-%d')
        exit_date = timestamps[trade['tick']].strftime('%Y-%m-%d')
        profit_str = f"{trade['profit_pct']:+.2f}%"
        result = "WIN" if trade['profit_pct'] > 0 else "LOSS"
        
        print(f"{i:<4} {entry_date:<12} {exit_date:<12} {trade['hold_time']:<6} {profit_str:<10} {result:<10}")

# Visualización detallada
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 1. Precio + Trades
ax1 = axes[0]
ax1.plot(timestamps, prices, label='XRP/USDT', color='black', linewidth=1.5)

# Marcar entries y exits
for trade in buy_trades:
    ax1.scatter(timestamps[trade['tick']], prices[trade['tick']], 
               color='green', marker='^', s=150, zorder=5)
    ax1.annotate('BUY', (timestamps[trade['tick']], prices[trade['tick']]),
                xytext=(0, -15), textcoords='offset points', ha='center',
                fontsize=8, color='green', weight='bold')

for trade in sell_trades:
    ax1.scatter(timestamps[trade['tick']], prices[trade['tick']], 
               color='red', marker='v', s=150, zorder=5)
    profit_text = f"{trade['profit_pct']:+.1f}%"
    ax1.annotate(profit_text, (timestamps[trade['tick']], prices[trade['tick']]),
                xytext=(0, 15), textcoords='offset points', ha='center',
                fontsize=8, color='red', weight='bold')

ax1.set_ylabel('Precio (USDT)')
ax1.set_title(f'XRP/USDT - Últimos 30 Días (Return: {return_pct:.2f}%)')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Equity Curve
ax2 = axes[1]
equity_timestamps = timestamps[:len(vnn.equity_curve)]
ax2.plot(equity_timestamps, vnn.equity_curve, color='blue', linewidth=2)
ax2.axhline(10000, color='gray', linestyle='--', alpha=0.5, label='Initial')
ax2.fill_between(equity_timestamps, 10000, vnn.equity_curve, 
                 where=np.array(vnn.equity_curve) >= 10000, 
                 color='green', alpha=0.2, label='Profit')
ax2.fill_between(equity_timestamps, 10000, vnn.equity_curve, 
                 where=np.array(vnn.equity_curve) < 10000, 
                 color='red', alpha=0.2, label='Loss')
ax2.set_ylabel('Equity ($)')
ax2.set_title(f'Equity Curve (Final: ${final_equity:,.2f})')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Daily Returns
ax3 = axes[2]
daily_returns = np.diff(vnn.equity_curve) / vnn.equity_curve[:-1] * 100
returns_timestamps = equity_timestamps[1:]
colors = ['green' if r > 0 else 'red' for r in daily_returns]
ax3.bar(returns_timestamps, daily_returns, color=colors, alpha=0.7)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylabel('Daily Return (%)')
ax3.set_xlabel('Fecha')
ax3.set_title('Retornos Diarios')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('daily_backtest_30days.png', dpi=150, bbox_inches='tight')
print(f"\nGráfico guardado: daily_backtest_30days.png")

# Guardar resultados
results = {
    'period': {
        'start': timestamps[0].strftime('%Y-%m-%d'),
        'end': timestamps[-1].strftime('%Y-%m-%d'),
        'days': len(prices)
    },
    'performance': {
        'initial_equity': 10000,
        'final_equity': float(final_equity),
        'profit': float(profit),
        'return_pct': float(return_pct),
        'fitness': float(fitness)
    },
    'trades': {
        'total': len(sell_trades),
        'wins': int(wins) if len(sell_trades) > 0 else 0,
        'losses': int(losses) if len(sell_trades) > 0 else 0,
        'win_rate': float(win_rate) if len(sell_trades) > 0 else 0,
        'avg_hold_days': float(np.mean([t['hold_time'] for t in sell_trades])) if len(sell_trades) > 0 else 0,
        'avg_profit_pct': float(np.mean([t['profit_pct'] for t in sell_trades])) if len(sell_trades) > 0 else 0
    },
    'trade_details': [
        {
            'entry_date': timestamps[t['tick'] - t['hold_time']].strftime('%Y-%m-%d'),
            'exit_date': timestamps[t['tick']].strftime('%Y-%m-%d'),
            'hold_days': t['hold_time'],
            'profit_pct': float(t['profit_pct'])
        }
        for t in sell_trades
    ]
}

with open('daily_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Resultados guardados: daily_backtest_results.json")
print(f"\n{'='*60}")
print("BACKTEST COMPLETADO")
print(f"{'='*60}")
