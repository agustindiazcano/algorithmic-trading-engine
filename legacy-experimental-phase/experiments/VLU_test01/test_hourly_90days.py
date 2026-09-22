"""
Backtest del genoma de 1h durante 90 días (2160 horas)
"""
import numpy as np
import json
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# Cargar mejor genoma de 1h
with open('pso_cma_best.json', 'r') as f:
    data = json.load(f)
best_genome = BounceHunterGenome.from_dict(data)

print("=" * 60)
print("BACKTEST 1H - ÚLTIMOS 90 DÍAS (2160 horas)")
print("=" * 60)
print(f"\nGenoma: Fitness {best_genome.fitness:.2f}%")
print(f"Radius: {best_genome.radius_base*100:.2f}%")
print(f"TP: {best_genome.take_profit_tolerance*100:.2f}%")
print(f"Stop: {best_genome.stop_loss_pct*100:.1f}%")

# Descargar últimos 90 días en velas de 1h
print(f"\nDescargando velas de 1h...")
url = "https://api.binance.com/api/v3/klines"

# Binance limita a 1000 velas, así que hacemos 3 requests
all_klines = []
for i in range(3):
    # Calcular timestamp de inicio para cada batch
    if i == 0:
        params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000}
    else:
        # Usar el timestamp del último kline del batch anterior
        end_time = all_klines[-1][0]
        params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000, "endTime": end_time - 1}
    
    response = requests.get(url, params=params)
    klines = response.json()
    all_klines = klines + all_klines  # Prepend para orden cronológico
    print(f"  Batch {i+1}: {len(klines)} velas")

# Tomar últimos 2160 (90 días * 24h)
all_klines = all_klines[-2160:]

prices = np.array([float(k[4]) for k in all_klines])
volumes = np.array([float(k[5]) for k in all_klines])
timestamps = [datetime.fromtimestamp(k[0]/1000) for k in all_klines]

print(f"\nTotal: {len(prices)} velas")
print(f"Periodo: {timestamps[0].strftime('%Y-%m-%d')} a {timestamps[-1].strftime('%Y-%m-%d')}")

# Ejecutar VNN
print("\nEjecutando VNN...")
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
    
    profits = [t['profit_pct'] for t in sell_trades]
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p <= 0]
    
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Hold Time: {np.mean([t['hold_time'] for t in sell_trades]):.1f} horas")
    print(f"Avg Profit: {np.mean(profits):.2f}%")
    print(f"Best Trade: {max(profits):.2f}%")
    print(f"Worst Trade: {min(profits):.2f}%")
    
    if winning_trades:
        print(f"Avg Win: {np.mean(winning_trades):.2f}%")
    if losing_trades:
        print(f"Avg Loss: {np.mean(losing_trades):.2f}%")
    
    # Profit Factor
    total_wins = sum(winning_trades) if winning_trades else 0
    total_losses = abs(sum(losing_trades)) if losing_trades else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Sharpe Ratio (aproximado)
    if len(sell_trades) > 1:
        returns = np.array(profits)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Detalles de cada trade
    print(f"\n{'='*60}")
    print("DETALLE DE TRADES")
    print(f"{'='*60}")
    print(f"\n{'#':<4} {'Entry Date':<12} {'Exit Date':<12} {'Hold':<7} {'Profit':<10} {'Type':<6}")
    print("-" * 60)
    
    for i, trade in enumerate(sell_trades, 1):
        entry_tick = trade['tick'] - trade['hold_time']
        entry_date = timestamps[entry_tick].strftime('%m-%d')
        exit_date = timestamps[trade['tick']].strftime('%m-%d')
        profit_str = f"{trade['profit_pct']:+.2f}%"
        result = "WIN" if trade['profit_pct'] > 0 else "LOSS"
        
        print(f"{i:<4} {entry_date:<12} {exit_date:<12} {trade['hold_time']:<7} {profit_str:<10} {result:<6}")

# Visualización
fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

# 1. Precio + Trades
ax1 = axes[0]
ax1.plot(timestamps, prices, label='XRP/USDT 1h', color='black', linewidth=0.8)

for trade in buy_trades:
    ax1.scatter(timestamps[trade['tick']], prices[trade['tick']], 
               color='green', marker='^', s=80, zorder=5, alpha=0.7)

for trade in sell_trades:
    color = 'darkgreen' if trade['profit_pct'] > 0 else 'darkred'
    ax1.scatter(timestamps[trade['tick']], prices[trade['tick']], 
               color=color, marker='v', s=80, zorder=5, alpha=0.7)

ax1.set_ylabel('Precio (USDT)')
ax1.set_title(f'XRP/USDT 1h - 90 Días (Return: {return_pct:.2f}%, Trades: {len(sell_trades)})')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Equity Curve
ax2 = axes[1]
equity_timestamps = timestamps[:len(vnn.equity_curve)]
ax2.plot(equity_timestamps, vnn.equity_curve, color='blue', linewidth=2)
ax2.axhline(10000, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(equity_timestamps, 10000, vnn.equity_curve, 
                 where=np.array(vnn.equity_curve) >= 10000, 
                 color='green', alpha=0.2)
ax2.fill_between(equity_timestamps, 10000, vnn.equity_curve, 
                 where=np.array(vnn.equity_curve) < 10000, 
                 color='red', alpha=0.2)
ax2.set_ylabel('Equity ($)')
ax2.set_title(f'Equity Curve (Final: ${final_equity:,.2f})')
ax2.grid(alpha=0.3)

# 3. Cumulative Returns
ax3 = axes[2]
cumulative_returns = ((np.array(vnn.equity_curve) / 10000) - 1) * 100
ax3.plot(equity_timestamps, cumulative_returns, color='purple', linewidth=2)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.fill_between(equity_timestamps, 0, cumulative_returns, 
                 where=cumulative_returns >= 0, color='green', alpha=0.2)
ax3.fill_between(equity_timestamps, 0, cumulative_returns, 
                 where=cumulative_returns < 0, color='red', alpha=0.2)
ax3.set_ylabel('Cumulative Return (%)')
ax3.set_xlabel('Fecha')
ax3.set_title('Retorno Acumulado')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hourly_90days_backtest.png', dpi=150, bbox_inches='tight')
print(f"\nGráfico guardado: hourly_90days_backtest.png")

# Guardar resultados
results = {
    'period': {
        'start': timestamps[0].strftime('%Y-%m-%d'),
        'end': timestamps[-1].strftime('%Y-%m-%d'),
        'hours': len(prices),
        'days': len(prices) / 24
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
        'avg_hold_hours': float(np.mean([t['hold_time'] for t in sell_trades])) if len(sell_trades) > 0 else 0,
        'avg_profit_pct': float(np.mean(profits)) if len(sell_trades) > 0 else 0,
        'profit_factor': float(profit_factor) if len(sell_trades) > 0 else 0
    }
}

with open('hourly_90days_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Resultados guardados: hourly_90days_results.json")
print(f"\n{'='*60}")
print("BACKTEST COMPLETADO")
print(f"{'='*60}")
