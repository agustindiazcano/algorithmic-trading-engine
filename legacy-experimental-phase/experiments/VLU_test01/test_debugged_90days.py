"""
CORRECTED Backtest of 1h genome for 90 days (2160 hours)
Fixes: Date mismatch, overlapping data, duplicate trades, mislabeled TP.
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
print("DEBUGGED BACKTEST 1H - ÚLTIMOS 90 DÍAS")
print("=" * 60)
print(f"\nGenoma:")
print(f"  Radius: {best_genome.radius_base*100:.2f}%")
print(f"  Center Tolerance (TP): {best_genome.take_profit_tolerance*100:.2f}% (Not fixed profit)")
print(f"  Stop Loss: {best_genome.stop_loss_pct*100:.1f}%")

# Descargar últimos 90 días en velas de 1h (Batch Fetching Correcto)
print(f"\nDescargando velas de 1h (3 Batches sin overlap)...")
url = "https://api.binance.com/api/v3/klines"

all_klines = []
end_time = None # Start with "Now"

for i in range(3):
    params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000}
    if end_time:
        params["endTime"] = end_time
    
    # Debug: print what we are asking for
    # print(f"  Requesting batch {i+1} with endTime={end_time}")
    
    response = requests.get(url, params=params)
    klines = response.json()
    
    if not klines:
        break
        
    # Prepend NEW fetched data (which is older) to our list
    # klines are returned [oldest ... newest]
    # So we want [Batch 3, Batch 2, Batch 1]
    all_klines = klines + all_klines
    
    # Next batch end time = timestamp of the OLDEST candle in this batch - 1ms
    first_candle_timestamp = klines[0][0]
    end_time = first_candle_timestamp - 1
    
    print(f"  Batch {i+1}: {len(klines)} velas. Rango: {datetime.fromtimestamp(klines[0][0]/1000)} -> {datetime.fromtimestamp(klines[-1][0]/1000)}")

# Tomar exactamente las últimas 2160 velas (90 días)
needed = 2160
if len(all_klines) > needed:
    all_klines = all_klines[-needed:]

prices = np.array([float(k[4]) for k in all_klines])
volumes = np.array([float(k[5]) for k in all_klines])
timestamps = [datetime.fromtimestamp(k[0]/1000) for k in all_klines]

print(f"\nTotal Velas: {len(prices)}")
print(f"Inicio Real: {timestamps[0]}")
print(f"Fin Real:    {timestamps[-1]}")
print(f"Duración:    {(timestamps[-1] - timestamps[0]).days} días")

# Ejecutar VNN
print("\nEjecutando simulación...")
vnn = BounceHunterVNN(best_genome, prices, volumes)
signals = vnn.run()
fitness = vnn.calculate_fitness() # Nota: este fitness usa la formula del VNN (return %)

# Resultados
final_equity = vnn.equity_curve[-1]
return_pct = ((final_equity / 10000) - 1) * 100
profit = final_equity - 10000

print(f"\n{'='*60}")
print("RESULTADOS FINALES")
print(f"{'='*60}")
print(f"Equity Inicial: $10,000.00")
print(f"Equity Final:   ${final_equity:,.2f}")
print(f"Return Total:   {return_pct:.2f}%")
print(f"Trades:         {len([t for t in vnn.trades if t['type'] == 'sell'])}")

# Analizar trades (Asegurar orden y unicidad)
sell_trades = [t for t in vnn.trades if t['type'] == 'sell']
# Sort by exit tick just in case
sell_trades.sort(key=lambda x: x['tick'])

print(f"\n{'='*60}")
print("DETALLE POR TRADE (Orden Cronológico)")
print("Nota: 'TP' en genoma = Tolerancia geométrica (no % fijo)")
print(f"{'='*60}")
print(f"{'#':<4} {'Entry Time':<18} {'Exit Time':<18} {'Hold(h)':<8} {'Profit %':<10} {'Trigger'}")
print("-" * 75)

for i, trade in enumerate(sell_trades, 1):
    exit_tick = trade['tick']
    entry_tick = exit_tick - trade['hold_time']
    
    entry_time_str = timestamps[entry_tick].strftime('%Y-%m-%d %H:%M')
    exit_time_str = timestamps[exit_tick].strftime('%Y-%m-%d %H:%M')
    
    # Inferir razón de salida (aproximada para display)
    reason = "Signal"
    if trade['profit_pct'] > 0:
        reason = "Target/Trail"
    else:
        reason = "Stop/Signal"
        
    print(f"{i:<4} {entry_time_str:<18} {exit_time_str:<18} {trade['hold_time']:<8} {trade['profit_pct']:>7.2f}%   {reason}")

# Visualización Correcta
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# 1. Price Chart
ax1 = axes[0]
ax1.plot(timestamps, prices, color='black', linewidth=0.8, label='XRP/USDT 1h')

# Markers match trades EXACTLY
for trade in sell_trades:
    exit_idx = trade['tick']
    entry_idx = exit_idx - trade['hold_time']
    
    # Buy Marker
    ax1.scatter(timestamps[entry_idx], prices[entry_idx], color='green', marker='^', s=60, zorder=3)
    # Sell Marker
    color = 'green' if trade['profit_pct'] > 0 else 'red'
    ax1.scatter(timestamps[exit_idx], prices[exit_idx], color=color, marker='v', s=60, zorder=3)
    
    # Connect
    ax1.plot([timestamps[entry_idx], timestamps[exit_idx]], [prices[entry_idx], prices[exit_idx]], 
             color=color, linestyle='--', alpha=0.5, linewidth=1)

ax1.set_title(f"XRP/USDT 90-Day Backtest ({timestamps[0].date()} to {timestamps[-1].date()})")
ax1.set_ylabel("Price")
ax1.grid(True, alpha=0.2)
ax1.legend()

# 2. Equity Curve
equity_ts = timestamps[:len(vnn.equity_curve)]
ax2 = axes[1]
ax2.plot(equity_ts, vnn.equity_curve, color='blue', linewidth=1.5, label='Equity')
ax2.axhline(10000, color='gray', linestyle='--')
ax2.set_ylabel("Equity ($)")
ax2.set_title(f"Final Equity: ${final_equity:,.2f} (+{return_pct:.2f}%)")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('debugged_backtest_90d.png')
print(f"\nGráfico guardado: debugged_backtest_90d.png")

# Guardar JSON limpio
clean_results = {
    "period_start": str(timestamps[0]),
    "period_end": str(timestamps[-1]),
    "total_return_pct": return_pct,
    "trades_count": len(sell_trades),
    "trades": [
        {
            "entry": str(timestamps[t['tick']-t['hold_time']]),
            "exit": str(timestamps[t['tick']]),
            "profit_pct": t['profit_pct'],
            "hold_hours": t['hold_time']
        }
        for t in sell_trades
    ]
}

with open('debugged_results.json', 'w') as f:
    json.dump(clean_results, f, indent=2)
print("Resultados guardados en debugged_results.json")
