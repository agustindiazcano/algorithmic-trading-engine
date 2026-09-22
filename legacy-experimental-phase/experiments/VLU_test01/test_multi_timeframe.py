"""
Multi-Timeframe Backtest
Testea el mejor genoma en 1m, 1h, 1d para ver robustez
"""
import numpy as np
import json
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
import requests
import matplotlib.pyplot as plt

# Cargar mejor genoma
with open('pso_cma_best.json', 'r') as f:
    data = json.load(f)
best_genome = BounceHunterGenome.from_dict(data)

print("=" * 60)
print("MULTI-TIMEFRAME BACKTEST")
print("=" * 60)
print(f"\nGenoma: Fitness {best_genome.fitness:.2f}%")
print(f"Radius: {best_genome.radius_base:.3f}")
print(f"TP: {best_genome.take_profit_tolerance*100:.2f}%")
print(f"Stop: {best_genome.stop_loss_pct*100:.1f}%\n")

# Timeframes a testear
timeframes = [
    ('1m', 1000, "1 minuto (scalping)"),
    ('5m', 1000, "5 minutos (day trading)"),
    ('15m', 1000, "15 minutos"),
    ('1h', 1000, "1 hora (swing)"),
    ('4h', 1000, "4 horas"),
    ('1d', 365, "1 dia (position)"),
]

results = []

for interval, limit, description in timeframes:
    print(f"\n{'='*60}")
    print(f"Testeando: {description}")
    print(f"{'='*60}")
    
    try:
        # Descargar datos
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "XRPUSDT", "interval": interval, "limit": limit}
        response = requests.get(url, params=params)
        klines = response.json()
        
        if not klines or len(klines) < 100:
            print(f"ERROR: No hay suficientes datos para {interval}")
            continue
        
        prices = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])
        
        print(f"OK: {len(prices)} velas descargadas")
        
        # Ejecutar VNN
        vnn = BounceHunterVNN(best_genome, prices, volumes)
        signals = vnn.run()
        fitness = vnn.calculate_fitness()
        
        # Calcular métricas
        final_equity = vnn.equity_curve[-1]
        return_pct = ((final_equity / 10000) - 1) * 100
        
        sell_trades = [t for t in vnn.trades if t['type'] == 'sell']
        
        if len(sell_trades) > 0:
            wins = sum(1 for t in sell_trades if t['profit_pct'] > 0)
            win_rate = wins / len(sell_trades) * 100
            avg_profit = np.mean([t['profit_pct'] for t in sell_trades])
            avg_hold = np.mean([t['hold_time'] for t in sell_trades])
        else:
            win_rate = 0
            avg_profit = 0
            avg_hold = 0
        
        # Guardar resultados
        result = {
            'timeframe': interval,
            'description': description,
            'fitness': fitness,
            'return_pct': return_pct,
            'trades': len(sell_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_hold': avg_hold,
            'final_equity': final_equity
        }
        results.append(result)
        
        # Mostrar resultados
        print(f"\nResultados:")
        print(f"  Fitness: {fitness:.2f}%")
        print(f"  Return: {return_pct:.2f}%")
        print(f"  Equity: ${final_equity:,.2f}")
        print(f"  Trades: {len(sell_trades)}")
        if len(sell_trades) > 0:
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Avg Profit: {avg_profit:.2f}%")
            print(f"  Avg Hold: {avg_hold:.1f} periodos")
        
    except Exception as e:
        print(f"ERROR en {interval}: {e}")
        continue

# Resumen final
print("\n" + "=" * 60)
print("RESUMEN MULTI-TIMEFRAME")
print("=" * 60)

if results:
    print(f"\n{'Timeframe':<15} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Avg Profit':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['description']:<15} {r['return_pct']:>7.2f}%  {r['trades']:>6}   {r['win_rate']:>7.1f}%   {r['avg_profit']:>9.2f}%")
    
    # Mejor timeframe
    best_tf = max(results, key=lambda x: x['return_pct'])
    print(f"\nMejor Timeframe: {best_tf['description']} ({best_tf['return_pct']:.2f}%)")
    
    # Peor timeframe
    worst_tf = min(results, key=lambda x: x['return_pct'])
    print(f"Peor Timeframe: {worst_tf['description']} ({worst_tf['return_pct']:.2f}%)")
    
    # Promedio
    avg_return = np.mean([r['return_pct'] for r in results])
    print(f"\nReturn Promedio: {avg_return:.2f}%")
    
    # Visualización
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gráfico 1: Return por timeframe
    ax1 = axes[0]
    timeframes_labels = [r['description'] for r in results]
    returns = [r['return_pct'] for r in results]
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    ax1.bar(timeframes_labels, returns, color=colors, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Return por Timeframe')
    ax1.grid(alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gráfico 2: Trades y Win Rate
    ax2 = axes[1]
    x = np.arange(len(results))
    width = 0.35
    
    trades = [r['trades'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, trades, width, label='Trades', color='blue', alpha=0.7)
    bars2 = ax2_twin.bar(x + width/2, win_rates, width, label='Win Rate %', color='orange', alpha=0.7)
    
    ax2.set_xlabel('Timeframe')
    ax2.set_ylabel('Trades', color='blue')
    ax2_twin.set_ylabel('Win Rate (%)', color='orange')
    ax2.set_title('Actividad y Win Rate por Timeframe')
    ax2.set_xticks(x)
    ax2.set_xticklabels(timeframes_labels)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_timeframe_backtest.png', dpi=150, bbox_inches='tight')
    print(f"\nGrafico guardado: multi_timeframe_backtest.png")
    
    # Guardar resultados
    with open('multi_timeframe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados: multi_timeframe_results.json")

else:
    print("\nNo se pudieron obtener resultados")

print("\nBacktest completado!")
