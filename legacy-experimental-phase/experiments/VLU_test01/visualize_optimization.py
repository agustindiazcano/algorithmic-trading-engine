"""
Visualization tools for hybrid optimizer analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from optimizers.cma_es_optimizer import CMAESOptimizer

def visualize_correlation_matrix(cma_optimizer: CMAESOptimizer, save_path: str = "correlation_matrix.png"):
    """Visualize learned parameter correlations from CMA-ES"""
    corr = cma_optimizer.get_correlation_matrix()
    
    params = [
        'radius_base', 'volat_mult', 'inertia_up', 'inertia_down',
        'vol_exhaust', 'take_profit', 'stop_loss', 'max_hold'
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=params, yticklabels=params,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title('CMA-ES Learned Parameter Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Correlation matrix saved: {save_path}")
    
    # Print top correlations
    print("\n🔗 Top Parameter Correlations:")
    print("-" * 50)
    correlations = []
    for i in range(len(params)):
        for j in range(i+1, len(params)):
            correlations.append((params[i], params[j], corr[i, j]))
    
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    for p1, p2, c in correlations[:10]:
        sign = "↑↑" if c > 0 else "↑↓"
        print(f"  {p1:15} {sign} {p2:15}: {c:+.3f}")


def compare_fitness_curves(csv_files: dict, save_path: str = "fitness_comparison.png"):
    """
    Compare fitness curves from different optimizers
    csv_files: dict like {'CMA-ES': 'cma_log.csv', 'Hybrid': 'hybrid_log.csv'}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, filepath) in enumerate(csv_files.items()):
        try:
            df = pd.read_csv(filepath)
            color = colors[i % len(colors)]
            
            # Best fitness
            ax1.plot(df['generation'], df['best'], label=name, 
                    color=color, linewidth=2, marker='o', markersize=3)
            
            # Average fitness
            ax2.plot(df['generation'], df['avg'], label=name,
                    color=color, linewidth=2, alpha=0.7)
        except Exception as e:
            print(f"⚠️  Could not load {filepath}: {e}")
    
    ax1.set_title('Best Fitness Evolution', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Average Fitness Evolution', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Average Fitness', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Fitness comparison saved: {save_path}")


def analyze_hybrid_phases(csv_file: str = "hybrid_evolution.csv", save_path: str = "hybrid_phases.png"):
    """Analyze performance by optimization phase"""
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color by method
    methods = df['method'].unique()
    colors = {'PSO + CMA-ES': '#1f77b4', 'Bayesian': '#ff7f0e', 'Simulated Annealing': '#2ca02c'}
    
    for method in methods:
        mask = df['method'] == method
        ax.scatter(df[mask]['generation'], df[mask]['best'], 
                  label=method, color=colors.get(method, 'gray'),
                  s=50, alpha=0.6)
    
    # Plot line
    ax.plot(df['generation'], df['best'], color='black', linewidth=1.5, alpha=0.3)
    
    # Phase boundaries
    ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='Phase Transition')
    ax.axvline(40, color='red', linestyle='--', alpha=0.5)
    
    # Annotations
    ax.text(10, ax.get_ylim()[1] * 0.95, 'Phase 1:\nPSO + CMA-ES', 
           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(30, ax.get_ylim()[1] * 0.95, 'Phase 2:\nBayesian', 
           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(45, ax.get_ylim()[1] * 0.95, 'Phase 3:\nSA', 
           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Hybrid Optimizer: Performance by Phase', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Phase analysis saved: {save_path}")
    
    # Print statistics by phase
    print("\n📈 Performance by Phase:")
    print("-" * 60)
    for method in methods:
        phase_data = df[df['method'] == method]
        print(f"\n{method}:")
        print(f"  Generations: {len(phase_data)}")
        print(f"  Best Fitness: {phase_data['best'].max():.2f}")
        print(f"  Avg Fitness: {phase_data['avg'].mean():.2f}")
        print(f"  Improvement: {phase_data['best'].iloc[-1] - phase_data['best'].iloc[0]:.2f}")


if __name__ == "__main__":
    print("🎨 Visualization Tools Ready")
    print("\nAvailable functions:")
    print("  - visualize_correlation_matrix(cma_optimizer)")
    print("  - compare_fitness_curves({'Method': 'file.csv', ...})")
    print("  - analyze_hybrid_phases('hybrid_evolution.csv')")
