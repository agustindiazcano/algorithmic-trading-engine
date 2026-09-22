"""
Evolutionary restart strategy - cada 50 generaciones resetea con el mejor
"""
import numpy as np
from bounce_hunter_vnn import BounceHunterGenome

def restart_population(best_genome: BounceHunterGenome, population_size: int, mutation_strength: float = 0.3):
    """
    Crea nueva población basada en el mejor genoma con mutaciones fuertes
    """
    new_population = []
    
    # Mantener el elite sin cambios
    new_population.append(best_genome)
    
    # Crear el resto con mutaciones progresivamente más fuertes
    for i in range(1, population_size):
        # Mutación más fuerte para los últimos (más exploración)
        mutation_rate = mutation_strength * (1 + i / population_size)
        
        # Copiar TODOS los 11 parámetros
        child = BounceHunterGenome(random_init=False)
        for attr in ['radius_base', 'volatility_multiplier', 'inertia_up', 'inertia_down',
                     'volume_exhaustion_threshold', 'take_profit_tolerance', 'stop_loss_pct',
                     'trailing_stop_trigger', 'trailing_stop_floor', 
                     'rsi_overbought_threshold', 'volume_spike_multiplier']:
            setattr(child, attr, getattr(best_genome, attr))
        
        # Mutar con fuerza
        child.mutate(mutation_rate=mutation_rate)
        new_population.append(child)
    
    return new_population


def should_restart(generation: int, restart_interval: int = 50) -> bool:
    """Determina si es momento de restart"""
    return generation > 0 and generation % restart_interval == 0


def calculate_diversity(population: list) -> float:
    """
    Calcula diversidad de la población
    Baja diversidad = todos son clones
    """
    if len(population) < 2:
        return 0.0
    
    # Comparar radius_base como proxy de diversidad
    radii = [g.radius_base for g in population]
    return np.std(radii)


if __name__ == "__main__":
    print("🔄 Restart Strategy Ready")
    print("  - Cada 50 generaciones: reset con mejor genoma")
    print("  - Mutación fuerte (30-60%) para exploración")
    print("  - Mantiene elite sin cambios")
