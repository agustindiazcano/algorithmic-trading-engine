"""
Simulated Annealing with adaptive cooling schedule
Escapes local minima through probabilistic acceptance
"""
import numpy as np
from typing import Callable
from bounce_hunter_vnn import BounceHunterGenome

class SimulatedAnnealing:
    """
    Simulated Annealing optimizer with exponential cooling
    """
    
    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.95, 
                 min_temp: float = 0.01):
        self.T = initial_temp
        self.T0 = initial_temp
        self.alpha = cooling_rate
        self.min_temp = min_temp
        
        # Current solution
        self.current_x = None
        self.current_fitness = -np.inf
        
        # Best solution found
        self.best_x = None
        self.best_fitness = -np.inf
        
        # Statistics
        self.iterations = 0
        self.acceptances = 0
        
    def initialize(self, x: np.ndarray, fitness: float):
        """Set initial solution"""
        self.current_x = x.copy()
        self.current_fitness = fitness
        self.best_x = x.copy()
        self.best_fitness = fitness
    
    def neighbor(self, x: np.ndarray, bounds: list) -> np.ndarray:
        """Generate neighbor solution"""
        neighbor = x.copy()
        
        # Perturb random subset of dimensions
        n_perturb = np.random.randint(1, len(x) + 1)
        dims = np.random.choice(len(x), n_perturb, replace=False)
        
        for dim in dims:
            # Adaptive step size based on temperature
            step_size = (bounds[dim][1] - bounds[dim][0]) * 0.1 * (self.T / self.T0)
            neighbor[dim] += np.random.normal(0, step_size)
            neighbor[dim] = np.clip(neighbor[dim], bounds[dim][0], bounds[dim][1])
        
        return neighbor
    
    def accept_probability(self, current_fit: float, neighbor_fit: float) -> float:
        """Calculate acceptance probability"""
        if neighbor_fit > current_fit:
            return 1.0
        
        if self.T == 0:
            return 0.0
        
        delta_E = neighbor_fit - current_fit
        return np.exp(delta_E / self.T)
    
    def step(self, neighbor_x: np.ndarray, neighbor_fitness: float, bounds: list) -> bool:
        """Perform one SA step, returns True if accepted"""
        self.iterations += 1
        
        # Acceptance criterion
        accept_prob = self.accept_probability(self.current_fitness, neighbor_fitness)
        
        if np.random.rand() < accept_prob:
            self.current_x = neighbor_x
            self.current_fitness = neighbor_fitness
            self.acceptances += 1
            
            # Update best
            if neighbor_fitness > self.best_fitness:
                self.best_x = neighbor_x.copy()
                self.best_fitness = neighbor_fitness
            
            accepted = True
        else:
            accepted = False
        
        # Cool down
        self.T = max(self.T * self.alpha, self.min_temp)
        
        return accepted
    
    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate"""
        if self.iterations == 0:
            return 0.0
        return self.acceptances / self.iterations
    
    def reheat(self, factor: float = 2.0):
        """Reheat temperature (for escaping plateaus)"""
        self.T = min(self.T * factor, self.T0)


if __name__ == "__main__":
    print("🧪 Testing Simulated Annealing...")
    
    sa = SimulatedAnnealing(initial_temp=50.0, cooling_rate=0.95)
    
    # Test function: Rastrigin (many local minima)
    def rastrigin(x):
        return -10 * len(x) - sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
    
    bounds = [(-5.12, 5.12)] * 2
    x = np.random.uniform(-5, 5, 2)
    fitness = rastrigin(x)
    sa.initialize(x, fitness)
    
    for i in range(500):
        neighbor = sa.neighbor(sa.current_x, bounds)
        neighbor_fit = rastrigin(neighbor)
        accepted = sa.step(neighbor, neighbor_fit, bounds)
        
        if i % 50 == 0:
            print(f"Iter {i}: T={sa.T:.2f}, Best={sa.best_fitness:.4f}, Accept Rate={sa.get_acceptance_rate():.2%}")
    
    print(f"\n✅ Best solution: {sa.best_x}, fitness={sa.best_fitness:.4f}")
