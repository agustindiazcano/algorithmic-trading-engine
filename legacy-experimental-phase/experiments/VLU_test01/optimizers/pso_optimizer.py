"""
Particle Swarm Optimization
Organic swarm intelligence with velocity-based search
"""
import numpy as np
from typing import List

class Particle:
    """Single particle in the swarm"""
    
    def __init__(self, dim: int, bounds: List[tuple]):
        self.dim = dim
        self.bounds = bounds
        
        # Position and velocity
        self.x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.v = np.zeros(dim)
        
        # Personal best
        self.p_best = self.x.copy()
        self.p_best_fitness = -np.inf
        
        # Current fitness
        self.fitness = -np.inf
    
    def update_velocity(self, g_best: np.ndarray, w: float, c1: float, c2: float):
        """Update velocity based on personal and global best"""
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        
        # Velocity update: v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        cognitive = c1 * r1 * (self.p_best - self.x)
        social = c2 * r2 * (g_best - self.x)
        self.v = w * self.v + cognitive + social
        
        # Velocity clamping
        v_max = np.array([(b[1] - b[0]) * 0.2 for b in self.bounds])
        self.v = np.clip(self.v, -v_max, v_max)
    
    def update_position(self):
        """Update position and enforce bounds"""
        self.x = self.x + self.v
        
        # Bounce off boundaries
        for i, (low, high) in enumerate(self.bounds):
            if self.x[i] < low:
                self.x[i] = low
                self.v[i] *= -0.5  # Reverse and dampen
            elif self.x[i] > high:
                self.x[i] = high
                self.v[i] *= -0.5
    
    def update_best(self, fitness: float):
        """Update personal best if improved"""
        self.fitness = fitness
        if fitness > self.p_best_fitness:
            self.p_best = self.x.copy()
            self.p_best_fitness = fitness


class PSOOptimizer:
    """
    Particle Swarm Optimization with adaptive inertia
    """
    
    def __init__(self, dim: int = 8, n_particles: int = 20, 
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 bounds: List[tuple] = None):
        self.dim = dim
        self.n_particles = n_particles
        
        # PSO parameters
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        
        self.bounds = bounds or [
            (0.08, 0.20), (1.5, 4.0), (0.95, 0.99), (0.65, 0.85),
            (0.6, 0.95), (0.005, 0.025), (0.02, 0.05),
            (0.01, 0.05), (0.003, 0.02), (65, 80), (1.5, 4.0)
        ]
        
        # Initialize swarm
        self.particles = [Particle(dim, self.bounds) for _ in range(n_particles)]
        
        # Global best
        self.g_best = None
        self.g_best_fitness = -np.inf
        
        # Adaptive inertia
        self.w_max = 0.9
        self.w_min = 0.4
        self.generation = 0
    
    def ask(self) -> List[np.ndarray]:
        """Get current particle positions"""
        return [p.x.copy() for p in self.particles]
    
    def tell(self, fitnesses: List[float]):
        """Update swarm based on fitness evaluations"""
        # Update particle bests
        for particle, fitness in zip(self.particles, fitnesses):
            particle.update_best(fitness)
            
            # Update global best
            if fitness > self.g_best_fitness:
                self.g_best = particle.x.copy()
                self.g_best_fitness = fitness
        
        # Adaptive inertia (linearly decreasing)
        self.w = self.w_max - (self.w_max - self.w_min) * (self.generation / 50)
        
        # Update velocities and positions
        for particle in self.particles:
            particle.update_velocity(self.g_best, self.w, self.c1, self.c2)
            particle.update_position()
        
        self.generation += 1
    
    def get_diversity(self) -> float:
        """Measure swarm diversity (average distance from centroid)"""
        positions = np.array([p.x for p in self.particles])
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return distances.mean()


if __name__ == "__main__":
    print("🧪 Testing PSO...")
    
    pso = PSOOptimizer(dim=2, n_particles=20, bounds=[(-5, 5), (-5, 5)])
    
    # Test function: Sphere
    def sphere(x):
        return -np.sum(x**2)
    
    for gen in range(30):
        positions = pso.ask()
        fitnesses = [sphere(x) for x in positions]
        pso.tell(fitnesses)
        
        if gen % 5 == 0:
            print(f"Gen {gen}: Best={pso.g_best_fitness:.4f}, Diversity={pso.get_diversity():.4f}, w={pso.w:.3f}")
    
    print(f"\n✅ Best solution: {pso.g_best}, fitness={pso.g_best_fitness:.4f}")
