"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
Learns correlations between parameters to intelligently explore fitness landscape
"""
import numpy as np
from typing import List, Tuple
from bounce_hunter_vnn import BounceHunterGenome

class CMAESOptimizer:
    """
    CMA-ES optimizer that adapts covariance matrix to learn parameter correlations
    """
    
    def __init__(self, dim: int = 8, sigma: float = 0.3, population_size: int = 20):
        self.dim = dim
        self.sigma = sigma
        self.lam = population_size  # λ (lambda) - offspring population size
        self.mu = population_size // 2  # μ (mu) - parents for recombination
        
        # Initialize mean in center of search space (AGGRESSIVE, 11D)
        self.mean = np.array([0.14, 2.75, 0.97, 0.75, 0.775, 0.015, 0.035, 0.03, 0.01, 72.5, 2.75])
        
        # Covariance matrix (starts as identity - no correlations)
        self.C = np.eye(dim)
        
        # Evolution paths for covariance matrix adaptation
        self.pc = np.zeros(dim)  # Evolution path for C
        self.ps = np.zeros(dim)  # Evolution path for sigma
        
        # Weights for recombination (favor better solutions)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        
        # Strategy parameters
        self.mueff = 1 / (self.weights ** 2).sum()
        self.cc = 4 / (dim + 4)  # Time constant for cumulation for C
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)  # For sigma
        self.c1 = 2 / ((dim + 1.3)**2 + self.mueff)  # Learning rate for rank-1 update
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        
        # Expectation of ||N(0,I)||
        self.chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        self.generation = 0
        
    def ask(self) -> List[np.ndarray]:
        """Generate new candidate solutions"""
        # Sample from multivariate normal N(mean, σ²C)
        samples = []
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(self.C)
                y = L @ z
            except np.linalg.LinAlgError:
                # If C is not positive definite, use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-8)
                y = eigvecs @ np.diag(np.sqrt(eigvals)) @ z
            
            x = self.mean + self.sigma * y
            samples.append(x)
        
        return samples
    
    def tell(self, solutions: List[np.ndarray], fitnesses: List[float]):
        """Update distribution based on fitness results"""
        # Sort by fitness (descending)
        idx = np.argsort(fitnesses)[::-1]
        solutions = np.array(solutions)[idx]
        
        # Select top mu solutions
        elite = solutions[:self.mu]
        
        # Recombination: new mean
        old_mean = self.mean.copy()
        self.mean = elite.T @ self.weights
        
        # Cumulation: update evolution paths
        mean_shift = (self.mean - old_mean) / self.sigma
        
        # Conjugate evolution path for C
        C_sqrt_inv = self._invsqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * C_sqrt_inv @ mean_shift
        
        # Cumulation for covariance matrix
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.generation + 1))) / self.chiN 
                < 1.4 + 2 / (self.dim + 1))
        
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * mean_shift
        
        # Adapt covariance matrix C
        artmp = (elite - old_mean) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) +
                  self.cmu * artmp.T @ np.diag(self.weights) @ artmp)
        
        # Adapt step size sigma
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        self.generation += 1
    
    def _invsqrt(self, C):
        """Compute C^(-1/2) using eigendecomposition"""
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
    
    def array_to_genome(self, x: np.ndarray) -> BounceHunterGenome:
        """Convert parameter array to genome (11 params)"""
        genome = BounceHunterGenome(random_init=False)
        genome.radius_base = np.clip(x[0], 0.08, 0.20)
        genome.volatility_multiplier = np.clip(x[1], 1.5, 4.0)
        genome.inertia_up = np.clip(x[2], 0.95, 0.99)
        genome.inertia_down = np.clip(x[3], 0.65, 0.85)
        genome.volume_exhaustion_threshold = np.clip(x[4], 0.6, 0.95)
        genome.take_profit_tolerance = np.clip(x[5], 0.005, 0.025)
        genome.stop_loss_pct = np.clip(x[6], 0.02, 0.05)
        genome.trailing_stop_trigger = np.clip(x[7], 0.01, 0.05)
        genome.trailing_stop_floor = np.clip(x[8], 0.003, 0.02)
        genome.rsi_overbought_threshold = np.clip(x[9], 65, 80)
        genome.volume_spike_multiplier = np.clip(x[10], 1.5, 4.0)
        return genome
    
    def genome_to_array(self, genome: BounceHunterGenome) -> np.ndarray:
        """Convert genome to parameter array (11 params)"""
        return np.array([
            genome.radius_base, genome.volatility_multiplier,
            genome.inertia_up, genome.inertia_down,
            genome.volume_exhaustion_threshold, genome.take_profit_tolerance,
            genome.stop_loss_pct, genome.trailing_stop_trigger,
            genome.trailing_stop_floor, genome.rsi_overbought_threshold,
            genome.volume_spike_multiplier
        ])
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get correlation matrix from covariance matrix"""
        std = np.sqrt(np.diag(self.C))
        corr = self.C / np.outer(std, std)
        return corr


if __name__ == "__main__":
    # Test CMA-ES
    print("🧪 Testing CMA-ES Optimizer...")
    
    opt = CMAESOptimizer(dim=8, sigma=0.3, population_size=10)
    
    # Dummy fitness function (sphere function)
    def fitness(x):
        return -np.sum((x - 0.5)**2)
    
    for gen in range(10):
        solutions = opt.ask()
        fitnesses = [fitness(s) for s in solutions]
        opt.tell(solutions, fitnesses)
        
        print(f"Gen {gen+1}: Best fitness = {max(fitnesses):.4f}, Sigma = {opt.sigma:.4f}")
    
    print("\n✅ CMA-ES test complete!")
    print(f"Final mean: {opt.mean}")
    print(f"\nCorrelation matrix:\n{opt.get_correlation_matrix()}")
