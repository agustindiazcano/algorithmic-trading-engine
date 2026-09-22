"""
Bayesian Optimization with Gaussian Processes
Uses Expected Improvement acquisition function
"""
import numpy as np
from typing import List, Tuple, Optional
from bounce_hunter_vnn import BounceHunterGenome
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Process surrogate model
    """
    
    def __init__(self, dim: int = 8, bounds: Optional[List[Tuple[float, float]]] = None):
        self.dim = dim
        self.bounds = bounds or [
            (0.05, 0.15), (1.0, 3.0), (0.95, 0.99), (0.70, 0.90),
            (0.7, 0.95), (0.01, 0.03), (0.03, 0.06), (3, 12)
        ]
        
        # Observed data
        self.X_observed = []
        self.y_observed = []
        
        # GP hyperparameters
        self.length_scale = 0.5
        self.signal_variance = 1.0
        self.noise_variance = 0.01
        
        # Best observed
        self.best_x = None
        self.best_y = -np.inf
        
    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF (Squared Exponential) kernel"""
        return self.signal_variance * np.exp(-0.5 * np.sum((x1 - x2)**2) / self.length_scale**2)
    
    def _compute_K(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix"""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel(X1[i], X2[j])
        return K
    
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict mean and std at point x"""
        if len(self.X_observed) == 0:
            return 0.0, 1.0
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Kernel matrices
        K = self._compute_K(X, X) + self.noise_variance * np.eye(len(X))
        k = np.array([self._kernel(x, xi) for xi in X])
        
        # GP prediction
        try:
            K_inv = np.linalg.inv(K)
            mean = k @ K_inv @ y
            variance = self.signal_variance - k @ K_inv @ k
            std = np.sqrt(max(variance, 1e-6))
        except np.linalg.LinAlgError:
            mean, std = 0.0, 1.0
        
        return mean, std
    
    def expected_improvement(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement acquisition function"""
        mean, std = self.predict(x)
        
        if std == 0:
            return 0.0
        
        # EI = E[max(0, f(x) - f_best - xi)]
        z = (mean - self.best_y - xi) / std
        ei = (mean - self.best_y - xi) * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
    
    def suggest_next(self) -> np.ndarray:
        """Suggest next point to evaluate using EI"""
        # Random restarts for optimization
        best_ei = -np.inf
        best_x = None
        
        for _ in range(10):
            # Random starting point
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
            
            # Maximize EI (minimize negative EI)
            result = minimize(
                lambda x: -self.expected_improvement(x),
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            ei = -result.fun
            if ei > best_ei:
                best_ei = ei
                best_x = result.x
        
        return best_x if best_x is not None else x0
    
    def update(self, x: np.ndarray, y: float):
        """Add observation to GP"""
        self.X_observed.append(x)
        self.y_observed.append(y)
        
        if y > self.best_y:
            self.best_y = y
            self.best_x = x
    
    def suggest_batch(self, n: int) -> List[np.ndarray]:
        """Suggest batch of points (with hallucinated observations)"""
        suggestions = []
        for _ in range(n):
            x = self.suggest_next()
            suggestions.append(x)
            # Hallucinate observation to encourage diversity
            mean, _ = self.predict(x)
            self.update(x, mean)
        
        # Remove hallucinated points
        self.X_observed = self.X_observed[:-n]
        self.y_observed = self.y_observed[:-n]
        
        return suggestions


if __name__ == "__main__":
    print("🧪 Testing Bayesian Optimizer...")
    
    opt = BayesianOptimizer(dim=2, bounds=[(0, 1), (0, 1)])
    
    # Test function: f(x,y) = -(x-0.7)^2 - (y-0.3)^2
    def fitness(x):
        return -((x[0] - 0.7)**2 + (x[1] - 0.3)**2)
    
    for i in range(15):
        x = opt.suggest_next()
        y = fitness(x)
        opt.update(x, y)
        print(f"Iter {i+1}: x={x}, f(x)={y:.4f}, best={opt.best_y:.4f}")
    
    print(f"\n✅ Found optimum near: {opt.best_x}")
