"""
Hybrid Meta-Optimizer
Coordinates CMA-ES, Bayesian, Simulated Annealing, and PSO
"""
import numpy as np
import requests
from typing import List, Tuple
import sys
sys.path.append('..')

from optimizers.cma_es_optimizer import CMAESOptimizer
from optimizers.bayesian_optimizer import BayesianOptimizer
from optimizers.simulated_annealing import SimulatedAnnealing
from optimizers.pso_optimizer import PSOOptimizer
from bounce_hunter_vnn import BounceHunterVNN, BounceHunterGenome
import csv

class HybridMetaOptimizer:
    """
    Meta-optimizer that intelligently switches between optimization strategies
    """
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        
        # Initialize all optimizers
        self.cma = CMAESOptimizer(dim=8, sigma=0.3, population_size=population_size)
        self.bayes = BayesianOptimizer(dim=8)
        self.sa = SimulatedAnnealing(initial_temp=50.0, cooling_rate=0.95)
        self.pso = PSOOptimizer(dim=8, n_particles=population_size)
        
        # Best solution tracking
        self.best_genome = None
        self.best_fitness = -np.inf
        
        # History
        self.fitness_history = []
        self.generation = 0
        
        # Bounds
        self.bounds = [
            (0.05, 0.15), (1.0, 3.0), (0.95, 0.99), (0.70, 0.90),
            (0.7, 0.95), (0.01, 0.03), (0.03, 0.06), (3, 12)
        ]
    
    def load_real_data(self, limit: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch real XRP data"""
        print(f"📡 Descargando {limit} velas...")
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": "XRPUSDT", "interval": "1h", "limit": limit}
            response = requests.get(url, params=params)
            data = response.json()
            
            prices = np.array([float(k[4]) for k in data])
            volumes = np.array([float(k[5]) for k in data]) + 1.0
            
            print(f"✅ {len(prices)} velas descargadas")
            return prices, volumes
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def array_to_genome(self, x: np.ndarray) -> BounceHunterGenome:
        """Convert parameter array to genome"""
        genome = BounceHunterGenome(random_init=False)
        genome.radius_base = np.clip(x[0], 0.05, 0.15)
        genome.volatility_multiplier = np.clip(x[1], 1.0, 3.0)
        genome.inertia_up = np.clip(x[2], 0.95, 0.99)
        genome.inertia_down = np.clip(x[3], 0.70, 0.90)
        genome.volume_exhaustion_threshold = np.clip(x[4], 0.7, 0.95)
        genome.take_profit_tolerance = np.clip(x[5], 0.01, 0.03)
        genome.stop_loss_pct = np.clip(x[6], 0.03, 0.06)
        genome.max_hold_time = int(np.clip(x[7], 3, 12))
        return genome
    
    def genome_to_array(self, genome: BounceHunterGenome) -> np.ndarray:
        """Convert genome to array"""
        return np.array([
            genome.radius_base, genome.volatility_multiplier,
            genome.inertia_up, genome.inertia_down,
            genome.volume_exhaustion_threshold, genome.take_profit_tolerance,
            genome.stop_loss_pct, float(genome.max_hold_time)
        ])
    
    def evaluate_genome(self, genome: BounceHunterGenome, scenarios: List) -> float:
        """Evaluate genome on all scenarios"""
        total_fitness = 0.0
        for prices, volumes, name in scenarios:
            vnn = BounceHunterVNN(genome, prices, volumes)
            vnn.run()
            fitness = vnn.calculate_fitness()
            total_fitness += fitness
        return total_fitness / len(scenarios)
    
    def evolve(self, generations: int = 50):
        """Run hybrid evolution"""
        print(f"\n🔥 HYBRID META-OPTIMIZER: {generations} generaciones\n")
        print("🧠 Estrategia adaptativa:")
        print("   Gen 1-20:  PSO + CMA-ES (exploración)")
        print("   Gen 21-40: Bayesian (refinamiento)")
        print("   Gen 41-50: Simulated Annealing (ajuste fino)\n")
        
        # Load training data
        scenarios = []
        for limit, name in [(1000, "xrp_1000h"), (500, "xrp_500h"), (300, "xrp_300h")]:
            prices, volumes = self.load_real_data(limit)
            if prices is not None:
                scenarios.append((prices, volumes, name))
        
        print(f"📚 Entrenando en {len(scenarios)} escenarios\n")
        
        # Initialize SA with random solution
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
        g0 = self.array_to_genome(x0)
        f0 = self.evaluate_genome(g0, scenarios)
        self.sa.initialize(x0, f0)
        
        for gen in range(generations):
            self.generation = gen
            print(f"🧬 Generación {gen+1}/{generations}")
            
            candidates = []
            method_name = ""
            
            # Phase-based strategy selection
            if gen < 20:
                # Phase 1: Exploration with PSO + CMA-ES
                method_name = "PSO + CMA-ES"
                pso_candidates = self.pso.ask()
                cma_candidates = self.cma.ask()
                candidates = pso_candidates[:10] + cma_candidates[:10]
                
            elif gen < 40:
                # Phase 2: Intelligent refinement with Bayesian
                method_name = "Bayesian"
                candidates = self.bayes.suggest_batch(15)
                
            else:
                # Phase 3: Local fine-tuning with SA
                method_name = "Simulated Annealing"
                for _ in range(10):
                    neighbor = self.sa.neighbor(self.sa.current_x, self.bounds)
                    candidates.append(neighbor)
            
            # Evaluate candidates
            genomes = [self.array_to_genome(x) for x in candidates]
            fitnesses = []
            
            for i, genome in enumerate(genomes):
                fitness = self.evaluate_genome(genome, scenarios)
                genome.fitness = fitness
                fitnesses.append(fitness)
                
                if (i + 1) % 5 == 0:
                    print(f"   Evaluando {i+1}/{len(genomes)}...", end='\r')
            
            # Update all optimizers (they learn from each other)
            if gen < 20:
                self.pso.tell(fitnesses[:10])
                self.cma.tell(candidates[10:], fitnesses[10:])
            elif gen < 40:
                for x, y in zip(candidates, fitnesses):
                    self.bayes.update(x, y)
            else:
                for x, y in zip(candidates, fitnesses):
                    self.sa.step(x, y, self.bounds)
            
            # Track best
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            avg_fitness = np.mean(fitnesses)
            
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_genome = genomes[best_idx]
                self.best_genome.save("hybrid_best.json")
                print(f"\n   💎 Nuevo mejor! Fitness: {best_fitness:.2f} ({method_name})")
            
            self.fitness_history.append({
                'generation': gen,
                'best': best_fitness,
                'avg': avg_fitness,
                'method': method_name
            })
            
            print(f"\n   Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Método: {method_name}")
            print()
        
        # Final report
        print("\n" + "="*60)
        print("🏆 OPTIMIZACIÓN HÍBRIDA COMPLETADA")
        print("="*60)
        print(f"Best Fitness: {self.best_fitness:.2f}")
        print(f"\nMejor genoma:")
        print(f"  Radio: {self.best_genome.radius_base:.3f}")
        print(f"  Inercia Up: {self.best_genome.inertia_up:.3f}")
        print(f"  Inercia Down: {self.best_genome.inertia_down:.3f}")
        print(f"  Take Profit: {self.best_genome.take_profit_tolerance:.3f}")
        print(f"  Guardado en: hybrid_best.json")
        
        # Save log
        with open('hybrid_evolution.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best', 'avg', 'method'])
            writer.writeheader()
            writer.writerows(self.fitness_history)
        print(f"📊 Log guardado: hybrid_evolution.csv\n")
        
        # Show CMA-ES learned correlations
        print("\n📊 Correlaciones aprendidas por CMA-ES:")
        corr = self.cma.get_correlation_matrix()
        params = ['radius', 'volat_mult', 'inertia_up', 'inertia_down', 
                  'vol_exhaust', 'take_profit', 'stop_loss', 'max_hold']
        print("\nTop correlaciones:")
        for i in range(len(params)):
            for j in range(i+1, len(params)):
                if abs(corr[i,j]) > 0.3:
                    print(f"  {params[i]} ↔ {params[j]}: {corr[i,j]:+.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Meta-Optimizer')
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--population', type=int, default=20)
    
    args = parser.parse_args()
    
    optimizer = HybridMetaOptimizer(population_size=args.population)
    optimizer.evolve(generations=args.generations)
