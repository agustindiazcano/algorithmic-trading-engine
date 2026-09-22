"""
PSO + CMA-ES Hybrid Optimizer
Best of both worlds: swarm exploration + covariance learning
"""
import numpy as np
import requests
from typing import List, Tuple
import sys
sys.path.append('..')

from optimizers.cma_es_optimizer import CMAESOptimizer
from optimizers.pso_optimizer import PSOOptimizer
from bounce_hunter_vnn import BounceHunterVNN, BounceHunterGenome
import csv

class PSOCMAHybrid:
    """
    Hybrid optimizer combining PSO and CMA-ES
    PSO explores broadly, CMA-ES refines with correlation learning
    """
    
    def __init__(self, population_size: int = 30):
        self.population_size = population_size
        
        # Split population between PSO and CMA-ES
        self.pso_size = population_size // 2
        self.cma_size = population_size - self.pso_size
        
        # Initialize optimizers (11 dimensions - evolvable exit rules)
        self.pso = PSOOptimizer(dim=11, n_particles=self.pso_size)
        self.cma = CMAESOptimizer(dim=11, sigma=0.3, population_size=self.cma_size)
        
        # Best solution tracking
        self.best_genome = None
        self.best_fitness = -np.inf
        
        # History
        self.fitness_history = []
        self.generation = 0
        
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
        """Convert parameter array to genome (11 params with evolvable exits)"""
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
        """Convert genome to array (11 params with evolvable exits)"""
        return np.array([
            genome.radius_base, genome.volatility_multiplier,
            genome.inertia_up, genome.inertia_down,
            genome.volume_exhaustion_threshold, genome.take_profit_tolerance,
            genome.stop_loss_pct, genome.trailing_stop_trigger,
            genome.trailing_stop_floor, genome.rsi_overbought_threshold,
            genome.volume_spike_multiplier
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
    
    def get_diversity(self) -> float:
        """Mide diversidad de la población PSO"""
        return self.pso.get_diversity()
    
    def evolve(self, generations: int = 50):
        """Run PSO + CMA-ES hybrid evolution"""
        print(f"\n🔥 PSO + CMA-ES HYBRID: {generations} generaciones\n")
        print(f"🧠 Estrategia:")
        print(f"   PSO: {self.pso_size} partículas (exploración global)")
        print(f"   CMA-ES: {self.cma_size} individuos (refinamiento + correlaciones)\n")
        
        # Load training data
        scenarios = []
        for limit, name in [(1000, "xrp_1000h"), (500, "xrp_500h"), (300, "xrp_300h")]:
            prices, volumes = self.load_real_data(limit)
            if prices is not None:
                scenarios.append((prices, volumes, name))
        
        print(f"📚 Entrenando en {len(scenarios)} escenarios\n")
        
        for gen in range(generations):
            self.generation = gen
            print(f"🧬 Generación {gen+1}/{generations}")
            
            # RESTART cada 50 generaciones
            if gen > 0 and gen % 50 == 0:
                print(f"\n🔄 RESTART! Reseteando población con el mejor genoma...")
                print(f"   Diversidad antes: {self.get_diversity():.4f}")
                
                # Reiniciar PSO y CMA-ES con mutaciones del mejor
                from restart_strategy import restart_population
                
                # Crear nueva población basada en el mejor
                new_genomes = restart_population(self.best_genome, self.population_size, mutation_strength=0.4)
                
                # Convertir a arrays y reiniciar optimizadores
                new_arrays = [self.genome_to_array(g) for g in new_genomes]
                
                # Reiniciar PSO
                self.pso = PSOOptimizer(dim=11, n_particles=self.pso_size)
                for i, particle in enumerate(self.pso.particles):
                    if i < len(new_arrays[:self.pso_size]):
                        particle.x = new_arrays[i]
                        particle.p_best = new_arrays[i].copy()
                
                # Reiniciar CMA-ES con sigma más grande (explorar)
                self.cma = CMAESOptimizer(dim=11, sigma=0.5, population_size=self.cma_size)
                self.cma.mean = self.genome_to_array(self.best_genome)
                
                print(f"   Diversidad después: {self.get_diversity():.4f}")
                print(f"   CMA-ES sigma: {self.cma.sigma:.4f}\n")
            
            # Get candidates from both optimizers
            pso_candidates = self.pso.ask()
            cma_candidates = self.cma.ask()
            
            all_candidates = pso_candidates + cma_candidates
            
            # Evaluate all
            genomes = [self.array_to_genome(x) for x in all_candidates]
            fitnesses = []
            
            for i, genome in enumerate(genomes):
                fitness = self.evaluate_genome(genome, scenarios)
                genome.fitness = fitness
                fitnesses.append(fitness)
                
                if (i + 1) % 5 == 0:
                    print(f"   Evaluando {i+1}/{len(genomes)}...", end='\r')
            
            # Split fitnesses
            pso_fitnesses = fitnesses[:self.pso_size]
            cma_fitnesses = fitnesses[self.pso_size:]
            
            # Update both optimizers
            self.pso.tell(pso_fitnesses)
            self.cma.tell(cma_candidates, cma_fitnesses)
            
            # Track best
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            avg_fitness = np.mean(fitnesses)
            
            # Determine source
            source = "PSO" if best_idx < self.pso_size else "CMA-ES"
            
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_genome = genomes[best_idx]
                self.best_genome.save("pso_cma_best.json")
                print(f"\n   💎 Nuevo mejor! Fitness: {best_fitness:.2f} (de {source})")
            
            self.fitness_history.append({
                'generation': gen,
                'best': best_fitness,
                'avg': avg_fitness,
                'pso_best': max(pso_fitnesses),
                'cma_best': max(cma_fitnesses),
                'source': source
            })
            
            # Show PSO diversity
            diversity = self.pso.get_diversity()
            
            print(f"\n   Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f}")
            print(f"   PSO Best: {max(pso_fitnesses):.2f} | CMA Best: {max(cma_fitnesses):.2f}")
            print(f"   PSO Diversity: {diversity:.4f} | CMA σ: {self.cma.sigma:.4f}")
            print()
        
        # Final report
        print("\n" + "="*60)
        print("🏆 PSO + CMA-ES OPTIMIZACIÓN COMPLETADA")
        print("="*60)
        print(f"Best Fitness: {self.best_fitness:.2f}")
        
        # Count wins
        pso_wins = sum(1 for h in self.fitness_history if h['source'] == 'PSO')
        cma_wins = sum(1 for h in self.fitness_history if h['source'] == 'CMA-ES')
        
        print(f"\n📊 Victorias:")
        print(f"   PSO: {pso_wins} generaciones")
        print(f"   CMA-ES: {cma_wins} generaciones")
        
        print(f"\n🧬 Mejor genoma:")
        print(f"  Radio: {self.best_genome.radius_base:.3f}")
        print(f"  Inercia Up: {self.best_genome.inertia_up:.3f}")
        print(f"  Inercia Down: {self.best_genome.inertia_down:.3f}")
        print(f"  Take Profit: {self.best_genome.take_profit_tolerance:.3f}")
        print(f"  Guardado en: pso_cma_best.json")
        
        # Save log
        with open('pso_cma_evolution.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best', 'avg', 'pso_best', 'cma_best', 'source'])
            writer.writeheader()
            writer.writerows(self.fitness_history)
        print(f"📊 Log guardado: pso_cma_evolution.csv\n")
        
        # Show CMA-ES learned correlations
        print("\n📊 Correlaciones aprendidas por CMA-ES:")
        corr = self.cma.get_correlation_matrix()
        params = ['radius', 'volat_mult', 'inertia_up', 'inertia_down', 
                  'vol_exhaust', 'take_profit', 'stop_loss', 'max_hold']
        print("\nTop correlaciones:")
        correlations = []
        for i in range(len(params)):
            for j in range(i+1, len(params)):
                correlations.append((params[i], params[j], corr[i,j]))
        
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        for p1, p2, c in correlations[:8]:
            sign = "↑↑" if c > 0 else "↑↓"
            print(f"  {p1:12} {sign} {p2:12}: {c:+.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PSO + CMA-ES Hybrid')
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--population', type=int, default=30)
    
    args = parser.parse_args()
    
    optimizer = PSOCMAHybrid(population_size=args.population)
    optimizer.evolve(generations=args.generations)
