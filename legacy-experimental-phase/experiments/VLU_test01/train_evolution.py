import numpy as np
import requests
import json
from evolutionary_vnn import NightmareMarketGenerator, VNNGenome, DualNeuronVNN
from typing import List, Tuple
import csv
from datetime import datetime

class EvolutionarySchool:
    """
    Manages population evolution through genetic algorithm
    with adversarial training scenarios
    """
    
    def __init__(self, population_size: int = 20, elite_count: int = 3, 
                 mutation_rate: float = 0.15, nightmare_ratio: float = 0.30):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.nightmare_ratio = nightmare_ratio
        
        self.population: List[VNNGenome] = []
        self.generation = 0
        self.best_genome = None
        self.fitness_history = []
        
        self.nightmare_gen = NightmareMarketGenerator()
        
    def load_real_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch real XRP data from Binance"""
        print("📡 Fetching real market data from Binance...")
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": "XRPUSDT", "interval": "1h", "limit": 1000}
            response = requests.get(url, params=params)
            data = response.json()
            
            prices = np.array([float(k[4]) for k in data])
            volumes = np.array([float(k[5]) for k in data])
            
            # Safety: ensure all volumes are positive
            volumes = np.abs(volumes) + 1.0  # Add 1.0 to avoid zeros
            
            print(f"✅ Loaded {len(prices)} real market ticks")
            return prices, volumes
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            print("⚠️ Using synthetic fallback...")
            prices = np.linspace(1.5, 1.7, 1000) + np.random.normal(0, 0.02, 1000)
            volumes = np.abs(np.random.normal(1.2e6, 3e5, 1000))
            return prices, volumes
    
    def generate_training_scenarios(self, base_price: float) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Generate mix of real and adversarial scenarios
        Returns: List of (prices, volumes, scenario_name)
        """
        scenarios = []
        
        # Real data (50%)
        real_p, real_v = self.load_real_data()
        scenarios.append((real_p, real_v, "real_market"))
        
        # Adversarial scenarios (30%)
        nightmare_count = int(self.nightmare_ratio * 3)  # 30% of total
        
        scenarios.append((
            *self.nightmare_gen.generate_flash_crash(base_price),
            "flash_crash"
        ))
        
        scenarios.append((
            *self.nightmare_gen.generate_bull_trap(base_price),
            "bull_trap"
        ))
        
        scenarios.append((
            *self.nightmare_gen.generate_choppy_hell(base_price, 800),
            "choppy_hell"
        ))
        
        scenarios.append((
            *self.nightmare_gen.generate_mixed_nightmare(base_price, 1000),
            "mixed_nightmare"
        ))
        
        return scenarios
    
    def evaluate_genome(self, genome: VNNGenome, scenarios: List) -> float:
        """Evaluate genome across all scenarios"""
        total_fitness = 0.0
        
        for prices, volumes, name in scenarios:
            vnn = DualNeuronVNN(genome, prices, volumes)
            vnn.run()
            fitness = vnn.calculate_fitness()
            total_fitness += fitness
        
        # Average fitness across scenarios
        return total_fitness / len(scenarios)
    
    def initialize_population(self):
        """Create initial random population"""
        print(f"🧬 Initializing population of {self.population_size} genomes...")
        self.population = [VNNGenome(random_init=True) for _ in range(self.population_size)]
    
    def select_elite(self):
        """Keep top performers"""
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        return self.population[:self.elite_count]
    
    def tournament_selection(self, tournament_size: int = 3) -> VNNGenome:
        """Select parent via tournament"""
        candidates = np.random.choice(self.population, tournament_size, replace=False)
        return max(candidates, key=lambda g: g.fitness)
    
    def breed_next_generation(self):
        """Create new generation through selection, crossover, and mutation"""
        elite = self.select_elite()
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child = VNNGenome.crossover(parent1, parent2)
            
            # Mutation
            child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
    
    def evolve(self, generations: int = 50):
        """Run evolution for N generations"""
        print(f"\n🔥 STARTING EVOLUTION: {generations} generations\n")
        
        # Initialize
        self.initialize_population()
        base_price = 1.6  # XRP typical price
        scenarios = self.generate_training_scenarios(base_price)
        
        print(f"📚 Training on {len(scenarios)} scenarios:")
        for _, _, name in scenarios:
            print(f"   - {name}")
        print()
        
        # Evolution loop
        for gen in range(generations):
            self.generation = gen
            print(f"🧬 Generation {gen+1}/{generations}")
            
            # Evaluate all genomes
            for i, genome in enumerate(self.population):
                fitness = self.evaluate_genome(genome, scenarios)
                genome.fitness = fitness
                
                if (i + 1) % 5 == 0:
                    print(f"   Evaluated {i+1}/{self.population_size} genomes...", end='\r')
            
            # Track best
            best = max(self.population, key=lambda g: g.fitness)
            avg = np.mean([g.fitness for g in self.population])
            worst = min(self.population, key=lambda g: g.fitness)
            
            self.fitness_history.append({
                'generation': gen,
                'best': best.fitness,
                'avg': avg,
                'worst': worst.fitness
            })
            
            print(f"\n   Best: {best.fitness:.2f} | Avg: {avg:.2f} | Worst: {worst.fitness:.2f}")
            
            # Save best genome
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
                self.best_genome.save("best_genome.json")
                print(f"   💎 New best genome saved! Fitness: {best.fitness:.2f}")
            
            # Breed next generation (except last)
            if gen < generations - 1:
                self.breed_next_generation()
            
            print()
        
        # Final report
        print("\n" + "="*60)
        print("🏆 EVOLUTION COMPLETE")
        print("="*60)
        print(f"Best Fitness: {self.best_genome.fitness:.2f}")
        print(f"Improvement: {self.fitness_history[-1]['best'] / self.fitness_history[0]['best']:.2f}x")
        print(f"\nBest Genome Parameters:")
        print(f"  Anchor Inertia: {self.best_genome.anchor_inertia:.3f}")
        print(f"  Hunter Inertia: {self.best_genome.hunter_inertia:.3f}")
        print(f"  Trauma Sensitivity: {self.best_genome.trauma_sensitivity:.3f}")
        print(f"  Saved to: best_genome.json")
        
        # Save evolution log
        self.save_evolution_log()
    
    def save_evolution_log(self):
        """Save fitness history to CSV"""
        with open('evolution_log.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best', 'avg', 'worst'])
            writer.writeheader()
            writer.writerows(self.fitness_history)
        print(f"📊 Evolution log saved to: evolution_log.csv\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolve VNN trading strategies')
    parser.add_argument('--generations', type=int, default=30, help='Number of generations')
    parser.add_argument('--population', type=int, default=15, help='Population size')
    parser.add_argument('--test-nightmare', action='store_true', help='Test on nightmare scenario')
    
    args = parser.parse_args()
    
    if args.test_nightmare:
        print("🔥 Testing best genome on nightmare scenario...")
        genome = VNNGenome.load("best_genome.json")
        gen = NightmareMarketGenerator()
        prices, vols = gen.generate_mixed_nightmare(1.6, 1000)
        
        vnn = DualNeuronVNN(genome, prices, vols)
        vnn.run()
        fitness = vnn.calculate_fitness()
        
        print(f"Nightmare Survival Fitness: {fitness:.2f}")
        print(f"Final Equity: ${vnn.equity_curve[-1]:.2f}")
        print(f"Trades: {len(vnn.trades)}, Rogue: {vnn.rogue_trades}")
    else:
        school = EvolutionarySchool(
            population_size=args.population,
            elite_count=3,
            mutation_rate=0.15,
            nightmare_ratio=0.30
        )
        school.evolve(generations=args.generations)
