import numpy as np
import requests
import json
from evolutionary_vnn import VNNGenome, DualNeuronVNN
from typing import List, Tuple
import csv

class RealMarketSchool:
    """
    Evolution focused 100% on REAL market data
    No synthetic nightmares - pure XRP trading
    """
    
    def __init__(self, population_size: int = 20, elite_count: int = 3, 
                 mutation_rate: float = 0.15):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        
        self.population: List[VNNGenome] = []
        self.generation = 0
        self.best_genome = None
        self.fitness_history = []
        
    def load_real_data(self, limit: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch real XRP data from Binance"""
        print(f"📡 Fetching {limit} candles from Binance...")
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": "XRPUSDT", "interval": "1h", "limit": limit}
            response = requests.get(url, params=params)
            data = response.json()
            
            prices = np.array([float(k[4]) for k in data])
            volumes = np.array([float(k[5]) for k in data]) + 1.0
            
            print(f"✅ Loaded {len(prices)} real market ticks")
            return prices, volumes
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def generate_training_scenarios(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Generate ONLY real market scenarios
        Uses different time windows of XRP data
        """
        scenarios = []
        
        # Full dataset (last 1000 hours)
        prices, volumes = self.load_real_data(1000)
        if prices is not None:
            scenarios.append((prices, volumes, "xrp_full_1000h"))
        
        # Recent volatile period (last 500 hours)
        prices, volumes = self.load_real_data(500)
        if prices is not None:
            scenarios.append((prices, volumes, "xrp_recent_500h"))
        
        # Very recent (last 200 hours)
        prices, volumes = self.load_real_data(200)
        if prices is not None:
            scenarios.append((prices, volumes, "xrp_recent_200h"))
        
        return scenarios
    
    def evaluate_genome(self, genome: VNNGenome, scenarios: List) -> float:
        """Evaluate genome across all real market scenarios"""
        total_fitness = 0.0
        
        for prices, volumes, name in scenarios:
            vnn = DualNeuronVNN(genome, prices, volumes)
            vnn.run()
            fitness = vnn.calculate_fitness()
            total_fitness += fitness
        
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
            
            child = VNNGenome.crossover(parent1, parent2)
            child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
    
    def evolve(self, generations: int = 50):
        """Run evolution for N generations"""
        print(f"\n🔥 REAL MARKET EVOLUTION: {generations} generations\n")
        
        self.initialize_population()
        scenarios = self.generate_training_scenarios()
        
        print(f"📚 Training on {len(scenarios)} REAL market scenarios:")
        for _, _, name in scenarios:
            print(f"   - {name}")
        print()
        
        for gen in range(generations):
            self.generation = gen
            print(f"🧬 Generation {gen+1}/{generations}")
            
            for i, genome in enumerate(self.population):
                fitness = self.evaluate_genome(genome, scenarios)
                genome.fitness = fitness
                
                if (i + 1) % 5 == 0:
                    print(f"   Evaluated {i+1}/{self.population_size} genomes...", end='\r')
            
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
            
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
                self.best_genome.save("best_genome_real.json")
                print(f"   💎 New best genome saved! Fitness: {best.fitness:.2f}")
            
            if gen < generations - 1:
                self.breed_next_generation()
            
            print()
        
        print("\n" + "="*60)
        print("🏆 REAL MARKET EVOLUTION COMPLETE")
        print("="*60)
        print(f"Best Fitness: {self.best_genome.fitness:.2f}")
        if self.fitness_history[0]['best'] > 0:
            improvement = self.fitness_history[-1]['best'] / self.fitness_history[0]['best']
            print(f"Improvement: {improvement:.2f}x")
        print(f"\nBest Genome Parameters:")
        print(f"  Anchor Inertia: {self.best_genome.anchor_inertia:.3f}")
        print(f"  Hunter Inertia: {self.best_genome.hunter_inertia:.3f}")
        print(f"  Trauma Sensitivity: {self.best_genome.trauma_sensitivity:.3f}")
        print(f"  Saved to: best_genome_real.json")
        
        self.save_evolution_log()
    
    def save_evolution_log(self):
        """Save fitness history to CSV"""
        with open('evolution_log_real.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best', 'avg', 'worst'])
            writer.writeheader()
            writer.writerows(self.fitness_history)
        print(f"📊 Evolution log saved to: evolution_log_real.csv\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolve VNN on REAL market data')
    parser.add_argument('--generations', type=int, default=30, help='Number of generations')
    parser.add_argument('--population', type=int, default=15, help='Population size')
    
    args = parser.parse_args()
    
    school = RealMarketSchool(
        population_size=args.population,
        elite_count=3,
        mutation_rate=0.15
    )
    school.evolve(generations=args.generations)
