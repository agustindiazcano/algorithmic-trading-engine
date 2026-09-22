import numpy as np
import requests
from bounce_hunter_vnn import BounceHunterVNN, BounceHunterGenome
from typing import List, Tuple
import csv

class BounceHunterSchool:
    """Escuela evolutiva para VNN de rebotes"""
    
    def __init__(self, population_size: int = 20, elite_count: int = 3, mutation_rate: float = 0.20):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate  # 20% para forzar exploración
        
        self.population: List[BounceHunterGenome] = []
        self.generation = 0
        self.best_genome = None
        self.fitness_history = []
    
    def load_real_data(self, limit: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Descarga datos reales de XRP"""
        print(f"📡 Descargando {limit} velas de XRP...")
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
    
    def generate_training_scenarios(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Genera escenarios de entrenamiento (solo datos reales)"""
        scenarios = []
        
        # Diferentes ventanas temporales
        for limit, name in [(1000, "xrp_1000h"), (500, "xrp_500h"), (300, "xrp_300h")]:
            prices, volumes = self.load_real_data(limit)
            if prices is not None:
                scenarios.append((prices, volumes, name))
        
        return scenarios
    
    def evaluate_genome(self, genome: BounceHunterGenome, scenarios: List) -> float:
        """Evalúa genoma en todos los escenarios"""
        total_fitness = 0.0
        
        for prices, volumes, name in scenarios:
            vnn = BounceHunterVNN(genome, prices, volumes)
            vnn.run()
            fitness = vnn.calculate_fitness()
            total_fitness += fitness
        
        return total_fitness / len(scenarios)
    
    def initialize_population(self):
        """Crea población inicial"""
        print(f"🧬 Inicializando {self.population_size} genomas...")
        self.population = [BounceHunterGenome(random_init=True) for _ in range(self.population_size)]
    
    def select_elite(self):
        """Selecciona los mejores"""
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        return self.population[:self.elite_count]
    
    def tournament_selection(self, tournament_size: int = 3) -> BounceHunterGenome:
        """Selección por torneo"""
        candidates = np.random.choice(self.population, tournament_size, replace=False)
        return max(candidates, key=lambda g: g.fitness)
    
    def breed_next_generation(self):
        """Crea nueva generación"""
        elite = self.select_elite()
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child = BounceHunterGenome.crossover(parent1, parent2)
            child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
    
    def evolve(self, generations: int = 50):
        """Ejecuta evolución"""
        print(f"\n🔥 BOUNCE HUNTER EVOLUTION: {generations} generaciones\n")
        print("🎯 Objetivo: Detectar divergencias de agotamiento y pescar rebotes")
        print("📊 Fitness: Profit / Tiempo en posición (francotirador)\n")
        
        self.initialize_population()
        scenarios = self.generate_training_scenarios()
        
        print(f"📚 Entrenando en {len(scenarios)} escenarios reales:")
        for _, _, name in scenarios:
            print(f"   - {name}")
        print()
        
        for gen in range(generations):
            self.generation = gen
            print(f"🧬 Generación {gen+1}/{generations}")
            
            for i, genome in enumerate(self.population):
                fitness = self.evaluate_genome(genome, scenarios)
                genome.fitness = fitness
                
                if (i + 1) % 5 == 0:
                    print(f"   Evaluando {i+1}/{self.population_size}...", end='\r')
            
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
                self.best_genome.save("bounce_hunter_best.json")
                print(f"   💎 Nuevo mejor genoma! Fitness: {best.fitness:.2f}")
            
            if gen < generations - 1:
                self.breed_next_generation()
            
            print()
        
        print("\n" + "="*60)
        print("🏆 EVOLUCIÓN COMPLETADA")
        print("="*60)
        print(f"Best Fitness: {self.best_genome.fitness:.2f}")
        if self.fitness_history[0]['best'] != 0:
            improvement = self.fitness_history[-1]['best'] / max(self.fitness_history[0]['best'], 0.01)
            print(f"Mejora: {improvement:.2f}x")
        
        print(f"\n📊 Parámetros del mejor genoma:")
        print(f"  Radio Base: {self.best_genome.radius_base:.3f}")
        print(f"  Inercia Up: {self.best_genome.inertia_up:.3f}")
        print(f"  Inercia Down: {self.best_genome.inertia_down:.3f}")
        print(f"  Vol Exhaustion: {self.best_genome.volume_exhaustion_threshold:.3f}")
        print(f"  Take Profit: {self.best_genome.take_profit_tolerance:.3f}")
        print(f"  Stop Loss: {self.best_genome.stop_loss_pct:.3f}")
        print(f"  Max Hold: {self.best_genome.max_hold_time}h")
        print(f"\n  Guardado en: bounce_hunter_best.json")
        
        self.save_evolution_log()
    
    def save_evolution_log(self):
        """Guarda log de evolución"""
        with open('bounce_hunter_evolution.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best', 'avg', 'worst'])
            writer.writeheader()
            writer.writerows(self.fitness_history)
        print(f"📊 Log guardado: bounce_hunter_evolution.csv\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolucionar Bounce Hunter VNN')
    parser.add_argument('--generations', type=int, default=50, help='Generaciones')
    parser.add_argument('--population', type=int, default=20, help='Tamaño población')
    
    args = parser.parse_args()
    
    school = BounceHunterSchool(
        population_size=args.population,
        elite_count=3,
        mutation_rate=0.20  # Alta mutación para explorar agresivamente
    )
    school.evolve(generations=args.generations)
