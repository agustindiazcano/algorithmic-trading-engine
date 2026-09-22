"""
Entrenamiento específico para SCALPING (1 minuto)
Parámetros ajustados para timeframes cortos
"""
import numpy as np
from bounce_hunter_vnn import BounceHunterGenome, BounceHunterVNN
from optimizers.pso_optimizer import PSOOptimizer
from optimizers.cma_es_optimizer import CMAESOptimizer
import requests
import csv
from typing import List
import argparse

class ScalpingGenome(BounceHunterGenome):
    """Genoma optimizado para SCALPING (1 minuto)"""
    
    def __init__(self, random_init: bool = True):
        if random_init:
            # PARÁMETROS PARA SCALPING (mucho más pequeños)
            self.radius_base = np.random.uniform(0.005, 0.02)  # 0.5-2% (vs 8-20%)
            self.volatility_multiplier = np.random.uniform(1.2, 2.5)  # Menos volátil
            
            # Inercia más rápida (scalping = reaccionar rápido)
            self.inertia_up = np.random.uniform(0.90, 0.95)  # Más rápido
            self.inertia_down = np.random.uniform(0.75, 0.85)
            
            # Detección
            self.volume_exhaustion_threshold = np.random.uniform(0.6, 0.95)
            
            # Exit rules para SCALPING
            self.take_profit_tolerance = np.random.uniform(0.002, 0.01)  # 0.2-1% TP
            self.stop_loss_pct = np.random.uniform(0.005, 0.015)  # 0.5-1.5% stop
            
            # Trailing stop más tight
            self.trailing_stop_trigger = np.random.uniform(0.003, 0.015)  # 0.3-1.5%
            self.trailing_stop_floor = np.random.uniform(0.001, 0.005)  # 0.1-0.5%
            
            # RSI más sensible para scalping
            self.rsi_overbought_threshold = np.random.uniform(60, 75)  # Más bajo
            
            # Volume spike más sensible
            self.volume_spike_multiplier = np.random.uniform(2.0, 4.0)
        else:
            # Defaults para scalping
            self.radius_base = 0.0125  # 1.25%
            self.volatility_multiplier = 1.85
            self.inertia_up = 0.925
            self.inertia_down = 0.80
            self.volume_exhaustion_threshold = 0.775
            self.take_profit_tolerance = 0.006  # 0.6%
            self.stop_loss_pct = 0.01  # 1%
            self.trailing_stop_trigger = 0.009  # 0.9%
            self.trailing_stop_floor = 0.003  # 0.3%
            self.rsi_overbought_threshold = 67.5
            self.volume_spike_multiplier = 3.0
        
        self.fitness = 0.0

class ScalpingTrainer:
    def __init__(self, population_size=40):
        self.population_size = population_size
        self.pso_size = population_size // 2
        self.cma_size = population_size - self.pso_size
        
        # Initialize optimizers con bounds de SCALPING
        self.pso = PSOOptimizer(dim=11, n_particles=self.pso_size, bounds=[
            (0.005, 0.02),   # radius_base
            (1.2, 2.5),      # volatility_multiplier
            (0.90, 0.95),    # inertia_up
            (0.75, 0.85),    # inertia_down
            (0.6, 0.95),     # volume_exhaustion
            (0.002, 0.01),   # take_profit
            (0.005, 0.015),  # stop_loss
            (0.003, 0.015),  # trailing_trigger
            (0.001, 0.005),  # trailing_floor
            (60, 75),        # rsi_threshold
            (2.0, 4.0)       # volume_spike
        ])
        
        # CMA-ES con mean para scalping
        self.cma = CMAESOptimizer(dim=11, sigma=0.3, population_size=self.cma_size)
        self.cma.mean = np.array([0.0125, 1.85, 0.925, 0.80, 0.775, 0.006, 0.01, 0.009, 0.003, 67.5, 3.0])
        
        self.best_genome = None
        self.best_fitness = -np.inf
        self.evolution_log = []
    
    def fetch_data_1m(self, limit=1000):
        """Descargar datos de 1 minuto"""
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "XRPUSDT", "interval": "1m", "limit": limit}
        response = requests.get(url, params=params)
        klines = response.json()
        
        prices = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])
        return prices, volumes
    
    def array_to_genome(self, x: np.ndarray) -> ScalpingGenome:
        genome = ScalpingGenome(random_init=False)
        genome.radius_base = np.clip(x[0], 0.005, 0.02)
        genome.volatility_multiplier = np.clip(x[1], 1.2, 2.5)
        genome.inertia_up = np.clip(x[2], 0.90, 0.95)
        genome.inertia_down = np.clip(x[3], 0.75, 0.85)
        genome.volume_exhaustion_threshold = np.clip(x[4], 0.6, 0.95)
        genome.take_profit_tolerance = np.clip(x[5], 0.002, 0.01)
        genome.stop_loss_pct = np.clip(x[6], 0.005, 0.015)
        genome.trailing_stop_trigger = np.clip(x[7], 0.003, 0.015)
        genome.trailing_stop_floor = np.clip(x[8], 0.001, 0.005)
        genome.rsi_overbought_threshold = np.clip(x[9], 60, 75)
        genome.volume_spike_multiplier = np.clip(x[10], 2.0, 4.0)
        return genome
    
    def genome_to_array(self, genome: ScalpingGenome) -> np.ndarray:
        return np.array([
            genome.radius_base, genome.volatility_multiplier,
            genome.inertia_up, genome.inertia_down,
            genome.volume_exhaustion_threshold, genome.take_profit_tolerance,
            genome.stop_loss_pct, genome.trailing_stop_trigger,
            genome.trailing_stop_floor, genome.rsi_overbought_threshold,
            genome.volume_spike_multiplier
        ])
    
    def evaluate_genome(self, genome: ScalpingGenome, scenarios: List) -> float:
        total_fitness = 0.0
        for prices, volumes in scenarios:
            vnn = BounceHunterVNN(genome, prices, volumes)
            vnn.run()
            fitness = vnn.calculate_fitness()
            total_fitness += fitness
        return total_fitness / len(scenarios)
    
    def train(self, generations=100):
        print(f"\n🔥 SCALPING TRAINING (1 minuto): {generations} generaciones\n")
        print("🧠 Estrategia:")
        print(f"   PSO: {self.pso_size} partículas")
        print(f"   CMA-ES: {self.cma_size} individuos\n")
        
        # Descargar 3 escenarios de 1 minuto
        print("📡 Descargando datos de 1 minuto...")
        scenarios = []
        for limit in [1000, 500, 300]:
            prices, volumes = self.fetch_data_1m(limit)
            scenarios.append((prices, volumes))
            print(f"✅ {len(prices)} velas descargadas")
        
        print(f"📚 Entrenando en {len(scenarios)} escenarios\n")
        
        for gen in range(1, generations + 1):
            print(f"🧬 Generación {gen}/{generations}")
            
            # PSO ask
            pso_arrays = self.pso.ask()
            pso_genomes = [self.array_to_genome(x) for x in pso_arrays]
            
            # CMA-ES ask
            cma_arrays = self.cma.ask()
            cma_genomes = [self.array_to_genome(x) for x in cma_arrays]
            
            # Evaluate all
            all_genomes = pso_genomes + cma_genomes
            print(f"   Evaluando {len(all_genomes)}/{len(all_genomes)}...", end='\r')
            
            fitnesses = [self.evaluate_genome(g, scenarios) for g in all_genomes]
            
            # Update genomes
            for g, f in zip(all_genomes, fitnesses):
                g.fitness = f
            
            # Tell optimizers
            pso_fitnesses = fitnesses[:self.pso_size]
            cma_fitnesses = fitnesses[self.pso_size:]
            
            self.pso.tell(pso_fitnesses)
            self.cma.tell(cma_arrays, cma_fitnesses)  # CMA-ES needs solutions + fitnesses
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best = all_genomes[gen_best_idx]
            
            if gen_best.fitness > self.best_fitness:
                self.best_fitness = gen_best.fitness
                self.best_genome = gen_best
                print(f"   💎 Nuevo mejor! Fitness: {self.best_fitness:.2f}% (de {'PSO' if gen_best_idx < self.pso_size else 'CMA-ES'})")
            
            # Stats
            avg_fitness = np.mean(fitnesses)
            pso_best = max(pso_fitnesses)
            cma_best = max(cma_fitnesses)
            
            print(f"\n   Best: {self.best_fitness:.2f}% | Avg: {avg_fitness:.2f}%")
            print(f"   PSO Best: {pso_best:.2f}% | CMA Best: {cma_best:.2f}%")
            print(f"   PSO Diversity: {self.pso.get_diversity():.4f} | CMA σ: {self.cma.sigma:.4f}\n")
            
            # Log
            self.evolution_log.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
                'pso_best': pso_best,
                'cma_best': cma_best
            })
        
        # Save results
        print("\n" + "="*60)
        print("OPTIMIZACIÓN COMPLETADA")
        print("="*60)
        print(f"\nBest Fitness: {self.best_fitness:.2f}%")
        print(f"\n🧬 Mejor genoma (SCALPING):")
        print(f"  Radius: {self.best_genome.radius_base*100:.2f}%")
        print(f"  TP: {self.best_genome.take_profit_tolerance*100:.2f}%")
        print(f"  Stop: {self.best_genome.stop_loss_pct*100:.2f}%")
        print(f"  Trailing Trigger: {self.best_genome.trailing_stop_trigger*100:.2f}%")
        
        self.best_genome.save('scalping_1m_best.json')
        print(f"\n  Guardado en: scalping_1m_best.json")
        
        # Save log
        with open('scalping_1m_evolution.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best_fitness', 'avg_fitness', 'pso_best', 'cma_best'])
            writer.writeheader()
            writer.writerows(self.evolution_log)
        print(f"📊 Log guardado: scalping_1m_evolution.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--population', type=int, default=40)
    args = parser.parse_args()
    
    trainer = ScalpingTrainer(population_size=args.population)
    trainer.train(generations=args.generations)
