import numpy as np
import json
from typing import Dict, List, Tuple

# --- ADVERSARIAL MARKET GENERATOR ---
class NightmareMarketGenerator:
    """Generates synthetic nightmare scenarios to stress-test VNN organisms"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_flash_crash(self, base_price: float, duration: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        -30% drop in 1 minute, followed by 10 hours of brownian noise
        Returns: (prices, volumes)
        """
        ticks = duration + 600  # 1h crash + 10h recovery
        
        # Flash crash phase
        crash_depth = base_price * 0.30
        crash_curve = np.linspace(0, crash_depth, duration)
        crash_prices = base_price - crash_curve
        
        # Recovery noise phase
        recovery_base = crash_prices[-1]
        noise = np.random.normal(0, base_price * 0.02, 600)
        recovery_prices = recovery_base + np.cumsum(noise)
        
        prices = np.concatenate([crash_prices, recovery_prices])
        
        # Volume spikes during panic
        crash_vol = np.linspace(1e6, 5e6, duration)
        recovery_vol = np.abs(np.random.normal(1.5e6, 3e5, 600))
        volumes = np.concatenate([crash_vol, recovery_vol])
        
        return prices, volumes
    
    def generate_bull_trap(self, base_price: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        +20% pump followed by immediate -35% dump
        Returns: (prices, volumes)
        """
        ticks = 200
        
        # Pump phase (100 ticks)
        pump = np.linspace(0, base_price * 0.20, 100)
        pump_prices = base_price + pump
        
        # Dump phase (100 ticks)
        dump_start = pump_prices[-1]
        dump = np.linspace(0, base_price * 0.35, 100)
        dump_prices = dump_start - dump
        
        prices = np.concatenate([pump_prices, dump_prices])
        
        # Volume: low during pump, massive during dump
        pump_vol = np.abs(np.random.normal(8e5, 1e5, 100))
        dump_vol = np.abs(np.random.normal(4e6, 5e5, 100))
        volumes = np.concatenate([pump_vol, dump_vol])
        
        return prices, volumes
    
    def generate_choppy_hell(self, base_price: float, duration: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure noise, no trend, maximum slippage conditions
        Returns: (prices, volumes)
        """
        # High-frequency oscillation with no net movement
        noise = np.random.normal(0, base_price * 0.015, duration)
        prices = base_price + np.cumsum(noise)
        # Force mean reversion
        prices = prices - (np.mean(prices) - base_price)
        
        # Erratic volume (use normal instead of lognormal to avoid scale issues)
        volumes = np.abs(np.random.normal(1.2e6, 3e5, duration))
        
        return prices, volumes
    
    def generate_mixed_nightmare(self, base_price: float, total_duration: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combines all nightmare scenarios into one hellish market
        """
        scenarios = [
            self.generate_choppy_hell(base_price, 300),
            self.generate_flash_crash(base_price * 0.95, 60),
            self.generate_bull_trap(base_price * 0.70),
            self.generate_choppy_hell(base_price * 0.65, 300)
        ]
        
        all_prices = np.concatenate([s[0] for s in scenarios])
        all_volumes = np.concatenate([s[1] for s in scenarios])
        
        # Trim or pad to exact duration
        if len(all_prices) > total_duration:
            all_prices = all_prices[:total_duration]
            all_volumes = all_volumes[:total_duration]
        elif len(all_prices) < total_duration:
            pad_len = total_duration - len(all_prices)
            pad_p = np.random.normal(all_prices[-1], base_price * 0.01, pad_len)
            pad_v = np.abs(np.random.normal(1e6, 2e5, pad_len))
            all_prices = np.concatenate([all_prices, pad_p])
            all_volumes = np.concatenate([all_volumes, pad_v])
        
        return all_prices, all_volumes


# --- VNN GENOME (DNA ENCODING) ---
class VNNGenome:
    """Encodes parameters for dual-neuron VNN with trauma response"""
    
    def __init__(self, random_init: bool = True):
        if random_init:
            # Anchor Neuron (Slow/Heavy)
            self.anchor_inertia = np.random.uniform(0.95, 0.99)
            self.anchor_radius_base = np.random.uniform(0.06, 0.12)
            self.anchor_persistence = int(np.random.uniform(8, 15))
            
            # Hunter Neuron (Fast/Light)
            self.hunter_inertia = np.random.uniform(0.85, 0.95)
            self.hunter_radius_base = np.random.uniform(0.02, 0.05)
            self.hunter_persistence = int(np.random.uniform(2, 6))
            self.hunter_volume_threshold = np.random.uniform(-0.3, 0.3)
            
            # Trauma Response
            self.trauma_sensitivity = np.random.uniform(0.0, 1.0)
            self.trauma_radius_multiplier = np.random.uniform(1.5, 3.0)
            
            # Shared parameters
            self.pressure_threshold = np.random.uniform(0.8, 2.0)
            self.radius_learning_rate = np.random.uniform(0.02, 0.10)
        
        self.fitness = 0.0
        self.used_trauma_response = False
    
    def to_dict(self) -> Dict:
        return {
            'anchor_inertia': float(self.anchor_inertia),
            'anchor_radius_base': float(self.anchor_radius_base),
            'anchor_persistence': int(self.anchor_persistence),
            'hunter_inertia': float(self.hunter_inertia),
            'hunter_radius_base': float(self.hunter_radius_base),
            'hunter_persistence': int(self.hunter_persistence),
            'hunter_volume_threshold': float(self.hunter_volume_threshold),
            'trauma_sensitivity': float(self.trauma_sensitivity),
            'trauma_radius_multiplier': float(self.trauma_radius_multiplier),
            'pressure_threshold': float(self.pressure_threshold),
            'radius_learning_rate': float(self.radius_learning_rate),
            'fitness': float(self.fitness)
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        genome = cls(random_init=False)
        for key, value in data.items():
            setattr(genome, key, value)
        return genome
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def mutate(self, mutation_rate: float = 0.15):
        """Apply gaussian mutation to genes with safe sigmas and per-gene bounds"""
        
        float_bounds = {
            'anchor_inertia': (0.95, 0.99),
            'anchor_radius_base': (0.06, 0.12),
            'hunter_inertia': (0.85, 0.95),
            'hunter_radius_base': (0.02, 0.05),
            'hunter_volume_threshold': (-0.3, 0.3),
            'trauma_sensitivity': (0.0, 1.0),
            'trauma_radius_multiplier': (1.5, 3.0),
            'pressure_threshold': (0.8, 2.0),
            'radius_learning_rate': (0.02, 0.10),
        }
        
        int_bounds = {
            'anchor_persistence': (1, 30),
            'hunter_persistence': (1, 20),
        }
        
        genes = [
            'anchor_inertia', 'anchor_radius_base', 'anchor_persistence',
            'hunter_inertia', 'hunter_radius_base', 'hunter_persistence',
            'hunter_volume_threshold', 'trauma_sensitivity',
            'trauma_radius_multiplier', 'pressure_threshold', 'radius_learning_rate'
        ]
        
        for gene in genes:
            if np.random.random() < mutation_rate:
                current = getattr(self, gene)
                
                # --- ints ---
                if isinstance(current, int):
                    lo, hi = int_bounds.get(gene, (1, 10**9))
                    step_sigma = 2
                    mutated = current + int(np.random.normal(0, step_sigma))
                    setattr(self, gene, int(np.clip(mutated, lo, hi)))
                    continue
                
                # --- floats ---
                lo, hi = float_bounds.get(gene, (None, None))
                
                # sigma basada en el rango (estable) y SIEMPRE positiva
                if lo is not None and hi is not None:
                    sigma = max((hi - lo) * 0.08, 1e-6)   # 8% del rango
                else:
                    sigma = max(abs(float(current)) * 0.1, 1e-6)
                
                mutated = float(current) + np.random.normal(0.0, sigma)
                
                # clamp por gen si hay bounds
                if lo is not None and hi is not None:
                    mutated = float(np.clip(mutated, lo, hi))
                
                setattr(self, gene, mutated)
    
    @staticmethod
    def crossover(parent1, parent2):
        """Create offspring by mixing parent genes"""
        child = VNNGenome(random_init=False)
        genes = [
            'anchor_inertia', 'anchor_radius_base', 'anchor_persistence',
            'hunter_inertia', 'hunter_radius_base', 'hunter_persistence',
            'hunter_volume_threshold', 'trauma_sensitivity', 
            'trauma_radius_multiplier', 'pressure_threshold', 'radius_learning_rate'
        ]
        
        for gene in genes:
            # 50/50 chance from each parent
            parent = parent1 if np.random.random() < 0.5 else parent2
            setattr(child, gene, getattr(parent, gene))
        
        return child


# --- DUAL-NEURON VNN (SYMBIOTIC ARCHITECTURE) ---
class DualNeuronVNN:
    """
    Two-neuron system:
    - Anchor: Slow, heavy, defines macro trend and safety zone
    - Hunter: Fast, light, executes trades only within Anchor's zone
    """
    
    def __init__(self, genome: VNNGenome, prices: np.ndarray, volumes: np.ndarray):
        self.genome = genome
        self.prices = prices
        self.volumes = volumes
        self.n_ticks = len(prices)
        
        # Anchor Neuron State
        self.anchor_center = float(prices[0])
        self.anchor_radius = float(prices[0]) * genome.anchor_radius_base
        self.anchor_persistence = 0
        
        # Hunter Neuron State
        self.hunter_center = float(prices[0])
        self.hunter_radius = float(prices[0]) * genome.hunter_radius_base
        self.hunter_persistence = 0
        
        # Trauma State
        self.consecutive_losses = 0
        self.trauma_active = False
        self.trauma_countdown = 0
        
        # Trading State
        self.signals = np.zeros(self.n_ticks)
        self.equity_curve = [10000.0]
        self.cash = 10000.0
        self.position = 0.0
        self.trades = []
        self.rogue_trades = 0
        
    def run(self):
        """Execute full simulation"""
        for i in range(25, self.n_ticks):
            self._update_neurons(i)
            signal = self._generate_signal(i)
            self._execute_trade(i, signal)
            self.signals[i] = signal
        
        return self.signals
    
    def _update_neurons(self, i: int):
        """Update both Anchor and Hunter neurons"""
        current_p = float(self.prices[i])
        current_v = float(self.volumes[i])
        volat = float(np.std(self.prices[i-20:i]))
        
        # --- ANCHOR NEURON (Macro Trend) ---
        # Slow breathing
        target_r_anchor = (current_p * self.genome.anchor_radius_base) + (volat * 2.0)
        self.anchor_radius = (1 - self.genome.radius_learning_rate) * self.anchor_radius + \
                             self.genome.radius_learning_rate * target_r_anchor
        
        # Heavy inertia
        self.anchor_center = self.genome.anchor_inertia * self.anchor_center + \
                             (1 - self.genome.anchor_inertia) * current_p
        
        # Anchor persistence (slow to change trend)
        p_press_anchor = (current_p - self.anchor_center) / self.anchor_radius
        if abs(p_press_anchor) > self.genome.pressure_threshold:
            if p_press_anchor < -1.0:
                self.anchor_persistence += 1
            elif p_press_anchor > 1.0:
                self.anchor_persistence -= 1
        else:
            self.anchor_persistence = int(self.anchor_persistence * 0.5)
        
        # --- HUNTER NEURON (Entry Timing) ---
        # Fast breathing
        target_r_hunter = (current_p * self.genome.hunter_radius_base) + (volat * 1.5)
        
        # Trauma Response: Widen radius after losses
        if self.trauma_active:
            target_r_hunter *= self.genome.trauma_radius_multiplier
            self.trauma_countdown -= 1
            if self.trauma_countdown <= 0:
                self.trauma_active = False
        
        self.hunter_radius = (1 - self.genome.radius_learning_rate * 2) * self.hunter_radius + \
                             (self.genome.radius_learning_rate * 2) * target_r_hunter
        
        # Light inertia
        self.hunter_center = self.genome.hunter_inertia * self.hunter_center + \
                             (1 - self.genome.hunter_inertia) * current_p
        
        # Hunter persistence (fast to react)
        p_press_hunter = (current_p - self.hunter_center) / self.hunter_radius
        v_press = (current_v - np.mean(self.volumes[i-20:i])) / (np.std(self.volumes[i-20:i]) + 1e-9)
        
        if abs(p_press_hunter) > self.genome.pressure_threshold:
            # Volume confirmation
            if v_press > self.genome.hunter_volume_threshold:
                if p_press_hunter < -1.0:
                    self.hunter_persistence += 1
                elif p_press_hunter > 1.0:
                    self.hunter_persistence -= 1
        else:
            self.hunter_persistence = int(self.hunter_persistence * 0.6)
    
    def _generate_signal(self, i: int) -> int:
        """Generate trading signal with symbiosis check"""
        signal = 0
        
        # Hunter wants to trade
        if self.hunter_persistence >= self.genome.hunter_persistence:
            signal = 1
            self.hunter_persistence = 0
        elif self.hunter_persistence <= -self.genome.hunter_persistence:
            signal = -1
            self.hunter_persistence = 0
        
        # Symbiosis Check: Is Hunter inside Anchor's safety zone?
        if signal != 0:
            current_p = float(self.prices[i])
            anchor_upper = self.anchor_center + self.anchor_radius
            anchor_lower = self.anchor_center - self.anchor_radius
            
            # If Anchor says "bearish" (price below center), don't buy
            # If Anchor says "bullish" (price above center), don't sell
            if signal == 1 and current_p < self.anchor_center - self.anchor_radius * 0.5:
                # Rogue buy in bearish zone
                self.rogue_trades += 1
                return 0
            elif signal == -1 and current_p > self.anchor_center + self.anchor_radius * 0.5:
                # Rogue sell in bullish zone
                self.rogue_trades += 1
                return 0
        
        return signal
    
    def _execute_trade(self, i: int, signal: int):
        """Execute trade and track equity"""
        current_p = float(self.prices[i])
        fee = 0.001
        
        if signal == 1 and self.cash > 10:
            # Buy
            shares = (self.cash * 0.99) / current_p
            self.position += shares
            self.cash -= shares * current_p * (1 + fee)
            self.trades.append({'tick': i, 'type': 'buy', 'price': current_p})
        
        elif signal == -1 and self.position > 0:
            # Sell
            revenue = self.position * current_p * (1 - fee)
            
            # Track profit/loss for trauma response
            entry_price = self.trades[-1]['price'] if self.trades else current_p
            if current_p < entry_price:
                self.consecutive_losses += 1
                if self.consecutive_losses >= 3 and self.genome.trauma_sensitivity > 0.5:
                    # Activate trauma response
                    self.trauma_active = True
                    self.trauma_countdown = 5
                    self.genome.used_trauma_response = True
            else:
                self.consecutive_losses = 0
            
            self.cash += revenue
            self.position = 0
            self.trades.append({'tick': i, 'type': 'sell', 'price': current_p})
        
        equity = self.cash + self.position * current_p
        self.equity_curve.append(equity)
    
    def calculate_fitness(self) -> float:
        """
        Enhanced fitness function:
        - Rewards profit relative to drawdown
        - Rewards win rate
        - Penalizes overtrading (log punishment)
        - Penalizes rogue trades
        - Bonus for adaptive trauma response
        """
        equity_curve = np.array(self.equity_curve)
        final_equity = equity_curve[-1]
        profit = final_equity - 10000
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 1.0
        max_dd = max(max_dd, 0.01)  # Avoid division by zero
        
        # Win rate
        wins = sum(1 for t in self.trades if t['type'] == 'sell' and 
                   t['price'] > self.trades[self.trades.index(t)-1]['price'])
        total_trades = len([t for t in self.trades if t['type'] == 'sell'])
        win_rate = wins / max(total_trades, 1)
        
        # Trade penalty (anti-overtrading)
        trade_penalty = 1 / np.log(max(total_trades, 2))
        
        # Cooperation bonus (anti-rogue)
        valid_trades = max(total_trades - self.rogue_trades, 1)
        cooperation_bonus = valid_trades / max(total_trades, 1)
        
        # Base fitness
        fitness = (profit / max_dd) * win_rate * trade_penalty * cooperation_bonus
        
        # Trauma bonus
        if self.genome.used_trauma_response:
            fitness *= 1.15
        
        return max(fitness, -1000)  # Floor to avoid extreme negatives


if __name__ == "__main__":
    # Test nightmare generator
    print("🔥 Testing Nightmare Market Generator...")
    gen = NightmareMarketGenerator()
    
    prices, vols = gen.generate_flash_crash(1.6)
    print(f"Flash Crash: {len(prices)} ticks, drop: {prices[0]:.2f} -> {prices[59]:.2f}")
    
    prices, vols = gen.generate_bull_trap(1.6)
    print(f"Bull Trap: {len(prices)} ticks, peak: {np.max(prices):.2f}, bottom: {np.min(prices):.2f}")
    
    prices, vols = gen.generate_mixed_nightmare(1.6, 1000)
    print(f"Mixed Nightmare: {len(prices)} ticks, volatility: {np.std(prices):.4f}")
    
    # Test genome
    print("\n🧬 Testing VNN Genome...")
    genome = VNNGenome()
    print(f"Anchor Inertia: {genome.anchor_inertia:.3f}")
    print(f"Hunter Persistence: {genome.hunter_persistence}")
    
    genome.save("test_genome.json")
    loaded = VNNGenome.load("test_genome.json")
    print(f"Loaded genome matches: {genome.anchor_inertia == loaded.anchor_inertia}")
    
    # Test DualNeuronVNN
    print("\n🧠 Testing DualNeuronVNN...")
    test_prices, test_vols = gen.generate_mixed_nightmare(1.6, 500)
    vnn = DualNeuronVNN(genome, test_prices, test_vols)
    signals = vnn.run()
    fitness = vnn.calculate_fitness()
    print(f"Trades: {len(vnn.trades)}, Rogue: {vnn.rogue_trades}, Fitness: {fitness:.2f}")
    print(f"Final Equity: ${vnn.equity_curve[-1]:.2f}")

