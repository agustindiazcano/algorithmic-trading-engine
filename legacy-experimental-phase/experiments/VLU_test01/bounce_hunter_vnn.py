import numpy as np
from typing import Dict, List, Tuple
import json

# --- MEAN REVERSION HUNTER VNN ---
class BounceHunterVNN:
    """
    VNN especializada en REBOTES en mercados bajistas
    Detecta divergencias de agotamiento y pesca suelos estructurales
    """
    
    def __init__(self, genome, prices: np.ndarray, volumes: np.ndarray):
        self.genome = genome
        self.prices = prices
        self.volumes = volumes
        self.n_ticks = len(prices)
        
        # Estado de la neurona
        self.center = float(prices[0])
        self.radius = float(prices[0]) * genome.radius_base
        
        # Indicadores técnicos
        self.rsi_history = []
        self.volume_ma = []
        
        # Trading state
        self.signals = np.zeros(self.n_ticks)
        self.equity_curve = [10000.0]
        self.cash = 10000.0
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_tick = 0
        self.trades = []
        
    def calculate_rsi(self, prices_window, period=14):
        """Calcula RSI"""
        if len(prices_window) < period + 1:
            return 50.0
        
        deltas = np.diff(prices_window)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_exhaustion_divergence(self, i: int) -> bool:
        """
        Versión MEJORADA (Z-Score + Elipsoide):
        1. Precio en zona de compra (tercio inferior del elipsoide).
        2. Forma geométrica de REBOTE (V-shape) detectada con Z-Score.
        """
        if i < 30:
            return False
            
        current_p = float(self.prices[i])
        
        # 1. Filtro de Zona: ¿Estamos "baratos"?
        lower_third_threshold = self.center - (self.radius * 0.33)
        if current_p > lower_third_threshold:
            return False
            
        # 2. Filtro de Forma (Z-Score): ¿Parece un rebote?
        # Ventana de 5 velas
        window = self.prices[i-4:i+1] # [t-4, t-3, t-2, t-1, t]
        if len(window) < 5: return False
        
        # Normalización Z-Score (Invariante a escala)
        w_mean = np.mean(window)
        w_std = np.std(window) + 1e-6
        z_window = (window - w_mean) / w_std
        
        # Patrón de "V-Shape" o "Cuchillo frenando" ideal (Z-Score aproximado)
        # [Alto, Bajo, MasBajo, Bajo, Alto] -> Rebote clásico en V
        # O simplemente: [Bajando, Bajando, Fondo, Subiendo]
        
        # Chequeo simple de geometría de rebote:
        # El último precio debe ser mayor que el penúltimo (empezó a subir)
        # Y el antepenúltimo debe ser bajo.
        is_bouncing = (z_window[-1] > z_window[-2]) and (z_window[-3] < z_window[-1])
        
        # Opcional: Calcular distancia a un patrón ideal si quisiéramos ser estrictos
        # v_shape = np.array([1, 0, -2, 0, 1], dtype=float)
        # v_shape = (v_shape - np.mean(v_shape)) / np.std(v_shape)
        # dist = np.linalg.norm(z_window - v_shape)
        # if dist < 2.0: return True
        
        return is_bouncing
    
    def should_take_profit(self, i: int) -> bool:
        """
        Reglas de salida INTELIGENTES con parámetros EVOLUCIONABLES
        """
        if self.position == 0:
            return False
        
        current_p = float(self.prices[i])
        
        # Salida 1: Precio tocó el centro (objetivo geométrico) - EVOLUCIONABLE
        if current_p >= self.center * (1 - self.genome.take_profit_tolerance):
            return True
        
        # Salida 2: Stop loss - EVOLUCIONABLE
        if current_p < self.entry_price * (1 - self.genome.stop_loss_pct):
            return True
        
        # Salida 3: Trailing stop - EVOLUCIONABLE
        profit_pct = (current_p / self.entry_price) - 1
        if profit_pct > self.genome.trailing_stop_trigger:  # Trigger evolucionable (1-5%)
            # Si cae por debajo del piso evolucionable (0.3-2%)
            if current_p < self.entry_price * (1 + self.genome.trailing_stop_floor):
                return True
        
        # Salida 4: Momentum reversal (RSI sobrecompra) - EVOLUCIONABLE
        if i >= self.entry_tick + 3:
            current_rsi = self.calculate_rsi(self.prices[max(0, i-20):i+1])
            if current_rsi > self.genome.rsi_overbought_threshold:  # Threshold evolucionable (65-80)
                return True
        
        # Salida 5: Volume spike - EVOLUCIONABLE
        if i >= self.entry_tick + 2:
            current_vol = self.volumes[i]
            avg_vol = np.mean(self.volumes[i-10:i])
            if current_vol > avg_vol * self.genome.volume_spike_multiplier:  # Multiplier evolucionable (1.5-4x)
                # Solo salir si el precio no está subiendo fuerte
                if current_p < self.prices[i-1] * 1.01:
                    return True
        
        return False
    
    def run(self):
        """Ejecuta la simulación completa"""
        for i in range(30, self.n_ticks):
            self._update_neuron(i)
            signal = self._generate_signal(i)
            self._execute_trade(i, signal)
            self.signals[i] = signal
        
        return self.signals
    
    def _update_neuron(self, i: int):
        """Actualiza centro y radio de la neurona"""
        current_p = float(self.prices[i])
        volat = float(np.std(self.prices[i-20:i]))
        
        # Respiración adaptativa (más sensible en suelos)
        target_r = (current_p * self.genome.radius_base) + (volat * self.genome.volatility_multiplier)
        self.radius = 0.9 * self.radius + 0.1 * target_r
        
        # Centro con inercia asimétrica
        # Lento para subir (no perseguir rallies), rápido para bajar (seguir caídas)
        if current_p < self.center:
            # Caída: inercia baja (seguir rápido)
            inertia = self.genome.inertia_down
        else:
            # Subida: inercia alta (no perseguir)
            inertia = self.genome.inertia_up
        
        self.center = inertia * self.center + (1 - inertia) * current_p
    
    def _generate_signal(self, i: int) -> int:
        """Genera señal de trading"""
        # Si ya estamos en posición, solo evaluar salida
        if self.position > 0:
            if self.should_take_profit(i):
                return -1  # SELL
            return 0  # HOLD
        
        # Si no hay posición, buscar entrada
        if self.detect_exhaustion_divergence(i):
            return 1  # BUY
        
        return 0  # WAIT
    
    def _execute_trade(self, i: int, signal: int):
        """Ejecuta trade y actualiza equity"""
        current_p = float(self.prices[i])
        fee = 0.001
        
        if signal == 1 and self.cash > 10:
            # COMPRA (entrada en rebote)
            shares = (self.cash * 0.99) / current_p
            self.position = shares
            self.cash -= shares * current_p * (1 + fee)
            self.entry_price = current_p
            self.entry_tick = i
            self.trades.append({'tick': i, 'type': 'buy', 'price': current_p})
        
        elif signal == -1 and self.position > 0:
            # VENTA (take profit / stop loss)
            revenue = self.position * current_p * (1 - fee)
            self.cash += revenue
            
            # Registrar trade completo
            hold_time = i - self.entry_tick
            profit_pct = ((current_p / self.entry_price) - 1) * 100
            
            self.trades.append({
                'tick': i,
                'type': 'sell',
                'price': current_p,
                'hold_time': hold_time,
                'profit_pct': profit_pct
            })
            
            self.position = 0
        
        equity = self.cash + self.position * current_p
        self.equity_curve.append(equity)
    
    def calculate_fitness(self) -> float:
        """
        Fitness = Return % total
        Simple, directo, intuitivo
        
        Fitness 15.5 = +15.5% return
        Fitness -5.0 = -5.0% loss
        """
        final_equity = self.equity_curve[-1]
        return_pct = ((final_equity / 10000) - 1) * 100
        
        # Penalizar inactividad severa
        complete_trades = len([t for t in self.trades if t['type'] == 'sell'])
        if complete_trades == 0:
            return -100  # Sin trades = muy malo
        
        return return_pct


# --- GENOMA PARA BOUNCE HUNTER ---
class BounceHunterGenome:
    """Genoma especializado para mean reversion"""
    
    def __init__(self, random_init: bool = True):
        if random_init:
            # Parámetros AGRESIVOS
            self.radius_base = np.random.uniform(0.08, 0.20)
            self.volatility_multiplier = np.random.uniform(1.5, 4.0)
            
            # Inercia asimétrica
            self.inertia_up = np.random.uniform(0.95, 0.99)
            self.inertia_down = np.random.uniform(0.65, 0.85)
            
            # Detección
            self.volume_exhaustion_threshold = np.random.uniform(0.6, 0.95)
            
            # Exit rules EVOLUCIONABLES
            self.take_profit_tolerance = np.random.uniform(0.005, 0.025)  # Geometric TP
            self.stop_loss_pct = np.random.uniform(0.02, 0.05)  # Stop loss
            
            # Trailing stop (evolucionable)
            self.trailing_stop_trigger = np.random.uniform(0.01, 0.05)  # 1-5% ganancia para activar
            self.trailing_stop_floor = np.random.uniform(0.003, 0.02)  # 0.3-2% piso
            
            # RSI reversal (evolucionable)
            self.rsi_overbought_threshold = np.random.uniform(65, 80)  # 65-80
            
            # Volume spike (evolucionable)
            self.volume_spike_multiplier = np.random.uniform(1.5, 4.0)  # 1.5-4x
        else:
            # Valores por defecto para genomas cargados
            self.radius_base = 0.14
            self.volatility_multiplier = 2.75
            self.inertia_up = 0.97
            self.inertia_down = 0.75
            self.volume_exhaustion_threshold = 0.775
            self.take_profit_tolerance = 0.015
            self.stop_loss_pct = 0.035
            self.trailing_stop_trigger = 0.03
            self.trailing_stop_floor = 0.01
            self.rsi_overbought_threshold = 72.5
            self.volume_spike_multiplier = 2.75
        
        self.fitness = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'radius_base': float(self.radius_base),
            'volatility_multiplier': float(self.volatility_multiplier),
            'inertia_up': float(self.inertia_up),
            'inertia_down': float(self.inertia_down),
            'volume_exhaustion_threshold': float(self.volume_exhaustion_threshold),
            'take_profit_tolerance': float(self.take_profit_tolerance),
            'stop_loss_pct': float(self.stop_loss_pct),
            'trailing_stop_trigger': float(self.trailing_stop_trigger),
            'trailing_stop_floor': float(self.trailing_stop_floor),
            'rsi_overbought_threshold': float(self.rsi_overbought_threshold),
            'volume_spike_multiplier': float(self.volume_spike_multiplier),
            'fitness': float(self.fitness)
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        genome = cls(random_init=False)
        for key, value in data.items():
            setattr(genome, key, value)
        
        # Backward compatibility: si faltan parámetros nuevos, usar defaults
        if not hasattr(genome, 'trailing_stop_trigger'):
            genome.trailing_stop_trigger = 0.03
        if not hasattr(genome, 'trailing_stop_floor'):
            genome.trailing_stop_floor = 0.01
        if not hasattr(genome, 'rsi_overbought_threshold'):
            genome.rsi_overbought_threshold = 72.5
        if not hasattr(genome, 'volume_spike_multiplier'):
            genome.volume_spike_multiplier = 2.75
        
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
        """Mutación con bounds específicos (11 parámetros evolucionables)"""
        float_bounds = {
            'radius_base': (0.08, 0.20),
            'volatility_multiplier': (1.5, 4.0),
            'inertia_up': (0.95, 0.99),
            'inertia_down': (0.65, 0.85),
            'volume_exhaustion_threshold': (0.6, 0.95),
            'take_profit_tolerance': (0.005, 0.025),
            'stop_loss_pct': (0.02, 0.05),
            # Exit rules evolucionables
            'trailing_stop_trigger': (0.01, 0.05),
            'trailing_stop_floor': (0.003, 0.02),
            'rsi_overbought_threshold': (65, 80),
            'volume_spike_multiplier': (1.5, 4.0),
        }
        
        genes = list(float_bounds.keys())
        
        for gene in genes:
            if np.random.random() < mutation_rate:
                current = getattr(self, gene)
                lo, hi = float_bounds[gene]
                sigma = (hi - lo) * 0.1
                mutated = current + np.random.normal(0, sigma)
                setattr(self, gene, float(np.clip(mutated, lo, hi)))
    
    @staticmethod
    def crossover(parent1, parent2):
        """Crossover entre dos genomas (11 genes)"""
        child = BounceHunterGenome(random_init=False)
        genes = [
            'radius_base', 'volatility_multiplier', 'inertia_up', 'inertia_down',
            'volume_exhaustion_threshold', 'take_profit_tolerance', 'stop_loss_pct',
            'trailing_stop_trigger', 'trailing_stop_floor', 
            'rsi_overbought_threshold', 'volume_spike_multiplier'
        ]
        
        for gene in genes:
            parent = parent1 if np.random.random() < 0.5 else parent2
            setattr(child, gene, getattr(parent, gene))
        
        return child


if __name__ == "__main__":
    # Test básico
    print("🧪 Testing Bounce Hunter VNN...")
    
    # Datos sintéticos de bear market con rebotes
    prices = np.linspace(100, 80, 500)  # Caída
    prices += np.sin(np.arange(500) * 0.3) * 3  # Rebotes
    volumes = np.random.normal(1e6, 2e5, 500)
    volumes = np.abs(volumes)
    
    genome = BounceHunterGenome()
    vnn = BounceHunterVNN(genome, prices, volumes)
    signals = vnn.run()
    fitness = vnn.calculate_fitness()
    
    print(f"Fitness: {fitness:.2f}")
    print(f"Trades: {len(vnn.trades)}")
    print(f"Final Equity: ${vnn.equity_curve[-1]:.2f}")
