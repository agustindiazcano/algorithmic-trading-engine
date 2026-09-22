import pandas as pd
import numpy as np
import requests
import random
import time
import matplotlib.pyplot as plt
import sys

# --- 1. CONFIGURACIÓN BASE (Valores Iniciales) ---
# Estos valores serán sobreescritos por el Optimizador Genético
DEFAULT_K = 1.0 
DEFAULT_ALPHA = 0.2639
SATURATION_THRESHOLD = 0.25  # Mercado Normal
EXTREME_TURBULENCE = 3.0     # Saturación Total (Reversión)

class DiazCanoFluidBot:
    def __init__(self, symbol='BTCUSDT', timeframe='1h', limit=1000):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.limit = limit
        self.df = pd.DataFrame()
        self.best_k = DEFAULT_K
        self.best_alpha = DEFAULT_ALPHA
    
    # --- A. MOTOR DE DATOS (Sin ccxt para máxima compatibilidad) ---
    def fetch_data(self):
        print(f"--- Conectando al oceano de liquidez de Binance ({self.symbol})...")
        base_url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': self.symbol,
            'interval': self.timeframe,
            'limit': self.limit
        }
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            # Estructura: [Open Time, Open, High, Low, Close, Volume, ...]
            self.df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
            ])
            self.df['close'] = pd.to_numeric(self.df['close'])
            self.df['volume'] = pd.to_numeric(self.df['volume'])
            self.df['high'] = pd.to_numeric(self.df['high'])
            self.df['low'] = pd.to_numeric(self.df['low'])
            self.df['ts'] = pd.to_datetime(self.df['open_time'], unit='ms')
            print(f" [OK] Descargadas {len(self.df)} velas.")
        except Exception as e:
            print(f" [ERROR] Error descargando datos: {e}")

    # --- B. FÍSICA DE FLUIDOS (Core Logic) ---
    def calculate_pressure(self, k, alpha):
        """
        Calcula el Índice de Presión de Diaz-Cano (DCPI) Normalizado.
        Nueva Lógica V2: Medir Potencia Relativa al Promedio Reciente.
        """
        # 1. Flujo Bruto: Potencia del movimiento actual
        delta_p = self.df['close'].diff().abs()
        flujo_bruto = delta_p * (self.df['volume'] ** alpha)
        
        # 2. Normalización Dinámica (Media Móvil de 24 periodos)
        # Esto nos dice qué es "mucho" flow en el contexto reciente
        flujo_promedio = flujo_bruto.rolling(window=24).mean()
        
        # 3. DCPI Relativo
        # DCPI = (Flujo Actual / Flujo Promedio) / K
        # Si da 1.0, el flujo es igual al promedio. Si da 4.0, es 4 veces más fuerte.
        pressure = (flujo_bruto / (flujo_promedio + 1e-9)) / k
        
        return pressure.fillna(0)

    # --- C. OPTIMIZADOR GENÉTICO ---
    def optimize_constants(self, generations=30, population_size=50):
        print(f"\n [ALGO] Iniciando Evolucion de Constantes para {self.symbol}...")
        
        # Población Inicial: Pares aleatorios de (k, alpha)
        # Rango K: 0.1 a 10.0 (Viscosidad)
        # Rango Alpha: 0.1 a 1.5 (Geometría Fractal)
        population = []
        for _ in range(population_size):
            population.append({
                'k': random.uniform(0.1, 5.0),
                'alpha': random.uniform(0.1, 0.8)
            })
            
        for gen in range(generations):
            scores = []
            for genome in population:
                profit = self.run_backtest(genome['k'], genome['alpha'])
                scores.append((profit, genome))
            
            # Ordenar por Profit (Fitness)
            scores.sort(key=lambda x: x[0], reverse=True)
            self.best_k = scores[0][1]['k']
            self.best_alpha = scores[0][1]['alpha']
            
            if gen % 5 == 0:
                print(f"   Gen {gen}: Mejor Profit {scores[0][0]:.2f}% | K={self.best_k:.4f} Alpha={self.best_alpha:.4f}")
            
            # Selección y Cruce (Elitismo: mantenemos el top 20%)
            survivors = scores[:int(population_size * 0.2)]
            new_population = [s[1] for s in survivors]
            
            while len(new_population) < population_size:
                parent = random.choice(survivors)[1]
                # Mutación
                child = {
                    'k': parent['k'] * random.uniform(0.8, 1.2),
                    'alpha': parent['alpha'] * random.uniform(0.9, 1.1)
                }
                # Límites físicos seguros
                child['k'] = max(0.01, child['k'])
                child['alpha'] = max(0.01, child['alpha'])
                new_population.append(child)
            
            population = new_population

        print(f" [FIN] Optimizacion Completada.")
        print(f"   CONSTANTES MAESTRAS: K={self.best_k:.4f}, Alpha={self.best_alpha:.4f}")

    # --- D. MOTOR DE SIMULACIÓN (Backtest Rápido) ---
    def run_backtest(self, k, alpha):
        dcpi = self.calculate_pressure(k, alpha)
        prices = self.df['close'].values
        # Detectar dirección local usando media móvil corta (Vectorizado)
        sma = self.df['close'].rolling(5).mean().ffill().values
        trend_up = prices > sma
        
        # Lógica Vectorizada Simplificada para Velocidad Genética
        # (En backtest real iteramos, pero para fitness aproximado usamos señales vectorizadas)
        
        # Señales de Entrada: Saturación Extrema (>3.0) en tendencia bajista
        entries = (dcpi > EXTREME_TURBULENCE) & (~trend_up)
        
        # Señales de Salida: Saturación Extrema (>3.0) en tendencia alcista 
        # O Pánico (Simplificado aquí con solo profit por vela para fitness rápido)
        exits = (dcpi > EXTREME_TURBULENCE) & (trend_up)
        
        # Simulación Aproximada de Profit (Suma de retornos en velas 'ideales')
        # Esto es un proxy de fitness mucho más rápido
        # Asumimos compra en 'entries' y venta en la siguiente 'exit' o reversión
        # Para el GA, maximizar la "calidad de entrada" es suficiente.
        
        # Calibración: Fitness = Suma de (Precio[Exit] - Precio[Entry]) 
        # Pero hacer match exacto vectorizado es complejo sin loop.
        # Volvemos al loop pero OPTIMIZADO con Numba o lógicas simples.
        # Mejor: Usamos loop simple pero pre-calculado.
        
        balance = 100.0
        position = 0
        entry_price = 0.0
        
        # Convertir a numpy para velocidad C
        dcpi_arr = dcpi.values
        
        for i in range(5, len(prices)):
            if position == 0:
                if entries[i]:
                    position = 1
                    entry_price = prices[i]
            elif position == 1:
                # Salida por señal o Stop Loss (5%)
                if exits[i] or (prices[i] < entry_price * 0.95):
                    position = 0
                    profit = (prices[i] - entry_price) / entry_price
                    balance *= (1.0 + profit)
                    
        return (balance - 100.0)

    # --- E. ANÁLISIS FINAL ---
    def analyze_market(self):
        print(f"\n [ANA] Analizando Mercado con Fisica Optimizada (DCPI Normalizado)...")
        self.df['DCPI'] = self.calculate_pressure(self.best_k, self.best_alpha)
        
        # Estados:
        # < 0.25: Laminar
        # > 1.0: Fuerte
        # > 3.0: Turbulencia Extrema (Saturado)
        conditions = [
            (self.df['DCPI'] < 0.25),
            (self.df['DCPI'] >= 0.25) & (self.df['DCPI'] < 3.0),
            (self.df['DCPI'] >= 3.0)
        ]
        choices = ["LAMINAR (Suave)", "FLUJO FUERTE", "SATURADO (Turbulencia)"]
        self.df['Estado'] = np.select(conditions, choices, default="NORMAL")
        
        last_rows = self.df.tail(10)
        print(last_rows[['ts', 'close', 'volume', 'DCPI', 'Estado']])
        
        last_dcpi = self.df['DCPI'].iloc[-1]
        
        print("\n [SENAL] EN TIEMPO REAL:")
        if last_dcpi > EXTREME_TURBULENCE:
            print(f" [ALERTA] TURBULENCIA EXTREMA (DCPI={last_dcpi:.2f}). Mercado sobre-extendido.")
            print("   Accion recomendada: CONTRARIAN (Operar contra tendencia).")
        elif last_dcpi > 1.0:
             print(f" [INFO] Flujo Fuerte (DCPI={last_dcpi:.2f}). Tendencia definida.")
        else:
            print(f" [OK] FLUJO LAMINAR (DCPI={last_dcpi:.2f}). Mercado calmo.")
        
        # Graficamos
        self.plot_results()

    def plot_results(self):
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'cyan'
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Precio', color=color)
        ax1.plot(self.df['ts'], self.df['close'], color=color, alpha=0.6, label='Precio')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'magenta'
        ax2.set_ylabel('Presión DCPI', color=color)
        ax2.plot(self.df['ts'], self.df['DCPI'], color=color, alpha=0.8, linewidth=1, label='Presión DCPI')
        ax2.axhline(SATURATION_THRESHOLD, color='green', linestyle=':', label='Laminar (0.25)')
        ax2.axhline(EXTREME_TURBULENCE, color='red', linestyle='--', label='Saturacion (3.0)')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Diaz-Cano Fluid Bot: {self.symbol} \n(K={self.best_k:.3f}, Alpha={self.best_alpha:.3f})")
        plt.show()

# --- EJECUCIÓN ---
if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
    bot = DiazCanoFluidBot(symbol=symbol, limit=1000)
    bot.fetch_data()
    
    if not bot.df.empty:
        # 1. Optimizar Constantes
        bot.optimize_constants()
        # 2. Analizar con las nuevas constantes
        bot.analyze_market()
