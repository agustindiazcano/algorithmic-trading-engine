import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime

# ==========================================
# 🔒 CONSTANTES DE INGENIERÍA (REALISTAS)
# ==========================================
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
PEPE_SATURACION_MIN = 15.0 
PEPE_SATURACION_MAX = 20.0 
DF_MIN = 1.40
DF_MAX = 1.70

# ⚙️ CONFIGURACIÓN DEL VIAJE EN EL TIEMPO
SYMBOL = "PEPEUSDT"
START_DATE = "2024-01-01"
LEVERAGE = 50.0  
FEE_TAKER = 0.0005  # 0.05% (Binance Futures Standard)

# STOP LOSS DE EMERGENCIA (Movimiento del precio)
# A x50, un -1.5% de precio es -75% de la cuenta. 
# Es el punto de "no retorno" antes de la liquidación total.
SL_PCT = 0.015 

def get_random_window(symbol):
    start_ts = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(time.time() * 1000) - (1000 * 60 * 1000) 
    
    random_start = random.randint(start_ts, end_ts)
    date_readable = datetime.fromtimestamp(random_start/1000).strftime('%Y-%m-%d %H:%M')
    print(f" [>] Viajando a: {date_readable} ...", end=" ")
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1000&startTime={random_start}"
    
    try:
        res = requests.get(url, timeout=10).json()
        if not res or isinstance(res, dict) or len(res) < 900: 
            print("❌ Error de datos")
            return None
        
        df = pd.DataFrame(res, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qa', 'tr', 'tb', 'tq', 'ig'])
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        print("✅")
        return df
    except Exception as e:
        print(f"❌ Error API: {e}")
        return None

def calculate_higuchi_df(series, k_max=10):
    N = len(series)
    if N < 20: return 1.5 
    L = []
    x = np.array(series)
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            indices = np.arange(m, N, k)
            if len(indices) < 2: continue
            L_m_k = np.sum(np.abs(np.diff(x[indices])))
            norm = (N - 1) / (len(indices) * k)
            Lk.append(L_m_k * norm / k)
        if Lk: L.append(np.mean(Lk))
    if len(L) < 2: return 1.5
    coeffs = np.polyfit(np.log(1/np.arange(1, k_max + 1)), np.log(L), 1)
    return coeffs[0]

def run_strategy_logic(df):
    # FÍSICA
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL) 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL
    
    # FRACTAL
    vals = [1.5]*100
    close_vals = df['close'].values
    # Optimización: solo calculamos si necesitamos (aunque Higuchi es pesado igual)
    for i in range(100, len(df)):
        window = close_vals[i-60:i]
        vals.append(calculate_higuchi_df(window))
    
    if len(vals) < len(df): vals += [1.5] * (len(df)-len(vals))
    df['df_vascular'] = pd.Series(vals, index=df.index[:len(vals)])

    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # SEÑAL BASE (Calculada al CIERRE de la vela)
    df['raw_signal'] = np.where(
        (df['dcpi'] > PEPE_SATURACION_MIN) & 
        (df['dcpi'] < PEPE_SATURACION_MAX) & 
        (df['df_vascular'] > DF_MIN) & 
        (df['df_vascular'] < DF_MAX),
        -df['trend'], 0
    )
    
    # CORRECCIÓN CRÍTICA: La señal de la vela 'i' se ejecuta en 'i+1'
    df['exec_signal'] = df['raw_signal'].shift(1).fillna(0)
    
    return df

def calculate_pnl(df):
    balance = 1000.0
    position = 0 # 1 (Long), -1 (Short), 0 (Cash)
    entry_price = 0
    trades_log = []
    
    # Iteramos desde el índice 1 porque usamos shift
    for i in range(1, len(df)):
        # DATOS DE LA VELA ACTUAL (REALIDAD)
        current_open = df['open'].iloc[i]  # Precio de entrada
        current_high = df['high'].iloc[i]  # Para chequear SL Short
        current_low = df['low'].iloc[i]    # Para chequear SL Long
        current_close = df['close'].iloc[i] # Para valorar al final
        
        signal = df['exec_signal'].iloc[i] # Señal generada AYER
        
        # 1. GESTIÓN DE POSICIÓN ABIERTA
        if position != 0:
            # A) CHEQUEO DE STOP LOSS / LIQUIDACIÓN INTRA-VELA
            pnl_pct_worst = 0
            if position == 1: # Long
                # El peor caso es que toque el Low
                pnl_pct_worst = (current_low - entry_price) / entry_price
            else: # Short
                # El peor caso es que toque el High
                pnl_pct_worst = (entry_price - current_high) / entry_price # Short inverso
            
            # Si en algún momento de la vela perdimos más del SL_PCT... MURIÓ
            if pnl_pct_worst <= -SL_PCT:
                # Asumimos salida al precio de SL (o peor)
                loss_usd = balance * (-SL_PCT) * LEVERAGE
                balance += loss_usd
                trades_log.append(loss_usd)
                position = 0 # Cierre forzoso
                if balance < 10: return 0.0, [-1000] # Cuenta quemada
                continue # Pasamos a la siguiente vela, ya cerramos esta

            # B) SALIDA POR SEÑAL CONTRARIA (FLIP)
            # Si tengo Long (1) y la señal es Short (-1) -> Cierro y Abro Short
            # Si tengo Short (-1) y la señal es Long (1) -> Cierro y Abro Long
            if (position == 1 and signal == -1) or (position == -1 and signal == 1):
                # Calculamos PnL al OPEN de esta vela (ejecución inmediata)
                # Nota: Si la señal cambió, cerramos al Open.
                pnl_pct = (current_open - entry_price) / entry_price * position
                pnl_usd = balance * pnl_pct * LEVERAGE
                
                # Restar comisión de salida
                commission = balance * LEVERAGE * FEE_TAKER
                balance += (pnl_usd - commission)
                trades_log.append(pnl_usd - commission)
                
                position = 0 # Temporalmente 0, abajo se reabre si aplica
        
        # 2. APERTURA DE NUEVA POSICIÓN
        # Si estamos cash o acabamos de cerrar (flip)
        if position == 0 and signal != 0:
            position = signal
            entry_price = current_open
            # Restar comisión de entrada INMEDIATAMENTE
            commission = balance * LEVERAGE * FEE_TAKER
            balance -= commission
            
    return balance, trades_log

# ==========================================
# 🧬 ALGORITMO GENÉTICO (V80 - PROYECCIÓN)
# ==========================================

# Configuración GA
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
ELITISM_COUNT = 5  # Los mejores 5 pasan directo

# Genes y sus rangos
GENE_RANGES = {
    'df_min': (1.40, 1.60),
    'df_max': (1.60, 2.00), # Siempre > df_min
    'k_uni': (0.10, 0.50),
    'alpha': (0.10, 0.50)
}

class Individual:
    def __init__(self, genes=None):
        if genes:
            self.genes = genes
        else:
            self.genes = {
                'df_min': random.uniform(*GENE_RANGES['df_min']),
                'df_max': random.uniform(*GENE_RANGES['df_max']),
                'k_uni': random.uniform(*GENE_RANGES['k_uni']),
                'alpha': random.uniform(*GENE_RANGES['alpha'])
            }
            # Corrección de consistencia
            if self.genes['df_max'] <= self.genes['df_min']:
                self.genes['df_max'] = self.genes['df_min'] + 0.1
                
        self.fitness = 0.0
        self.roi = 0.0
        self.wr = 0.0
        self.survived = True

def run_strategy_logic_dynamic(df, genes):
    """Versión dinámica que acepta genes"""
    # FÍSICA
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['volume'] ** genes['alpha']) 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    # Evitamos división por cero con un epsilon más robusto
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / genes['k_uni']
    
    # FRACTAL (A diferencia de la física, el DF NO depende de los genes K/Alpha, 
    # pero sí se usa para filtrar. Podríamos pre-calcularlo si no dependiera de nada,
    # pero aquí asumimos que ya viene calculado en 'df_vascular' del CSV o dataset)
    
    # Si no está calculado, lo calculamos (solo la primera vez o si es dataset crudo)
    if 'df_vascular' not in df.columns or df['df_vascular'].iloc[-1] == 1.5:
         # Nota: Esto es lento. En GA idealmente usamos pre-cálculo.
         # Para este script, asumiremos que "df_vascular" ESTÁ en el dataframe
         # O que el usuario usa el CSV pre-procesado.
         pass 

    df['sma_5'] = df['close'].rolling(5).mean()
    df['trend'] = np.where(df['close'] > df['sma_5'], 1, -1)
    
    # SEÑAL CON GENES
    # Saturación de PEPE (Hardcodeada por ahora o añadida a genes si se quiere)
    SAT_MIN = 15.0
    SAT_MAX = 20.0
    
    df['raw_signal'] = np.where(
        (df['dcpi'] > SAT_MIN) & 
        (df['dcpi'] < SAT_MAX) & 
        (df['df_vascular'] > genes['df_min']) & 
        (df['df_vascular'] < genes['df_max']),
        -df['trend'], 0
    )
    
    df['exec_signal'] = df['raw_signal'].shift(1).fillna(0)
    return df

def evaluate_fitness(individual, dataset_pool):
    """
    Evalúa al individuo en 5 escenarios aleatorios del pool.
    Fitness = Promedio ROI * Factor Supervivencia
    """
    rois = []
    wins = 0
    total_ops = 0
    liquidations = 0
    
    # Testeamos en 5 ventanas aleatorias
    # dataset_pool es una lista de DataFrames precargados
    sample_windows = random.sample(dataset_pool, min(5, len(dataset_pool)))
    
    for df_window in sample_windows:
        # Copia ligera para no manchar el original con columnas temporales
        df_test = df_window.copy()
        
        # Ejecutar lógica con genes
        df_test = run_strategy_logic_dynamic(df_test, individual.genes)
        
        # Simular PnL
        bal, logs = calculate_pnl(df_test)
        
        # Métricas
        roi = ((bal / 1000.0) - 1.0) * 100.0
        if bal <= 10.0:
            liquidations += 1
            roi = -100.0 # Castigo máximo
            
        rois.append(roi)
        
        # Winrate
        ops_ganadoras = len([x for x in logs if x > 0])
        wins += ops_ganadoras
        total_ops += len(logs)

    # Cálculo Final
    avg_roi = sum(rois) / len(rois)
    survival_rate = 1.0 - (liquidations / len(sample_windows))
    
    # El fitness premia ROI pero requiere supervivencia.
    # Si muere mucho, el fitness se va al suelo.
    individual.fitness = avg_roi * (survival_rate ** 2)
    individual.roi = avg_roi
    individual.survived = (survival_rate == 1.0)
    
    if total_ops > 0:
        individual.wr = (wins / total_ops) * 100.0
    else:
        individual.wr = 0.0

def crossover(p1, p2):
    """Mezcla genética uniforme"""
    child_genes = {}
    for key in GENE_RANGES:
        if random.random() > 0.5:
            child_genes[key] = p1.genes[key]
        else:
            child_genes[key] = p2.genes[key]
            
    # Validación
    if child_genes['df_max'] <= child_genes['df_min']:
        child_genes['df_max'] = child_genes['df_min'] + 0.1
        
    return Individual(child_genes)

def mutate(ind):
    """Pequeña mutación aleatoria"""
    new_genes = ind.genes.copy()
    gene_to_mutate = random.choice(list(GENE_RANGES.keys()))
    
    # Valor actual
    val = new_genes[gene_to_mutate]
    # Rango
    min_v, max_v = GENE_RANGES[gene_to_mutate]
    
    # Mutación +/- 10% del rango
    mutation_strength = (max_v - min_v) * 0.1
    change = random.uniform(-mutation_strength, mutation_strength)
    
    new_val = val + change
    # Clip al rango
    new_val = max(min_v, min(new_val, max_v))
    
    new_genes[gene_to_mutate] = new_val
    
    # Validación post-mutación
    if new_genes['df_max'] <= new_genes['df_min']:
         if gene_to_mutate == 'df_max':
             new_genes['df_max'] = new_genes['df_min'] + 0.1
         else:
             new_genes['df_min'] = new_genes['df_max'] - 0.1
             
    ind.genes = new_genes

def load_master_dataset():
    """Intenta cargar el dataset maestro o genera error"""
    files = ["PEPE_MASTER_VL_500K.csv", "MASTER_DATA_PEPEUSDT_500K.csv"]
    df = None
    for f in files:
        try:
            print(f"📂 Buscando {f}...")
            df = pd.read_csv(f)
            
            # ADAPTACIÓN DINÁMICA DE COLUMNAS
            # El archivo generado PEPE_MASTER... tiene: ts, close, vol, dcpi, df_vascular
            # El bot espera: open, high, low, close, volume (para PnL preciso)
            
            # 1. Normalizar nombres
            df.rename(columns={'vol': 'volume', 'Vol': 'volume'}, inplace=True)
            
            # 2. Rellenar OHLC si falta (Simulación de velas planas para backtest rápido)
            if 'open' not in df.columns:
                print("⚠️  OHLC incompleto. Usando Close para todo (Open=High=Low=Close).")
                df['open'] = df['close']
                df['high'] = df['close']
                df['low'] = df['close']
            
            # Limpieza básica
            cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'df_vascular']
            for c in cols_to_numeric:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # Asegurar que existan, si df_vascular no está, avisar
            if 'df_vascular' not in df.columns:
                print("⚠️ Advertencia: 'df_vascular' no encontrado. Se usará 1.5 por defecto (resultado pobre).")
                df['df_vascular'] = 1.5
                
            df['ts'] = pd.to_datetime(df.get('ts', pd.Series()), errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            
            # Reset index para que los slices funcionen bien
            df.reset_index(drop=True, inplace=True)
            
            print(f"✅ Dataset cargado: {len(df)} velas.")
            break
        except FileNotFoundError:
            continue
            
    if df is None:
        print("❌ ERROR: No se encontró ningún CSV maestro. Ejecutá primero el recolector.")
        return None
        
    # Pre-cortar el dataset en ventanas de 1000 velas para el pool
    # Esto simula los "viajes en el tiempo" pero mucho más rápido (en RAM)
    pool = []
    # Generamos 100 ventanas aleatorias del historial
    max_idx = len(df) - 1000
    if max_idx < 0: return [df] # Si es muy chico devolvemos el entero
    
    print("✂️  Generando pool de entrenamiento (100 escenarios)...")
    for _ in range(100):
        start = random.randint(0, max_idx)
        # Copia profunda para no alterar original
        window = df.iloc[start : start+1000].copy().reset_index(drop=True)
        pool.append(window)
        
    return pool

def run_genetic_algorithm():
    print(f"\n🧬 INICIANDO EVOLUCIÓN GENÉTICA ({GENERATIONS} gens x {POPULATION_SIZE} inds)...")
    
    # 1. Cargar Datos
    training_pool = load_master_dataset()
    if not training_pool: return

    # 2. Población Inicial
    population = [Individual() for _ in range(POPULATION_SIZE)]
    
    best_overall = None
    
    for gen in range(1, GENERATIONS + 1):
        # A. Evaluar
        for ind in population:
            evaluate_fitness(ind, training_pool)
            
        # B. Ordenar por fitness descendente
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        current_best = population[0]
        if best_overall is None or current_best.fitness > best_overall.fitness:
            best_overall = current_best
            
        # Reporte Generacional
        print(f"Gen {gen:03} | Mejor ROI: {current_best.roi:6.2f}% | WR: {current_best.wr:4.1f}% | "
              f"Genes: DF[{current_best.genes['df_min']:.2f}-{current_best.genes['df_max']:.2f}] "
              f"K[{current_best.genes['k_uni']:.3f}] A[{current_best.genes['alpha']:.3f}]")
        
        # C. Selección y Evolución
        next_gen = []
        
        # Elitismo
        next_gen.extend(population[:ELITISM_COUNT])
        
        # Reproducción (Torneo simple)
        while len(next_gen) < POPULATION_SIZE:
            # Torneo de 3
            candidates = random.sample(population, 3)
            parent1 = max(candidates, key=lambda x: x.fitness)
            candidates = random.sample(population, 3)
            parent2 = max(candidates, key=lambda x: x.fitness)
            
            child = crossover(parent1, parent2)
            
            if random.random() < MUTATION_RATE:
                mutate(child)
                
            next_gen.append(child)
            
        population = next_gen

    print("\n✅ EVOLUCIÓN FINALIZADA")
    print("="*60)
    print("🏆 CAMPEÓN ABSOLUTO:")
    print(f" fitness: {best_overall.fitness:.4f}")
    print(f" ROI Promedio: {best_overall.roi:.2f}%")
    print(f" WinRate: {best_overall.wr:.2f}%")
    print(" 🧬 GENOMA GANADOR:")
    for k, v in best_overall.genes.items():
        print(f"   {k}: {v:.4f}")
    
    # 3. Guardar Cerebro
    import json
    with open("best_genome_pepe.json", "w") as f:
        json.dump(best_overall.genes, f, indent=4)
    print(f"\n💾 Cerebro guardado en 'best_genome_pepe.json'")
    print("="*60)
    
if __name__ == "__main__":
    # Si querés correr solo el GA:
    run_genetic_algorithm()
    # Si querés correr el test manual antiguo:
    # run_quantum_leap()