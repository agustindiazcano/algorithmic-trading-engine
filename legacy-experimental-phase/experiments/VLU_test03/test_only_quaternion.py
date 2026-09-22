import numpy as np
import time
import requests 
import random
import sys
from scipy.spatial.transform import Rotation as R

# Forzar UTF-8 en consola para Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURACIÓN DE TRADING
TRADING_MODE = "LONG"  # Opciones: "LONG", "SHORT", "BOTH"

# ==============================================================================
# 🧬 NEURONA VOLUMÉTRICA 3D CON CUATERNIONES (S³)
# ==============================================================================

class QuaternionNeuron3D:
    def __init__(self, genome_slice):
        self.params = genome_slice
        self.center = self.params[0:3]
        
        self.alpha = 10.0
        self.beta = 1.0
        self.confidence = 0.5
    
    def activate_3d_scan(self, market_trajectory, current_temp):
        r_base = self.params[3]
        r_eff = r_base * (1.0 + current_temp * 3.0)
        
        quat = self.params[8:12]
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation_matrix = R.from_quat(quat_norm).as_matrix()
        
        stretch = self.params[5:8]
        S_inv = np.diag(1.0 / (stretch + 1e-6))
        M = S_inv @ rotation_matrix.T
        
        worm_length = int(np.clip(self.params[12], 3, 50))
        adaptive_trajectory = market_trajectory[-worm_length:] if len(market_trajectory) >= worm_length else market_trajectory
        
        max_overlap = 0.0
        for point in adaptive_trajectory:
            diff = point - self.center
            local_diff = M @ diff
            dist = np.linalg.norm(local_diff)
            overlap = np.maximum(0, 1 - dist / (r_eff + 1e-6))
            
            if overlap > max_overlap:
                max_overlap = overlap
        
        self.confidence = self.alpha / (self.alpha + self.beta)
        weight = self.params[4]
        
        return weight * max_overlap * self.confidence
    
    def learn_from_result(self, pnl):
        learning_rate = 0.1
        
        if pnl > 0:
            self.alpha += learning_rate
        else:
            self.beta += learning_rate
        
        if self.alpha + self.beta > 20:
            self.alpha *= 0.9
            self.beta *= 0.9

# ==============================================================================
# 🧠 CEREBRO CUATERNIÓNICO
# ==============================================================================

class QuaternionBrain:
    def __init__(self, n_neurons=10):
        self.n_neurons = n_neurons
        self.neurons = []
        self.worm_history = []
    
    def load_genome(self, genome_array):
        self.neurons = []
        for genome_slice in genome_array:
            self.neurons.append(QuaternionNeuron3D(genome_slice))
    
    def predict(self, market_point, current_volatility):
        self.worm_history.append(market_point)
        if len(self.worm_history) > 100:
            self.worm_history.pop(0)
        
        max_activation = 0.0
        winner_activation = 0.0
        
        for neuron in self.neurons:
            activation = neuron.activate_3d_scan(self.worm_history, current_volatility)
            
            if abs(activation) > abs(max_activation):
                max_activation = activation
                winner_activation = activation
        
        return np.tanh(winner_activation)
    
    def feedback_all(self, pnl):
        for neuron in self.neurons:
            neuron.learn_from_result(pnl)

# ==============================================================================
# 🌍 MERCADO REAL
# ==============================================================================

def fetch_extended_data(symbol="PEPEUSDT", interval="1m", total_candles=20000, custom_end_time=None):
    if custom_end_time is None:
        current_end_time = int(time.time() * 1000)
        print(f"📡 Bajando {total_candles} velas recientes de {symbol}...")
    else:
        current_end_time = int(custom_end_time)
        readable_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_end_time/1000))
        print(f"📡 Bajando {total_candles} velas HISTÓRICAS terminando en {readable_date}...")
    
    limit_per_call = 1000
    all_closes = []
    all_volumes = []
    all_highs = []
    all_lows = []
    
    calls = total_candles // limit_per_call
    
    for i in range(calls):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit_per_call}&endTime={current_end_time}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            # Guard contra errores de Binance (dict en vez de list)
            if not isinstance(data, list):
                print(f"⚠️  Binance error: {data}")
                break
            
            if not data:
                print(f"⚠️  Sin datos en llamada {i+1}")
                break
            
            closes = [float(k[4]) for k in data]
            volumes = [float(k[5]) for k in data]
            highs = [float(k[2]) for k in data]
            lows = [float(k[3]) for k in data]
            
            all_closes = closes + all_closes
            all_volumes = volumes + all_volumes
            all_highs = highs + all_highs
            all_lows = lows + all_lows
            
            current_end_time = int(data[0][0]) - 1
            
            # DEBUG: Imprimir rango real bajado en este batch
            start_batch = time.strftime('%Y-%m-%d %H:%M', time.localtime(data[0][0]/1000))
            end_batch = time.strftime('%Y-%m-%d %H:%M', time.localtime(data[-1][0]/1000))
            print(f"   Batch {i+1}: {start_batch} -> {end_batch}")
            
        except Exception as e:
            print(f"❌ Error en llamada {i+1}: {e}")
            break
    
    if len(all_closes) < 100:
        print(f"❌ Datos insuficientes: {len(all_closes)} velas")
        return None, None, None
    
    print(f"✅ DATA CARGADA: {len(all_closes)} velas.")
    
    prices = np.array(all_closes)
    volumes = np.array(all_volumes)
    highs = np.array(all_highs)
    lows = np.array(all_lows)
    volatilities = (highs - lows) / prices
    
    return prices, volumes, volatilities

def prepare_inputs_3d(prices, volumes, volatilities):
    if prices is None:
        return None, None, None
    
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    acceleration = np.diff(returns)
    acceleration = np.concatenate([[0], acceleration])
    
    window = 100
    
    ret_norm = np.zeros_like(returns)
    vol_norm = np.zeros_like(volumes)
    acc_norm = np.zeros_like(acceleration)
    
    for i in range(len(returns)):
        start_idx = max(0, i - window)
        
        ret_window = returns[start_idx:i+1]
        ret_mean = np.mean(ret_window)
        ret_std = np.std(ret_window) + 1e-8
        ret_norm[i] = (returns[i] - ret_mean) / ret_std
        
        vol_window = volumes[start_idx:i+1]
        vol_mean = np.mean(vol_window)
        vol_std = np.std(vol_window) + 1e-8
        vol_norm[i] = (volumes[i] - vol_mean) / vol_std
        
        acc_window = acceleration[start_idx:i+1]
        acc_mean = np.mean(acc_window)
        acc_std = np.std(acc_window) + 1e-8
        acc_norm[i] = (acceleration[i] - acc_mean) / acc_std
    
    inputs = np.column_stack([ret_norm, vol_norm, acc_norm])
    
    vol_window = 20
    rolling_vol = np.array([np.std(returns[max(0, i-vol_window):i+1]) for i in range(len(returns))])
    
    MIN_TEMP = 0.001
    rolling_vol = np.maximum(rolling_vol, MIN_TEMP)
    
    return inputs, prices, rolling_vol

# ==============================================================================
# 🧪 TEST
# ==============================================================================

def run_test_simulation(genome, threshold, inputs, prices, rolling_volatility):
    brain = QuaternionBrain(n_neurons=10)
    brain.load_genome(genome)
    brain.decision_threshold = threshold
    
    INITIAL_BALANCE = 10000.0
    LEVERAGE = 50.0
    BET_PERCENTAGE = 0.10
    SLIPPAGE = 0.002  # 0.2% (más conservador para memecoin)
    FEE = 0.0004      # 0.04% (taker fee sin descuentos)
    
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    current_bet = 0
    trades = 0
    actions = []
    equity_curve = []
    activations = []  # <--- NEW
    
    print("\n⏯️  Ejecutando simulación...")
    
    for t in range(len(inputs)):
        if balance <= 0:
            break
        
        signal = brain.predict(inputs[t], rolling_volatility[t])
        activations.append(abs(signal)) # <--- NEW
        current_price = prices[t]
        act = 0
        
        # Cierre
        if (signal > brain.decision_threshold and position == -1) or \
           (signal < -brain.decision_threshold and position == 1):
            pnl_pct = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                      else (entry_price - current_price) / entry_price * LEVERAGE
            profit = current_bet * pnl_pct
            cost = (current_bet * LEVERAGE) * FEE
            balance += current_bet + (profit - cost)
            
            brain.feedback_all(profit)
            
            position = 0
            current_bet = 0
        
        # Apertura
        if position == 0:
            if signal > brain.decision_threshold and TRADING_MODE in ["LONG", "BOTH"]:
                position = 1
                entry_price = current_price * (1 + SLIPPAGE)
                current_bet = balance * BET_PERCENTAGE
                balance -= current_bet
                balance -= (current_bet * LEVERAGE) * FEE
                trades += 1
                act = 1
            elif signal < -brain.decision_threshold and TRADING_MODE in ["SHORT", "BOTH"]:
                position = -1
                entry_price = current_price * (1 - SLIPPAGE)
                current_bet = balance * BET_PERCENTAGE
                balance -= current_bet
                balance -= (current_bet * LEVERAGE) * FEE
                trades += 1
                act = -1
        
        # Liquidación (igual que en entrenamiento)
        if position != 0:
            unrealized = (current_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                         else (entry_price - current_price) / entry_price * LEVERAGE
            if unrealized <= -0.9:
                # El margen ya fue descontado al abrir, solo cerrar posición
                position = 0
                current_bet = 0
        
        actions.append(act)
        equity_curve.append(balance)
    
    # Cierre final
    if position != 0:
        final_price = prices[-1]
        pnl_pct = (final_price - entry_price) / entry_price * LEVERAGE if position == 1 \
                  else (entry_price - final_price) / entry_price * LEVERAGE
        profit = current_bet * pnl_pct
        cost = (current_bet * LEVERAGE) * FEE
        balance += current_bet + (profit - cost)
    
    # Calcular promedio de activaciones (certeza del experto)
    avg_confidence = np.mean(activations) if activations else 0.0
    
    return balance, actions, equity_curve, trades, avg_confidence

# ==============================================================================
# 🚀 MAIN - SOLO TEST
# ==============================================================================

if __name__ == "__main__":
    print("🧪 CARGANDO GENOMA GUARDADO...")
    
    try:
        genome = np.load("best_genome_quaternion_FINAL.npy")
        # threshold = np.load("best_threshold_quaternion_FINAL.npy")
        threshold = 0.8
        print(f"✅ Genoma cargado: {genome.shape[0]} neuronas")
        print(f"✅ Umbral de decisión: {threshold:.3f}")
    except FileNotFoundError:
        print("❌ No se encontraron archivos de genoma guardados.")
        print("   Ejecuta primero el entrenamiento para generar:")
        print("   - best_genome_quaternion_FINAL.npy")
        print("   - best_threshold_quaternion_FINAL.npy")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎲 --- TEST CIEGO (10 PERIODOS ALEATORIOS) ---")
    print("="*60)
    
    # Timestamp de inicio de 2023
    start_2023_ms = int(time.mktime(time.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
    now_ms = int(time.time() * 1000)
    
    test_results = []
    
    for test_num in range(1, 11):
        print(f"\n🧪 TEST {test_num}/10:")
        
        # Generar tiempo aleatorio desde 2023
        random_test_time = random.randint(start_2023_ms, now_ms)
        
        test_prices, test_vols, test_volatilities = fetch_extended_data(
            "PEPEUSDT", "1m", 5000, custom_end_time=random_test_time
        )
        if test_prices is not None and len(test_prices) > 0:
            print(f"   � Datos descargados ok: {len(test_prices)} velas")
            
        test_inputs, test_prices_aligned, test_rolling_vol = prepare_inputs_3d(test_prices, test_vols, test_volatilities)
        
        if test_inputs is not None:
            final_bal, acts, equity, test_trades, test_conf = run_test_simulation(genome, threshold, test_inputs, test_prices_aligned, test_rolling_vol)
            
            profit_pct = ((final_bal - 10000) / 10000) * 100
            test_results.append({
                'balance': final_bal,
                'profit_pct': profit_pct,
                'trades': test_trades
            })
            
            print(f"   Balance Final: ${final_bal:,.2f} | P/L: {profit_pct:+.2f}% | Trades: {test_trades}")
            print(f"   🧠 Similitud Promedio: {test_conf*100:.1f}% (Nivel de confianza)")
    
    # Resumen final
    if test_results:
        avg_balance = np.mean([r['balance'] for r in test_results])
        avg_profit = np.mean([r['profit_pct'] for r in test_results])
        avg_trades = np.mean([r['trades'] for r in test_results])
        
        wins = sum(1 for r in test_results if r['profit_pct'] > 0)
        losses = len(test_results) - wins
        
        print("\n" + "="*60)
        print("📊 RESUMEN DE 10 TESTS:")
        print("="*60)
        print(f"   Balance Promedio:  ${avg_balance:,.2f}")
        print(f"   P/L Promedio:      {avg_profit:+.2f}%")
        print(f"   Trades Promedio:   {avg_trades:.1f}")
        print(f"   Tests Ganadores:   {wins}/10")
        print(f"   Tests Perdedores:  {losses}/10")
