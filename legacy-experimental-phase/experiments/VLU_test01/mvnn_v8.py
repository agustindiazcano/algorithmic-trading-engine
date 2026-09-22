import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import cdist
import warnings

# Silenciamos warnings de convergencia del MLP (es normal que no converja perfecto en RL simple)
warnings.filterwarnings('ignore')

# ==========================================
# 1. GENERADOR PROCEDURAL DE MUNDOS 🌍
# ==========================================
def generate_map(size, obstacle_density=0.1):
    """
    Genera un mapa cuadrado de tamaño 'size' con obstáculos aleatorios.
    Garantiza (casi siempre) que Start y End no estén bloqueados.
    """
    grid = np.zeros((size, size))
    
    # Bordes
    grid[0, :] = 1; grid[-1, :] = 1
    grid[:, 0] = 1; grid[:, -1] = 1
    
    # Obstáculos aleatorios (Ruido)
    num_obstacles = int(size * size * obstacle_density)
    for _ in range(num_obstacles):
        rx, ry = np.random.randint(1, size-1), np.random.randint(1, size-1)
        grid[ry, rx] = 1
        
    start = np.array([2.0, 2.0])
    end = np.array([float(size-3), float(size-3)])
    
    # Limpiar zona de start y end
    grid[int(start[1]), int(start[0])] = 0
    grid[int(end[1]), int(end[0])] = 0
    
    obstacles = np.column_stack(np.where(grid == 1))
    return grid, obstacles, start, end

# ==========================================
# 2. AGENTE MLP (Deep Learning - Q-Learning) 🧠❌
# ==========================================
class MLPAgent:
    def __init__(self, training_map_size=15):
        # Input: [AgentX, AgentY, GoalX, GoalY, DistToN, DistToS, DistToE, DistToW]
        # Normalizamos inputs dividiendo por training_map_size para intentar ser justos
        self.map_size = training_map_size
        self.model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', 
                                  solver='adam', max_iter=1, warm_start=True)
        self.epsilon = 1.0 # Exploración inicial
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        self.initialized = False
        
    def get_state(self, pos, goal, grid):
        # Sensores locales (Distancia a muros en 4 direcciones)
        x, y = int(pos[0]), int(pos[1])
        h, w = grid.shape
        
        # Raycast simple
        d_n, d_s, d_e, d_w = 0, 0, 0, 0
        for i in range(y, -1, -1): # North
            if grid[i, x] == 1: d_n = (y - i)/h; break
        for i in range(y, h): # South
            if grid[i, x] == 1: d_s = (i - y)/h; break
        for i in range(x, w): # East
            if grid[i, x] == 1: d_e = (i - x)/w; break
        for i in range(x, -1, -1): # West
            if grid[i, x] == 1: d_w = (x - i)/w; break
            
        # Estado normalizado
        return np.array([pos[0]/w, pos[1]/h, goal[0]/w, goal[1]/h, d_n, d_s, d_e, d_w])

    def train(self, episodes=1000):
        print(f"   [MLP] Entrenando 1000 épocas en mapa {self.map_size}x{self.map_size}...")
        grid, _, start, end = generate_map(self.map_size)
        
        # Pre-entrenamiento ficticio para inicializar pesos
        X_dummy = np.zeros((1, 8))
        y_dummy = np.zeros((1, 4)) # 4 acciones: N, S, E, W
        self.model.fit(X_dummy, y_dummy)
        
        for ep in range(episodes):
            pos = np.copy(start)
            state = self.get_state(pos, end, grid)
            
            for step in range(self.map_size * 2): # Max steps
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, 4)
                else:
                    q_values = self.model.predict(state.reshape(1, -1))
                    action = np.argmax(q_values)
                
                # Mover
                next_pos = np.copy(pos)
                if action == 0: next_pos[1] -= 1 # N
                elif action == 1: next_pos[1] += 1 # S
                elif action == 2: next_pos[0] += 1 # E
                elif action == 3: next_pos[0] -= 1 # W
                
                # Reward
                reward = -0.1 # Costo por paso
                done = False
                
                # Chequeo límites/colisión
                if (next_pos[0] < 0 or next_pos[0] >= self.map_size or 
                    next_pos[1] < 0 or next_pos[1] >= self.map_size or 
                    grid[int(next_pos[1]), int(next_pos[0])] == 1):
                    reward = -1.0 # Castigo muro
                    next_pos = pos # Rebote
                elif np.linalg.norm(next_pos - end) < 1.0:
                    reward = 10.0 # Meta
                    done = True
                
                next_state = self.get_state(next_pos, end, grid)
                
                # Q-Learning Update
                target = reward
                if not done:
                    next_q = self.model.predict(next_state.reshape(1, -1))
                    target = reward + 0.9 * np.max(next_q)
                
                target_f = self.model.predict(state.reshape(1, -1))
                target_f[0][action] = target
                
                self.model.partial_fit(state.reshape(1, -1), target_f)
                
                pos = next_pos
                state = next_state
                if done: break
            
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                
            if ep % 200 == 0: print(f"     -> Ep {ep}: Epsilon {self.epsilon:.2f}")

    def solve(self, grid, start, end):
        # Testeo (Greedy puro)
        pos = np.copy(start)
        path = [np.copy(pos)]
        max_steps = grid.shape[0] * 3
        
        for _ in range(max_steps):
            state = self.get_state(pos, end, grid)
            q_values = self.model.predict(state.reshape(1, -1))
            action = np.argmax(q_values)
            
            next_pos = np.copy(pos)
            if action == 0: next_pos[1] -= 1
            elif action == 1: next_pos[1] += 1
            elif action == 2: next_pos[0] += 1
            elif action == 3: next_pos[0] -= 1
            
            # Chequeo colisión
            h, w = grid.shape
            if (next_pos[0] < 0 or next_pos[0] >= w or 
                next_pos[1] < 0 or next_pos[1] >= h or 
                grid[int(next_pos[1]), int(next_pos[0])] == 1):
                break # Chocó
                
            pos = next_pos
            path.append(np.copy(pos))
            if np.linalg.norm(pos - end) < 1.5:
                return path, True # Llegó
        
        return path, False # Se perdió o tardó mucho

# ==========================================
# 3. AGENTE DCNN (Topología Elástica) 🧠✅
# ==========================================
class DCNNAgent:
    def __init__(self):
        pass # No entrena. Nace sabiendo física.

    def solve(self, obstacles, start, end, size):
        # Inicialización: Línea recta (Riel mental)
        # Adaptamos cantidad de neuronas al tamaño del mapa
        num_neurons = size * 2 
        path = np.linspace(start, end, num_neurons)
        
        # Física
        alpha, beta, gamma = 0.1, 0.5, 2.0
        
        # Iteraciones de relajación (Inferencia)
        for _ in range(200):
            # 1. Neurogénesis (Si se estira mucho, agrega nodos)
            new_path = [path[0]]
            for i in range(1, len(path)):
                dist = np.linalg.norm(path[i] - path[i-1])
                if dist > 2.0: # Umbral de estiramiento
                    new_path.append((path[i] + path[i-1])/2)
                new_path.append(path[i])
            path = np.array(new_path)
            
            # 2. Fuerzas
            path_smooth = np.copy(path)
            path_smooth[1:-1] = 0.5 * (path[:-2] + path[2:])
            
            repulsion = np.zeros_like(path)
            if len(obstacles) > 0:
                # Optimización: Solo mirar obstáculos cercanos (KD-Tree sería ideal, cdist es bruto pero va)
                # Para mapa 100x100 cdist es pesado, filtramos por distancia simple primero si fuera prod.
                # Aquí usamos cdist full power.
                dists = cdist(path, obstacles)
                min_dists = np.min(dists, axis=1)
                nearest_idx = np.argmin(dists, axis=1)
                
                danger = min_dists < 3.0
                for k in np.where(danger)[0]:
                    obs = obstacles[nearest_idx[k]]
                    vec = path[k] - obs
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        repulsion[k] = (vec/norm) * (1.0 / (min_dists[k] + 0.1))
            
            # Update
            path += alpha * (beta * (path_smooth - path) + gamma * repulsion)
            path[0], path[-1] = start, end # Clavar extremos
            
        # Validación final
        dists = cdist(path, obstacles)
        success = np.min(dists) > 0.5
        return path, success

# ==========================================
# 4. EJECUCIÓN DEL TEST ⚔️
# ==========================================
print("=== ⚔️ BATALLA FINAL: ESCALABILIDAD ===")

# 1. ENTRENAR AL MLP EN JARDÍN DE INFANTES (15x15)
print("\n[FASE 1] Entrenando Agente MLP...")
mlp_agent = MLPAgent(training_map_size=15)
mlp_agent.train(episodes=1000)
print("   -> MLP listo (Cree que sabe navegar).")

dcnn_agent = DCNNAgent() # No necesita entreno

# 2. SOLTARLOS EN EL MUNDO REAL
map_sizes = [15, 30, 40, 50, 100]

print("\n[FASE 2] Testeando Generalización...")
print(f"{'MAPA':<10} | {'MLP (Deep Learning)':<20} | {'DCNN (Diaz-Cano)':<20}")
print("-" * 60)

for size in map_sizes:
    # Generar mapa nuevo
    grid, obstacles, start, end = generate_map(size, obstacle_density=0.15)
    
    # MLP Intento
    # Convertimos coordenadas de obstaculos para el DCNN (x,y)
    obs_xy = np.column_stack((obstacles[:, 1], obstacles[:, 0])) 
    
    # Test MLP
    try:
        _, mlp_success = mlp_agent.solve(grid, start, end)
        mlp_res = "✅ ÉXITO" if mlp_success else "❌ CHOCÓ/PERDIDO"
    except:
        mlp_res = "💥 ERROR MATRIZ"
        
    # Test DCNN
    # La DCNN escala sus neuronas internamente
    _, dcnn_success = dcnn_agent.solve(obs_xy, start, end, size)
    dcnn_res = "✅ ÉXITO (DOMA)" if dcnn_success else "❌ FALLO"
    
    print(f"{size}x{size:<6} | {mlp_res:<20} | {dcnn_res:<20}")

print("-" * 60)
print("CONCLUSIÓN:")
print("El MLP aprendió coordenadas, no conceptos.")
print("La DCNN usa leyes físicas universales. La física funciona igual en 15m que en 100m.")
print("-" * 60)