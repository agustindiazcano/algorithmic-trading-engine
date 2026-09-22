import pygame
import time
import random
import math

# --- CONFIGURACIÓN ---
ANCHO, ALTO = 800, 600
FPS = 15
TAMAÑO_CELDA = 20

# Colores
NEGRO = (10, 10, 10)
VERDE_RETRO = (0, 255, 0)      # Para el Snake Clásico
AZUL_W = (0, 120, 255)   # Para el Snake Volumétrico
ROJO = (255, 50, 50)
BLANCO = (255, 255, 255)

# ==========================================
# 1. LÓGICA TRADICIONAL (GRID / ARRAY)
# ==========================================
class SnakeTradicional:
    def __init__(self):
        # En tradicional, pensamos en [columna, fila], no en pixeles
        self.body = [[10, 10], [9, 10], [8, 10]] 
        self.direction = [1, 0] # [x, y] en celdas
        self.food = self._spawn_food()
        self.score = 0
        self.dead = False

    def _spawn_food(self):
        return [random.randint(0, (ANCHO//TAMAÑO_CELDA)-1), 
                random.randint(0, (ALTO//TAMAÑO_CELDA)-1)]

    def update(self):
        if self.dead: return
        
        # Calcular nueva cabeza (Matemática entera simple)
        new_head = [self.body[0][0] + self.direction[0], 
                    self.body[0][1] + self.direction[1]]

        # Colisión con Paredes (Lógica de IFs rígidos)
        if new_head[0] < 0 or new_head[0] >= ANCHO//TAMAÑO_CELDA or \
           new_head[1] < 0 or new_head[1] >= ALTO//TAMAÑO_CELDA:
            self.dead = True
            return

        # Colisión con cuerpo
        if new_head in self.body:
            self.dead = True
            return

        self.body.insert(0, new_head)

        # Colisión con Comida (Coordenada exacta)
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.body.pop() # Borrar cola si no comió

    def draw(self, surface):
        for segment in self.body:
            rect = pygame.Rect(segment[0]*TAMAÑO_CELDA, segment[1]*TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA)
            pygame.draw.rect(surface, VERDE_RETRO, rect) # Dibuja Cuadrados
            pygame.draw.rect(surface, NEGRO, rect, 1) # Borde
        
        # Comida
        rect_food = pygame.Rect(self.food[0]*TAMAÑO_CELDA, self.food[1]*TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA)
        pygame.draw.rect(surface, ROJO, rect_food)

# ==========================================
# 2. LÓGICA W (ESFERAS / FÍSICA)
# ==========================================
class WSphere:
    def __init__(self, x, y, r):
        self.pos = pygame.Vector2(x, y)
        self.r = r

class SnakeW:
    def __init__(self):
        # En W, pensamos en Espacio Continuo (Floats)
        self.body = [
            WSphere(200, 200, TAMAÑO_CELDA/2),
            WSphere(180, 200, TAMAÑO_CELDA/2),
            WSphere(160, 200, TAMAÑO_CELDA/2)
        ]
        self.direction = pygame.Vector2(TAMAÑO_CELDA, 0)
        self.food = self._spawn_food()
        self.score = 0
        self.dead = False

    def _spawn_food(self):
        # La comida puede estar en CUALQUIER lugar (no alineado a grilla)
        return WSphere(random.randint(50, ANCHO-50), random.randint(50, ALTO-50), TAMAÑO_CELDA/2 + 5) # Comida más grande

    def check_collision(self, s1, s2):
        # LÓGICA W: Interferencia de Radios
        # No importa si es un cuadrado o un hexágono, importa la distancia.
        dist = s1.pos.distance_to(s2.pos)
        return dist < (s1.r + s2.r)

    def update(self):
        if self.dead: return
        
        # Mover cuerpo (sigue al anterior)
        for i in range(len(self.body)-1, 0, -1):
            self.body[i].pos = self.body[i-1].pos.copy()
        
        # Mover cabeza
        self.body[0].pos += self.direction
        head = self.body[0]

        # Colisión Comida (Geometría)
        if self.check_collision(head, self.food):
            self.score += 1
            tail = self.body[-1]
            self.body.append(WSphere(tail.pos.x, tail.pos.y, tail.r))
            self.food = self._spawn_food()

        # Colisión Paredes (Límites físicos)
        if not (0 < head.pos.x < ANCHO and 0 < head.pos.y < ALTO):
            self.dead = True

        # Auto-colisión
        for i in range(4, len(self.body)):
            if self.check_collision(head, self.body[i]):
                self.dead = True

    def draw(self, surface):
        for s in self.body:
            # Dibuja Círculos (Esferas)
            pygame.draw.circle(surface, AZUL_W, (int(s.pos.x), int(s.pos.y)), int(s.r))
        
        # Comida
        pygame.draw.circle(surface, ROJO, (int(self.food.pos.x), int(self.food.pos.y)), int(self.food.r))

# ==========================================
# 3. BENCHMARKER (LA PRUEBA DE FUEGO)
# ==========================================
def correr_benchmark():
    ITERACIONES = 1_000_000
    print(f"--- INICIANDO W BENCHMARK ({ITERACIONES} ciclos) ---")
    
    # 1. Test Tradicional (Comparación de Enteros)
    pos1 = [10, 10]
    pos2 = [10, 10] # Colisión
    start_t = time.perf_counter()
    for _ in range(ITERACIONES):
        # La lógica clásica es comparar dos arrays
        colision = (pos1[0] == pos2[0] and pos1[1] == pos2[1])
    end_t = time.perf_counter()
    tiempo_trad = end_t - start_t
    print(f"TRADICIONAL (Grid Logic): {tiempo_trad:.5f} seg")

    # 2. Test W (Distancia Euclidiana)
    v1 = pygame.Vector2(100.0, 100.0)
    v2 = pygame.Vector2(100.0, 100.0)
    r = 10.0
    start_w = time.perf_counter()
    for _ in range(ITERACIONES):
        # La lógica W es Pitágoras (o .distance_to)
        # Optimizamos manual para ser justos: sqrt((x2-x1)^2 + ...)
        dist = v1.distance_to(v2)
        colision = dist < (r + r)
    end_w = time.perf_counter()
    tiempo_weyl = end_w - start_w
    print(f"W (Sphere Logic):   {tiempo_weyl:.5f} seg")
    
    return tiempo_trad, tiempo_weyl

# ==========================================
# MAIN APP
# ==========================================
def main():
    # EJECUTAR BENCHMARK ANTES DE ABRIR VENTANA
    t_trad, t_weyl = correr_benchmark()
    
    pygame.init()
    screen = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("W Systems: Protocolo Comparativo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    # Instanciamos AMBOS juegos
    game_trad = SnakeTradicional()
    game_weyl = SnakeW()
    
    mode = "W" # Empezamos con el tuyo

    running = True
    while running:
        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: mode = "TRADICIONAL"
                if event.key == pygame.K_2: mode = "W"
                if event.key == pygame.K_SPACE:
                    game_trad = SnakeTradicional()
                    game_weyl = SnakeW()

                # Controles compartidos
                current_game = game_trad if mode == "TRADICIONAL" else game_weyl
                if not current_game.dead:
                    if event.key == pygame.K_UP: current_game.direction = [0, -1] if mode=="TRADICIONAL" else pygame.Vector2(0, -TAMAÑO_CELDA)
                    if event.key == pygame.K_DOWN: current_game.direction = [0, 1] if mode=="TRADICIONAL" else pygame.Vector2(0, TAMAÑO_CELDA)
                    if event.key == pygame.K_LEFT: current_game.direction = [-1, 0] if mode=="TRADICIONAL" else pygame.Vector2(-TAMAÑO_CELDA, 0)
                    if event.key == pygame.K_RIGHT: current_game.direction = [1, 0] if mode=="TRADICIONAL" else pygame.Vector2(TAMAÑO_CELDA, 0)

        # Update
        if mode == "TRADICIONAL": game_trad.update()
        else: game_weyl.update()

        # Draw
        screen.fill(NEGRO)
        
        if mode == "TRADICIONAL": 
            game_trad.draw(screen)
            color_ui = VERDE_RETRO
        else: 
            game_weyl.draw(screen)
            color_ui = AZUL_W

        # UI DE DATOS
        pygame.draw.rect(screen, (20,20,20), (0, 0, ANCHO, 60))
        ui_texts = [
            f"MODO ACTIVO: {mode} (Presiona 1 o 2 para cambiar)",
            f"SCORE: {game_trad.score if mode=='TRADICIONAL' else game_weyl.score}",
            f"BENCHMARK (1M ops): Trad={t_trad:.4f}s | W={t_weyl:.4f}s"
        ]
        
        for i, txt in enumerate(ui_texts):
            surface = font.render(txt, True, color_ui if i == 0 else BLANCO)
            screen.blit(surface, (10, 5 + i*18))

        if (mode == "TRADICIONAL" and game_trad.dead) or (mode == "W" and game_weyl.dead):
            over_txt = font.render("GAME OVER - SPACE PARA REINICIAR", True, ROJO)
            screen.blit(over_txt, (ANCHO//2 - 150, ALTO//2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()