import pygame
import random
import time
import json
import math

# --- CONFIGURACIÓN DEL LABORATORIO ---
DURACION_TEST = 60  # Segundos
CANTIDAD_PARTICULAS = 800  # Carga pesada para estresar la CPU
ANCHO, ALTO = 1200, 600    # Pantalla ancha para split-screen
MITAD_ANCHO = ANCHO // 2

# Colores
NEGRO = (10, 10, 15)
BLANCO = (200, 200, 200)
ROJO_TRAD = (255, 60, 60)    # El pasado
AZUL_WEYL = (0, 140, 255)    # El futuro
GRIS_LINEA = (50, 50, 50)

# --- CLASE PARTÍCULA HÍBRIDA ---
class Particula:
    def __init__(self, x, y, vx, vy, radio):
        # Estado base (Física)
        self.pos_trad = pygame.Vector2(x, y)
        self.vel_trad = pygame.Vector2(vx, vy)
        
        self.pos_weyl = pygame.Vector2(x + MITAD_ANCHO, y) # Offset para pantalla derecha
        self.vel_weyl = pygame.Vector2(vx, vy)
        
        self.radio = radio
        
        # Estado Lógico
        self.col_trad = False
        self.col_weyl = False

    def update_traditional(self, core_rect):
        # 1. Física simple
        self.vel_trad.y += 0.1 # Gravedad
        self.pos_trad += self.vel_trad
        
        # 2. Lógica AABB (Cajas)
        # Creamos la caja de la partícula
        my_rect = pygame.Rect(self.pos_trad.x - self.radio, self.pos_trad.y - self.radio, 
                              self.radio*2, self.radio*2)
        
        self.col_trad = False
        if my_rect.colliderect(core_rect):
            self.col_trad = True
            # Rebote "Caja" (Inexacto)
            if abs(my_rect.centerx - core_rect.centerx) > abs(my_rect.centery - core_rect.centery):
                self.vel_trad.x *= -0.8
            else:
                self.vel_trad.y *= -0.8
        
        # Límites pantalla izquierda
        if self.pos_trad.y > ALTO: self.pos_trad.y = 0
        if self.pos_trad.x < 0 or self.pos_trad.x > MITAD_ANCHO: self.vel_trad.x *= -1

    def update_w(self, core_pos, core_radius):
        # 1. Física simple
        self.vel_weyl.y += 0.1
        self.pos_weyl += self.vel_weyl
        
        # 2. Lógica Volumétrica (Esferas)
        dist = self.pos_weyl.distance_to(core_pos)
        min_dist = self.radio + core_radius
        
        self.col_weyl = False
        if dist < min_dist:
            self.col_weyl = True
            # Rebote Vectorial (Exacto)
            normal = (self.pos_weyl - core_pos).normalize()
            self.pos_weyl = core_pos + normal * min_dist # Corrección de posición
            self.vel_weyl = self.vel_weyl.reflect(normal) * 0.8
            
        # Límites pantalla derecha
        if self.pos_weyl.y > ALTO: self.pos_weyl.y = 0
        if self.pos_weyl.x < MITAD_ANCHO or self.pos_weyl.x > ANCHO: self.vel_weyl.x *= -1

    def draw(self, surface):
        # LADO IZQUIERDO (TRADICIONAL)
        color_t = ROJO_TRAD if self.col_trad else (100, 100, 100)
        # Dibujamos la caja para evidenciar el AABB
        rect_t = pygame.Rect(self.pos_trad.x - self.radio, self.pos_trad.y - self.radio, 
                             self.radio*2, self.radio*2)
        pygame.draw.rect(surface, color_t, rect_t, 1)
        
        # LADO DERECHO (W)
        color_w = AZUL_WEYL if self.col_weyl else BLANCO
        pygame.draw.circle(surface, color_w, (int(self.pos_weyl.x), int(self.pos_weyl.y)), self.radio)


# --- MAIN ---
def run_benchmark():
    pygame.init()
    screen = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("W Systems | Comparative Benchmark Protocol")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    font_big = pygame.font.SysFont("monospace", 30, bold=True)

    # Inicializar Partículas (Gemelas)
    particulas = []
    for _ in range(CANTIDAD_PARTICULAS):
        r = random.randint(5, 15)
        x = random.randint(50, MITAD_ANCHO - 50)
        y = random.randint(-400, -50)
        vx = random.uniform(-3, 3)
        vy = random.uniform(2, 5)
        particulas.append(Particula(x, y, vx, vy, r))

    # Variables de Datos
    start_time = time.time()
    frame_data = []
    core_radius = 60
    
    running = True
    while running:
        current_time = time.time()
        elapsed = current_time - start_time
        remaining = DURACION_TEST - elapsed

        # Input y Salida automática
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        if remaining <= 0:
            running = False # Fin del test

        # --- CONTROL DEL NÚCLEO (MÁSTER) ---
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Clampear mouse a la pantalla izquierda para control
        if mouse_x > MITAD_ANCHO: mouse_x -= MITAD_ANCHO
        
        # Posiciones de los núcleos
        core_pos_trad = (mouse_x, mouse_y)
        core_rect_trad = pygame.Rect(mouse_x - core_radius, mouse_y - core_radius, 
                                     core_radius*2, core_radius*2)
        
        core_pos_weyl = pygame.Vector2(mouse_x + MITAD_ANCHO, mouse_y)

        # --- MEDICIÓN DE PERFORMANCES (MICROSEGUNDOS) ---
        
        # 1. Medir Lógica Tradicional
        t0 = time.perf_counter()
        collisions_trad = 0
        for p in particulas:
            p.update_traditional(core_rect_trad)
            if p.col_trad: collisions_trad += 1
        t1 = time.perf_counter()
        cpu_trad = (t1 - t0) * 1000 # a ms

        # 2. Medir Lógica W
        t2 = time.perf_counter()
        collisions_weyl = 0
        for p in particulas:
            p.update_w(core_pos_weyl, core_radius)
            if p.col_weyl: collisions_weyl += 1
        t3 = time.perf_counter()
        cpu_weyl = (t3 - t2) * 1000 # a ms

        # Guardar datos del frame
        metrics = {
            "time_stamp": round(elapsed, 2),
            "cpu_traditional_ms": round(cpu_trad, 4),
            "cpu_w_ms": round(cpu_weyl, 4),
            "collisions_active_trad": collisions_trad,
            "collisions_active_weyl": collisions_weyl
        }
        frame_data.append(metrics)

        # --- RENDER ---
        screen.fill(NEGRO)
        
        # Línea divisoria
        pygame.draw.line(screen, GRIS_LINEA, (MITAD_ANCHO, 0), (MITAD_ANCHO, ALTO), 2)
        
        # Dibujar Núcleos
        # Izquierda (Caja)
        pygame.draw.rect(screen, (40, 40, 40), core_rect_trad)
        pygame.draw.rect(screen, ROJO_TRAD, core_rect_trad, 2)
        # Derecha (Esfera)
        pygame.draw.circle(screen, (40, 40, 50), (int(core_pos_weyl.x), int(core_pos_weyl.y)), core_radius)
        pygame.draw.circle(screen, AZUL_WEYL, (int(core_pos_weyl.x), int(core_pos_weyl.y)), core_radius, 2)

        # Dibujar Partículas
        for p in particulas:
            p.draw(screen)

        # UI
        # Timer
        timer_txt = font_big.render(f"T-MINUS: {remaining:.1f}s", True, BLANCO)
        screen.blit(timer_txt, (ANCHO//2 - 100, 10))
        
        # Stats Izquierda
        txt_t1 = font.render(f"AABB LOGIC (Legacy)", True, ROJO_TRAD)
        txt_t2 = font.render(f"CPU Time: {cpu_trad:.3f}ms", True, BLANCO)
        screen.blit(txt_t1, (10, 10))
        screen.blit(txt_t2, (10, 30))
        
        # Stats Derecha
        txt_w1 = font.render(f"W LOGIC (Volumetric)", True, AZUL_WEYL)
        txt_w2 = font.render(f"CPU Time: {cpu_weyl:.3f}ms", True, BLANCO)
        screen.blit(txt_w1, (MITAD_ANCHO + 10, 10))
        screen.blit(txt_w2, (MITAD_ANCHO + 10, 30))

        pygame.display.flip()
        clock.tick(60)

    # --- EXPORTAR JSON AL FINALIZAR ---
    print("Test finalizado. Generando reporte...")
    
    reporte = {
        "metadata": {
            "protocol": "W Dual-Core Stress Test",
            "particle_count": CANTIDAD_PARTICULAS,
            "duration_seconds": DURACION_TEST,
            "system": "Python/Pygame Engine"
        },
        "summary": {
            "avg_cpu_traditional": sum(d['cpu_traditional_ms'] for d in frame_data) / len(frame_data),
            "avg_cpu_w": sum(d['cpu_w_ms'] for d in frame_data) / len(frame_data),
            "total_frames_analyzed": len(frame_data)
        },
        "frame_by_frame_data": frame_data
    }
    
    with open("w_benchmark.json", "w") as f:
        json.dump(reporte, f, indent=4)
        
    print(f"Reporte guardado en: w_benchmark.json")
    pygame.quit()

if __name__ == "__main__":
    run_benchmark()