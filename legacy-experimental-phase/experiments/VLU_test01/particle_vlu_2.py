import pygame
import random
import time
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURACIÓN VISUAL Y DE ESTRÉS ---
CANTIDAD_PARTICULAS = 5000 # Ajustá esto según tu PC. 1200 se ve increíble.
ANCHO, ALTO = 1400, 700    # Pantalla ancha para ver la comparación
MITAD = ANCHO // 2
RADIO_NUCLEO = 100
DURACION_DEMO = 30         # Segundos antes de generar el reporte

# Colores Corporativos
NEGRO = (10, 12, 16)
ROJO_FALLO = (255, 50, 50)
AZUL_W = (0, 190, 255)
GRIS_CAJA = (60, 60, 60)
BLANCO = (220, 220, 220)

# --- CLASE PARTÍCULA ---
class Particula:
    def __init__(self, x, y, vx, vy, r):
        self.r = r
        # Estado Izquierdo (Tradicional)
        self.pos_t = pygame.Vector2(x, y)
        self.vel_t = pygame.Vector2(vx, vy)
        self.col_t = False
        
        # Estado Derecho (W) - Mismas condiciones iniciales
        self.pos_w = pygame.Vector2(x + MITAD, y)
        self.vel_w = pygame.Vector2(vx, vy)
        self.col_w = False

    def update_traditional(self, core_rect):
        # Mover
        self.pos_t += self.vel_t
        
        # Crear la CAJA (La mentira geométrica)
        my_rect = pygame.Rect(self.pos_t.x - self.r, self.pos_t.y - self.r, 
                              self.r*2, self.r*2)
        
        self.col_t = False
        # COLISIÓN AABB (Caja contra Caja)
        if my_rect.colliderect(core_rect):
            self.col_t = True
            # Rebote simple (Inexacto)
            if abs(my_rect.centerx - core_rect.centerx) > abs(my_rect.centery - core_rect.centery):
                self.vel_t.x *= -1
            else:
                self.vel_t.y *= -1
        
        # Paredes
        if self.pos_t.x < 0 or self.pos_t.x > MITAD: self.vel_t.x *= -1
        if self.pos_t.y < 0 or self.pos_t.y > ALTO: self.vel_t.y *= -1

    def update_w(self, core_pos, core_radius):
        # Mover
        self.pos_w += self.vel_w
        
        # COLISIÓN VOLUMÉTRICA (Esfera contra Esfera)
        dist = self.pos_w.distance_to(core_pos)
        min_dist = self.r + core_radius
        
        self.col_w = False
        if dist < min_dist:
            self.col_w = True
            # Rebote Vectorial (Física Real)
            normal = (self.pos_w - core_pos).normalize()
            self.pos_w = core_pos + normal * min_dist # Corrección antisolapamiento
            self.vel_w = self.vel_w.reflect(normal)
            
        # Paredes
        if self.pos_w.x < MITAD or self.pos_w.x > ANCHO: self.vel_w.x *= -1
        if self.pos_w.y < 0 or self.pos_w.y > ALTO: self.vel_w.y *= -1

    def draw(self, surface):
        # --- LADO IZQUIERDO (TRADICIONAL) ---
        # Dibujamos la CAJA GRIS para mostrar "la hitbox invisible"
        rect_t = pygame.Rect(self.pos_t.x - self.r, self.pos_t.y - self.r, self.r*2, self.r*2)
        color_t = ROJO_FALLO if self.col_t else GRIS_CAJA
        
        # Si hay colisión, rellenamos el cuadrado para que se vea el error
        width = 0 if self.col_t else 1 
        pygame.draw.rect(surface, color_t, rect_t, width)
        
        # El puntito central
        pygame.draw.circle(surface, color_t, (int(self.pos_t.x), int(self.pos_t.y)), 2)

        # --- LADO DERECHO (W) ---
        color_w = AZUL_W if self.col_w else (100, 100, 120)
        # Aquí dibujamos ESFERAS reales
        pygame.draw.circle(surface, color_w, (int(self.pos_w.x), int(self.pos_w.y)), self.r)

# --- EJECUCIÓN ---
def run_live_demo():
    pygame.init()
    screen = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("W Systems | Visual Benchmark Protocol")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)
    font_hud = pygame.font.SysFont("monospace", 12)

    # Crear enjambre
    particulas = []
    for _ in range(CANTIDAD_PARTICULAS):
        vx = random.uniform(-4, 4)
        vy = random.uniform(-4, 4)
        r = random.randint(4, 10)
        # Nacen arriba al centro de cada mitad
        particulas.append(Particula(random.randint(50, MITAD-50), random.randint(50, 300), vx, vy, r))

    # Núcleos
    core_rect = pygame.Rect(MITAD//2 - RADIO_NUCLEO, ALTO//2 - RADIO_NUCLEO, RADIO_NUCLEO*2, RADIO_NUCLEO*2)
    core_pos = pygame.Vector2(MITAD + MITAD//2, ALTO//2)

    # Datos para gráfico
    data_log = {"t": [], "cpu_t": [], "cpu_w": [], "cols_t": [], "cols_w": []}
    start_time = time.time()

    running = True
    while running:
        # 1. INPUT
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: running = False

        elapsed = time.time() - start_time
        if elapsed > DURACION_DEMO: running = False

        # 2. LÓGICA & MEDICIÓN (Separadas para precisión)
        
        # Medir Tradicional
        t0 = time.perf_counter()
        cols_trad_count = 0
        for p in particulas:
            p.update_traditional(core_rect)
            if p.col_t: cols_trad_count += 1
        cpu_trad = (time.perf_counter() - t0) * 1000

        # Medir W
        t1 = time.perf_counter()
        cols_weyl_count = 0
        for p in particulas:
            p.update_w(core_pos, RADIO_NUCLEO)
            if p.col_w: cols_weyl_count += 1
        cpu_weyl = (time.perf_counter() - t1) * 1000

        # Loguear
        data_log["t"].append(elapsed)
        data_log["cpu_t"].append(cpu_trad)
        data_log["cpu_w"].append(cpu_weyl)
        data_log["cols_t"].append(cols_trad_count)
        data_log["cols_w"].append(cols_weyl_count)

        # 3. RENDERIZADO (El Show)
        screen.fill(NEGRO)
        
        # Línea divisoria
        pygame.draw.line(screen, (50, 50, 50), (MITAD, 0), (MITAD, ALTO), 3)

        # Dibujar Núcleos
        # Izquierda: La CAJA (Visiblemente cuadrada)
        pygame.draw.rect(screen, (40, 0, 0), core_rect)
        pygame.draw.rect(screen, ROJO_FALLO, core_rect, 2)
        
        # Derecha: La ESFERA (Visiblemente redonda)
        pygame.draw.circle(screen, (0, 20, 40), (int(core_pos.x), int(core_pos.y)), RADIO_NUCLEO)
        pygame.draw.circle(screen, AZUL_W, (int(core_pos.x), int(core_pos.y)), RADIO_NUCLEO, 2)

        # Dibujar Partículas
        for p in particulas:
            p.draw(screen)

        # 4. HUD (DATOS EN VIVO)
        # Izquierda
        lbl_t1 = font.render("MÉTODO TRADICIONAL (AABB)", True, ROJO_FALLO)
        lbl_t2 = font_hud.render(f"Colisiones Activas: {cols_trad_count} (Incluye Phantom Hits)", True, BLANCO)
        lbl_t3 = font_hud.render(f"CPU Time: {cpu_trad:.3f} ms", True, BLANCO)
        screen.blit(lbl_t1, (20, 20))
        screen.blit(lbl_t2, (20, 50))
        screen.blit(lbl_t3, (20, 70))
        
        # Derecha
        lbl_w1 = font.render("MÉTODO W (VOLUMETRIC)", True, AZUL_W)
        lbl_w2 = font_hud.render(f"Colisiones Activas: {cols_weyl_count} (Física Real)", True, BLANCO)
        lbl_w3 = font_hud.render(f"CPU Time: {cpu_weyl:.3f} ms", True, BLANCO)
        screen.blit(lbl_w1, (MITAD + 20, 20))
        screen.blit(lbl_w2, (MITAD + 20, 50))
        screen.blit(lbl_w3, (MITAD + 20, 70))

        # Timer
        lbl_time = font.render(f"TIEMPO RESTANTE: {DURACION_DEMO - int(elapsed)}s", True, BLANCO)
        screen.blit(lbl_time, (MITAD - 100, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    generar_reporte(data_log)

def generar_reporte(data):
    print("Generando gráfico final...")
    df = pd.DataFrame(data)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Suavizado para mejor lectura
    window = 20
    ax.plot(df["t"], df["cpu_t"].rolling(window).mean(), color="#ff4444", label="Tradicional (AABB)", linewidth=2)
    ax.plot(df["t"], df["cpu_w"].rolling(window).mean(), color="#00aaff", label="W (Volumetric)", linewidth=2)
    
    ax.set_title(f"BENCHMARK FINAL - {CANTIDAD_PARTICULAS} OBJETOS")
    ax.set_ylabel("Tiempo de CPU (ms)")
    ax.set_xlabel("Segundos")
    ax.legend()
    ax.grid(alpha=0.2)
    
    plt.savefig("w_live_report.png")
    plt.show()

if __name__ == "__main__":
    run_live_demo()