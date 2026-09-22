import pygame
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import json  # <--- IMPORTANTE PARA EXPORTAR

# --- CONFIGURACIÓN VISUAL Y DE ESTRÉS ---
CANTIDAD_PARTICULAS = 2000 # 5000 es brutal, probemos 2000 para que sea fluido y se vea el error.
ANCHO, ALTO = 1400, 700    
MITAD = ANCHO // 2
RADIO_NUCLEO = 100
DURACION_DEMO = 30         

# Colores Corporativos
NEGRO = (10, 12, 16)
ROJO_FALLO = (255, 50, 50)
AZUL_W = (0, 190, 255)
GRIS_CAJA = (60, 60, 60) # Para mostrar la "mentira"
BLANCO = (220, 220, 220)

# --- CLASE PARTÍCULA ---
class Particula:
    def __init__(self, x, y, vx, vy, r):
        self.r = r
        # Estado Izquierdo (Tradicional)
        self.pos_t = pygame.Vector2(x, y)
        self.vel_t = pygame.Vector2(vx, vy)
        self.col_t = False
        
        # Estado Derecho (W)
        self.pos_w = pygame.Vector2(x + MITAD, y)
        self.vel_w = pygame.Vector2(vx, vy)
        self.col_w = False

    def update_traditional(self, core_rect):
        self.pos_t += self.vel_t
        
        # HITBOX CUADRADA (La trampa estándar de la industria)
        my_rect = pygame.Rect(self.pos_t.x - self.r, self.pos_t.y - self.r, 
                              self.r*2, self.r*2)
        
        self.col_t = False
        # COLISIÓN AABB
        if my_rect.colliderect(core_rect):
            self.col_t = True
            # Rebote de caja
            if abs(my_rect.centerx - core_rect.centerx) > abs(my_rect.centery - core_rect.centery):
                self.vel_t.x *= -1
            else:
                self.vel_t.y *= -1
        
        # Paredes
        if self.pos_t.x < 0 or self.pos_t.x > MITAD: self.vel_t.x *= -1
        if self.pos_t.y < 0 or self.pos_t.y > ALTO: self.vel_t.y *= -1

    def update_w(self, core_pos, core_radius):
        self.pos_w += self.vel_w
        
        # COLISIÓN VOLUMÉTRICA (Física Real)
        dist = self.pos_w.distance_to(core_pos)
        min_dist = self.r + core_radius
        
        self.col_w = False
        if dist < min_dist:
            self.col_w = True
            # Rebote Vectorial
            normal = (self.pos_w - core_pos).normalize()
            self.pos_w = core_pos + normal * min_dist
            self.vel_w = self.vel_w.reflect(normal)
            
        # Paredes
        if self.pos_w.x < MITAD or self.pos_w.x > ANCHO: self.vel_w.x *= -1
        if self.pos_w.y < 0 or self.pos_w.y > ALTO: self.vel_w.y *= -1

    def draw(self, surface):
        # IZQUIERDA (TRADICIONAL)
        # Dibujamos visualmente una esfera, pero lógicamente es un cuadrado.
        # Si choca (col_t es True), se pone ROJA aunque visualmente no toque el núcleo.
        color_t = ROJO_FALLO if self.col_t else (100, 100, 100)
        pygame.draw.circle(surface, color_t, (int(self.pos_t.x), int(self.pos_t.y)), self.r)

        # DERECHA (W)
        color_w = AZUL_W if self.col_w else (100, 100, 120)
        pygame.draw.circle(surface, color_w, (int(self.pos_w.x), int(self.pos_w.y)), self.r)

# --- EJECUCIÓN ---
def run_live_demo():
    pygame.init()
    screen = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("W Systems | The Hitbox Lie Protocol")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)
    font_hud = pygame.font.SysFont("monospace", 12)

    particulas = []
    for _ in range(CANTIDAD_PARTICULAS):
        vx = random.uniform(-4, 4)
        vy = random.uniform(-4, 4)
        r = random.randint(4, 10)
        particulas.append(Particula(random.randint(50, MITAD-50), random.randint(50, 300), vx, vy, r))

    # Definición de Núcleos
    # Izquierda: Rect para lógica, Pos para visual
    core_rect = pygame.Rect(MITAD//2 - RADIO_NUCLEO, ALTO//2 - RADIO_NUCLEO, RADIO_NUCLEO*2, RADIO_NUCLEO*2)
    core_pos_trad = pygame.Vector2(MITAD//2, ALTO//2)
    
    # Derecha: Pos para lógica y visual
    core_pos_weyl = pygame.Vector2(MITAD + MITAD//2, ALTO//2)

    # Estructura de datos para JSON
    metrics_data = {
        "metadata": {
            "test_name": "W Truth Test (Sphere vs AABB)",
            "particles": CANTIDAD_PARTICULAS,
            "duration": DURACION_DEMO
        },
        "frames": []
    }
    
    start_time = time.time()
    running = True
    
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: running = False

        elapsed = time.time() - start_time
        if elapsed > DURACION_DEMO: running = False

        # MEDICIÓN
        t0 = time.perf_counter()
        cols_trad = 0
        for p in particulas:
            p.update_traditional(core_rect)
            if p.col_t: cols_trad += 1
        cpu_trad = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        cols_weyl = 0
        for p in particulas:
            p.update_w(core_pos_weyl, RADIO_NUCLEO)
            if p.col_w: cols_weyl += 1
        cpu_weyl = (time.perf_counter() - t1) * 1000

        # Guardar Frame
        metrics_data["frames"].append({
            "timestamp": round(elapsed, 2),
            "cpu_trad_ms": round(cpu_trad, 4),
            "cpu_weyl_ms": round(cpu_weyl, 4),
            "active_collisions_trad": cols_trad,
            "active_collisions_weyl": cols_weyl
        })

        # RENDER
        screen.fill(NEGRO)
        pygame.draw.line(screen, (50, 50, 50), (MITAD, 0), (MITAD, ALTO), 3)

        # --- DIBUJAR NÚCLEOS ---
        
        # IZQUIERDA (LA MENTIRA)
        # 1. Dibujamos la caja invisible MUY suave para que se entienda el error
        pygame.draw.rect(screen, (30, 30, 30), core_rect, 1) 
        # 2. Dibujamos la Esfera Visual (lo que el jugador ve)
        pygame.draw.circle(screen, (50, 0, 0), (int(core_pos_trad.x), int(core_pos_trad.y)), RADIO_NUCLEO)
        pygame.draw.circle(screen, ROJO_FALLO, (int(core_pos_trad.x), int(core_pos_trad.y)), RADIO_NUCLEO, 2)
        
        # DERECHA (LA VERDAD)
        pygame.draw.circle(screen, (0, 30, 60), (int(core_pos_weyl.x), int(core_pos_weyl.y)), RADIO_NUCLEO)
        pygame.draw.circle(screen, AZUL_W, (int(core_pos_weyl.x), int(core_pos_weyl.y)), RADIO_NUCLEO, 2)

        # Partículas
        for p in particulas:
            p.draw(screen)

        # HUD
        # Izquierda
        screen.blit(font.render("VISUAL: ESFERA | LÓGICA: CAJA (AABB)", True, ROJO_FALLO), (20, 20))
        screen.blit(font_hud.render("Nota: Las partículas chocan con el 'aire' (esquinas invisibles)", True, BLANCO), (20, 45))
        screen.blit(font_hud.render(f"Colisiones Falsas + Reales: {cols_trad}", True, BLANCO), (20, 65))
        
        # Derecha
        screen.blit(font.render("VISUAL: ESFERA | LÓGICA: W", True, AZUL_W), (MITAD + 20, 20))
        screen.blit(font_hud.render("Física Volumétrica 1:1", True, BLANCO), (MITAD + 20, 45))
        screen.blit(font_hud.render(f"Colisiones Reales: {cols_weyl}", True, BLANCO), (MITAD + 20, 65))
        
        # Timer
        screen.blit(font.render(f"TIEMPO: {int(DURACION_DEMO - elapsed)}s", True, BLANCO), (MITAD - 50, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    
    # --- EXPORTAR JSON AL FINAL ---
    print("Guardando datos de telemetría...")
    nombre_archivo = "w_telemetry.json"
    with open(nombre_archivo, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"¡ÉXITO! Datos guardados en {nombre_archivo}")
    
    # Generar Gráfico Rápido
    generar_grafico_final(metrics_data)

def generar_grafico_final(data):
    df = pd.DataFrame(data["frames"])
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar Colisiones para mostrar los "Phantom Hits"
    ax.plot(df["timestamp"], df["active_collisions_trad"], color="#ff4444", label="AABB (Incluye choques fantasma)", linestyle="--")
    ax.plot(df["timestamp"], df["active_collisions_weyl"], color="#00aaff", label="W (Solo contactos reales)")
    
    ax.fill_between(df["timestamp"], df["active_collisions_trad"], df["active_collisions_weyl"], color="#ff4444", alpha=0.2, label="Basura Computacional")
    
    ax.set_title("COMPARATIVA DE PRECISIÓN DE IMPACTO")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Eventos de Colisión")
    ax.legend()
    
    plt.savefig("w_truth_graph.png")
    plt.show()

if __name__ == "__main__":
    run_live_demo()