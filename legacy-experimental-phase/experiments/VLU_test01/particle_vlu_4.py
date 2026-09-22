import pygame
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import json

# --- CONFIGURACIÓN ---
CANTIDAD_PARTICULAS = 1500 # Un buen número para ver el flujo
ANCHO, ALTO = 1400, 700    
MITAD = ANCHO // 2
RADIO_NUCLEO = 100 # Esto ahora será la mitad del ancho del cuadrado
DURACION_DEMO = 30         

# Colores
NEGRO = (10, 12, 16)
ROJO_TRAD = (255, 50, 50)
AZUL_W = (0, 190, 255)
BLANCO = (220, 220, 220)

# --- CLASE PARTÍCULA (Lógica idéntica al anterior) ---
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
        # Hitbox de la partícula
        my_rect = pygame.Rect(self.pos_t.x - self.r, self.pos_t.y - self.r, 
                              self.r*2, self.r*2)
        self.col_t = False
        # COLISIÓN AABB (Caja vs Caja)
        if my_rect.colliderect(core_rect):
            self.col_t = True
            # Rebote simple
            if abs(my_rect.centerx - core_rect.centerx) > abs(my_rect.centery - core_rect.centery):
                self.vel_t.x *= -1
            else:
                self.vel_t.y *= -1
        # Paredes
        if self.pos_t.x < 0 or self.pos_t.x > MITAD: self.vel_t.x *= -1
        if self.pos_t.y < 0 or self.pos_t.y > ALTO: self.vel_t.y *= -1

    def update_w(self, core_pos, core_radius):
        self.pos_w += self.vel_w
        # COLISIÓN VOLUMÉTRICA (Esfera vs Esfera)
        # La lógica sigue pensando que el núcleo es REDONDO
        dist = self.pos_w.distance_to(core_pos)
        min_dist = self.r + core_radius
        
        self.col_w = False
        if dist < min_dist:
            self.col_w = True
            # Rebote Vectorial Realista
            normal = (self.pos_w - core_pos).normalize()
            self.pos_w = core_pos + normal * min_dist
            self.vel_w = self.vel_w.reflect(normal)
            
        # Paredes
        if self.pos_w.x < MITAD or self.pos_w.x > ANCHO: self.vel_w.x *= -1
        if self.pos_w.y < 0 or self.pos_w.y > ALTO: self.vel_w.y *= -1

    def draw(self, surface):
        # Dibujamos las partículas
        color_t = ROJO_TRAD if self.col_t else (100, 100, 100)
        pygame.draw.circle(surface, color_t, (int(self.pos_t.x), int(self.pos_t.y)), self.r)
        
        color_w = AZUL_W if self.col_w else (100, 100, 120)
        pygame.draw.circle(surface, color_w, (int(self.pos_w.x), int(self.pos_w.y)), self.r)

# --- EJECUCIÓN ---
def run_live_demo():
    pygame.init()
    screen = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("W Systems | Square Visuals Protocol")
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
    # Izquierda: Rect para lógica AABB
    core_rect_trad = pygame.Rect(MITAD//2 - RADIO_NUCLEO, ALTO//2 - RADIO_NUCLEO, RADIO_NUCLEO*2, RADIO_NUCLEO*2)
    
    # Derecha: Posición central para lógica Esférica
    core_pos_weyl = pygame.Vector2(MITAD + MITAD//2, ALTO//2)
    # Rect visual para el lado derecho (para dibujarlo cuadrado)
    core_rect_weyl_visual = pygame.Rect(core_pos_weyl.x - RADIO_NUCLEO, core_pos_weyl.y - RADIO_NUCLEO, RADIO_NUCLEO*2, RADIO_NUCLEO*2)

    # Datos para JSON
    metrics_data = {"frames": []}
    start_time = time.time()
    running = True
    
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: running = False

        elapsed = time.time() - start_time
        if elapsed > DURACION_DEMO: running = False

        # Lógica & Medición
        cols_trad = 0
        for p in particulas:
            p.update_traditional(core_rect_trad)
            if p.col_t: cols_trad += 1

        cols_weyl = 0
        for p in particulas:
            p.update_w(core_pos_weyl, RADIO_NUCLEO)
            if p.col_w: cols_weyl += 1

        # Guardar Frame para JSON
        metrics_data["frames"].append({
            "timestamp": round(elapsed, 2),
            "active_collisions_trad": cols_trad,
            "active_collisions_weyl": cols_weyl
        })

        # RENDER
        screen.fill(NEGRO)
        pygame.draw.line(screen, (50, 50, 50), (MITAD, 0), (MITAD, ALTO), 3)

        # --- DIBUJAR NÚCLEOS (AHORA AMBOS CUADRADOS) ---
        
        # IZQUIERDA (TRADICIONAL) - Visual Cuadrado = Lógica Cuadrada
        pygame.draw.rect(screen, (50, 0, 0), core_rect_trad) # Relleno
        pygame.draw.rect(screen, ROJO_TRAD, core_rect_trad, 3) # Borde
        
        # DERECHA (W) - Visual Cuadrado != Lógica Esférica
        pygame.draw.rect(screen, (0, 30, 60), core_rect_weyl_visual) # Relleno
        pygame.draw.rect(screen, AZUL_W, core_rect_weyl_visual, 3) # Borde
        
        # OPCIONAL: Descomentá esto para ver la esfera lógica invisible que causa el "clipping"
        # pygame.draw.circle(screen, (0, 100, 255), (int(core_pos_weyl.x), int(core_pos_weyl.y)), RADIO_NUCLEO, 1)

        # Partículas
        for p in particulas:
            p.draw(screen)

        # HUD
        # Izquierda
        screen.blit(font.render("VISUAL: CUADRADO | LÓGICA: CUADRADO (AABB)", True, ROJO_TRAD), (20, 20))
        screen.blit(font_hud.render("Nota: Coincidencia perfecta. Ideal para bloques estáticos.", True, BLANCO), (20, 45))
        screen.blit(font_hud.render(f"Colisiones Activas: {cols_trad}", True, BLANCO), (20, 65))
        
        # Derecha
        screen.blit(font.render("VISUAL: CUADRADO | LÓGICA: ESFERA (W)", True, AZUL_W), (MITAD + 20, 20))
        screen.blit(font_hud.render("Nota: Desajuste visual. La lógica esférica 'corta' las esquinas.", True, BLANCO), (MITAD + 20, 45))
        screen.blit(font_hud.render(f"Colisiones Activas: {cols_weyl}", True, BLANCO), (MITAD + 20, 65))
        
        # Timer
        screen.blit(font.render(f"TIEMPO: {int(DURACION_DEMO - elapsed)}s", True, BLANCO), (MITAD - 50, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    
    # Exportar JSON
    print("Guardando datos...")
    with open("w_square_test.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    print("Datos guardados en w_square_test.json")

if __name__ == "__main__":
    run_live_demo()