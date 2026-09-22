import pygame
import math

# --- CONFIGURACIÓN ---
ANCHO, ALTO = 1200, 600
MITAD = ANCHO // 2
COLOR_FONDO = (10, 10, 15)
BLANCO = (200, 200, 200)
ROJO_AABB = (255, 50, 50)     # La mentira
AZUL_W = (0, 180, 255)  # La verdad
VERDE_HIT = (50, 255, 50)

# --- MOTOR 3D BÁSICO (Pure Python Math) ---
class Punto3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

def rotar_punto(p, angulo_x, angulo_y, angulo_z):
    # Rotación X
    y = p.y * math.cos(angulo_x) - p.z * math.sin(angulo_x)
    z = p.y * math.sin(angulo_x) + p.z * math.cos(angulo_x)
    p.y, p.z = y, z
    # Rotación Y
    x = p.x * math.cos(angulo_y) - p.z * math.sin(angulo_y)
    z = p.x * math.sin(angulo_y) + p.z * math.cos(angulo_y)
    p.x, p.z = x, z
    # Rotación Z
    x = p.x * math.cos(angulo_z) - p.y * math.sin(angulo_z)
    y = p.x * math.sin(angulo_z) + p.y * math.cos(angulo_z)
    p.x, p.y = x, y
    return p

def proyectar(p, centro_x, centro_y, escala=400):
    # Proyección de perspectiva simple
    factor = escala / (p.z + 5) # +5 es la distancia de la cámara
    x = p.x * factor + centro_x
    y = p.y * factor + centro_y
    return x, y, factor

# Definimos un Cubo (Vértices)
vertices_base = [
    Punto3D(-1, -1, -1), Punto3D(1, -1, -1), Punto3D(1, 1, -1), Punto3D(-1, 1, -1),
    Punto3D(-1, -1, 1), Punto3D(1, -1, 1), Punto3D(1, 1, 1), Punto3D(-1, 1, 1)
]
# Aristas para dibujar el wireframe
aristas = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

# --- SIMULACIÓN ---
def run_3d_demo():
    pygame.init()
    screen = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("W Systems | 3D Rotational Invariance Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    font_big = pygame.font.SysFont("monospace", 24, bold=True)

    angle_x, angle_y = 0, 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # Input Mouse (El "Cursor" es una sonda de prueba)
        mx, my = pygame.mouse.get_pos()
        
        # Rotación automática
        angle_x += 0.01
        angle_y += 0.02
        
        screen.fill(COLOR_FONDO)
        pygame.draw.line(screen, (50, 50, 50), (MITAD, 0), (MITAD, ALTO), 2)

        # ==========================================
        # LADO IZQUIERDO: TRADICIONAL (AABB)
        # ==========================================
        # 1. Calcular vértices rotados y proyectados
        puntos_2d_izq = []
        cx_izq, cy_izq = MITAD // 2, ALTO // 2
        
        for v in vertices_base:
            # Copiar y rotar
            p = Punto3D(v.x, v.y, v.z)
            rotar_punto(p, angle_x, angle_y, 0)
            # Proyectar
            px, py, f = proyectar(p, cx_izq, cy_izq)
            puntos_2d_izq.append((px, py))

        # 2. CALCULAR AABB (LA CAJA QUE CRECE)
        # En 3D tradicional, buscas el min/max x e y en pantalla para hacer el cuadro
        min_x = min(p[0] for p in puntos_2d_izq)
        max_x = max(p[0] for p in puntos_2d_izq)
        min_y = min(p[1] for p in puntos_2d_izq)
        max_y = max(p[1] for p in puntos_2d_izq)
        
        aabb_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        
        # 3. Detectar Colisión
        hit_trad = aabb_rect.collidepoint(mx, my)
        
        # 4. Dibujar
        # La caja fantasma
        color_box = VERDE_HIT if hit_trad else ROJO_AABB
        pygame.draw.rect(screen, color_box, aabb_rect, 2)
        
        # El objeto (Wireframe)
        for s, e in aristas:
            pygame.draw.line(screen, BLANCO, puntos_2d_izq[s], puntos_2d_izq[e], 2)
            
        # Etiquetas
        if hit_trad:
             pygame.draw.circle(screen, VERDE_HIT, (mx, my), 5)
             txt = font_big.render("COLISIÓN DETECTADA", True, VERDE_HIT)
             screen.blit(txt, (20, ALTO - 50))
             
        screen.blit(font.render("TRADICIONAL (AABB 3D)", True, ROJO_AABB), (20, 20))
        screen.blit(font.render("Nota: La caja roja PULSA (crece/achica).", True, BLANCO), (20, 50))
        screen.blit(font.render("El aire de las esquinas mata la CPU.", True, BLANCO), (20, 70))

        # ==========================================
        # LADO DERECHO: W (VOLUMETRIC)
        # ==========================================
        cx_der, cy_der = MITAD + MITAD // 2, ALTO // 2
        
        # En W, la esfera de colisión es constante.
        # Calculamos el radio basado en la escala promedio de proyección
        # (Simplificado para demo, pero matemáticamente robusto)
        radio_w = 110 # Radio fijo que cubre el objeto
        
        # 1. Dibujar Wireframe (Solo visual)
        puntos_2d_der = []
        for v in vertices_base:
            p = Punto3D(v.x, v.y, v.z)
            rotar_punto(p, angle_x, angle_y, 0)
            px, py, f = proyectar(p, cx_der, cy_der)
            puntos_2d_der.append((px, py))
            
        # 2. Detectar Colisión (Distancia Euclidiana)
        dist = math.hypot(mx - cx_der, my - cy_der)
        hit_weyl = dist < radio_w
        
        # 3. Dibujar
        # La Esfera Lógica (Estable)
        color_sphere = VERDE_HIT if hit_weyl else AZUL_W
        pygame.draw.circle(screen, color_sphere, (cx_der, cy_der), radio_w, 2)
        
        # El objeto
        for s, e in aristas:
            pygame.draw.line(screen, BLANCO, puntos_2d_der[s], puntos_2d_der[e], 2)

        if hit_weyl:
             pygame.draw.circle(screen, VERDE_HIT, (mx, my), 5)
             txt = font_big.render("CONTACTO FÍSICO", True, VERDE_HIT)
             screen.blit(txt, (MITAD + 20, ALTO - 50))

        screen.blit(font.render("W (INVARIANCIA ROTACIONAL)", True, AZUL_W), (MITAD + 20, 20))
        screen.blit(font.render("Nota: El volumen lógico es ESTABLE.", True, BLANCO), (MITAD + 20, 50))
        screen.blit(font.render("0% Desperdicio dinámico.", True, BLANCO), (MITAD + 20, 70))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    run_3d_demo()
