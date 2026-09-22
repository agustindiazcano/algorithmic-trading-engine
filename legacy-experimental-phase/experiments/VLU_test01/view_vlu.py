import pygame
import math

# --- CONFIGURACIÓN DE W SYSTEMS ---
ANCHO, ALTO = 800, 600
FPS = 60

# Colores "Corporativos" W
NEGRO_FONDO = (20, 20, 30) # Gris oscuro aséptico
BLANCO_TITANIO = (220, 220, 220)
AZUL_W = (0, 120, 255) # Color de la "Intención"
ROJO_FILTRADO = (200, 50, 50) # Cuando la lógica bloquea
VERDE_FLUJO = (50, 200, 50)   # Cuando la lógica pasa

# Inicializar Pygame
pygame.init()
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("W Systems - Volumetric Logic Proof of Concept")
reloj = pygame.time.Clock()
fuente_ui = pygame.font.SysFont("monospace", 20, bold=True)

# --- CLASES DE LA LÓGICA VOLUMÉTRICA ---

class WSphere:
    def __init__(self, x, y, radius, mass):
        self.x = x
        self.y = y
        self.radius = radius  # Incertidumbre (Volumen)
        self.mass = mass      # Prioridad (Inercia)

    def dibujar(self, superficie):
        # Dibujamos el "Volumen de Incertidumbre" con transparencia
        superficie_transparente = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(superficie_transparente, (AZUL_W[0], AZUL_W[1], AZUL_W[2], 100), (self.radius, self.radius), self.radius)
        superficie.blit(superficie_transparente, (self.x - self.radius, self.y - self.radius))
        
        # Dibujamos el "Centro de Intención" sólido
        pygame.draw.circle(superficie, BLANCO_TITANIO, (self.x, self.y), 5)

class VolumetricGate:
    def __init__(self, x, y, aperture):
        self.x = x
        self.y = y
        self.aperture = aperture # Ancho del canal

    def dibujar(self, superficie):
        # Dibujamos el canal como dos bloques físicos
        grosor_bloque = 20
        alto_bloque = 150
        
        # Bloque superior
        pygame.draw.rect(superficie, BLANCO_TITANIO, (self.x - grosor_bloque//2, self.y - self.aperture - alto_bloque, grosor_bloque, alto_bloque))
        # Bloque inferior
        pygame.draw.rect(superficie, BLANCO_TITANIO, (self.x - grosor_bloque//2, self.y + self.aperture, grosor_bloque, alto_bloque))
        
        # Línea central del canal (guía visual)
        pygame.draw.line(superficie, (50, 50, 50), (self.x, self.y - self.aperture), (self.x, self.y + self.aperture), 2)

# --- EL NÚCLEO MATEMÁTICO (La "Tostadora" Logic) ---
def calcular_flujo_w(esfera, puerta):
    # 1. Distancia Euclidiana simple (barata de calcular)
    dist_x = abs(esfera.x - puerta.x)
    # Solo nos importa la distancia en el eje del canal para esta demo simple
    
    # 2. Interferencia Geométrica: ¿El volumen de incertidumbre cabe en la apertura?
    # Si el radio de la esfera es mayor que la distancia al borde del canal, hay solapamiento.
    # Calculamos el margen que queda entre el centro de la esfera y el borde del canal
    margen_disponible = puerta.aperture - abs(esfera.y - puerta.y)
    
    # Si el radio de la esfera es menor que el margen, pasa limpio. Si es mayor, hay "fricción".
    # Esta fórmula calcula qué porcentaje del "volumen" logra pasar.
    # Usamos max(0, ...) para que no dé negativo.
    interferencia = max(0, margen_disponible / esfera.radius) if esfera.radius > 0 else 0

    # 3. Factor de Masa (Prioridad):
    # Una masa alta puede "forzar" el paso incluso con interferencia.
    # Una masa baja (ruido) es filtrada fácilmente.
    factor_masa = math.log1p(esfera.mass) # Usamos log para suavizar el efecto

    flujo_final = interferencia * factor_masa
    
    # Normalizamos a 0.0 - 1.0 para visualizar
    return min(1.0, flujo_final)

# --- BUCLE PRINCIPAL DEL JUEGO ---

# Inicializar objetos
mi_esfera = WSphere(ANCHO//4, ALTO//2, radius=40.0, mass=2.0)
la_puerta = VolumetricGate(ANCHO//2 + 100, ALTO//2, aperture=50.0)
flujo_actual = 0.0

ejecutando = True
while ejecutando:
    # 1. Manejo de Eventos (Input)
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False

    # Control de la esfera con el mouse
    mouse_pos = pygame.mouse.get_pos()
    mi_esfera.x, mi_esfera.y = mouse_pos

    # Control de parámetros W con teclado
    teclas = pygame.key.get_pressed()
    # Flechas ARRIBA/ABAJO cambian el RADIO (Incertidumbre/Ruido)
    if teclas[pygame.K_UP]: mi_esfera.radius = min(150, mi_esfera.radius + 1)
    if teclas[pygame.K_DOWN]: mi_esfera.radius = max(10, mi_esfera.radius - 1)
    # Flechas IZQ/DER cambian la MASA (Prioridad/Inercia)
    if teclas[pygame.K_RIGHT]: mi_esfera.mass = min(10.0, mi_esfera.mass + 0.1)
    if teclas[pygame.K_LEFT]: mi_esfera.mass = max(0.1, mi_esfera.mass - 0.1)

    # 2. ACTUALIZAR LÓGICA (El Cerebro W)
    # Aquí ocurre la magia: una sola línea de matemática decide todo.
    flujo_actual = calcular_flujo_w(mi_esfera, la_puerta)

    # 3. DIBUJAR (Render)
    pantalla.fill(NEGRO_FONDO)

    la_puerta.dibujar(pantalla)
    mi_esfera.dibujar(pantalla)

    # --- DIBUJAR UI (Dashboard W) ---
    # Barra de Flujo (Output del sistema)
    color_barra = VERDE_FLUJO if flujo_actual > 0.8 else (ROJO_FILTRADO if flujo_actual < 0.3 else AZUL_W)
    pygame.draw.rect(pantalla, (50, 50, 50), (50, ALTO - 80, ANCHO - 100, 30)) # Fondo barra
    pygame.draw.rect(pantalla, color_barra, (50, ALTO - 80, (ANCHO - 100) * flujo_actual, 30)) # Relleno barra
    
    texto_flujo = fuente_ui.render(f"W FLOW OUTPUT: {flujo_actual:.4f} ({flujo_actual*100:.1f}%)", True, BLANCO_TITANIO)
    pantalla.blit(texto_flujo, (50, ALTO - 110))

    # Textos de Parámetros
    info_radio = fuente_ui.render(f"[UP/DOWN] Radio (Incertidumbre): {mi_esfera.radius:.1f}", True, BLANCO_TITANIO)
    info_masa = fuente_ui.render(f"[LEFT/RIGHT] Masa (Prioridad): {mi_esfera.mass:.1f}", True, BLANCO_TITANIO)
    instrucciones = fuente_ui.render("Mueve el mouse. Intenta pasar la señal por el canal.", True, AZUL_W)

    pantalla.blit(info_radio, (50, 20))
    pantalla.blit(info_masa, (50, 50))
    pantalla.blit(instrucciones, (50, 90))

    # Actualizar pantalla
    pygame.display.flip()
    reloj.tick(FPS)

pygame.quit()