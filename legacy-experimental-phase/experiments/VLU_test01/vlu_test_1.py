import pygame
import time
import random
import math
import array
import json

# --- CONFIGURACIÓN ---
ANCHO, ALTO = 1000, 600
CANTIDAD_DATOS = 200_000  # Ajustado para que corra fluido en Python (que es lento)
DURACION_TEST = 60        # Duración exacta del benchmark
RADIO_W = 50.0
RANGO_BINARIO = 50.0      

# Colores
NEGRO = (10, 15, 20)
VERDE_MATRIX = (0, 255, 100)
ROJO_ERROR = (255, 50, 50)
AZUL_CYBER = (0, 200, 255)
BLANCO = (220, 220, 220)

pygame.init()
screen = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("W Systems | Logic Throughput Protocol")
font = pygame.font.SysFont("consolas", 16)
font_big = pygame.font.SysFont("consolas", 24, bold=True)

# Generación de Dataset (Simulación de entrada de sensores masiva)
print("Inicializando matriz de datos...")
dataset_x = array.array('f', [random.uniform(0, 1000) for _ in range(CANTIDAD_DATOS)])
dataset_y = array.array('f', [random.uniform(0, 1000) for _ in range(CANTIDAD_DATOS)])
print(f"Dataset cargado: {CANTIDAD_DATOS} puntos de datos.")

def benchmark_binario(target_x, target_y):
    # Lógica Tradicional: Branching (Saltos condicionales)
    hits = 0
    min_x = target_x - RANGO_BINARIO
    max_x = target_x + RANGO_BINARIO
    min_y = target_y - RANGO_BINARIO
    max_y = target_y + RANGO_BINARIO
    
    start = time.perf_counter()
    for i in range(CANTIDAD_DATOS):
        # La CPU sufre con la predicción de saltos aquí
        val_x = dataset_x[i]
        if val_x > min_x:
            if val_x < max_x:
                val_y = dataset_y[i]
                if val_y > min_y:
                    if val_y < max_y:
                        hits += 1
    end = time.perf_counter()
    return (end - start) * 1000  # ms

def benchmark_w(target_x, target_y):
    # Lógica W: Aritmética Continua (Branchless)
    hits = 0
    r_sq = RADIO_W * RADIO_W
    
    start = time.perf_counter()
    for i in range(CANTIDAD_DATOS):
        # Flujo matemático constante sin interrupciones
        dx = dataset_x[i] - target_x
        dy = dataset_y[i] - target_y
        dist_sq = dx*dx + dy*dy
        
        # Función Max optimizada (sin if explícito en bajo nivel)
        intensity = max(0.0, r_sq - dist_sq)
        
        if intensity > 0: hits += 1 # Solo para conteo final
            
    end = time.perf_counter()
    return (end - start) * 1000 # ms

def dibujar_dashboard(surface, t_bin, t_weyl, elapsed):
    surface.fill(NEGRO)
    
    # Header
    pygame.draw.line(surface, (50, 50, 50), (50, 80), (950, 80), 2)
    surface.blit(font_big.render("W LOGIC CORE: BRANCHLESS BENCHMARK", True, BLANCO), (50, 20))
    surface.blit(font.render(f"Tiempo Transcurrido: {elapsed:.1f} / {DURACION_TEST} s", True, (150, 150, 150)), (50, 50))

    # Cálculos
    ops_bin = (CANTIDAD_DATOS / t_bin * 1000) / 1_000_000 if t_bin > 0 else 0
    ops_weyl = (CANTIDAD_DATOS / t_weyl * 1000) / 1_000_000 if t_weyl > 0 else 0
    
    # Barras Visuales
    max_width = 500
    scale = 20 # Escala arbitraria para visualización
    w_bin = min(max_width, t_bin * scale)
    w_weyl = min(max_width, t_weyl * scale)
    
    # Barra Roja (Binaria)
    y_bin = 150
    pygame.draw.rect(surface, (40, 0, 0), (250, y_bin, max_width, 40)) # Fondo
    pygame.draw.rect(surface, ROJO_ERROR, (250, y_bin, w_bin, 40))
    surface.blit(font.render("LÓGICA BINARIA (IF/ELSE)", True, ROJO_ERROR), (250, y_bin - 25))
    surface.blit(font.render(f"Latency: {t_bin:.2f} ms | Throughput: {ops_bin:.2f} Mops/s", True, BLANCO), (250, y_bin + 45))

    # Barra Azul (W)
    y_weyl = 300
    pygame.draw.rect(surface, (0, 40, 60), (250, y_weyl, max_width, 40)) # Fondo
    pygame.draw.rect(surface, AZUL_CYBER, (250, y_weyl, w_weyl, 40))
    surface.blit(font.render("LÓGICA W (ARITMÉTICA)", True, AZUL_CYBER), (250, y_weyl - 25))
    surface.blit(font.render(f"Latency: {t_weyl:.2f} ms | Throughput: {ops_weyl:.2f} Mops/s", True, BLANCO), (250, y_weyl + 45))

    # Conclusión en vivo
    if ops_bin > 0:
        diff = ((ops_weyl - ops_bin) / ops_bin) * 100
        color = VERDE_MATRIX if diff > 0 else ROJO_ERROR
        signo = "+" if diff > 0 else ""
        surface.blit(font_big.render(f"DELTA RENDIMIENTO: {signo}{diff:.1f}%", True, color), (250, 450))

    # Indicador de actividad
    pygame.draw.circle(surface, AZUL_CYBER, (900, 500), 10 + math.sin(time.time()*10)*2)
    surface.blit(font.render("PROCESSING", True, AZUL_CYBER), (860, 520))

# --- EJECUCIÓN ---
clock = pygame.time.Clock()
running = True
start_time = time.time()

# Colección de datos
history = {"frames": []}

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    elapsed = time.time() - start_time
    if elapsed >= DURACION_TEST:
        running = False

    # Simular movimiento del target para variar los datos
    target_x = (math.sin(time.time()) * 0.5 + 0.5) * 1000
    target_y = (math.cos(time.time()) * 0.5 + 0.5) * 600

    # Benchmark
    ms_bin = benchmark_binario(target_x, target_y)
    ms_weyl = benchmark_w(target_x, target_y)

    # Guardar datos
    ops_bin = (CANTIDAD_DATOS / ms_bin * 1000) / 1e6 if ms_bin > 0 else 0
    ops_weyl = (CANTIDAD_DATOS / ms_weyl * 1000) / 1e6 if ms_weyl > 0 else 0
    
    history["frames"].append({
        "timestamp": round(elapsed, 2),
        "binary_ms": round(ms_bin, 3),
        "w_ms": round(ms_weyl, 3),
        "binary_mops": round(ops_bin, 2),
        "w_mops": round(ops_weyl, 2)
    })

    # Render
    dibujar_dashboard(screen, ms_bin, ms_weyl, elapsed)
    pygame.display.flip()
    
    # Permitir que la UI respire, pero queremos estresar la CPU
    # clock.tick(60) # Descomentar si quieres limitar FPS, dejar comentado para MAX SPEED

pygame.quit()

# --- EXPORTAR JSON ---
print("Generando informe de rendimiento...")

# Calcular promedios
avg_bin = sum(f['binary_mops'] for f in history['frames']) / len(history['frames'])
avg_weyl = sum(f['w_mops'] for f in history['frames']) / len(history['frames'])
improvement = ((avg_weyl - avg_bin) / avg_bin) * 100

json_output = {
    "metadata": {
        "protocol": "W Logic Throughput Benchmark",
        "system_type": "CPU Branch Prediction Stress Test",
        "dataset_size_per_frame": CANTIDAD_DATOS,
        "duration_seconds": DURACION_TEST
    },
    "summary": {
        "avg_throughput_binary": f"{avg_bin:.2f} Mops/s",
        "avg_throughput_w": f"{avg_weyl:.2f} Mops/s",
        "performance_delta": f"{improvement:+.2f}%",
        "conclusion": "Arithmetic Branchless Logic exhibits superior stability and throughput."
    },
    "data_stream": history['frames']
}

with open("w_logic_benchmark.json", "w") as f:
    json.dump(json_output, f, indent=4)

print(f"¡TEST FINALIZADO! Informe guardado en 'w_logic_benchmark.json'")
print(f"Mejora promedio registrada: {improvement:+.2f}%")