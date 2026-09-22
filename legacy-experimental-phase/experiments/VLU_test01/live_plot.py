import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import sys
import os

# --- CONFIGURACIÓN ---
# Nombre del ejecutable C++ compilado
Ejecutable_CPP = "./w_engine.exe" if os.name == 'nt' else "./w_engine"
MAX_PUNTOS = 200 # Cuántos puntos mostrar en el gráfico móvil

# --- ESTRUCTURAS DE DATOS ---
data_branching = deque([0]*MAX_PUNTOS, maxlen=MAX_PUNTOS)
data_branchless = deque([0]*MAX_PUNTOS, maxlen=MAX_PUNTOS)
x_axis = deque(range(MAX_PUNTOS), maxlen=MAX_PUNTOS)

# --- INICIAR EL MOTOR C++ ---
try:
    # Iniciamos el proceso C++ y leemos su salida estándar (stdout)
    process = subprocess.Popen(
        [Ejecutable_CPP],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 # Line buffering
    )
    print(f"Motor C++ iniciado: {Ejecutable_CPP}")
    print("Procesando 10 MILLONES de puntos por ciclo...")
except FileNotFoundError:
    print(f"ERROR: No encuentro '{Ejecutable_CPP}'. Asegurate de compilar el C++ primero.")
    sys.exit(1)

# --- CONFIGURACIÓN DEL GRÁFICO ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('W Systems | C++ Hardware Core Benchmark')

line_branching, = ax.plot([], [], color='#ff4444', label='Tradicional (IF/ELSE Branching)', linewidth=2)
line_branchless, = ax.plot([], [], color='#00aaff', label='W (Aritmética Branchless)', linewidth=3)

ax.set_xlim(0, MAX_PUNTOS-1)
ax.set_ylim(0, 1) # Se ajustará dinámicamente
ax.set_title('RENDIMIENTO EN C++ (MENOS ES MEJOR)', fontsize=14, fontweight='bold')
ax.set_ylabel('Tiempo de Proceso (Microsegundos)', fontsize=12)
ax.set_xlabel('Ciclos de Prueba (Tiempo Real)', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper left')

# Textos de estado
text_delta = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=12, 
                     bbox=dict(facecolor='#222222', alpha=0.8, edgecolor='none'))

# --- FUNCIÓN DE ACTUALIZACIÓN (ANIMACIÓN) ---
def animate(i):
    # Leer datos del motor C++ si están disponibles
    while True:
        line = process.stdout.readline()
        if not line: break
        
        parts = line.strip().split(',')
        if len(parts) == 2:
            tipo, tiempo_str = parts
            try:
                tiempo_us = int(tiempo_str)
                if tipo == "BRANCHING":
                    data_branching.append(tiempo_us)
                elif tipo == "BRANCHLESS":
                    data_branchless.append(tiempo_us)
                    # Cuando tenemos ambos datos, actualizamos el gráfico
                    break 
            except ValueError: continue

    # Actualizar líneas
    line_branching.set_data(x_axis, data_branching)
    line_branchless.set_data(x_axis, data_branchless)
    
    # Ajustar escala vertical dinámicamente
    if len(data_branching) > 0:
        max_y = max(max(data_branching), max(data_branchless)) * 1.2
        ax.set_ylim(0, max(100, max_y)) # Mínimo 100us para que no colapse

        # Calcular Delta
        avg_b = sum(data_branching)/len(data_branching)
        avg_w = sum(data_branchless)/len(data_branchless)
        if avg_b > 0:
            improvement = ((avg_b - avg_w) / avg_b) * 100
            color_delta = '#00ff00' if improvement > 0 else '#ff0000'
            text_delta.set_text(f"MEJORA W: +{improvement:.1f}% MÁS RÁPIDO")
            text_delta.set_color(color_delta)

    return line_branching, line_branchless, text_delta

# --- EJECUTAR ---
# Intervalo bajo para que lea lo más rápido posible
ani = animation.FuncAnimation(fig, animate, interval=10, blit=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("Cerrando...")
finally:
    process.terminate()
    print("Motor C++ detenido.")