import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CONFIGURACIÓN ---
N_POINTS = 500       # Puntos por luna
NOISE = 0.05         # Ruido "fuzziness" (para confundir MLPs)
RADIUS = 1.0         # Radio de la luna
SPEED = 0.02         # Velocidad de rotación global

def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def generate_moon(n_points, noise, tilt_angle_y, tilt_angle_z, phase_shift):
    # 1. Crear la forma base (Semiciírculo en plano XY)
    # Usamos np.linspace para crear el arco
    t = np.linspace(0, np.pi, n_points) 
    
    # Coordenadas base (forma de banana/luna)
    x = np.cos(t) * RADIUS
    y = np.sin(t) * RADIUS
    z = np.zeros(n_points)
    
    # 2. Agregar Ruido (jitter)
    x += np.random.normal(scale=noise, size=n_points)
    y += np.random.normal(scale=noise, size=n_points)
    z += np.random.normal(scale=noise, size=n_points)
    
    # Apilar en una matriz (3, N)
    points = np.vstack([x, y, z])
    
    # 3. Rotar la luna individualmente para ubicarla en 3D
    # Primero rotamos en Y para darle inclinación
    R_y = rotation_matrix_y(tilt_angle_y)
    # Luego rotamos en Z para distribuirla en el círculo
    R_z = rotation_matrix_z(tilt_angle_z)
    
    points = R_y @ points # Aplicar tilt
    points = R_z @ points # Aplicar distribución angular
    
    return points

# --- GENERACIÓN DE DATOS ---
def get_frame_data(frame_idx):
    all_moons = []
    labels = []
    
    # Generamos 5 lunas distribuidas esféricamente
    for i in range(5):
        # Ángulos mágicos para distribuir 5 objetos en 3D (aprox)
        tilt_y = np.pi / 4  # Inclinación de 45 grados
        tilt_z = (2 * np.pi / 5) * i # 72 grados de separación entre cada una
        
        moon = generate_moon(N_POINTS, NOISE, tilt_y, tilt_z, 0)
        
        # 4. ROTACIÓN GLOBAL (SIMULACIÓN DE TIEMPO)
        # Todo el sistema gira complejo en 3 ejes para que no sea fácil predecir
        global_rot = rotation_matrix_z(frame_idx * SPEED) @ rotation_matrix_x(frame_idx * SPEED * 0.5)
        moon_rotated = global_rot @ moon
        
        all_moons.append(moon_rotated)
        labels.append(np.full(N_POINTS, i)) # Guardamos la etiqueta (Clase 0-4)

    return np.hstack(all_moons), np.hstack(labels)

# --- VISUALIZACIÓN ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Colores para las 5 clases
colors = ['red', 'blue', 'green', 'orange', 'purple']

def update(frame):
    ax.clear()
    data, labels = get_frame_data(frame)
    
    # Ploteamos cada clase
    for i in range(5):
        mask = labels == i
        ax.scatter(data[0, mask], data[1, mask], data[2, mask], 
                   s=10, c=colors[i], alpha=0.6, label=f'Clase {i}')
    
    ax.set_title(f"5 Moons 3D - Frame {frame}\nTopología Dinámica")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X (Espacio)')
    ax.set_ylabel('Y (Espacio)')
    ax.set_zlabel('Z (Espacio)')
    # ax.legend()

ani = FuncAnimation(fig, update, frames=200, interval=50)
plt.show()