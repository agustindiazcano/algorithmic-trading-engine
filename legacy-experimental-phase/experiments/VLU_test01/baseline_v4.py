import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generador de la Forma "L" (Tetris) ---
def create_L_shape(origin=(0,0), scale=1.0):
    # Definimos la L con 4 puntos (esferas)
    # 0
    # 0
    # 0 0
    coords = np.array([
        [0, 0],     # Esquina
        [1, 0],     # Derecha
        [0, 1],     # Arriba 1
        [0, 2]      # Arriba 2
    ]) * scale
    return coords + np.array(origin)

# --- 2. El Motor Lógico Molecular (2D) ---
class MolecularNeuron2D:
    def __init__(self, template_coords, radius=0.3):
        # El "Molde" interno (Memoria a Largo Plazo)
        self.center = np.mean(template_coords, axis=0)
        self.rel_coords = template_coords - self.center # Coordenadas relativas (Rigidez)
        self.r = radius
        
    def get_rotation_matrix(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def activate_static(self, x):
        """ Lógica Estática: ¿Encajan los puntos EXACTAMENTE donde los recuerdo? """
        # Reconstruimos la posición absoluta original
        expected_pos = self.rel_coords + self.center
        
        # Inferencia Volumétrica (Promedio de intersecciones) [cite: 339, 2071]
        score = 0
        for p_obs in x:
            # Distancia al átomo más cerca del molde
            dists = np.linalg.norm(expected_pos - p_obs, axis=1)
            min_dist = np.min(dists)
            # Función de proximidad radial [cite: 340, 2072]
            score += max(0, 1 - min_dist / self.r)
            
        return score / len(x)

    def activate_molecular(self, x, scan_steps=36):
        """ 
        Lógica Molecular: Busca en el grupo SE(2) (Rotaciones)
        "¿Existe algún ángulo donde esto tenga sentido?" 
        """
        best_score = 0
        best_angle = 0
        
        # El centro de la observación (asumimos que la atención está centrada en el objeto)
        obs_center = np.mean(x, axis=0)
        
        # Barrido de rotación (Simulación Mental)
        # En una VNN real esto es optimización, aquí es fuerza bruta para demo
        angles = np.linspace(0, 2*np.pi, scan_steps)
        
        for theta in angles:
            R = self.get_rotation_matrix(theta)
            # Transformamos el MOLDE para ver si encaja con los DATOS
            # Hipótesis: Molde Rotado + Centro Observado
            hypothesis = (self.rel_coords @ R.T) + obs_center
            
            # Medir encaje
            current_score = 0
            for p_obs in x:
                dists = np.linalg.norm(hypothesis - p_obs, axis=1)
                current_score += max(0, 1 - np.min(dists) / self.r)
            current_score /= len(x)
            
            if current_score > best_score:
                best_score = current_score
                best_angle = theta
                
        return best_score, best_angle

# --- 3. Ejecución del Experimento ---

# A. Crear la "Memoria" (Template)
template = create_L_shape(scale=0.5)
neuron = MolecularNeuron2D(template, radius=0.25)

# B. Crear el "Mundo Real" (Objeto rotado)
# Rotamos la L real 90 grados (pi/2)
theta_real = np.pi / 2
R_real = np.array([[0, -1], [1, 0]])
# Centramos en (0,0) para rotar y luego movemos
data_centered = template - np.mean(template, axis=0)
data_rotated = (data_centered @ R_real.T) + np.array([2, 2]) # Lejos del origen original

# C. Evaluar Motores Lógicos
score_static = neuron.activate_static(data_rotated)
score_molecular, detected_angle = neuron.activate_molecular(data_rotated)

print(f"Decisión Motor Estático: {score_static:.4f} (¿Es mi L? No sé)")
print(f"Decisión Motor Molecular: {score_molecular:.4f} (¿Es mi L? SÍ, rotada {np.degrees(detected_angle):.0f}°)")

# --- 4. Visualización ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Visión Estática
ax[0].set_title(f"Visión Estática (Score: {score_static:.2f})")
ax[0].scatter(data_rotated[:,0], data_rotated[:,1], c='blue', s=100, label="Datos (L Rotada)")
# Dibujamos dónde la neurona ESPERA que esté la L
expected = neuron.rel_coords + neuron.center
ax[0].scatter(expected[:,0], expected[:,1], c='red', alpha=0.3, s=300, label="Memoria (L Original)")
for p in expected:
    circle = plt.Circle(p, neuron.r, color='red', fill=False, linestyle='--')
    ax[0].add_artist(circle)
ax[0].legend()
ax[0].set_xlim(-1, 3); ax[0].set_ylim(-1, 3); ax[0].grid(True)

# Plot 2: Visión Molecular
ax[1].set_title(f"Visión Molecular (Score: {score_molecular:.2f})")
ax[1].scatter(data_rotated[:,0], data_rotated[:,1], c='blue', s=100, label="Datos")
# Dibujamos la neurona ADAPTADA (Rotada por el motor lógico)
R_detected = neuron.get_rotation_matrix(detected_angle)
adapted = (neuron.rel_coords @ R_detected.T) + np.mean(data_rotated, axis=0)
ax[1].scatter(adapted[:,0], adapted[:,1], c='green', alpha=0.5, s=300, label="Inferencia (Match)")
for p in adapted:
    circle = plt.Circle(p, neuron.r, color='green', fill=False, linestyle='-')
    ax[1].add_artist(circle)
ax[1].legend()
ax[1].set_xlim(-1, 3); ax[1].set_ylim(-1, 3); ax[1].grid(True)

plt.show()