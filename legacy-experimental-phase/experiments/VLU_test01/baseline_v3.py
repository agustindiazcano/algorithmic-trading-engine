import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Generador de Formas (La "Luna") ---
def generate_moon_3d(n_points=200, radius=1.0, noise=0.05):
    # Generamos un arco en 3D
    t = np.linspace(0, np.pi, n_points)
    x = np.cos(t) * radius
    y = np.sin(t) * radius
    z = np.random.normal(0, 0.1, n_points) # Grosor en Z
    
    # Ruido
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    
    return np.vstack([x, y, z]).T

# --- 2. Neurona Molecular (El Enjambre Rígido) ---
class MolecularNeuron:
    def __init__(self, template_cloud, atom_radius=0.15):
        """
        Crea una neurona 'capturando' la forma de una nube de puntos.
        En lugar de guardar los puntos crudos, guarda 'átomos' (esferas) relativos al centro.
        """
        self.center = np.mean(template_cloud, axis=0)
        self.atom_radius = atom_radius
        
        # Guardamos la estructura RELATIVA (Rigidez) [cite: 1720, 1721]
        # p_local = p_global - centro
        self.relative_atoms = template_cloud - self.center
        
        # Estado actual (Pose) en el grupo SE(3)
        self.current_rotation = np.eye(3) # Matriz identidad
        self.current_position = self.center

    def transform(self, rotation_matrix, translation_vec):
        """ Aplica una transformación rígida temporal para ver si encaja """
        # Formula: P_global = R * p_local + T
        transformed_atoms = (self.relative_atoms @ rotation_matrix.T) + translation_vec
        return transformed_atoms

    def evaluate_fit(self, target_cloud, rotation_matrix, translation_vec):
        """
        Mide qué tan bien encaja la neurona en los datos (Inferencia por Registro).
        Calculamos cuántos puntos de la nube caen dentro de nuestros átomos.
        """
        # 1. Movemos la neurona mentalmente (Simulación)
        transformed_atoms = self.transform(rotation_matrix, translation_vec)
        
        # 2. Chequeo de colisión masiva (Broad phase)
        # Para simplificar, medimos distancia promedio al átomo más cerca (Chamfer-like)
        # O simplemente contamos "Inliers" (puntos atrapados por alguna esfera)
        
        inliers = 0
        for point in target_cloud:
            # Distancia al átomo transformado más cerca
            dists = np.sum((transformed_atoms - point)**2, axis=1) # L2^2 eficiente
            if np.min(dists) < self.atom_radius**2:
                inliers += 1
                
        # Score: % de puntos cubiertos
        return inliers / len(target_cloud)

# --- 3. Ejecución del Test ---

# A. ENTRENAMIENTO (One-Shot)
print(">>> Fase 1: Aprendiendo la forma 'Luna'...")
template_moon = generate_moon_3d()
# La neurona "cristaliza" la forma relativa
neuron = MolecularNeuron(template_moon, atom_radius=0.2)
print(f"Neurona Molecular creada con {len(template_moon)} átomos rígidos.")

# B. EL DESAFÍO (Rotación Desconocida)
print("\n>>> Fase 2: Aparece una Luna rotada 90 grados...")
# Rotamos la luna original 90 grados en Z
theta = np.pi / 2
R_challenge = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])
challenge_moon = (template_moon @ R_challenge.T) + np.array([0.5, 0.5, 0]) # Rotada + Trasladada

# C. INFERENCIA (Búsqueda en SE(3))
print(">>> Fase 3: La neurona intenta 'encajar'...")

# Prueba 1: Sin rotar (Naive)
score_naive = neuron.evaluate_fit(challenge_moon, np.eye(3), np.mean(challenge_moon, axis=0))
print(f"Fit sin rotar: {score_naive*100:.1f}% (Fallo esperado)")

# Prueba 2: Rotando la neurona (Solución Molecular)
# En un sistema real, esto se hace con optimización. Aquí "probamos" la rotación correcta.
score_rotated = neuron.evaluate_fit(challenge_moon, R_challenge, np.mean(challenge_moon, axis=0))
print(f"Fit rotando la estructura interna: {score_rotated*100:.1f}% (Éxito)")

# --- 4. Visualización 3D ---
fig = plt.figure(figsize=(12, 5))

# Plot 1: Lo que ve la neurona "ingenua" (Estática)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title(f"VNN Estática (Fit: {score_naive*100:.0f}%)")
ax1.scatter(challenge_moon[:,0], challenge_moon[:,1], challenge_moon[:,2], c='blue', alpha=0.2, label="Datos (Target)")
# Dibujamos la neurona en su pose original (no encaja)
neuron_naive = neuron.transform(np.eye(3), np.mean(challenge_moon, axis=0))
ax1.scatter(neuron_naive[:,0], neuron_naive[:,1], neuron_naive[:,2], c='red', s=5, label="Neurona (Template)")
ax1.legend()

# Plot 2: Lo que ve la neurona "Molecular" (Dinámica)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title(f"VNN Molecular (Fit: {score_rotated*100:.0f}%)")
ax2.scatter(challenge_moon[:,0], challenge_moon[:,1], challenge_moon[:,2], c='blue', alpha=0.2)
# Dibujamos la neurona rotada
neuron_fitted = neuron.transform(R_challenge, np.mean(challenge_moon, axis=0))
ax2.scatter(neuron_fitted[:,0], neuron_fitted[:,1], neuron_fitted[:,2], c='green', s=5, label="Neurona Rotada")
ax2.legend()

plt.show()