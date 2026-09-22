import matplotlib.pyplot as plt
import numpy as np

# TU GENOMA
data = {
  "Anchor Inertia": 0.969,
  "Anchor Radius": 0.651, # Multipliqué x10 para que se vea en el gráfico
  "Anchor Persist": 0.700, # Normalizado (14/20) para escala
  "Hunter Inertia": 0.901,
  "Hunter Radius": 0.384, # Multipliqué x10
  "Hunter Persist": 0.100, # Normalizado (2/20)
  "Trauma Sens": 0.429,
  "Trauma Mult": 0.806, # Normalizado (1.61 / 2)
  "Pressure Thresh": 0.927,
  "Radius Learn": 0.869 # Multiplicado x10
}

# Preparamos datos para el radar
categories = list(data.keys())
values = list(data.values())
values += values[:1] # Cerrar el círculo
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1] # Cerrar el círculo

# Configuración del Gráfico
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Dibujar
ax.plot(angles, values, color='#00ff00', linewidth=2, linestyle='solid')
ax.fill(angles, values, color='#00ff00', alpha=0.25)

# Etiquetas
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10, color="white", weight="bold")

# Estética "Matrix/Cyberpunk"
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#0d0d0d')
ax.spines['polar'].set_color('#333333')
ax.tick_params(axis='y', colors='#005500')
ax.grid(color='#333333', linestyle='--')
plt.title("DCNN GENOME VISUALIZER v1", size=15, color="white", pad=20)

plt.show()