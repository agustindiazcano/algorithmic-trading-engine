import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import time
import json
import os

# --- CONFIGURACIÓN ---
# Nota: Como el environment puede no tener GPU, JAX correrá en CPU
# pero XLA optimizará igual las instrucciones vectoriales (AVX).
N = 1000000  # 1 Millón de partículas
key = random.PRNGKey(0)

# Datos aleatorios
pos = random.uniform(key, (N, 3))
target = jnp.array([0.5, 0.5, 0.5])

# ==========================================
# 1. MÉTODO TRADICIONAL (Branching Logic)
# ==========================================
# Intentamos simular lógica condicional:
# "Si está cerca, empujá fuerte. Si está lejos, ignorá."
# Usamos jnp.where que es la forma vectorizada del IF.
# Aunque JAX lo intenta optimizar, esto suele generar lectura/escritura extra de memoria.
@jit
def update_traditional(p, tgt):
    dist = jnp.linalg.norm(p - tgt, axis=1)
    
    # Lógica discreta: IF dist < 0.1 THEN force = 10 ELSE force = 0
    # Esto obliga a evaluar condiciones y enmascarar memoria.
    force_magnitude = jnp.where(dist < 0.1, 10.0, 0.0)
    
    # Aplicar fuerza
    direction = (p - tgt) / (dist[:, None] + 1e-6)
    return p + direction * force_magnitude[:, None] * 0.01

# ==========================================
# 2. MÉTODO W (Pure Arithmetic Fusion)
# ==========================================
# Sin condiciones. Sin máscaras. Matemática pura.
# XLA ama esto y debería fusionarlo en un solo bloque de instrucciones PTX/AVX.
@jit
def update_w(p, tgt):
    diff = p - tgt
    dist_sq = jnp.sum(diff**2, axis=1)
    
    # Lógica Volumétrica: Fuerza continua
    # F = 1 / (1 + dist^2) * 10
    # Sin cortes. Diferenciable. Fusionable.
    force_magnitude = 10.0 / (1.0 + dist_sq / 0.01)
    
    # Aplicar fuerza (Matemática vectorial directa)
    # Evitamos sqrt() usando la distancia cuadrada en la logica
    return p + diff * force_magnitude[:, None] * 0.01

# --- BENCHMARK ---
print(f"=== W SYSTEMS: JAX FUSION REACTOR ===")
# print(f"Backend: {jax.lib.xla_bridge.get_backend().platform}")
try:
    print(f"Backend: {jax.devices()[0].platform}")
except:
    print("Backend: Unknown (CPU likely)")
print(f"Simulando {N} partículas con fusión XLA...")

# Warmup (Compilación JIT)
print("Compilando Kernels...")
_ = update_traditional(pos, target).block_until_ready()
_ = update_w(pos, target).block_until_ready()
print("Compilación lista.\n")

iterations = 100

# Test Tradicional
start = time.time()
for _ in range(iterations): # pasos de simulación
    _ = update_traditional(pos, target).block_until_ready()
end = time.time()
avg_trad = (end-start)*1000/iterations
print(f"Tradicional (jnp.where): {avg_trad:.4f} ms por paso")

# Test W
start = time.time()
for _ in range(iterations):
    _ = update_w(pos, target).block_until_ready()
end = time.time()
avg_weyl = (end-start)*1000/iterations
print(f"W (Aritmética):    {avg_weyl:.4f} ms por paso")

speedup = avg_trad / avg_weyl

print(f"\nSPEEDUP FACTOR: {speedup:.2f}x")

# JSON Export
results = {
    "test_type": "JAX Fusion Reactor (XLA Compilation Stress)",
    "particles": N,
    "backend": jax.lib.xla_bridge.get_backend().platform,
    "avg_time_traditional_ms": avg_trad,
    "avg_time_w_ms": avg_weyl,
    "speedup_factor": speedup
}

with open("w_jax_results.json", "w") as f:
    json.dump(results, f, indent=4)
    print("Reporte JSON guardado: w_jax_results.json")
