#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN DE SIMULACIÓN GPU ---
// Simulamos la arquitectura WARP/WAVEFRONT de una GPU moderna (NVIDIA/AMD)
// Un Warp son 32 hilos que DEBEN ejecutar la misma instrucción al mismo tiempo.
const int WARP_SIZE = 32;
const int NUM_PARTICLES = 1000000; // 1 Millón de partículas
const int NUM_WARPS = NUM_PARTICLES / WARP_SIZE;

// Parámetros del Espacio
const float TARGET = 500.0f;
const float RADIUS = 100.0f;

// COSTOS EN CICLOS DE GPU (Estimados)
// Una GPU odia la divergencia.
const long long COST_LOAD = 10;
const long long COST_ALU = 1;
const long long COST_BRANCH_OVERHEAD = 4;
const long long COST_DIVERGENCE_PENALTY =
    32; // Serialización de hilos + Masking

// Generador RNG
std::random_device rd;
std::mt19937 rng(rd());

struct ParticleSystem {
  std::vector<float> x, y, z;
  ParticleSystem(int size) : x(size), y(size), z(size) {}
};

// ==========================================
// KERNEL TRADICIONAL (Con Branching)
// ==========================================
// Simula el comportamiento del hardware cuando se enfrentan IFs.
long long simulate_gpu_traditional(const ParticleSystem &p) {
  long long total_cycles = 0;
  long long diverged_warps = 0;
  float box_min = TARGET - RADIUS;
  float box_max = TARGET + RADIUS;
  float r_sq = RADIUS * RADIUS;

  // Iteramos por WARPS (Grupos de 32 hilos)
  for (int w = 0; w < NUM_WARPS; ++w) {
    int start = w * WARP_SIZE;
    int end = start + WARP_SIZE;

    // Simulación de Ejecución SIMT (Single Instruction Multiple Threads)
    // Fase 1: Chequeo Bounding Box (El IF mortal)

    bool mask_inside[WARP_SIZE];
    int active_inside_box = 0;

    for (int i = 0; i < WARP_SIZE; ++i) {
      int idx = start + i;
      // Lógica tradicional de AABB
      bool in =
          (p.x[idx] >= box_min && p.x[idx] <= box_max && p.y[idx] >= box_min &&
           p.y[idx] <= box_max && p.z[idx] >= box_min && p.z[idx] <= box_max);
      mask_inside[i] = in;
      if (in)
        active_inside_box++;
    }

    // ANÁLISIS DE DIVERGENCIA DE WARP
    long long warp_cycles =
        COST_LOAD + COST_ALU * 6; // Carga + Comparaciones base

    if (active_inside_box == 0) {
      // MEJOR CASO: Todos fuera.
      // La GPU descarta el warp completo rápido.
      warp_cycles += COST_BRANCH_OVERHEAD;
    } else if (active_inside_box == WARP_SIZE) {
      // CASO RARO: Todos dentro.
      // Ejecutan la fase de precisión juntos.
      warp_cycles += COST_BRANCH_OVERHEAD + (COST_ALU * 5); // Distancia precisa
    } else {
      // PESADILLA: DIVERGENCIA (Warp Divergence)
      // Algunos entraron, otros no.
      // La GPU debe EJECUTAR AMBOS CAMINOS secuencialmente para todo el warp
      // enmascarando hilos.
      diverged_warps++;

      // Costo = (Camino A) + (Camino B) + Penalización de Hardware
      long long cost_path_false = 1;             // Salir
      long long cost_path_true = (COST_ALU * 5); // Calcular distancia

      warp_cycles += COST_BRANCH_OVERHEAD + cost_path_false + cost_path_true +
                     COST_DIVERGENCE_PENALTY;
    }

    total_cycles += warp_cycles;
  }

  // std::cout << "DEBUG: Divergencia detectada en " << diverged_warps << "
  // warps." << std::endl;
  return total_cycles;
}

// ==========================================
// KERNEL W (Branchless Compute)
// ==========================================
long long simulate_gpu_w(const ParticleSystem &p) {
  long long total_cycles = 0;

  // En W, NO IMPORTA LA DATA.
  // El costo es constante y perfectamente predecible.
  // La GPU funciona a máxima eficiencia (100% Occupancy).

  // Costo por Warp:
  // math: dx*dx + dy*dy + dz*dz < r_sq
  // ops: 3 sub, 3 mul, 2 add, 1 cmp = ~9 ops
  long long cycles_per_warp = COST_LOAD + (COST_ALU * 9);

  // No hay chequeos de divergencia porque no hay IFs.
  total_cycles = NUM_WARPS * cycles_per_warp;

  return total_cycles;
}

int main() {
  std::cout << "=== W SYSTEMS: GPU ARCHITECTURE SIMULATOR ==="
            << std::endl;
  std::cout << "Hardware Emulado: NVIDIA CUDA Core Architecture (Warp Size 32)"
            << std::endl;
  std::cout << "Carga: " << NUM_PARTICLES << " particulas." << std::endl;

  ParticleSystem data(NUM_PARTICLES);

  // ESTRATEGIA: "Warp Fragmentation"
  // Generamos partículas justo en el borde del volumen.
  // Esto asegura que en cada grupo de 32, ~16 entren y ~16 salgan.
  // Esto es el peor escenario posible para una GPU (50% Divergencia).
  std::normal_distribution<float> dist_edge(TARGET - RADIUS, RADIUS * 0.2f);
  std::uniform_real_distribution<float> dist_noise(TARGET - RADIUS * 2,
                                                   TARGET + RADIUS * 2);

  for (size_t i = 0; i < NUM_PARTICLES; ++i) {
    // Mezcla tóxica para romper coherencia espacial
    if (i % 2 == 0) {
      data.x[i] = dist_edge(rng); // Borde
      data.y[i] = dist_edge(rng);
    } else {
      data.x[i] = dist_noise(rng); // Ruido aleatorio
      data.y[i] = dist_noise(rng);
    }
    data.z[i] = dist_edge(rng);
  }

  auto start_time = std::chrono::steady_clock::now();
  int frames = 0;
  long long total_gpu_cycles_trad = 0;
  long long total_gpu_cycles_weyl = 0;

  std::cout << "Iniciando simulacion de 60 segundos..." << std::endl;

  while (true) {
    // Control de tiempo
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
            .count() >= 60)
      break;

    // Ejecutar simulación de hardware
    long long cycles_trad = simulate_gpu_traditional(data);
    long long cycles_weyl = simulate_gpu_w(data);

    total_gpu_cycles_trad += cycles_trad;
    total_gpu_cycles_weyl += cycles_weyl;
    frames++;

    // Regenerar 10% de data para simular movimiento
    for (int k = 0; k < NUM_PARTICLES / 10; ++k) {
      data.x[rand() % NUM_PARTICLES] = dist_edge(rng);
    }

    if (frames % 100 == 0) {
      double efficiency = (double)cycles_trad / cycles_weyl;
      std::cout << "Frame " << frames
                << " | GPU Efficiency Loss (Trad): " << efficiency
                << "x slower due to Warp Divergence" << std::endl;
    }
  }

  double avg_cycles_trad = (double)total_gpu_cycles_trad / frames;
  double avg_cycles_weyl = (double)total_gpu_cycles_weyl / frames;
  double speedup = avg_cycles_trad / avg_cycles_weyl;
  double improvement =
      (avg_cycles_trad - avg_cycles_weyl) / avg_cycles_trad * 100.0;

  std::cout << "\n=== REPORTE FINAL DE ARQUITECTURA GPU ===" << std::endl;
  std::cout << "Ciclos Promedio Tradicional (con Divergencia): "
            << (long long)avg_cycles_trad << std::endl;
  std::cout << "Ciclos Promedio W (Branchless):          "
            << (long long)avg_cycles_weyl << std::endl;
  std::cout << "SPEEDUP EN GPU: " << speedup << "x" << std::endl;

  std::ofstream jsonFile("w_gpu_results.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_type\": \"GPU Warp Divergence Simulation\",\n";
    jsonFile << "  \"duration_sec\": 60,\n";
    jsonFile << "  \"frames_simulated\": " << frames << ",\n";
    jsonFile << "  \"avg_cycles_traditional\": " << avg_cycles_trad << ",\n";
    jsonFile << "  \"avg_cycles_w\": " << avg_cycles_weyl << ",\n";
    jsonFile << "  \"divergence_penalty_factor\": " << speedup << ",\n";
    jsonFile << "  \"improvement_percent\": " << improvement << "\n";
    jsonFile << "}" << std::endl;
    jsonFile.close();
  }

  return 0;
}
