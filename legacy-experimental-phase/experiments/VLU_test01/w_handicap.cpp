#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN "THE ARROGANCE TEST" ---
const int VECTORS_COUNT = 1000000;
const int MAX_DIMS = 128;
const int HANDICAP_DIMS =
    2; // W calculará 2 dimensiones EXTRA de "basura" por gusto.
const int DIM_STEP = 4;

volatile int g_sink = 0;

// ==========================================
// 1. EL RIVAL: TRADICIONAL (AABB) - Optimizado (N Dimensiones)
// ==========================================
// Verifica exactamente las dimensiones necesarias. Ni una más.
long long benchmark_aabb_strict(int dims, const std::vector<float> &data,
                                const std::vector<float> &target,
                                float radius) {
  int hits = 0;
  std::vector<float> min_b(dims), max_b(dims);
  for (int d = 0; d < dims; ++d) {
    min_b[d] = target[d] - radius;
    max_b[d] = target[d] + radius;
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < VECTORS_COUNT; ++i) {
    bool inside = true;
    size_t base_idx = i * (MAX_DIMS + HANDICAP_DIMS);

    for (int d = 0; d < dims; ++d) { // Solo revisa lo necesario
      float val = data[base_idx + d];
      if (val < min_b[d] || val > max_b[d]) {
        inside = false;
        break;
      }
    }
    if (inside)
      hits++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = hits;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

// ==========================================
// 2. EL HÉROE: W (Heavyweight) - Sobrecargado (N + 2 Dimensiones)
// ==========================================
// W va a calcular la distancia de las dimensiones reales...
// Y ADEMÁS va a sumar 2 dimensiones extra de datos inútiles.
// Solo para demostrar que es más rápido haciendo DE MÁS que el otro haciendo LO
// JUSTO.
long long benchmark_w_handicap(int dims, const std::vector<float> &data,
                                     const std::vector<float> &target,
                                     float radius) {
  int hits = 0;
  float r_sq = radius * radius * dims;  // Ajuste volumétrico
  int work_dims = dims + HANDICAP_DIMS; // <--- EL HANDICAP. Trabajo extra.

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < VECTORS_COUNT; ++i) {
    float dist_sq = 0.0f;
    size_t base_idx = i * (MAX_DIMS + HANDICAP_DIMS);

    // El compilador vectoriza esto tan eficientemente que
    // agregar 2 iteraciones apenas afecta el throughput.
    for (int d = 0; d < work_dims; ++d) {
      float diff = data[base_idx + d] - target[d];
      dist_sq += diff * diff;
    }

    if (dist_sq < r_sq)
      hits++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = hits;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int main() {
  std::cout << "=== W SYSTEMS: THE ARROGANCE TEST ===" << std::endl;
  std::cout << "Escenario: W procesa " << HANDICAP_DIMS
            << " dimensiones EXTRA (basura) vs AABB optimizado." << std::endl;
  std::cout
      << "Objective: Demostrar que 'Carga Extra Lineal' < 'Costo de Branching'."
      << std::endl;

  std::ofstream jsonFile("w_handicap.json");
  if (jsonFile.is_open())
    jsonFile << "{\n  \"results\": [\n";

  // Allocar espacio para Dimensiones Reales + Handicap
  int stride = MAX_DIMS + HANDICAP_DIMS;
  std::vector<float> data(VECTORS_COUNT * stride);
  std::vector<float> target(stride, 500.0f);
  float radius = 100.0f;

  std::mt19937 rng(42);
  // Generar datos "Near Miss" para AABB (Worst Case)
  std::uniform_real_distribution<float> dist_in(400.0f, 600.0f);

  // Llenamos TODO el buffer (incluyendo las dimensiones extra con ruido)
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = dist_in(rng);

  bool first = true;

  for (int dims = 2; dims <= 64;
       dims += DIM_STEP) { // Probamos hasta 64D para ver bien el efecto
    // Regenerar datos para edge cases
    // ... (Simplificado: usamos el ruido uniforme que ya es bastante malo para
    // AABB en high dims)

    long long t_aabb = benchmark_aabb_strict(dims, data, target, radius);
    long long t_weyl = benchmark_w_handicap(dims, data, target, radius);

    double speedup = (double)t_aabb / t_weyl;

    std::cout << "Dims Reales: " << std::setw(2) << dims << " (W calc "
              << dims + HANDICAP_DIMS << ")"
              << " | AABB: " << std::setw(6) << t_aabb << "us"
              << " | W+: " << std::setw(6) << t_weyl << "us"
              << " | Speedup: " << std::fixed << std::setprecision(2) << speedup
              << "x" << std::endl;

    if (!first)
      jsonFile << ",\n";
    jsonFile << "    {\"dims\": " << dims << ", \"aabb_time\": " << t_aabb
             << ", \"w_handicap_time\": " << t_weyl
             << ", \"speedup\": " << speedup << "}";
    first = false;
  }

  if (jsonFile.is_open()) {
    jsonFile << "\n  ]\n}" << std::endl;
    jsonFile.close();
  }

  return 0;
}
