#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN HYPER-DIMENSIONAL ---
const int NUM_VECTORS =
    100000; // 100k vectores (para mantener cache pressure manejable y ver CPU)
const int DIMENSIONS = 128;    // 128 dimensiones (Embeddings típicos de IA)
const float THRESHOLD = 50.0f; // Distancia de similitud
volatile int g_sink = 0;

// ==========================================
// 1. MÉTODO TRADICIONAL (Early Exit / Scalar Optimization)
// ==========================================
// La "Astucia" algorítmica clásica:
// "Si mientras sumo la distancia ya me pasé del límite, paro y descarto."
// Suena genial en papel. En hardware moderno es suicidio.
// 1. Rompe la Vectorización (SIMD).
// 2. Introduce un salto condicional dentro del bucle interno más caliente.
long long benchmark_early_exit(const std::vector<float> &flat_data,
                               const std::vector<float> &target) {
  int matches = 0;
  float r_sq = THRESHOLD * THRESHOLD;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NUM_VECTORS; ++i) {
    float dist_sq = 0.0f;
    int base_idx = i * DIMENSIONS;
    bool match = true;

    for (int d = 0; d < DIMENSIONS; ++d) {
      float diff = flat_data[base_idx + d] - target[d];
      dist_sq += diff * diff;

      // EL FRENO DE MANO: Early Exit
      // "Optimización" que impide usar registros AVX de 256 bits
      if (dist_sq > r_sq) {
        match = false;
        break; // Rompe el flujo, vacía el pipeline
      }
    }

    if (match)
      matches++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = matches;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

// ==========================================
// 2. MÉTODO W (Full SIMD Brute Force)
// ==========================================
// Fuerza bruta inteligente.
// Calcula las 128 dimensiones SIEMPRE.
// Permite al compilador usar instrucciones AVX (procesar 8 floats por ciclo).
long long benchmark_w_simd(const std::vector<float> &flat_data,
                                 const std::vector<float> &target) {
  int matches = 0;
  float r_sq = THRESHOLD * THRESHOLD;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NUM_VECTORS; ++i) {
    float dist_sq = 0.0f;
    int base_idx = i * DIMENSIONS;

    // Bucle interno vectorizable al 100%
    // El compilador desenrolla esto y usa YMM registers
    for (int d = 0; d < DIMENSIONS; ++d) {
      float diff = flat_data[base_idx + d] - target[d];
      dist_sq += diff * diff;
    }

    // Un solo Branch por vector (vs 128 potenciales arriba)
    if (dist_sq <= r_sq)
      matches++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = matches;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int main() {
  std::cout << "=== W SYSTEMS: HYPER-DIMENSIONAL SEARCH TEST ==="
            << std::endl;
  std::cout << "Escenario: Busqueda de Similitud en Embeddings de IA (128 Dims)"
            << std::endl;
  std::cout << "Vectores: " << NUM_VECTORS << " | Dimensiones: " << DIMENSIONS
            << std::endl;

  // Generar Datos Aleatorios (Simulando Embeddings normalizados)
  std::vector<float> flat_data(NUM_VECTORS * DIMENSIONS);
  std::vector<float> target(DIMENSIONS);

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::cout << "Generando " << (NUM_VECTORS * DIMENSIONS) << " floats..."
            << std::endl;
  for (auto &val : flat_data)
    val = dist(rng);
  for (auto &val : target)
    val = dist(rng);

  // Warmup
  benchmark_early_exit(flat_data, target);
  benchmark_w_simd(flat_data, target);

  long long total_early = 0;
  long long total_w = 0;
  int samples = 0;

  auto global_start = std::chrono::steady_clock::now();
  std::cout << "Corriendo Benchmarks (60s)..." << std::endl;

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - global_start)
            .count() >= 60)
      break;

    // Rotar target para evitar cache exacta
    target[rand() % DIMENSIONS] = dist(rng);

    long long t_early = benchmark_early_exit(flat_data, target);
    long long t_weyl = benchmark_w_simd(flat_data, target);

    total_early += t_early;
    total_w += t_weyl;
    samples++;

    if (samples % 10 == 0) {
      std::cout << "Sample " << samples << ": Tradition=" << t_early
                << "us | W=" << t_weyl
                << "us | Speedup: " << (double)t_early / t_weyl << "x"
                << std::endl;
    }
  }

  double avg_early = (double)total_early / samples;
  double avg_weyl = (double)total_w / samples;
  double speedup = avg_early / avg_weyl;
  double improvement = (avg_early - avg_weyl) / avg_early * 100.0;

  std::cout << "\n=== RESULTADOS HYPER-DIMENSIONALES ===" << std::endl;
  std::cout << "Avg Traditional (Early Exit): " << avg_early << " us"
            << std::endl;
  std::cout << "Avg W (SIMD Brute):     " << avg_weyl << " us"
            << std::endl;
  std::cout << "SPEEDUP FACTOR:               " << speedup << "x" << std::endl;

  std::ofstream jsonFile("w_hyper_results.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_type\": \"Hyper-Dimensional Search (128D)\",\n";
    jsonFile << "  \"vectors\": " << NUM_VECTORS << ",\n";
    jsonFile << "  \"dimensions\": " << DIMENSIONS << ",\n";
    jsonFile << "  \"avg_traditional_early_exit_us\": " << avg_early << ",\n";
    jsonFile << "  \"avg_w_simd_us\": " << avg_weyl << ",\n";
    jsonFile << "  \"improvement_percent\": " << improvement << ",\n";
    jsonFile << "  \"speedup_factor\": " << speedup << "\n";
    jsonFile << "}" << std::endl;
    jsonFile.close();
  }

  return 0;
}
