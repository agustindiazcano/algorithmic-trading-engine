#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN "HYPER-CUBE MASSACRE" ---
const int VECTORS_COUNT = 1000000; // 1 Millón de puntos por dimensión
const int MAX_DIMS = 128;          // Hasta 128 dimensiones
const int DIM_STEP = 4;            // Saltos de 4 dimensiones

volatile int g_sink = 0;

// ==========================================
// 1. MÉTODO TRADICIONAL (N-Dimensional AABB)
// ==========================================
// El "Hyper-Box" check.
// Requiere verificar 2 condiciones por dimensión (min y max).
// Total de saltos potenciales: 2 * Dimensiones.
// En 128D, eso son 256 decisiones por punto.
long long benchmark_aabb_ndim(int dims, const std::vector<float> &data,
                              const std::vector<float> &target, float radius) {
  int hits = 0;

  // Límites de la caja (Hyper-Cube)
  std::vector<float> min_bounds(dims);
  std::vector<float> max_bounds(dims);
  for (int d = 0; d < dims; ++d) {
    min_bounds[d] = target[d] - radius;
    max_bounds[d] = target[d] + radius;
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < VECTORS_COUNT; ++i) {
    bool inside = true;
    size_t base_idx = i * dims;

    // La maldición de la dimensionalidad hecha código
    for (int d = 0; d < dims; ++d) {
      float val = data[base_idx + d];
      // Dos IFs por dimensión.
      // Si falla, rompe (Early Exit).
      if (val < min_bounds[d] || val > max_bounds[d]) {
        inside = false;
        break; // Pipeline Flush si la predicción falló
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
// 2. MÉTODO W (N-Dimensional Sphere)
// ==========================================
// Distancia Euclídea Pura.
// Sin 'if' dentro del bucle de dimensiones.
// El compilador puede usar SIMD reduction loops.
long long benchmark_w_ndim(int dims, const std::vector<float> &data,
                                 const std::vector<float> &target,
                                 float radius) {
  int hits = 0;
  float r_sq =
      radius * radius * dims; // Aproximamos volumen equivalente escalando radio
                              // (heurística simple para test)

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < VECTORS_COUNT; ++i) {
    float dist_sq = 0.0f;
    size_t base_idx = i * dims;

    // Bucle puramente aritmético.
    // Compilador: "Oh, puedo usar procesadores vectoriales aquí".
    for (int d = 0; d < dims; ++d) {
      float diff = data[base_idx + d] - target[d];
      dist_sq += diff * diff;
    }

    // Un solo chequeo al final de todo el hiper-espacio
    if (dist_sq < r_sq)
      hits++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = hits;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int main() {
  std::cout << "=== W SYSTEMS: THE HYPER-CUBE MASSACRE ===" << std::endl;
  std::cout
      << "Objective: Demonstrate Dimensionality Curse vs Vector Invariance"
      << std::endl;

  std::ofstream jsonFile("hyperdim_results.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n  \"results\": [\n";
  }

  // Buffer gigante para reusar memoria (evitar allocs en el loop)
  std::vector<float> data(VECTORS_COUNT * MAX_DIMS);
  std::vector<float> target(MAX_DIMS, 500.0f);
  float radius = 100.0f;

  std::mt19937 rng(42);
  // Generador para datos "Tóxicos"
  // Queremos que pasen casi todos los filtros y fallen al final.
  std::uniform_real_distribution<float> dist_inside(target[0] - radius + 1.0f,
                                                    target[0] + radius - 1.0f);
  std::uniform_real_distribution<float> dist_outside_far(0.0f, 1000.0f);

  bool first_entry = true;

  for (int dims = 2; dims <= MAX_DIMS; dims += DIM_STEP) {
    // --- GENERACIÓN DE DATOS ADVERSARIOS ---
    // Estrategia: "Near Miss Deep"
    // Para cada vector:
    // - Dimensiones 0 hasta N-2: SIEMPRE DENTRO (Pasan los IFs)
    // - Dimensión N-1 (Última): 50% Fuera / 50% Dentro.
    // Esto obliga a la CPU a ejecutar el bucle completo y fallar al final.
    // Es el "Worst Case" para AABB.

#pragma omp parallel for
    for (size_t i = 0; i < VECTORS_COUNT; ++i) {
      size_t base_idx =
          i * MAX_DIMS; // Usamos stride maximo para simplificar punteros

      // Llenar dimensiones "seguras"
      for (int d = 0; d < dims - 1; ++d) {
        data[base_idx + d] = dist_inside(rng);
      }

      // Llenar dimensión "traicionera"
      if (i % 2 == 0) {
        data[base_idx + dims - 1] = dist_inside(rng); // Entra
      } else {
        data[base_idx + dims - 1] = dist_outside_far(rng); // Falla al final
      }
    }

    // Ejecutar Benchmarks
    // Nota: Pasamos el vector completo, pero las funciones saben usar el stride
    // correcto (simulado) Para simular correctamente el stride variable,
    // tendríamos que re-packear, pero para efectos de CPU branch prediction,
    // leer linealmente es lo mejor que le puede pasar a AABB y aún así va a
    // perder. Ajustaremos los punteros en la llamada es complejo, así que
    // simplificaremos: Usamos los primeros 'dims' datos contiguous de cada
    // bloque de MAX_DIMS. Espera, eso rompería la localidad de caché para dims
    // pequeños. Mejor: Re-generar dataset compacto para cada dimensión para ser
    // justos con la caché.

    std::vector<float> compact_data(VECTORS_COUNT * dims);
    for (size_t i = 0; i < VECTORS_COUNT; ++i) {
      for (int d = 0; d < dims; ++d) {
        compact_data[i * dims + d] = data[i * MAX_DIMS + d];
      }
    }

    long long time_aabb =
        benchmark_aabb_ndim(dims, compact_data, target, radius);
    long long time_weyl =
        benchmark_w_ndim(dims, compact_data, target, radius);

    double speedup = (double)time_aabb / time_weyl;

    std::cout << "Dims: " << std::setw(3) << dims << " | AABB: " << std::setw(8)
              << time_aabb << "us"
              << " | W: " << std::setw(8) << time_weyl << "us"
              << " | Speedup: " << std::fixed << std::setprecision(2) << speedup
              << "x" << std::endl;

    if (!first_entry)
      jsonFile << ",\n";
    jsonFile << "    {\"dims\": " << dims << ", \"aabb_time\": " << time_aabb
             << ", \"w_time\": " << time_weyl
             << ", \"speedup\": " << speedup << "}";
    first_entry = false;
  }

  if (jsonFile.is_open()) {
    jsonFile << "\n  ]\n}" << std::endl;
    jsonFile.close();
  }

  std::cout << "Benchmark Complete. Results saved to hyperdim_results.json"
            << std::endl;
  return 0;
}
