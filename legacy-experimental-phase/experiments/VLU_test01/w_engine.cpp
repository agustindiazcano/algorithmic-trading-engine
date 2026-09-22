#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN ---
const int DATA_SIZE = 10000000; // 10 Millones de datos
const float TARGET_X = 500.0f;
const float TARGET_Y = 500.0f;
const float RADIUS = 100.0f;
const float RANGE = 100.0f; // Radio del Bounding Box

// Inicialización del generador de azar
std::random_device rd;
std::mt19937 rng(rd());

volatile int g_sink = 0;

// ==========================================
// 1. MÉTODO TRADICIONAL (Branching - La "Optimización" Ingenua)
// ==========================================
// Este método intenta ser "inteligente" descartando puntos lejanos con IFs.
// Esta es la trampa: Con datos aleatorios, el Branch Predictor falla
// masivamente.
long long benchmark_branching(const std::vector<float> &x,
                              const std::vector<float> &y) {
  int hits = 0;
  float r_sq = RADIUS * RADIUS;
  float min_x = TARGET_X - RADIUS;
  float max_x = TARGET_X + RADIUS;
  float min_y = TARGET_Y - RADIUS;
  float max_y = TARGET_Y + RADIUS;

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < DATA_SIZE; ++i) {
    // La "Optimización" clásica: Bounding Box check
    if (x[i] >= min_x) {
      if (x[i] <= max_x) {
        if (y[i] >= min_y) {
          if (y[i] <= max_y) {
            // Solo calculamos distancia si estamos cerca
            float dx = x[i] - TARGET_X;
            float dy = y[i] - TARGET_Y;
            if (dx * dx + dy * dy < r_sq) {
              hits++;
            }
          }
        }
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = hits;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

// ==========================================
// 2. MÉTODO W (Branchless - Fuerza Bruta Matemática)
// ==========================================
long long benchmark_branchless(const std::vector<float> &x,
                               const std::vector<float> &y) {
  int hits = 0;
  float r_sq = RADIUS * RADIUS;

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < DATA_SIZE; ++i) {
    // Sin IFs, sin piedad. Todo cálculo vectorizable.
    float dx = x[i] - TARGET_X;
    float dy = y[i] - TARGET_Y;

    // El compilador puede usar instrucciones AVX/SSE aquí
    // La condición booleana (dist < r) se evalúa como 0 o 1 sin saltos
    hits += (dx * dx + dy * dy < r_sq);
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = hits;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int main() {
  std::cout << "=== W SYSTEMS: STRESS TEST PROTOCOL ===" << std::endl;
  std::cout
      << "Configuracion: 60 Segundos | Datos Toxicos | Branch Prediction Stress"
      << std::endl;

  std::vector<float> data_x(DATA_SIZE);
  std::vector<float> data_y(DATA_SIZE);

  // ESTRATEGIA DE ANIQUILACIÓN RANDOMIZADA
  // Llenamos todo el espacio de "ruido" para maximizar la entropía
  // y hacer imposible la predicción.
  std::uniform_real_distribution<float> dist_stress(TARGET_X - RADIUS * 1.5f,
                                                    TARGET_X + RADIUS * 1.5f);

  // Generar datos iniciales
  for (size_t i = 0; i < DATA_SIZE; ++i) {
    data_x[i] = dist_stress(rng);
    data_y[i] = dist_stress(rng);
  }

  long long total_branching = 0;
  long long total_branchless = 0;
  int samples = 0;

  auto global_start = std::chrono::steady_clock::now();

  while (true) {
    // Verificar tiempo (60 segundos)
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - global_start)
            .count() >= 60) {
      break;
    }

    // Re-randomizar parcialmente los datos cada ciclo para evitar caché
    // "caliente" demasiado optimista (Simula flujo de datos en tiempo real)
    for (size_t i = 0; i < 1000; ++i) {
      data_x[rand() % DATA_SIZE] = dist_stress(rng);
    }

    long long t_b = benchmark_branching(data_x, data_y);
    long long t_w = benchmark_branchless(data_x, data_y);

    total_branching += t_b;
    total_branchless += t_w;
    samples++;

    std::cout << "Muestra " << samples << ": TRAD=" << t_b
              << "us | W=" << t_w << "us" << std::endl;

    // Feedback visual inmediato si hay paliza (Branching > 1.5x Branchless)
    if (t_b > t_w * 1.5)
      std::cout << ">>> Branching DESTROYED (+"
                << (int)((double)t_b / t_w * 100 - 100) << "%)" << std::endl;
  }

  double avg_branching = (double)total_branching / samples;
  double avg_branchless = (double)total_branchless / samples;
  double improvement = (avg_branching - avg_branchless) / avg_branching * 100.0;
  double speedup = avg_branching / avg_branchless;

  std::cout << "\n=== RESULTADOS FINALES (60s) ===" << std::endl;
  std::cout << "Promedio Tradicional: " << avg_branching << " us" << std::endl;
  std::cout << "Promedio W:     " << avg_branchless << " us" << std::endl;
  std::cout << "Speedup Factor:       " << speedup << "x" << std::endl;

  std::ofstream jsonFile("w_benchmark.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_duration_sec\": 60,\n";
    jsonFile << "  \"samples\": " << samples << ",\n";
    jsonFile << "  \"avg_branching_us\": " << avg_branching << ",\n";
    jsonFile << "  \"avg_branchless_us\": " << avg_branchless << ",\n";
    jsonFile << "  \"improvement_percent\": " << improvement << ",\n";
    jsonFile << "  \"speedup_factor\": " << speedup << "\n";
    jsonFile << "}" << std::endl;
    jsonFile.close();
    std::cout << "Reporte guardado en w_benchmark.json" << std::endl;
  }

  return 0;
}