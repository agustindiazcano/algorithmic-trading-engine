/*
═══════════════════════════════════════════════════════════════════════
🔥 C++ FINAL SHOWDOWN: W OPTIMIZED vs CONDITIONAL 🔥
Reglas:
1. MAXIMA ENTROPÍA (Datos aleatorios 50/50)
2. SIN SALIDA RÁPIDA (Siempre hay trabajo)
3. MATEMÁTICA BARATA (Prohibido exp, sin, cos, sqrt)
═══════════════════════════════════════════════════════════════════════
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <immintrin.h> // Para dar pistas de vectorización
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


using namespace std;
using namespace chrono;

struct Result {
  string name;
  double time_cond;
  double time_weyl;
  double speedup;
};
vector<Result> results;

class Timer {
  high_resolution_clock::time_point start;

public:
  Timer() : start(high_resolution_clock::now()) {}
  double elapsed() {
    return duration_cast<duration<double>>(high_resolution_clock::now() - start)
        .count();
  }
};

// ===================================
// FUNCIONES W OPTIMIZADAS (FAST MATH)
// ===================================

// Fast Sigmoid (x / (1 + |x|)) -> Reemplaza a tanh
// Costo: 1 abs, 1 suma, 1 div. NO exp.
inline float fast_sigmoid(float x) { return x / (1.0f + std::abs(x)); }

// Pseudo-Conditional Masking
// Convierte bool a float 1.0/0.0 sin IF
inline float to_mask(bool condition) { return condition ? 1.0f : 0.0f; }

// ===================================
// TEST 1: EL INFIERNO DE LA DIVERGENCIA
// Descripción: Operación matemática compleja diferente según el signo.
// Contexto: Shaders, Audio Processing, Image Filters.
// ===================================
void test_divergence() {
  cout << "\n=== TEST 1: DIVERGENCIA SIMD / BRANCHING (100M items) ===\n";
  const size_t N = 100'000'000;
  vector<float> data(N);
  vector<float> res_cond(N);
  vector<float> res_weyl(N);

  // Entropía Máxima: -1 a 1 uniforme
  mt19937 gen(42);
  uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < N; i++)
    data[i] = dist(gen);

  // --- CONDICIONAL ---
  Timer t1;
  for (size_t i = 0; i < N; i++) {
    float v = data[i];
    if (v > 0.0f) {
      // Rama A (Positiva): Amplificar
      res_cond[i] = v * v + 0.5f;
    } else {
      // Rama B (Negativa): Atenuar e invertir
      res_cond[i] = v * 0.5f - 0.1f;
    }
  }
  double time_cond = t1.elapsed();

  // --- W (Branchless Mix) ---
  Timer t2;
  for (size_t i = 0; i < N; i++) {
    float v = data[i];
    // Autopista SIMD: Calculamos ambass y mezclamos.
    // float mask = (v > 0.0f); // 1.0 o 0.0
    // TRUCO: v > 0 genera 1, v <= 0 genera 0.
    // Compute both paths
    float branch_A = v * v + 0.5f;
    float branch_B = v * 0.5f - 0.1f;

    // Select without branching (Mix)
    // Usamos una lógica algebraica simple para el mask
    // mask = 0.5 * (sign(v) + 1) -> 0 o 1
    // Fast sign extraction:
    float mask =
        (v > 0.0f) ? 1.0f : 0.0f; // Compilador optimiza esto a CMOV o máscaras

    res_weyl[i] = branch_A * mask + branch_B * (1.0f - mask);
  }
  double time_weyl = t2.elapsed();

  double speedup = time_cond / time_weyl;
  results.push_back({"SIMD Divergence", time_cond, time_weyl, speedup});
  cout << "Conditional: " << time_cond << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

// ===================================
// TEST 2: FLUIDO DENSO (INTERACCIÓN SIN ESCAPE)
// Descripción: Todas las partículas interactúan.
// La condicional "if (dist < r)" falla porque r es grande.
// ===================================
struct Particle {
  float x, y;
};

void test_dense_fluid() {
  cout << "\n=== TEST 2: FLUIDO DENSO (Sin Salida Rápida) ===\n";
  const int N_P =
      20000; // 20k partículas (O(N^2) sería mucho, usamos radio grande local)
  // Para simplificar test CPU: Interacción todos contra todos en batch pequeño
  // Simulamos un bucket de 2048 partículas MUY APRETADAS.
  const int N_BUCKET = 2048;

  vector<Particle> p(N_BUCKET);
  mt19937 gen(42);
  uniform_real_distribution<float> dist(0.0f, 10.0f); // Caja pequeña 10x10
  for (int i = 0; i < N_BUCKET; i++) {
    p[i] = {dist(gen), dist(gen)};
  }

  const float RADIUS = 8.0f; // Radio enorme. Cubre casi toda la caja.
  const float RADIUS_SQ = RADIUS * RADIUS;

  // --- CONDICIONAL ---
  Timer t1;
  float energy_cond = 0;
  for (int i = 0; i < N_BUCKET; i++) {
    for (int j = 0; j < N_BUCKET; j++) {
      float dx = p[i].x - p[j].x;
      float dy = p[i].y - p[j].y;
      float d2 = dx * dx + dy * dy;

      // EL IF MALDITO: Casi siempre es true, pero el CPU tiene que chequear
      if (d2 < RADIUS_SQ && d2 > 0.0001f) {
        // Física LJ simple (12-6 potential simplificado)
        float inv_d2 = 1.0f / d2;
        energy_cond += inv_d2 * inv_d2 * inv_d2 - inv_d2 * inv_d2;
      }
    }
  }
  double time_cond = t1.elapsed();

  // --- W (Fast Math Field) ---
  Timer t2;
  float energy_weyl = 0;
  for (int i = 0; i < N_BUCKET; i++) {
    for (int j = 0; j < N_BUCKET; j++) {
      float dx = p[i].x - p[j].x;
      float dy = p[i].y - p[j].y;
      float d2 = dx * dx + dy * dy + 0.0001f; // Evitar div/0 sin IF

      // Fórmula continua (sin corte)
      // Asumimos interacción infinita suave (Gravity/Electro)
      // O un kernel suave SPH: (1 - d2/r2)^3
      // Usaremos Fast Sigmoid para modular la fuerza en lugar de IF

      // Pura aritmética:
      float inv_d2 = 1.0f / d2;
      float raw_force = inv_d2 * inv_d2 * inv_d2 - inv_d2 * inv_d2;

      // Aplicar soft-cut (Algebraic masking)
      // Mask = 1 si d2 < R2, 0 si d2 > R2 (aproximado)
      // Fast clamp:
      float diff = RADIUS_SQ - d2;
      // ReLU algebraico rápido (max(0, diff))
      // mask = 0 si diff < 0.
      // Para ser 100% branchless en CPU a veces usas (diff + abs(diff))*0.5
      float active = (diff + std::abs(diff)) * 0.5f;

      // Si active > 0, queremos que la fuerza cuente.
      // Normalizar active es caro, así que usamos el raw force ponderado
      // Pero para ser justos con 'Condicional', usaremos una función de peso
      // simple Weight = diff / (1 + abs(diff)) -> Fast Sigmoid escalada

      float weight = fast_sigmoid(active);
      energy_weyl += raw_force * weight;
    }
  }
  double time_weyl = t2.elapsed();

  double speedup = time_cond / time_weyl;
  results.push_back({"Dense Fluid Interaction", time_cond, time_weyl, speedup});
  cout << "Conditional: " << time_cond << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

// ===================================
// TEST 3: ACTIVATION / GAIN CONTROL (Fast Sigmoid)
// Descripción: Reemplazar una lógica de control por matemática rápida
// ===================================
void test_fast_math() {
  cout << "\n=== TEST 3: FAST MATH (100M Gain Control) ===\n";
  const size_t N = 100'000'000;
  vector<float> data(N);
  vector<float> data_weyl(N);

  mt19937 gen(99);
  normal_distribution<float> dist(0.0f, 2.0f);
  for (size_t i = 0; i < N; i++) {
    data[i] = dist(gen);
    data_weyl[i] = data[i];
  }

  // --- CONDICIONAL (Hard Clipping) ---
  Timer t1;
  for (size_t i = 0; i < N; i++) {
    if (data[i] > 1.0f)
      data[i] = 1.0f;
    else if (data[i] < -1.0f)
      data[i] = -1.0f;
  }
  double time_cond = t1.elapsed();

  // --- W (Soft Saturation / Fast Sigmoid) ---
  // x / (1 + |x|)
  Timer t2;
  for (size_t i = 0; i < N; i++) {
    float x = data_weyl[i];
    data_weyl[i] =
        x / (1.0f + std::abs(x) * 0.5f); // Scale to match roughly range
  }
  double time_weyl = t2.elapsed();

  double speedup = time_cond / time_weyl;
  results.push_back({"Fast Math Control", time_cond, time_weyl, speedup});

  cout << "Conditional: " << time_cond << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

int main() {
  cout << "╔══════════════════════════════════════════╗\n";
  cout << "║ W OPTIMIZED BENCHMARK (THE TAMER)  ║\n";
  cout << "╚══════════════════════════════════════════╝\n";

  test_divergence();
  test_dense_fluid();
  test_fast_math();

  // Export JSON
  ofstream j("w_final_battle.json");
  j << "{\n  \"results\": [\n";
  for (size_t i = 0; i < results.size(); i++) {
    j << "    { \"test\": \"" << results[i].name << "\", "
      << "\"conditional_s\": " << results[i].time_cond << ", "
      << "\"w_s\": " << results[i].time_weyl << ", "
      << "\"speedup\": " << results[i].speedup << " }"
      << (i < results.size() - 1 ? "," : "") << "\n";
  }
  j << "  ]\n}\n";

  return 0;
}
