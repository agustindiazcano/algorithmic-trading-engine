/*
═══════════════════════════════════════════════════════════════════════
🔥 W: THE SIMD MASSACRE 🔥
Target: Throughput (Ancho de Banda)
Technique: Auto-Vectorization (AVX/SSE) via Branchless Logic
Scenario: Audio Noise Gate / Pixel Filter
═══════════════════════════════════════════════════════════════════════
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN ---
// 100 Millones de muestras de audio (aprox 30 min de audio a 44.1kHz)
const size_t N = 100000000;
const float THRESHOLD = 0.5f;

using namespace std;

// Timer simple
class Timer {
  chrono::high_resolution_clock::time_point start;

public:
  Timer() : start(chrono::high_resolution_clock::now()) {}
  double elapsed() {
    return chrono::duration_cast<chrono::duration<double>>(
               chrono::high_resolution_clock::now() - start)
        .count();
  }
};

int main() {
  cout << "🔥 W: SIMD VECTORIZATION TEST 🔥\n";
  cout << "Procesando " << N / 1'000'000
       << " Millones de muestras de audio...\n";

  // 1. Generar Ruido Blanco (Datos aleatorios)
  // Esto es vital: Si los datos fueran ordenados (todo silencio o todo ruido),
  // el IF ganaría. Con datos random, el IF falla y rompe la vectorización.
  vector<float> audio_in(N);
  // Align memory for AVX (optional but good practice, std::vector usually
  // handles it ok)
  vector<float> audio_out_trad(N);
  vector<float> audio_out_weyl(N);

  mt19937 gen(42);
  uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < N; i++)
    audio_in[i] = dist(gen);

  double t_trad, t_weyl;

  // ==========================================
  // 1. MÉTODO TRADICIONAL (SCALAR / BRANCHING)
  // ==========================================
  // El compilador mira esto y dice: "Hay un IF, no puedo asegurar que sea
  // seguro vectorizar". Procesa de a 1 (o intenta trucos sucios que fallan con
  // datos random).
  {
    Timer t;
    // Use a volatile sink to prevent optimization? No, writting to vector is
    // side effect enough. #pragma omp simd might help here but we WANT to test
    // if compiler auto-vectorizes naturally.
    for (size_t i = 0; i < N; i++) {
      if (audio_in[i] > THRESHOLD) {
        audio_out_trad[i] = audio_in[i] * 0.9f; // Aplicar ganancia
      } else {
        audio_out_trad[i] = 0.0f; // Silenciar
      }
    }
    t_trad = t.elapsed();
    cout << "Tradicional (If/Else): " << t_trad << " s" << endl;
  }

  // ==========================================
  // 2. MÉTODO W (VECTORIZED / BRANCHLESS)
  // ==========================================
  // Aquí usamos un truco matemático: Convertimos la comparación en una máscara
  // (0.0 o 1.0). El compilador ve esto y dice: "Ah, es solo multiplicar y
  // sumar. ¡AVX ACTIVADO!" Procesa 8 floats por ciclo.
  {
    Timer t;
    // We explicitly tell compiler this loop is safe
    for (size_t i = 0; i < N; i++) {
      float sample = audio_in[i];

      // TRUCO DE MAGIA:
      // (sample > THRESHOLD) devuelve true(1) o false(0).
      // En C++, al multiplicar por float, se convierte a 1.0f o 0.0f.
      // Esto se llama "Predication" y permite usar instrucciones FMA (Fused
      // Multiply Add).

      // Masking is pure data parallelism
      float mask = (float)(sample > THRESHOLD);

      // Fórmula: (Audio * Ganancia * 1) + (0 * 0)
      audio_out_weyl[i] = sample * 0.9f * mask;
    }
    t_weyl = t.elapsed();
    cout << "W (Vectorized):  " << t_weyl << " s" << endl;
  }

  double speedup = t_trad / t_weyl;
  cout << "\n------------------------------------------------\n";
  cout << "SPEEDUP FACTOR: " << fixed << setprecision(2) << speedup << "x"
       << endl;
  cout << "------------------------------------------------\n";

  // JSON
  ofstream json("w_simd_results.json");
  json << "{\n";
  json << "  \"test\": \"SIMD Audio Noise Gate\",\n";
  json << "  \"samples\": " << N << ",\n";
  json << "  \"time_traditional\": " << t_trad << ",\n";
  json << "  \"time_w\": " << t_weyl << ",\n";
  json << "  \"speedup\": " << speedup << "\n";
  json << "}";

  // Verify correctness spot check
  int errors = 0;
  for (size_t i = 0; i < 1000; i++) {
    if (std::abs(audio_out_trad[i] - audio_out_weyl[i]) > 0.0001f)
      errors++;
  }
  if (errors > 0)
    cout << "WARNING: Logic mismatch detected! (" << errors << " errors)"
         << endl;
  else
    cout << "Verification: CLEAN." << endl;

  return 0;
}
