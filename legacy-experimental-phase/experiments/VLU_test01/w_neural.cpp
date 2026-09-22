#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN DE IA (CONTEXT WINDOW) ---
// Simulamos una ventana de contexto grande (4096 tokens)
// Esto genera una matriz de atención de 4096 x 4096 = 16.7 Millones de
// interacciones por pasada.
const int BATCH_SIZE = 4096;
const int EMBED_DIM = 128; // Dimensiones del vector (Head Size standard)

// Estructura de Tensor Simple
struct Tensor {
  std::vector<float> data;
  Tensor(int size) : data(size) {}
};

// ==========================================
// 1. MÉTODO TRADICIONAL (Softmax Attention)
// ==========================================
// La fórmula estándar: Attention(Q, K) = exp(Q * K^T)
// El costo oculto es std::exp(). Es una función trascendental.
// La CPU tiene que calcular series de Taylor. Es lentísimo comparado con
// sumar/multiplicar.
long long benchmark_softmax(const Tensor &Q, const Tensor &K,
                            std::vector<float> &scores) {
  auto start = std::chrono::high_resolution_clock::now();

  // Simulamos la operación crítica: Dot Product + Exponencial
  for (int i = 0; i < BATCH_SIZE; ++i) {
    // Punteros crudos para ayudar al compilador a vectorizar
    const float *q_ptr = &Q.data[i * EMBED_DIM];

    for (int j = 0; j < BATCH_SIZE; ++j) {
      const float *k_ptr = &K.data[j * EMBED_DIM];
      float dot = 0.0f;

      // Producto Punto (SIMD Friendly)
      for (int d = 0; d < EMBED_DIM; ++d) {
        dot += q_ptr[d] * k_ptr[d];
      }

      // EL CUELLO DE BOTELLA: EXPONENCIAL
      // Esto mata el pipeline porque es una operación compleja (no es 1 ciclo)
      // Además suele requerir chequeos de rango (IFs ocultos en la FPU)
      scores[i * BATCH_SIZE + j] = std::exp(dot * 0.125f); // Scaling factor
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

// ==========================================
// 2. MÉTODO W (Volumetric Attention)
// ==========================================
// La fórmula W: Attention(Q, K) = 1 / (1 + Distance^2)
// Algebraica pura. Solo suma, resta, multiplicación y una división.
// Mapea la similitud de 0 a 1 igual que Softmax, pero sin costo trascendental.
long long benchmark_w(const Tensor &Q, const Tensor &K,
                            std::vector<float> &scores) {
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < BATCH_SIZE; ++i) {
    const float *q_ptr = &Q.data[i * EMBED_DIM];

    for (int j = 0; j < BATCH_SIZE; ++j) {
      const float *k_ptr = &K.data[j * EMBED_DIM];
      float dist_sq = 0.0f;

      // Distancia Euclidiana al Cuadrado (SIMD Friendly)
      // El compilador usa instrucciones FMA (Fused Multiply-Add)
      for (int d = 0; d < EMBED_DIM; ++d) {
        float diff = q_ptr[d] - k_ptr[d];
        dist_sq += diff * diff;
      }

      // W KERNEL: Inversión Algebraica
      // 4 ciclos de reloj vs 100+ ciclos de exp()
      scores[i * BATCH_SIZE + j] = 1.0f / (1.0f + dist_sq);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int main() {
  std::cout << "=== W SYSTEMS: NEURAL CORE STRESS TEST ===" << std::endl;
  std::cout << "Target: Attention Mechanism (The Brain of GPT/Llama)"
            << std::endl;
  std::cout << "Matrix Size: " << BATCH_SIZE << "x" << BATCH_SIZE << " ("
            << (long long)BATCH_SIZE * BATCH_SIZE << " interactions per pass)"
            << std::endl;
  std::cout << "Duration: 60 Seconds Endurance Test" << std::endl;

  // Inicializar Datos
  Tensor Q(BATCH_SIZE * EMBED_DIM);
  Tensor K(BATCH_SIZE * EMBED_DIM);
  std::vector<float> scores_buffer(BATCH_SIZE * BATCH_SIZE);

  // Generar datos aleatorios (Embeddings)
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : Q.data)
    val = dist(rng);
  for (auto &val : K.data)
    val = dist(rng);

  long long total_time_softmax = 0;
  long long total_time_w = 0;
  long long passes_softmax = 0;
  long long passes_w = 0;

  auto global_start = std::chrono::steady_clock::now();

  std::cout << "\n>>> INICIANDO FASE 1: SOFTMAX (TRADICIONAL) <<<" << std::endl;
  while (true) {
    long long t = benchmark_softmax(Q, K, scores_buffer);
    total_time_softmax += t;
    passes_softmax++;

    // Feedback visual simple
    if (passes_softmax % 1 == 0)
      std::cout << "." << std::flush;

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - global_start)
            .count() >= 30)
      break;
  }
  std::cout << " DONE." << std::endl;

  // Reset clock para la fase 2
  global_start = std::chrono::steady_clock::now();

  std::cout << "\n>>> INICIANDO FASE 2: W (ALGEBRAIC) <<<" << std::endl;
  while (true) {
    long long t = benchmark_w(Q, K, scores_buffer);
    total_time_w += t;
    passes_w++;

    if (passes_w % 2 == 0)
      std::cout << "#" << std::flush;

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - global_start)
            .count() >= 30)
      break;
  }
  std::cout << " DONE." << std::endl;

  // --- CÁLCULOS FINALES ---
  double avg_softmax_ms = (double)total_time_softmax / passes_softmax / 1000.0;
  double avg_w_ms = (double)total_time_w / passes_w / 1000.0;

  // Evitar division por cero si algo falla
  if (passes_softmax == 0)
    avg_softmax_ms = 0.001;

  double tps_softmax = (double)passes_softmax * BATCH_SIZE /
                       30.0; // Tokens processed per second (aprox)
  double tps_w = (double)passes_w * BATCH_SIZE / 30.0;

  double speedup = avg_softmax_ms / avg_w_ms;
  double improvement =
      (avg_softmax_ms - avg_w_ms) / avg_softmax_ms * 100.0;

  std::cout << "\n=== RESULTADOS DEEP LEARNING (60s TEST) ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Softmax Total Passes: " << passes_softmax << std::endl;
  std::cout << "W Total Passes: " << passes_w << std::endl;
  std::cout << "Softmax Avg Latency: " << avg_softmax_ms << " ms" << std::endl;
  std::cout << "W Avg Latency: " << avg_w_ms << " ms" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "SPEEDUP FACTOR: " << speedup << "x" << std::endl;
  std::cout << "LATENCY REDUCTION: " << improvement << "%" << std::endl;

  // JSON Export
  std::ofstream jsonFile("w_neural_results.json");
  jsonFile << "{\n";
  jsonFile
      << "  \"test_type\": \"Neural Attention Mechanism (Transformer)\",\n";
  jsonFile << "  \"context_window\": " << BATCH_SIZE << ",\n";
  jsonFile << "  \"embedding_dim\": " << EMBED_DIM << ",\n";
  jsonFile << "  \"avg_latency_softmax_ms\": " << avg_softmax_ms << ",\n";
  jsonFile << "  \"avg_latency_w_ms\": " << avg_w_ms << ",\n";
  jsonFile << "  \"tokens_per_sec_softmax\": " << tps_softmax << ",\n";
  jsonFile << "  \"tokens_per_sec_w\": " << tps_w << ",\n";
  jsonFile << "  \"speedup_factor\": " << speedup << "\n";
  jsonFile << "}" << std::endl;

  std::cout << "Reporte JSON generado: w_neural_results.json"
            << std::endl;

  return 0;
}
