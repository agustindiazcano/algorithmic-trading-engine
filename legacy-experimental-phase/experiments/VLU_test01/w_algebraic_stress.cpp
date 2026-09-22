/*
═══════════════════════════════════════════════════════════════════════
🔥 C++ STRESS TESTS: ALGEBRAICO (W) VS BOOLEANO (TRADICIONAL) 🔥
Scenario: "The logic war"
═══════════════════════════════════════════════════════════════════════
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>


// Try to include OpenMP, if not available, macros will be ignored or we define
// stubs
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

using namespace std;
using namespace chrono;

// ============================================================================
// RESULTADOS ESTRUCTURA
// ============================================================================
struct TestResult {
  string name;
  double time_bool;
  double time_alg; // W
  double score_bool;
  double score_alg;
  double speedup;
};
vector<TestResult> results;

// ============================================================================
// UTILIDADES
// ============================================================================

class Timer {
  high_resolution_clock::time_point start;

public:
  Timer() : start(high_resolution_clock::now()) {}

  double elapsed() {
    auto end = high_resolution_clock::now();
    return duration_cast<duration<double>>(end - start).count();
  }
};

// Funciones algebraicas suaves (W Logic)
inline float smooth_step(float x) {
  return 0.5f + 0.5f * tanh(2.0f * x); // Reduced steepness for stability
}

inline float smooth_greater(float a, float b) { return smooth_step(a - b); }

inline float soft_clamp(float x, float vmin, float vmax) {
  float center = (vmax + vmin) * 0.5f;
  float width = (vmax - vmin) * 0.5f;
  if (width < 0.0001f)
    width = 0.0001f;
  float normalized = (x - center) / width;
  return tanh(normalized) * width + center;
}

// ============================================================================
// TEST 1: CLASIFICACIÓN BINARIA (Branch-heavy)
// ============================================================================

void test_1_binary_classification() {
  cout << "\n" << string(60, '-') << "\n";
  cout << "TEST 1: CLASIFICACIÓN BINARIA (50M comparaciones)\n";

  const size_t N = 50'000'000;
  vector<float> data(N);

  // Generar datos random (Alta entropía = Branch Misprediction probable)
  random_device rd;
  mt19937 gen(42);
  uniform_real_distribution<float> dist(-100.0f, 100.0f);

  for (size_t i = 0; i < N; i++)
    data[i] = dist(gen);

  const float threshold = 0.0f;

  // MÉTODO BOOLEANO (TRADICIONAL)
  Timer t1;
  long long count_bool = 0;

#pragma omp parallel for reduction(+ : count_bool)
  for (size_t i = 0; i < N; i++) {
    if (data[i] > threshold) {
      count_bool++;
    }
  }
  double time_bool = t1.elapsed();

  // MÉTODO ALGEBRAICO (W)
  Timer t2;
  double count_alg = 0.0;

#pragma omp parallel for reduction(+ : count_alg)
  for (size_t i = 0; i < N; i++) {
    // En lugar de branching, sumamos probabilities 0..1
    count_alg += smooth_greater(data[i], threshold);
  }
  double time_alg = t2.elapsed();

  double speedup = time_bool / time_alg;
  results.push_back({"Binary Classification", time_bool, time_alg,
                     (double)count_bool, count_alg, speedup});

  cout << "Booleano:   " << time_bool << "s\n";
  cout << "W:    " << time_alg << "s\n";
  cout << "Speedup:    " << speedup << "x\n";
}

// ============================================================================
// TEST 2: CLIPPING/SATURACIÓN
// ============================================================================

void test_2_clipping() {
  cout << "\n" << string(60, '-') << "\n";
  cout << "TEST 2: CLIPPING (50M valores)\n";

  const size_t N = 50'000'000;
  vector<float> data(N);

  mt19937 gen(42);
  normal_distribution<float> dist(0.0f, 5.0f);
  for (size_t i = 0; i < N; i++)
    data[i] = dist(gen);

  vector<float> result_bool(N);
  vector<float> result_alg(N);
  const float vmin = -1.0f, vmax = 1.0f;

  // MÉTODO BOOLEANO
  Timer t1;
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    float x = data[i];
    if (x < vmin)
      x = vmin;
    else if (x > vmax)
      x = vmax;
    result_bool[i] = x;
  }
  double time_bool = t1.elapsed();

  // MÉTODO ALGEBRAICO
  Timer t2;
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    result_alg[i] = soft_clamp(data[i], vmin, vmax);
  }
  double time_alg = t2.elapsed();

  double speedup = time_bool / time_alg;
  results.push_back({"Clipping", time_bool, time_alg, 0, 0, speedup});

  cout << "Booleano:   " << time_bool << "s\n";
  cout << "W:    " << time_alg << "s\n";
  cout << "Speedup:    " << speedup << "x\n";
}

// ============================================================================
// TEST 3: RELU (Deep Learning)
// ============================================================================

void test_3_relu() {
  cout << "\n" << string(60, '-') << "\n";
  cout << "TEST 3: ReLU ACTIVATION (50M neuronas)\n";

  const size_t N = 50'000'000;
  vector<float> activations(N);

  mt19937 gen(42);
  normal_distribution<float> dist(0.0f, 2.0f);
  for (size_t i = 0; i < N; i++)
    activations[i] = dist(gen);

  vector<float> result_bool(N);
  vector<float> result_alg(N);

  // MÉTODO BOOLEANO (max(0, x))
  Timer t1;
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    float v = activations[i];
    result_bool[i] = (v > 0) ? v : 0;
  }
  double time_bool = t1.elapsed();

  // MÉTODO ALGEBRAICO (Softplus aprox)
  Timer t2;
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    float x = activations[i];
    // Swish / Sigmoid approx
    result_alg[i] = x / (1.0f + exp(-2.0f * x));
  }
  double time_alg = t2.elapsed();

  double speedup = time_bool / time_alg;
  results.push_back({"ReLU Activation", time_bool, time_alg, 0, 0, speedup});

  cout << "Booleano:   " << time_bool << "s\n";
  cout << "W:    " << time_alg << "s\n";
  cout << "Speedup:    " << speedup << "x\n";
}

// ============================================================================
// TEST 4: CADENAS DE DECISIONES
// ============================================================================

void test_4_decision_chains() {
  cout << "\n" << string(60, '-') << "\n";
  cout << "TEST 4: CADENAS DE DECISIONES (Complex Branching)\n";

  const size_t N = 10'000'000;
  vector<float> data(N);
  mt19937 gen(42);
  uniform_real_distribution<float> dist(0.0f, 100.0f);
  for (size_t i = 0; i < N; i++)
    data[i] = dist(gen);

  const float thresholds[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95};

  // MÉTODO BOOLEANO
  Timer t1;
  long long score_bool = 0;
#pragma omp parallel for reduction(+ : score_bool)
  for (size_t i = 0; i < N; i++) {
    int score = 0;
    float x = data[i];
    if (x > 10)
      score++;
    if (x > 20)
      score++;
    if (x > 30)
      score++;
    if (x > 40)
      score++;
    if (x > 50)
      score++;
    if (x > 60)
      score++;
    if (x > 70)
      score++;
    if (x > 80)
      score++;
    if (x > 90)
      score++;
    if (x > 95)
      score++;
    score_bool += score;
  }
  double time_bool = t1.elapsed();

  // MÉTODO ALGEBRAICO
  Timer t2;
  double score_alg = 0.0;
#pragma omp parallel for reduction(+ : score_alg)
  for (size_t i = 0; i < N; i++) {
    float x = data[i];
    float s = 0.0f;
    // Unrollable, vectorizable math
    for (int j = 0; j < 10; j++) {
      s += smooth_greater(x, thresholds[j]);
    }
    score_alg += s;
  }
  double time_alg = t2.elapsed();

  double speedup = time_bool / time_alg;
  results.push_back({"Decision Chains", time_bool, time_alg, (double)score_bool,
                     score_alg, speedup});

  cout << "Booleano:   " << time_bool << "s\n";
  cout << "W:    " << time_alg << "s\n";
  cout << "Speedup:    " << speedup << "x\n";
}

// ============================================================================
// TEST 6: COLLISION DETECTION (Slightly reduced for runtime)
// ============================================================================

struct Ship {
  float x, y, z;
};
struct Mine {
  float x, y, z, radius;
};

void test_6_collision() {
  cout << "\n" << string(60, '-') << "\n";
  cout << "TEST 6: COLLISION DETECTION (High Density)\n";

  const size_t N_SHIPS = 100'000;
  const size_t N_MINES = 512;

  vector<Ship> ships(N_SHIPS);
  vector<Mine> mines(N_MINES);

  mt19937 gen(42);
  uniform_real_distribution<float> pos_dist(-500.0f, 500.0f);
  uniform_real_distribution<float> rad_dist(10.0f, 50.0f);

  for (size_t i = 0; i < N_SHIPS; i++)
    ships[i] = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
  for (size_t i = 0; i < N_MINES; i++)
    mines[i] = {pos_dist(gen), pos_dist(gen), pos_dist(gen), rad_dist(gen)};

  // MÉTODO BOOLEANO
  Timer t1;
  long long collisions_bool = 0;
#pragma omp parallel for reduction(+ : collisions_bool)
  for (size_t i = 0; i < N_SHIPS; i++) {
    for (size_t j = 0; j < N_MINES; j++) {
      float dx = ships[i].x - mines[j].x;
      float dy = ships[i].y - mines[j].y;
      float dz = ships[i].z - mines[j].z;
      float dist_sq = dx * dx + dy * dy + dz * dz;
      float threshold_sq = mines[j].radius * mines[j].radius;
      // Branch!
      if (dist_sq < threshold_sq) {
        collisions_bool++;
      }
    }
  }
  double time_bool = t1.elapsed();

  // MÉTODO ALGEBRAICO
  Timer t2;
  double collisions_alg = 0.0;
#pragma omp parallel for reduction(+ : collisions_alg)
  for (size_t i = 0; i < N_SHIPS; i++) {
    for (size_t j = 0; j < N_MINES; j++) {
      float dx = ships[i].x - mines[j].x;
      float dy = ships[i].y - mines[j].y;
      float dz = ships[i].z - mines[j].z;
      float dist_sq = dx * dx + dy * dy + dz * dz;
      float threshold_sq = mines[j].radius * mines[j].radius;

      // Continuous logic (Branchless)
      float diff = threshold_sq - dist_sq;
      // Sigmoid-like activation for collision overlap
      collisions_alg += 1.0f / (1.0f + exp(-0.1f * diff));
    }
  }
  double time_alg = t2.elapsed();

  double speedup = time_bool / time_alg;
  results.push_back({"Collision Detection", time_bool, time_alg,
                     (double)collisions_bool, collisions_alg, speedup});

  cout << "Booleano:   " << time_bool << "s\n";
  cout << "W:    " << time_alg << "s\n";
  cout << "Speedup:    " << speedup << "x\n";
}

int main() {
  cout << "=== W ALGEBRAIC STRESS TESTS ===\n";
#ifdef _OPENMP
  cout << "OpenMP Enabled. Max Threads: " << omp_get_max_threads() << "\n";
#endif

  test_1_binary_classification();
  test_2_clipping();
  test_3_relu();
  test_4_decision_chains();
  // test_5_pairwise skipped to keep runtime short, O(N^2) sucks
  test_6_collision(); // O(N*M) heavy check

  // JSON Export
  ofstream json("w_algebraic_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Algebraic (W) vs Boolean "
          "(Traditional)\",\n";
  json << "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); i++) {
    const auto &r = results[i];
    json << "    {\n";
    json << "      \"test\": \"" << r.name << "\",\n";
    json << "      \"time_traditional_s\": " << r.time_bool << ",\n";
    json << "      \"time_w_s\": " << r.time_alg << ",\n";
    json << "      \"speedup\": " << r.speedup << ",\n";
    json << "      \"score_traditional\": " << r.score_bool << ",\n";
    json << "      \"score_w\": " << r.score_alg << "\n";
    json << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
  }
  json << "  ]\n";
  json << "}\n";

  cout << "\nResults saved to w_algebraic_results.json\n";

  return 0;
}
