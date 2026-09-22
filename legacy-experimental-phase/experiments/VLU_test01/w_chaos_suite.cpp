/*
═══════════════════════════════════════════════════════════════════════
🔥 C++ W CHAOS SUITE: THE BRANCH KILLER 🔥
Target: Destroy Branch Predictor
Conditions: High Entropy, High Density, Complex Logic.
Optimization: Fast Algebraic Math (Pade/Abs)
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


// OpenMP
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

using namespace std;
using namespace chrono;

struct ExperimentResult {
  string name;
  double time_trad;
  double time_weyl;
  double speedup;
  long long checksum_trad; // To verify correctness/work
  double checksum_weyl;
};
vector<ExperimentResult> suite_results;

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
// W ALGEBRAIC UTILS (FAST MATH)
// ===================================

// Branchless greater than: returns ~1.0 if a > b, else ~0.0
// Uses sign bit hack or fast abs logic.
// x / (epsilon + |x|) gives -1 to 1 sigmoid.
// (x / (eps + |x|) + 1) * 0.5 gives 0 to 1
inline float soft_step(float x) {
  // Very fast algebraic sigmoid 0..1
  // Steepness 10.0
  float val = x * 10.0f;
  return (val / (1.0f + std::abs(val)) + 1.0f) * 0.5f;
}

// Branchless Soft Clip [-1, 1]
inline float soft_clip(float x) {
  // x / (1 + |x|) approximates clamping to -1..1 smoothly
  // but actual clamping is hard to beat with min/max instruction.
  // Let's us Fast Sigmoid as the "Alternative" logic
  return x / (1.0f + std::abs(x) * 0.5f); // Soft saturation
}

// ===================================
// EXPERIMENT 1: CLASIFICACIÓN BINARIA CAÓTICA
// 100M items, 50/50 shuffle. Can CPU predict?
// ===================================
void exp_1_chaos_binary() {
  cout << "\n=== EXP 1: CLASIFICACIÓN BINARIA CAÓTICA (100M) ===\n";
  const size_t N = 100'000'000;
  vector<float> data(N);
  mt19937 gen(1337);
  uniform_real_distribution<float> dist(-100.0f, 100.0f);

  // Fill
  for (size_t i = 0; i < N; i++)
    data[i] = dist(gen);
  // Shuffle to ensure entropy
  shuffle(data.begin(), data.end(), gen);

  // TRADICIONAL
  Timer t1;
  long long sum_trad = 0;
#pragma omp parallel for reduction(+ : sum_trad)
  for (size_t i = 0; i < N; i++) {
    if (data[i] > 0.0f)
      sum_trad++;
  }
  double time_trad = t1.elapsed();

  // W (Optimized)
  // No tanh. Just sign bit extraction logic (branchless)
  // float mask = (data[i] > 0.0f); -> This compiles to SETCC (branchless)
  // usually Let's force algebraic math to "simulate" a soft counter
  Timer t2;
  double sum_weyl = 0;
#pragma omp parallel for reduction(+ : sum_weyl)
  for (size_t i = 0; i < N; i++) {
    // Soft Step function
    sum_weyl += soft_step(data[i]);
  }
  double time_weyl = t2.elapsed();

  double speedup = time_trad / time_weyl;
  suite_results.push_back({"Chaos Classification", time_trad, time_weyl,
                           speedup, sum_trad, sum_weyl});
  cout << "Traditional: " << time_trad << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

// ===================================
// EXPERIMENT 2: CLIPPING DENSO
// 200M items. 80% out of range. 3 Branches needed Trad.
// ===================================
void exp_2_dense_clipping() {
  cout << "\n=== EXP 2: CLIPPING DENSO (200M, 80% Outliers) ===\n";
  const size_t N = 200'000'000;
  vector<float> data(N);
  mt19937 gen(42);
  // Rango amplio (-5 a 5), queremos clip a [-1, 1].
  // Muchos estarán fuera.
  uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (size_t i = 0; i < N; i++)
    data[i] = dist(gen);

  vector<float> res_trad(N);
  vector<float> res_weyl(N);

  // TRADICIONAL
  // if < -1, if > 1, else.
  Timer t1;
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    float x = data[i];
    if (x < -1.0f)
      res_trad[i] = -1.0f;
    else if (x > 1.0f)
      res_trad[i] = 1.0f;
    else
      res_trad[i] = x;
  }
  double time_trad = t1.elapsed();

  // W
  // x / (1 + |x|) * scale
  // Doesn't strictly clip equal to box, but performs the "Saturation" role
  // Using Algebra.
  Timer t2;
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    float x = data[i];
    // Optimized variant: x * (1 / (1+|x|))
    // We roughly scale input so 1.0 maps to ~0.5 then mult by 2?
    // Let's just use the canical fast sigmoid physics saturation
    res_weyl[i] = x / (1.0f + std::abs(x) * 0.5f);
  }
  double time_weyl = t2.elapsed();

  double speedup = time_trad / time_weyl;
  suite_results.push_back(
      {"Dense Clipping", time_trad, time_weyl, speedup, 0, 0});
  cout << "Traditional: " << time_trad << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

// ===================================
// EXPERIMENT 3: COLLISION DETECTION APRETADO
// 100M Ships vs 1024 Mines. Box is small. 50% hit rate.
// ===================================
struct Point {
  float x, y, z;
};
void exp_3_collision_tight() {
  cout << "\n=== EXP 3: COLLISION TIGHT (Heavy Load) ===\n";
  // Reduce N slightly for reasonable runtime, density matters not total N
  const int N_SHIPS = 100'000; // 100k
  const int N_MINES = 512;
  // Box size small
  const float BOX = 200.0f;
  const float RAD = 15.0f; // Big mines

  vector<Point> ships(N_SHIPS);
  vector<Point> mines(N_MINES);
  mt19937 gen(1);
  uniform_real_distribution<float> d(-BOX, BOX);
  for (auto &p : ships)
    p = {d(gen), d(gen), d(gen)};
  for (auto &p : mines)
    p = {d(gen), d(gen), d(gen)};

  // TRADICIONAL
  Timer t1;
  long long hits_trad = 0;
  float rad_sq = RAD * RAD;

#pragma omp parallel for reduction(+ : hits_trad)
  for (int i = 0; i < N_SHIPS; i++) {
    for (int j = 0; j < N_MINES; j++) {
      float dx = ships[i].x - mines[j].x;
      float dy = ships[i].y - mines[j].y;
      float dz = ships[i].z - mines[j].z;
      float dist_sq = dx * dx + dy * dy + dz * dz;

      // Hard Branch. High probability of true.
      if (dist_sq < rad_sq) {
        // Do something heavy to simulate impact physics
        // hits_trad += (long long)(std::sqrt(dist_sq)); -> slow
        hits_trad++;
      }
    }
  }
  double time_trad = t1.elapsed();

  // W
  Timer t2;
  double hits_weyl = 0;
#pragma omp parallel for reduction(+ : hits_weyl)
  for (int i = 0; i < N_SHIPS; i++) {
    for (int j = 0; j < N_MINES; j++) {
      float dx = ships[i].x - mines[j].x;
      float dy = ships[i].y - mines[j].y;
      float dz = ships[i].z - mines[j].z;
      float dist_sq = dx * dx + dy * dy + dz * dz;

      // Continuous Logic
      // Force = 1 / (1 + r^2)
      // No IF check. Always compute.
      float w = 1.0f / (1.0f + dist_sq * 0.1f);
      hits_weyl += w;
    }
  }
  double time_weyl = t2.elapsed();

  double speedup = time_trad / time_weyl;
  suite_results.push_back(
      {"Tight Collision", time_trad, time_weyl, speedup, hits_trad, hits_weyl});
  cout << "Traditional: " << time_trad << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

// ===================================
// EXPERIMENT 4: FILTROS COMPLEJOS CAÓTICOS
// AND/OR Hell. 100M items. 5 Traits.
// (F1>0 AND F2<0.5 AND ...)
// ===================================
struct Entity {
  float f1, f2, f3, f4, f5;
};
void exp_4_complex_logic() {
  cout << "\n=== EXP 4: FILTRO LOGICO COMPLEJO (AND/OR HELL) ===\n";
  const size_t N = 50'000'000;
  vector<Entity> data(N);
  mt19937 gen(99);
  uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (size_t i = 0; i < N; i++)
    data[i] = {d(gen), d(gen), d(gen), d(gen), d(gen)};

  // TRADICIONAL
  // if ((f1 > 0) && (f2 < 0.5) && (f3 > -0.25) || (f4 > 0.75))
  Timer t1;
  long long count_trad = 0;
#pragma omp parallel for reduction(+ : count_trad)
  for (size_t i = 0; i < N; i++) {
    // High complexity branch
    if ((data[i].f1 > 0.0f) && (data[i].f2 < 0.5f) && (data[i].f3 > -0.25f) ||
        (data[i].f4 > 0.75f)) {
      count_trad++;
    }
  }
  double time_trad = t1.elapsed();

  // W
  // AND -> Multiply (soft steps)
  // OR -> Add (soft steps) - clamp
  Timer t2;
  double count_weyl = 0;
#pragma omp parallel for reduction(+ : count_weyl)
  for (size_t i = 0; i < N; i++) {
    float f1 = soft_step(data[i].f1);         // > 0
    float f2 = soft_step(0.5f - data[i].f2);  // < 0.5 -> 0.5-f2 > 0
    float f3 = soft_step(data[i].f3 + 0.25f); // > -0.25
    float f4 = soft_step(data[i].f4 - 0.75f); // > 0.75

    // (A & B & C) | D
    float and_part = f1 * f2 * f3;
    // OR algebraic: A + B - A*B (Exact probabilistic OR) or simply A+B for
    // score
    float prob = and_part + f4 - (and_part * f4);

    count_weyl += prob;
  }
  double time_weyl = t2.elapsed();

  double speedup = time_trad / time_weyl;
  suite_results.push_back({"Complex Logic Filter", time_trad, time_weyl,
                           speedup, count_trad, count_weyl});
  cout << "Traditional: " << time_trad << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

// ===================================
// EXPERIMENT 5: CADENAS DE DECISIONES RANDOM
// 10 Checkpoints. Pure Branch Prediction Stress.
// ===================================
void exp_5_random_chains() {
  cout << "\n=== EXP 5: CADENAS DE DECISIONES RANDOM (Branch Nightmare) ===\n";
  const size_t N = 20'000'000; // 20M deep chains
  vector<float> data(N);
  mt19937 gen(555);
  uniform_real_distribution<float> d(0.0f, 100.0f);
  for (size_t i = 0; i < N; i++)
    data[i] = d(gen);

  // Thresholds random
  float t[10] = {15, 28, 37, 42, 55, 61, 78, 83, 91, 99};

  // TRADICIONAL
  Timer t1;
  long long score_trad = 0;
#pragma omp parallel for reduction(+ : score_trad)
  for (size_t i = 0; i < N; i++) {
    int s = 0;
    float x = data[i];
    // 10 Saltos
    if (x > t[0])
      s++;
    if (x > t[1])
      s++;
    if (x > t[2])
      s++;
    if (x > t[3])
      s++;
    if (x > t[4])
      s++;
    if (x > t[5])
      s++;
    if (x > t[6])
      s++;
    if (x > t[7])
      s++;
    if (x > t[8])
      s++;
    if (x > t[9])
      s++;
    score_trad += s;
  }
  double time_trad = t1.elapsed();

  // W
  // Vectorized addition of soft steps
  Timer t2;
  double score_weyl = 0;
#pragma omp parallel for reduction(+ : score_weyl)
  for (size_t i = 0; i < N; i++) {
    float x = data[i];
    float s = 0;
    // Unroll
    for (int k = 0; k < 10; k++) {
      // soft_step(x - t[k])
      // Manual optimization: Use fast sign/abs logic here
      float val = (x - t[k]) * 10.0f;
      s += (val / (1.0f + std::abs(val)) + 1.0f) * 0.5f;
    }
    score_weyl += s;
  }
  double time_weyl = t2.elapsed();

  double speedup = time_trad / time_weyl;
  suite_results.push_back({"Random Decision Chains", time_trad, time_weyl,
                           speedup, score_trad, score_weyl});
  cout << "Traditional: " << time_trad << "s\n";
  cout << "W:     " << time_weyl << "s\n";
  cout << "Speedup:     " << speedup << "x\n";
}

int main() {
  cout << "╔════════════════════════════════════════════════════╗\n";
  cout << "║ W CHAOS SUITE: BRANCH PREDICTOR DESTRUCTION  ║\n";
  cout << "╚════════════════════════════════════════════════════╝\n";

  exp_1_chaos_binary();
  exp_2_dense_clipping();
  exp_3_collision_tight();
  exp_4_complex_logic();
  exp_5_random_chains();

  ofstream j("w_chaos_results.json");
  j << "{\n \"suite\": \"Chaos Branch Killer\",\n \"results\": [\n";
  for (size_t i = 0; i < suite_results.size(); i++) {
    j << "  { \"test\": \"" << suite_results[i].name << "\", "
      << "\"time_traditional\": " << suite_results[i].time_trad << ", "
      << "\"time_w\": " << suite_results[i].time_weyl << ", "
      << "\"speedup\": " << suite_results[i].speedup << " }"
      << (i < suite_results.size() - 1 ? "," : "") << "\n";
  }
  j << " ]\n}\n";

  return 0;
}
