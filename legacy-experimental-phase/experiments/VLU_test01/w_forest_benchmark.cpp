/*
═══════════════════════════════════════════════════════════════════════
🔥 EXPERIMENTO 8: LA FÁBRICA DE DECISIONES (Random Forest Vectorized) 🔥
Escenario: 10M Samples, 100 Trees (Depth 8 for manageability in raw code)
Target: Machine Learning Inference Throughput
Comparison: Branching Path vs Oblivious Evaluation (Vectorized)
═══════════════════════════════════════════════════════════════════════
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

using namespace std;
using namespace chrono;

// ===================================
// CONFIGURACIÓN
// ===================================
const size_t NUM_SAMPLES = 5'000'000; // 5M Samples
const int NUM_TREES = 50;             // 50 Trees
const int TREE_DEPTH = 6;             // Depth 6 = 64 leaves
// Note: Full depth 20 oblivious tree is 2^20 nodes (too big for RAM cache).
// Oblivious trees work best for shallow depths (ensemble boosting usually 4-8).
const int NUM_FEATURES = 16;
const int NUM_NODES_PER_TREE = (1 << TREE_DEPTH) - 1; // 63 nodes

struct TreeNode {
  int feature_idx;
  float threshold;
  float left_val; // Used for leaf values or ignored
  float right_val;
  bool is_leaf;
};

// ===================================
// TREE GEN
// ===================================
// Simple random tree structure
vector<TreeNode> generate_tree(int depth) {
  int count = (1 << depth) - 1;
  vector<TreeNode> nodes(count);
  mt19937 gen(1234);
  uniform_int_distribution<int> f_dist(0, NUM_FEATURES - 1);
  uniform_real_distribution<float> t_dist(0.0f, 1.0f);

  for (int i = 0; i < count; i++) {
    nodes[i].feature_idx = f_dist(gen);
    nodes[i].threshold = t_dist(gen);
    nodes[i].is_leaf = false; // We just compute paths, simple structure
                              // Leaves are implicit at indices >= count/2
  }
  return nodes;
}

// ===================================
// UTILS
// ===================================
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
// MAIN
// ===================================
int main() {
  cout << "╔════════════════════════════════════════════════════════╗\n";
  cout << "║ EXPERIMENTO 8: LA FÁBRICA DE DECISIONES (ML INFERENCE) ║\n";
  cout << "╚════════════════════════════════════════════════════════╝\n";
  cout << "Muestras: " << NUM_SAMPLES << "\n";
  cout << "Árboles: " << NUM_TREES << " (Depth " << TREE_DEPTH << ")\n";

  // 1. Data Gen
  // Features array: [Sample0_F0..F15, Sample1_F0..F15...] (Row major)
  // For Traditional: Row major is okay (one sample at a time)
  // For W: To vectorize across samples, we'd prefer SoA, but let's stick
  // to standard layout to show auto-vectorization limits or use gathering?
  // Actually, W "Oblivious" usually vectorizes NODE evaluation across one
  // sample OR multiple samples. Let's optimize W for Batch Processing:
  // Process 8 samples at once.

  vector<float> features(NUM_SAMPLES * NUM_FEATURES);
  mt19937 gen(42);
  uniform_real_distribution<float> d(0.0f, 1.0f);
#pragma omp parallel for
  for (size_t i = 0; i < features.size(); i++)
    features[i] = d(gen);

  // Forest
  vector<vector<TreeNode>> forest(NUM_TREES);
  for (int i = 0; i < NUM_TREES; i++)
    forest[i] = generate_tree(TREE_DEPTH);

  // Results
  vector<float> predictions_trad(NUM_SAMPLES, 0.0f);
  vector<float> predictions_weyl(NUM_SAMPLES, 0.0f);

  // ==============================================================
  // 1. TRADICIONAL (Pointer Chasing / Branching)
  // ==============================================================
  double t_trad;
  {
    Timer t;
#pragma omp parallel for
    for (size_t s = 0; s < NUM_SAMPLES; s++) {
      float total_score = 0;
      size_t base_idx = s * NUM_FEATURES;

      for (int t = 0; t < NUM_TREES; t++) {
        // Navigate Pointer/Index Array
        int curr = 0; // Root index 0
        // Traverse depth
        for (int d = 0; d < TREE_DEPTH - 1; d++) {
          const TreeNode &node = forest[t][curr];
          float val = features[base_idx + node.feature_idx];

          if (val <= node.threshold) {
            curr = 2 * curr + 1; // Left child
          } else {
            curr = 2 * curr + 2; // Right child
          }
        }
        // Leaf value (simulated by threshold of leaf node)
        total_score += forest[t][curr].threshold;
      }
      predictions_trad[s] = total_score;
    }
    t_trad = t.elapsed();
    cout << ">> Tradicional (Branching): " << t_trad << " s" << endl;
  }

  // ==============================================================
  // 2. W (Oblivious Execution / Predication)
  // ==============================================================
  // Concept: Compare ALL nodes at each layer.
  // Layer 0: Root Compare -> Output Mask L, Mask R
  // Layer 1: Compare Node L, Node R. Combine with Mask.
  // This is "Oblivious Tree".
  // For depth 6, we evaluate 63 comparisons instead of 6 branches.
  // The win comes if SIMD > 10x scalar branch cost.

  // To minimize memory bandwidth, we'll flatten forest to arrays
  // But for fair comparison of Logic, we use same structure logic with masks.

  double t_weyl;
  {
    Timer t;
#pragma omp parallel for
    for (size_t s = 0; s < NUM_SAMPLES; s++) {
      float total_score = 0;
      size_t base_idx = s * NUM_FEATURES;

      for (int t = 0; t < NUM_TREES; t++) {
        // Oblivious Traversal
        // We maintain a "current probability" of being at each node in the
        // layer Depth 0: Prob=1.0 at Node 0.

        // Simplified Implementation for Benchmark:
        // We track the index using math instead of branching.
        // idx = idx*2 + (val > thresh) + 1
        // This is still dependent chain, but branchless.
        // Auto-vectorization might pick this up for multiple samples?

        int curr = 0;
        for (int d = 0; d < TREE_DEPTH - 1; d++) {
          const TreeNode &node = forest[t][curr];
          float val = features[base_idx + node.feature_idx];

          // Branchless Index Update
          // (val > thresh) -> 0 or 1.
          // left (<=) -> +1
          // right (>) -> +2
          // Formula: 2*curr + 1 + (val > thresh)
          int go_right = (val > node.threshold);
          curr = 2 * curr + 1 + go_right;
        }
        total_score += forest[t][curr].threshold;
      }
      predictions_weyl[s] = total_score;
    }
    t_weyl = t.elapsed();
    cout << ">> W (Predicate idx): " << t_weyl << " s" << endl;
  }

  double speedup = t_trad / t_weyl;
  cout << "\nResultados Finales:" << endl;
  cout << "-----------------------------------" << endl;
  cout << "Tiempo Tradicional: " << t_trad << "s" << endl;
  cout << "Tiempo W:     " << t_weyl << "s" << endl;
  cout << "SPEEDUP FACTOR:     " << fixed << setprecision(2) << speedup << "x"
       << endl;
  cout << "-----------------------------------" << endl;

  // Check
  int diffs = 0;
  for (int i = 0; i < 1000; i++)
    if (abs(predictions_trad[i] - predictions_weyl[i]) > 0.001f)
      diffs++;
  if (diffs > 0)
    cout << "WARN: CHECKSUM FAIL" << endl;

  ofstream json("w_forest_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Decision Factory (Random Forest)\",\n";
  json << "  \"samples\": " << NUM_SAMPLES << ",\n";
  json << "  \"trees\": " << NUM_TREES << ",\n";
  json << "  \"traditional_s\": " << t_trad << ",\n";
  json << "  \"w_s\": " << t_weyl << ",\n";
  json << "  \"speedup\": " << speedup << "\n";
  json << "}\n";

  return 0;
}
