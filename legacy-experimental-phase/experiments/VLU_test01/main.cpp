#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string> // Added include for string
#include <vector>


#ifdef _OPENMP
#include <omp.h>
#endif

// Platform specific includes for CPU ID if needed, skipping for
// portability/simplicity or using simple macros.

// --- CONFIGURATION ---
struct Config {
  size_t N = 20000000;
  int iters = 20;
  int warmup = 5;
  int threads = 1; // Default 1, if OpenMP available will create team,
                   // but we control main loop logic or omp_set_num_threads
  bool trig = true;
  bool adversarial = false;
};

// --- DATA ---
struct SimData {
  std::vector<float> r; // roughness
  std::vector<float> p; // pitch
  std::vector<float> v; // velocity

  // Output placeholders (optional, to verify writes or just checksum
  // accumulator) We will accumulate checksum to avoid memory bandwidth
  // bottleneck on writes if possible, or write to a small array? The robust way
  // is to write to an output array or verify accumulator. The requirement says
  // "acumular un double checksum". We will do that in the loop.
};

// --- UTILS ---
// Branchless Clamp 0..1
inline float b_clamp01(float x) { return std::max(0.0f, std::min(1.0f, x)); }

// Logic helpers for Volumetric
// "Gate" functions that return [0..1]
// Standard soft step or linear ramp? Linear for speed/simplicity as per req:
// "simple algebra".
inline float ramp_up(float val, float thresh, float range) {
  // 0 if val < thresh, 1 if val > thresh+range
  return b_clamp01((val - thresh) / range);
}

inline float ramp_down(float val, float thresh, float range) {
  // 1 if val < thresh, 0 if val > thresh+range
  return 1.0f - ramp_up(val, thresh, range);
}

// --- CORE LOGIC ---

// Primitives (Actions per mode)
// Shared by both implementations to ensure logic fairness
// returns a checksum-able value (e.g. mag of output vector) or struct?
// Let's return a single float "action_val" for simplicity in checksumming,
// representing the result of the complex calculation.
// OR compute x,y,z and return sum.

struct Vec3 {
  float x, y, z;
};

inline Vec3 action_stabilize(float r, float p, float v, bool use_trig) {
  float damping = 1.0f - p; // higher pitch = less stability? or whatever
  float vy = v * -0.1f * p; // try to stop fall
  // Penalize slope
  return {r * damping, vy, p * 2.0f};
}

inline Vec3 action_walk(float r, float p, float v, bool use_trig) {
  float step = use_trig ? std::sin(v) : (v * 0.5f - 1.0f);
  return {v, step, r};
}

inline Vec3 action_trot(float r, float p, float v, bool use_trig) {
  float bounce = use_trig ? std::sqrt(v * v + 1.0f) : (v * v * 0.1f);
  return {v * 1.5f, bounce, p};
}

inline Vec3 action_sprint(float r, float p, float v, bool use_trig) {
  float thrust = v * 2.0f;
  float lift = p * p * 5.0f;
  return {thrust, lift, r * v};
}

inline double vec_sum(const Vec3 &v) { return (double)(v.x + v.y + v.z); }

// ----------------------------------------------------
// IMPLEMENTATION A: BOOLEAN / BRANCHY
// ----------------------------------------------------
// Single pass with if/else chains.
double run_branchy(const SimData &data, const Config &cfg) {
  size_t N = cfg.N;
  double total_checksum = 0.0;
  const float *__restrict r_ptr = data.r.data();
  const float *__restrict p_ptr = data.p.data();
  const float *__restrict v_ptr = data.v.data();
  bool use_trig = cfg.trig;

#pragma omp parallel for reduction(+ : total_checksum) num_threads(cfg.threads)
  for (size_t i = 0; i < N; i++) {
    float r = r_ptr[i];
    float p = p_ptr[i];
    float v = v_ptr[i];

    Vec3 out;

    // BRANCHES
    // Logic:
    // si v < 2 o p > 0.75 => mode 0 (Stabilize)
    // sino si v > 7 y p < 0.35 y r < 0.45 => mode 3 (Sprint)
    // sino si v > 4 y r < 0.75 => mode 2 (Trot)
    // sino => mode 1 (Walk)

    if (v < 2.0f || p > 0.75f) {
      out = action_stabilize(r, p, v, use_trig);
    } else if (v > 7.0f && p < 0.35f && r < 0.45f) {
      out = action_sprint(r, p, v, use_trig);
    } else if (v > 4.0f && r < 0.75f) {
      out = action_trot(r, p, v, use_trig);
    } else {
      out = action_walk(r, p, v, use_trig);
    }

    total_checksum += vec_sum(out);
  }
  return total_checksum;
}

// ----------------------------------------------------
// IMPLEMENTATION B: VOLUMETRIC / BRANCHLESS
// ----------------------------------------------------
// Calculate all 4 weights, normalize, calc all 4 actions, blend.
double run_volumetric(const SimData &data, const Config &cfg) {
  size_t N = cfg.N;
  double total_checksum = 0.0;
  const float *__restrict r_ptr = data.r.data();
  const float *__restrict p_ptr = data.p.data();
  const float *__restrict v_ptr = data.v.data();
  bool use_trig = cfg.trig;

#pragma omp parallel for reduction(+ : total_checksum) num_threads(cfg.threads)
  for (size_t i = 0; i < N; i++) {
    float r = r_ptr[i];
    float p = p_ptr[i];
    float v = v_ptr[i];

    // 1. Calculate Unnormalized Gates (Continuous Logic)

    // g0 (stabilize): high pitch OR low velocity
    // Logic: v < 2 OR p > 0.75
    // Smooth formulation:
    // g_v_low = ramp_down(v, 2.0, 0.5)  (1 at 2.0, 0 at 2.5)
    // g_p_high = ramp_up(p, 0.75, 0.1)  (0 at 0.75, 1 at 0.85)
    // g0 = max(g_v_low, g_p_high)
    float g_v_low = (v < 2.5f) ? (1.0f - (v - 1.5f))
                               : 0.0f; // Simplified linear ramp manual
    // Branchless manual ramp:
    // ramp down v from 2.5 to 1.5.
    // 1.5 -> 1.0, 2.5 -> 0.0
    // val = 1.0 - (v - 1.5) = 2.5 - v. Clamp 0..1
    float w_stab_v = b_clamp01(2.5f - v);
    float w_stab_p = b_clamp01((p - 0.70f) * 10.0f); // Fast ramp up at 0.7
    float g0 = std::max(w_stab_v, w_stab_p);

    // g3 (sprint): v > 7 AND p < 0.35 AND r < 0.45
    // AND = mult (or min)
    float w_spr_v = b_clamp01(v - 6.5f);            // 6.5->0, 7.5->1
    float w_spr_p = b_clamp01((0.40f - p) * 10.0f); // 0.4->0, 0.3->1
    float w_spr_r = b_clamp01((0.50f - r) * 10.0f); // 0.5->0, 0.4->1
    float g3 = w_spr_v * w_spr_p * w_spr_r;

    // g2 (trot): v > 4 AND r < 0.75
    float w_trot_v = b_clamp01(v - 3.5f);           // 3.5->0, 4.5->1
    float w_trot_r = b_clamp01((0.80f - r) * 5.0f); // 0.8->0, 0.6->1
    float g2 = w_trot_v * w_trot_r;

    // g1 (walk): remainder base
    // To behave similar to "Else", it should be high when others are low.
    // Simple heuristic: g1 = 0.5 constant or 1.0?
    // Normalization will handle balance. Let's give it a base weight.
    float g1 = 1.0f; // Base state

    // 2. Normalize
    // To prioritize higher order logic (like IF chain),
    // we can suppress lower logic if higher logic is active?
    // But "Volumetric" means blending. Let's strictly normalize sum.
    // Wait, standard volumetric usually creates a specialized blend.
    // The prompt asks for "sum = ... gk /= sum".

    float sum = g0 + g1 + g2 + g3 + 1e-6f;
    float inv_sum = 1.0f / sum;
    g0 *= inv_sum;
    g1 *= inv_sum;
    g2 *= inv_sum;
    g3 *= inv_sum;

    // 3. Compute All Actions
    Vec3 f0 = action_stabilize(r, p, v, use_trig);
    Vec3 f3 = action_sprint(r, p, v, use_trig);
    Vec3 f2 = action_trot(r, p, v, use_trig);
    Vec3 f1 = action_walk(r, p, v, use_trig); // Walk is default

    // 4. Blend
    Vec3 out;
    out.x = g0 * f0.x + g1 * f1.x + g2 * f2.x + g3 * f3.x;
    out.y = g0 * f0.y + g1 * f1.y + g2 * f2.y + g3 * f3.y;
    out.z = g0 * f0.z + g1 * f1.z + g2 * f2.z + g3 * f3.z;

    total_checksum += vec_sum(out);
  }
  return total_checksum;
}

// --- MAIN RUNNER ---
int main(int argc, char **argv) {
  Config cfg;

  // Simple CLI parser
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--N" && i + 1 < argc)
      cfg.N = std::stoull(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc)
      cfg.iters = std::stoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc)
      cfg.warmup = std::stoi(argv[++i]);
    else if (arg == "--threads" && i + 1 < argc)
      cfg.threads = std::stoi(argv[++i]);
    else if (arg == "--trig" && i + 1 < argc)
      cfg.trig = (std::stoi(argv[++i]) != 0);
    else if (arg == "--adversarial" && i + 1 < argc)
      cfg.adversarial = (std::stoi(argv[++i]) != 0);
  }

  std::cout << "--- BENCHMARK CONFIG ---" << std::endl;
  std::cout << "N: " << cfg.N << std::endl;
  std::cout << "Threads: " << cfg.threads << std::endl;
  std::cout << "Trig Math: " << (cfg.trig ? "ON" : "OFF") << std::endl;
  std::cout << "Adversarial Data: " << (cfg.adversarial ? "ON" : "OFF")
            << std::endl;

  // Gen Data
  std::cout << "Generating Data..." << std::endl;
  SimData data;
  data.r.resize(cfg.N);
  data.p.resize(cfg.N);
  data.v.resize(cfg.N);

  std::mt19937 rng(42);
  // Roughness 0..1, Pitch 0..1, Vel 0..10

  if (cfg.adversarial) {
    // Force lots of transitions?
    // Generate values right on the boundaries of logic
    std::uniform_real_distribution<float> d_r(0.3f, 0.8f);
    std::uniform_real_distribution<float> d_p(0.3f, 0.8f);
    std::uniform_real_distribution<float> d_v(1.5f, 7.5f);
    for (size_t i = 0; i < cfg.N; i++) {
      data.r[i] = d_r(rng);
      data.p[i] = d_p(rng);
      data.v[i] = d_v(rng);
    }
  } else {
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);
    std::uniform_real_distribution<float> d_v(0.0f, 10.0f);
    for (size_t i = 0; i < cfg.N; i++) {
      data.r[i] = d01(rng);
      data.p[i] = d01(rng);
      data.v[i] = d_v(rng);
    }
  }

  // --- WARMUP ---
  std::cout << "Warming up..." << std::endl;
  double dummy = 0;
  for (int i = 0; i < cfg.warmup; i++) {
    dummy += run_branchy(data, cfg);
    dummy += run_volumetric(data, cfg);
  }

  // --- MEASURE BRANCHY ---
  std::cout << "Running Branchy..." << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  double cs_branchy = 0;
  for (int i = 0; i < cfg.iters; i++)
    cs_branchy += run_branchy(data, cfg);
  auto t2 = std::chrono::high_resolution_clock::now();
  double dt_branchy =
      std::chrono::duration<double, std::milli>(t2 - t1).count() /
      cfg.iters; // ms per iter

  // --- MEASURE VOLUMETRIC ---
  std::cout << "Running Volumetric..." << std::endl;
  auto t3 = std::chrono::high_resolution_clock::now();
  double cs_volumetric = 0;
  for (int i = 0; i < cfg.iters; i++)
    cs_volumetric += run_volumetric(data, cfg);
  auto t4 = std::chrono::high_resolution_clock::now();
  double dt_volumetric =
      std::chrono::duration<double, std::milli>(t4 - t3).count() /
      cfg.iters; // ms per iter

  // Stats
  double branchy_mps = (cfg.N / 1e6) / (dt_branchy / 1000.0);
  double vol_mps = (cfg.N / 1e6) / (dt_volumetric / 1000.0);

  // Normalize cheksums (avg per iter)
  cs_branchy /= cfg.iters;
  cs_volumetric /= cfg.iters;

  std::cout << "\n--- RESULTS ---\n";
  std::cout << "Method             | Time (ms) | M Samples/s | Checksum\n";
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Boolean (Branchy)  | " << std::setw(9) << dt_branchy << " | "
            << std::setw(11) << branchy_mps << " | " << cs_branchy << "\n";
  std::cout << "Volumetric (Blend) | " << std::setw(9) << dt_volumetric << " | "
            << std::setw(11) << vol_mps << " | " << cs_volumetric << "\n";
  std::cout << "--------------------------------------------------------\n";
  std::cout << "Speedup (Branchy/Volumetric): " << dt_branchy / dt_volumetric
            << "x\n";

  return 0;
}
