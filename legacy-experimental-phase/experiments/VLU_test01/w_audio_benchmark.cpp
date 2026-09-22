/*
═══════════════════════════════════════════════════════════════════════
🔥 EXPERIMENTO 3: LA PESADILLA DEL AUDIO (Mixer 192 Loops) 🔥
Escenario: 192 Canales x 48kHz x 10 Efectos en Cadena
Target: Audio DSP Throughput & Real-time Stability
Optimization: Branchless Math vs Conditional Logic
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
// 1 Segundo de audio a 48kHz
const int SAMPLE_RATE = 48000;
const int NUM_CHANNELS = 192;
const int NUM_FRAMES = SAMPLE_RATE * 10; // 10 segundos de audio total
const int EFFECT_CHAIN_LENGTH = 10;

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
// DSP KERNELS
// ===================================

// -- TRADICIONAL (CONDITIONAL MESS) --
// Implements: Gate -> Compressor (Multi-zone) -> Saturation (Hard Clip)
inline float process_sample_traditional(float sample) {
  float out = sample;

  // Simulate 10 effects chain by repeating logic
  for (int i = 0; i < EFFECT_CHAIN_LENGTH; i++) {
    // 1. Noise Gate
    if (std::abs(out) < 0.05f) {
      out = 0.0f;
    }
    // 2. Compressor / Limiter Logic
    else if (out > 0.8f) {
      // Hard compression
      out = 0.8f + (out - 0.8f) * 0.1f;
    } else if (out > 0.5f) {
      // Soft compression
      out = out * 0.8f;
    } else if (out < -0.8f) {
      out = -0.8f + (out + 0.8f) * 0.1f;
    }
    // 3. Distortion / Saturation (Hard Clip)
    if (out > 1.0f)
      out = 1.0f;
    if (out < -1.0f)
      out = -1.0f;

    // Gain stage
    out *= 1.1f;
  }
  return out;
}

// -- W (ALGEBRAIC FLOW) --
// Implements: Soft Gate -> Soft Knee Compressor -> Soft Saturation
// Uses only Auto-Vectorizable Math (min, max, abs, ops)
inline float process_sample_w(float sample) {
  float out = sample;

  // 10 Effects Chain
  for (int i = 0; i < EFFECT_CHAIN_LENGTH; i++) {
    // 1. Soft Gate using Squared Weighting (x * (x^2 / (x^2 + eps)))
    // Effectively silences low signals smoothly
    float x2 = out * out;
    float gate = x2 / (x2 + 0.0025f); // 0.05^2 = 0.0025
    out *= gate;

    // 2. Soft Knee Compressor (Algebraic)
    // Gain reduction = 1 / (1 + max(0, |x|-thresh) * ratio)
    float abs_x = std::abs(out);
    float over = std::max(0.0f, abs_x - 0.5f);
    float gain_reduction = 1.0f / (1.0f + over * 2.0f); // Ratio approx
    out *= gain_reduction;

    // 3. Soft Saturation (Fast Sigmoid)
    // x / (1 + |x|/k) -> Smoothly limits to k
    out = out / (1.0f + std::abs(out) * 0.5f); // Soft limit near 2.0

    // Gain stage
    out *= 1.1f;
  }
  return out;
}

// ===================================
// MAIN BENCHMARK
// ===================================
int main() {
  cout << "╔══════════════════════════════════════════════════════╗\n";
  cout << "║ EXPERIMENTO 3: LA PESADILLA DEL AUDIO (DSP MIXER)    ║\n";
  cout << "╚══════════════════════════════════════════════════════╝\n";
  cout << "Canales: " << NUM_CHANNELS << " (192 tracks)\n";
  cout << "Duración: 10s (480k muestras/canal)\n";
  cout << "Efectos: 10 por canal\n";

  // 1. Generar Audio Multicanal (Caótico)
  // Vector lineal gigante: [CH1_S1, CH1_S2... CH2_S1...]
  size_t total_samples = (size_t)NUM_CHANNELS * NUM_FRAMES;
  vector<float> audio_buffer(total_samples);

  mt19937 gen(44100);
  // Audio signals are typically -1 to 1 but dynamic (silence + loud busts)
  // We simulate this with a distribution that creates "peaks"
  normal_distribution<float> dist(0.0f, 0.4f);

#pragma omp parallel for
  for (size_t i = 0; i < total_samples; i++) {
    // Add some "DC offset" or chaotic bursts to random channels
    float burst = (i % 5000 < 500) ? 2.0f : 1.0f; // Bursts
    audio_buffer[i] = dist(gen) * burst;
  }

  vector<float> out_trad(total_samples);
  vector<float> out_weyl(total_samples);

  // ==============================================================
  // 1. TRADICIONAL (Scalar Processing due to Branching)
  // ==============================================================
  double t_trad;
  {
    Timer t;
// Process channel by channel (parallel channels)
// Inside channel, loop is sequential/scalar
#pragma omp parallel for
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
      size_t offset = (size_t)ch * NUM_FRAMES;
      for (int i = 0; i < NUM_FRAMES; i++) {
        out_trad[offset + i] =
            process_sample_traditional(audio_buffer[offset + i]);
      }
    }
    t_trad = t.elapsed();
    cout << ">> Tradicional (If/Else):   " << t_trad << " s" << endl;
  }

  // ==============================================================
  // 2. W (Vectorized Processing)
  // ==============================================================
  double t_weyl;
  {
    Timer t;
// Compiler auto-vectorization should eat this inner loop alive
// because math is continuous.
#pragma omp parallel for
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
      size_t offset = (size_t)ch * NUM_FRAMES;
      // #pragma omp simd // Hint to enforce AVX
      for (int i = 0; i < NUM_FRAMES; i++) {
        out_weyl[offset + i] = process_sample_w(audio_buffer[offset + i]);
      }
    }
    t_weyl = t.elapsed();
    cout << ">> W (Algebraic):     " << t_weyl << " s" << endl;
  }

  double speedup = t_trad / t_weyl;

  cout << "\nResultados Finales:" << endl;
  cout << "-----------------------------------" << endl;
  cout << "Tiempo Tradicional: " << t_trad << "s" << endl;
  cout << "Tiempo W:     " << t_weyl << "s" << endl;
  cout << "SPEEDUP FACTOR:     " << fixed << setprecision(2) << speedup << "x"
       << endl;
  cout << "-----------------------------------" << endl;

  ofstream json("w_audio_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Audio Nightmare (192 Ch Mixer)\",\n";
  json << "  \"total_samples\": " << total_samples << ",\n";
  json << "  \"traditional_s\": " << t_trad << ",\n";
  json << "  \"w_s\": " << t_weyl << ",\n";
  json << "  \"speedup\": " << speedup << "\n";
  json << "}\n";

  return 0;
}
