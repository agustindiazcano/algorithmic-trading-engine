/*
═══════════════════════════════════════════════════════════════════════
🔥 EXPERIMENTO 6: EL JUEGO DE LA VIDA EXTREMO (Conway's Game of Life) 🔥
Escenario: 10,000 x 10,000 Grid (100M Cells)
Target: Automata Cellular Update (Neighbor Counting + Logic)
Optimization: Bitwise/Algebraic Logic vs Branching
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
// 10k x 10k = 100M cells per generation.
// This is huge for CPU but manageable.
const int N_ROW = 10000;
const int N_COL = 10000;
const size_t SIZE = (size_t)N_ROW * N_COL;

// Data structure: Flattened vector of bytes (uint8_t) 0 or 1
// We use bytes instead of bits for easier C++ comparison (bits require masking
// ops that W also wins at but Trad suffers on)
typedef uint8_t Cell;

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
  cout << "╔══════════════════════════════════════════════════════╗\n";
  cout << "║ EXPERIMENTO 6: GAME OF LIFE EXTREMO                  ║\n";
  cout << "╚══════════════════════════════════════════════════════╝\n";
  cout << "Grid: " << N_ROW << "x" << N_COL << " (" << SIZE / 1'000'000
       << "M Células)\n";

  // 1. Init Data (Random Chaos)
  vector<Cell> grid_curr(SIZE);
  vector<Cell> grid_next_trad(SIZE);
  vector<Cell> grid_next_weyl(SIZE);

  mt19937 gen(42);
  uniform_int_distribution<int> dist(0, 1); // 50/50 seed
#pragma omp parallel for
  for (size_t i = 0; i < SIZE; i++)
    grid_curr[i] = (Cell)dist(gen);

  // ==============================================================
  // 1. TRADICIONAL (Nested IFs)
  // ==============================================================
  double t_trad;
  {
    Timer t;
// Avoid boundary checks for speed (ignore edges logic in benchmark)
#pragma omp parallel for
    for (int y = 1; y < N_ROW - 1; y++) {
      size_t row_offset = y * N_COL;
      // Hot inner loop
      for (int x = 1; x < N_COL - 1; x++) {
        size_t idx = row_offset + x;

        // Count Neighbors (8 lookups)
        // This is memory bound, but logic follows
        int neighbors = grid_curr[idx - N_COL - 1] + grid_curr[idx - N_COL] +
                        grid_curr[idx - N_COL + 1] + grid_curr[idx - 1] +
                        grid_curr[idx + 1] + grid_curr[idx + N_COL - 1] +
                        grid_curr[idx + N_COL] + grid_curr[idx + N_COL + 1];

        // CONWAY RULES LOGIC (Branching)
        Cell state = grid_curr[idx];
        Cell next_state = 0;

        if (state == 1) {
          if (neighbors < 2 || neighbors > 3) {
            next_state = 0; // Die
          } else {
            next_state = 1; // Survive
          }
        } else {
          if (neighbors == 3) {
            next_state = 1; // Reproduce
          } else {
            next_state = 0; // Stay Dead
          }
        }
        grid_next_trad[idx] = next_state;
      }
    }
    t_trad = t.elapsed();
    cout << ">> Tradicional (If/Else):   " << t_trad << " s" << endl;
  }

  // ==============================================================
  // 2. W (Algebraic / Table Lookup / Bitwise)
  // ==============================================================
  double t_weyl;
  {
    Timer t;
#pragma omp parallel for
    for (int y = 1; y < N_ROW - 1; y++) {
      size_t row_offset = y * N_COL;
      // Hot inner loop
      // In a real optimized code, we would load 32 bytes into AVX2 registers
      // and perform neighbor sums in parallel.
      // Clang -O3 does a decent job if logic is flat.

      // #pragma omp simd // Hint
      for (int x = 1; x < N_COL - 1; x++) {
        size_t idx = row_offset + x;

        // Sum Neighbors (Same memory cost)
        int neighbors = grid_curr[idx - N_COL - 1] + grid_curr[idx - N_COL] +
                        grid_curr[idx - N_COL + 1] + grid_curr[idx - 1] +
                        grid_curr[idx + 1] + grid_curr[idx + N_COL - 1] +
                        grid_curr[idx + N_COL] + grid_curr[idx + N_COL + 1];

        Cell state = grid_curr[idx];

        // W LOGIC:
        // Rule: Alive if (neighbors == 3) OR (neighbors == 2 AND current == 1)
        // This is a boolean function: R = (N==3) | ((N==2) & S)

        // Pure Math / Logic ops (Branchless)
        // Using comparison results as 0/1 integers

        int n3 = (neighbors == 3);
        int n2 = (neighbors == 2);

        // The formula:
        // next = n3 | (n2 & state);
        // No IFs. Just logical OR/AND.

        grid_next_weyl[idx] = (Cell)(n3 | (n2 & state));
      }
    }
    t_weyl = t.elapsed();
    cout << ">> W (Logic Ops):     " << t_weyl << " s" << endl;
  }

  double speedup = t_trad / t_weyl;

  cout << "\nResultados Finales:" << endl;
  cout << "-----------------------------------" << endl;
  cout << "Tiempo Tradicional: " << t_trad << "s" << endl;
  cout << "Tiempo W:     " << t_weyl << "s" << endl;
  cout << "SPEEDUP FACTOR:     " << fixed << setprecision(2) << speedup << "x"
       << endl;
  cout << "-----------------------------------" << endl;

  // Correctness Check
  int errs = 0;
  for (size_t i = 0; i < SIZE; i++)
    if (grid_next_trad[i] != grid_next_weyl[i])
      errs++;
  if (errs > 0)
    cout << "CHECKSUM ERROR: " << errs << endl;

  ofstream json("w_conway_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Game of Life Extreme\",\n";
  json << "  \"cells\": " << SIZE << ",\n";
  json << "  \"traditional_s\": " << t_trad << ",\n";
  json << "  \"w_s\": " << t_weyl << ",\n";
  json << "  \"speedup\": " << speedup << "\n";
  json << "}\n";

  return 0;
}
