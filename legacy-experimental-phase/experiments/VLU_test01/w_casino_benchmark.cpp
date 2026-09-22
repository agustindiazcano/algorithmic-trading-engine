/*
═══════════════════════════════════════════════════════════════════════
🔥 EXPERIMENTO 2: EL CASINO CUÁNTICO (Montecarlo Simulation) 🔥
Escenario: 500M Tiradas | 20 Estrategias Concurrentes | Pura Entropía
Target: Branch Prediction Nightmare (Random Logic Chains)
Optimization: Algebraic Predication (Multiplying by bool masks)
═══════════════════════════════════════════════════════════════════════
*/

#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// OpenMP
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

using namespace std;
using namespace chrono;

// ===================================
// CONFIGURACIÓN
// ===================================
const size_t NUM_SPINS =
    100'000'000; // 100M (Reduced from 500M to finish faster in standard env)
const int NUM_STRATEGIES = 20;

struct SpinResult {
  int color;  // 0=Red, 1=Black, 2=Green
  int number; // 0-36
};

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
// W ALGEBRAIC HELPERS
// ===================================
// Convert bool expression to 1.0f or 0.0f branchless
inline float to_float_mask(bool cond) {
  return (float)cond; // C++ Standard guarantees true->1, false->0
}

// ===================================
// MAIN
// ===================================
int main() {
  cout << "╔════════════════════════════════════════════════════╗\n";
  cout << "║ EXPERIMENTO 2: EL CASINO CUÁNTICO (W)        ║\n";
  cout << "╚════════════════════════════════════════════════════╝\n";
  cout << "Tiradas: " << NUM_SPINS << "\n";
  cout << "Estrategias Concurrentes: " << NUM_STRATEGIES << "\n";

  // 1. Data Gen (The Entropy)
  // Usamos int array simple para la ruleta
  vector<int> spins(NUM_SPINS);
  // Simple PRNG para inicializar
  mt19937 gen(777);
  uniform_int_distribution<int> dist(0, 36);
  for (size_t i = 0; i < NUM_SPINS; i++)
    spins[i] = dist(gen);

  // Estrategias (Simples balances)
  // Para simplificar, calculamos la ganancia neta total de las 20 strats
  // Tradicional usa arrays, W usa acumuladores vectoriales.

  // ==============================================================
  // 1. TRADICIONAL (Branching Logic Hell)
  // ==============================================================
  double t_trad = 0;
  long long balance_trad = 0;

  {
    Timer t;
// Parallel loop
#pragma omp parallel reduction(+ : balance_trad)
    {
      long long local_balance = 0;
#pragma omp for
      for (size_t i = 0; i < NUM_SPINS; i++) {
        int num = spins[i];
        int color = (num == 0) ? 2
                    : ((num >= 1 && num <= 10) || (num >= 19 && num <= 28))
                        ? (num % 2 == 0 ? 1 : 0)
                        : (num % 2 == 0 ? 0 : 1);
        // 0=Red, 1=Black, 2=Green (Simplified rules logic)
        // Real roulette logic is messy branches.

        // Strategy 1: Bet Red
        if (color == 0)
          local_balance += 1;
        else
          local_balance -= 1;
        // Strategy 2: Bet Black
        if (color == 1)
          local_balance += 1;
        else
          local_balance -= 1;
        // Strategy 3: Bet Even
        if (num != 0 && num % 2 == 0)
          local_balance += 1;
        else
          local_balance -= 1;
        // Strategy 4: Bet Odd
        if (num != 0 && num % 2 != 0)
          local_balance += 1;
        else
          local_balance -= 1;
        // Strategy 5: First Dozen (1-12)
        if (num >= 1 && num <= 12)
          local_balance += 2;
        else
          local_balance -= 1;
        // Strategy 6: Number 17 (Specific)
        if (num == 17)
          local_balance += 35;
        else
          local_balance -= 1;

        // ... Repeat variations of branches for 20 strategies ...
        // Let's loop the logic to simulate 20 conds
        for (int k = 7; k < 20; k++) {
          // Conditional mess
          if (num % (k + 2) == 0)
            local_balance += 1;
          else
            local_balance -= 1;
        }
      }
      balance_trad += local_balance;
    }
    t_trad = t.elapsed();
    cout << ">> Tradicional (Branching): " << t_trad << " s" << endl;
  }

  // ==============================================================
  // 2. W (Algebraic / Predication)
  // ==============================================================
  double t_weyl = 0;
  long long balance_weyl = 0;

  {
    Timer t;
#pragma omp parallel reduction(+ : balance_weyl)
    {
      long long local_balance = 0;
#pragma omp for
      for (size_t i = 0; i < NUM_SPINS; i++) {
        int num = spins[i];
        // Compute conditions as masks (0 or 1 integers)
        // Compiler vectorizes this part efficiently
        int is_zero = (num == 0);
        int is_even = (num % 2 == 0);

        // Color Login (Branchless ternary)
        // RED rules... logic logic...
        // Simplified: red mask
        int val_idx = (num >= 1 && num <= 10) || (num >= 19 && num <= 28);
        int is_black = (!is_zero) & (val_idx ? is_even : !is_even);
        int is_red = (!is_zero) & (!is_black);

        // Strategies (Algebraic Accumulation)
        // Balance += (WinMask * WinAmt) + (LoseMask * LoseAmt)
        // Simplify: Balance += WinMask * (Win - Lose) + Lose
        // Example 1: Red (Win +1, Lose -1) -> Mask*2 - 1

        int step = 0;
        step += is_red * 2 - 1;   // Strat 1
        step += is_black * 2 - 1; // Strat 2

        int num_even = (!is_zero & is_even);
        int num_odd = (!is_zero & !is_even);
        step += num_even * 2 - 1; // Strat 3
        step += num_odd * 2 - 1;  // Strat 4

        int dozen = (num >= 1) & (num <= 12);
        step += dozen * 3 - 1; // Strat 5 (Win +2, Bet 1)

        int is_17 = (num == 17);
        step += is_17 * 36 - 1; // Strat 6 (Win +35, Bet 1)

        // Loop 7-20
        for (int k = 7; k < NUM_STRATEGIES; k++) {
          int cond = (num % (k + 2) == 0);
          step += cond * 2 - 1;
        }

        local_balance += step;
      }
      balance_weyl += local_balance;
    }
    t_weyl = t.elapsed();
    cout << ">> W (Algebraic):     " << t_weyl << " s" << endl;
  }

  // ==============================================================
  // REPORT
  // ==============================================================
  double speedup = t_trad / t_weyl;

  cout << "\nResultados Finales:" << endl;
  cout << "-----------------------------------" << endl;
  cout << "Tiempo Tradicional: " << t_trad << "s" << endl;
  cout << "Tiempo W:     " << t_weyl << "s" << endl;
  cout << "SPEEDUP FACTOR:     " << fixed << setprecision(2) << speedup << "x"
       << endl;
  cout << "-----------------------------------" << endl;

  if (balance_trad != balance_weyl)
    cout << "WARN: CHECKSUM MISMATCH!" << endl;

  ofstream json("w_casino_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Quantum Casino (Montecarlo)\",\n";
  json << "  \"spins\": " << NUM_SPINS << ",\n";
  json << "  \"strategies\": " << NUM_STRATEGIES << ",\n";
  json << "  \"traditional_s\": " << t_trad << ",\n";
  json << "  \"w_s\": " << t_weyl << ",\n";
  json << "  \"speedup\": " << speedup << "\n";
  json << "}\n";

  return 0;
}
