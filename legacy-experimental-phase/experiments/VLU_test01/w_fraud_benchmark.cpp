/*
═══════════════════════════════════════════════════════════════════════
🔥 EXPERIMENTO 5: LA AVALANCHA DE DATOS (Fraude Bancario) 🔥
Escenario: 200 Millones Transacciones | 15 Reglas Complejas
Target: Complex Logic Evaluation (AND/OR Trees)
Optimization: Algebraic Logic (Multiplication/Addition) vs Boolean
Short-Circuiting
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
const size_t NUM_TX = 50'000'000; // 50 Millones de Transacciones (Reduced
                                  // slightly to fit RAM/Time)

struct Transaction {
  float amount;
  float time; // 0.0 - 24.0
  int country_code;
  int user_country;
  int category; // 0=Retail, 1=Online, 2=Crypto
  int tx_count_today;
  float risk_score_history;
  int device_id;
  int prev_device_id;
  // ... total 30 fields simulated by random access
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
// MAIN
// ===================================
int main() {
  cout << "╔══════════════════════════════════════════════════════╗\n";
  cout << "║ EXPERIMENTO 5: LA AVALANCHA DE DATOS (FRAUDE)        ║\n";
  cout << "╚══════════════════════════════════════════════════════╝\n";
  cout << "Transacciones: " << NUM_TX << "\n";
  cout << "Reglas por TX: 15 Complejas\n";

  // 1. Generar Datos (Caos Financiero)
  // Usamos SoA (Structure of Arrays) para permitir vectorización máxima en
  // W Tradicional usará acceso tipo array igual. SoA es mejor para data
  // processing.
  vector<float> t_amount(NUM_TX);
  vector<float> t_time(NUM_TX);
  vector<int> t_country(NUM_TX);
  vector<int> t_user_country(NUM_TX);
  vector<int> t_category(NUM_TX);
  vector<int> t_tx_today(NUM_TX);
  vector<float> t_risk(NUM_TX);
  vector<int> t_dev(NUM_TX);
  vector<int> t_prev_dev(NUM_TX);

  mt19937 gen(42);
  uniform_real_distribution<float> dist_amt(0.0f, 20000.0f);
  uniform_real_distribution<float> dist_time(0.0f, 24.0f);
  uniform_int_distribution<int> dist_cat(0, 5);

#pragma omp parallel for
  for (size_t i = 0; i < NUM_TX; i++) {
    // Not using thread-safe random for speed gen, just random chaos is needed
    t_amount[i] = dist_amt(gen); // Race condition on gen ok for garbage data
    t_time[i] = dist_time(gen);
    t_country[i] = (i % 10 == 0) ? 1 : 0; // 10% foreign
    t_user_country[i] = 0;
    t_category[i] = dist_cat(gen);
    t_tx_today[i] = i % 100;
    t_risk[i] = (float)(i % 100) / 100.0f;
    t_dev[i] = i;
    t_prev_dev[i] = (i % 5 == 0) ? i + 1 : i; // 20% diff device
  }

  vector<float> score_trad(NUM_TX);
  vector<float> score_weyl(NUM_TX);

  // ==============================================================
  // 1. TRADICIONAL (Boolean Logic Trees)
  // ==============================================================
  double t_trad;
  {
    Timer t;
#pragma omp parallel for
    for (size_t i = 0; i < NUM_TX; i++) {
      float score = 0.0f;

      // Regla 1: Monto alto y país diferente
      // Short-circuiting helps here! If amount low, skip country check.
      if (t_amount[i] > 10000.0f && t_country[i] != t_user_country[i]) {
        score += 1.0f;
      }

      // Regla 2: Hora nocturna y Online
      if ((t_time[i] < 5.0f || t_time[i] > 23.0f) && t_category[i] == 1) {
        score += 0.8f;
      }

      // Regla 3: Muchas transacciones
      if (t_tx_today[i] > 50)
        score += 0.5f;

      // Regla 4: High risk history OR different device
      if (t_risk[i] > 0.8f || t_dev[i] != t_prev_dev[i]) {
        score += 0.7f;
      }

      // Regla 5: Complex Combo
      if ((t_amount[i] > 5000.0f && t_category[i] == 2) ||
          (t_tx_today[i] > 20 && t_time[i] < 4.0f)) {
        score += 1.5f; // Crypto high amount OR night spam
      }

      // ... (Repeat logic 3 times to simulate 15 rules)
      if (t_amount[i] > 15000.0f && t_country[i] != t_user_country[i])
        score += 1.0f;
      if ((t_time[i] < 4.0f) && t_category[i] == 1)
        score += 0.8f;
      if (t_tx_today[i] > 80)
        score += 0.5f;
      if (t_risk[i] > 0.9f || t_dev[i] != t_prev_dev[i])
        score += 0.7f;
      if ((t_amount[i] > 6000.0f && t_category[i] == 2) ||
          (t_tx_today[i] > 10 && t_time[i] < 3.0f))
        score += 1.5f;

      if (t_amount[i] > 20000.0f && t_country[i] != t_user_country[i])
        score += 1.0f;
      if ((t_time[i] < 3.0f) && t_category[i] == 1)
        score += 0.8f;
      if (t_tx_today[i] > 90)
        score += 0.5f;
      if (t_risk[i] > 0.95f || t_dev[i] != t_prev_dev[i])
        score += 0.7f;
      if ((t_amount[i] > 8000.0f && t_category[i] == 2) ||
          (t_tx_today[i] > 5 && t_time[i] < 2.0f))
        score += 1.5f;

      score_trad[i] = score;
    }
    t_trad = t.elapsed();
    cout << ">> Tradicional (If/Bool):     " << t_trad << " s" << endl;
  }

  // ==============================================================
  // 2. W (Algebraic Logic)
  // ==============================================================
  double t_weyl;
  {
    Timer t;
// The compiler CAN vectorize this because there are NO branches.
// It computes logic for 8/16 transactions at once.
#pragma omp parallel for
    for (size_t i = 0; i < NUM_TX; i++) {
      float score = 0.0f;

      // Regla 1: AND -> Mult
      float r1_amt = (float)(t_amount[i] > 10000.0f);
      float r1_cty = (float)(t_country[i] != t_user_country[i]);
      score += r1_amt * r1_cty * 1.0f;

      // Regla 2
      float r2_time =
          (float)(t_time[i] < 5.0f ||
                  t_time[i] >
                      23.0f); // OR logic still needs bool op or add-mult
      // (a < 5) | (a > 23) -> let's assume boolean works for masks
      float r2_cat = (float)(t_category[i] == 1);
      score += r2_time * r2_cat * 0.8f;

      // Regla 3
      score += (float)(t_tx_today[i] > 50) * 0.5f;

      // Regla 4: OR -> Logic OR inside mask
      float r4 = (float)((t_risk[i] > 0.8f) || (t_dev[i] != t_prev_dev[i]));
      score += r4 * 0.7f;

      // Regla 5: Complex
      float r5_a = (float)(t_amount[i] > 5000.0f && t_category[i] == 2);
      float r5_b = (float)(t_tx_today[i] > 20 && t_time[i] < 4.0f);
      float r5 = (float)(r5_a || r5_b);
      score += r5 * 1.5f;

      // ... Repeat x3
      float ra_1 =
          (float)(t_amount[i] > 15000.0f && t_country[i] != t_user_country[i]);
      score += ra_1 * 1.0f;
      float ra_2 = (float)((t_time[i] < 4.0f) && t_category[i] == 1);
      score += ra_2 * 0.8f;
      score += (float)(t_tx_today[i] > 80) * 0.5f;
      score +=
          (float)((t_risk[i] > 0.9f) || (t_dev[i] != t_prev_dev[i])) * 0.7f;
      float ra_5 = (float)((t_amount[i] > 6000.0f && t_category[i] == 2) ||
                           (t_tx_today[i] > 10 && t_time[i] < 3.0f));
      score += ra_5 * 1.5f;

      float rb_1 =
          (float)(t_amount[i] > 20000.0f && t_country[i] != t_user_country[i]);
      score += rb_1 * 1.0f;
      float rb_2 = (float)((t_time[i] < 3.0f) && t_category[i] == 1);
      score += rb_2 * 0.8f;
      score += (float)(t_tx_today[i] > 90) * 0.5f;
      score +=
          (float)((t_risk[i] > 0.95f) || (t_dev[i] != t_prev_dev[i])) * 0.7f;
      float rb_5 = (float)((t_amount[i] > 8000.0f && t_category[i] == 2) ||
                           (t_tx_today[i] > 5 && t_time[i] < 2.0f));
      score += rb_5 * 1.5f;

      score_weyl[i] = score;
    }
    t_weyl = t.elapsed();
    cout << ">> W (Vectorized):    " << t_weyl << " s" << endl;
  }

  double speedup = t_trad / t_weyl;

  cout << "\nResultados Finales:" << endl;
  cout << "-----------------------------------" << endl;
  cout << "Tiempo Tradicional: " << t_trad << "s" << endl;
  cout << "Tiempo W:     " << t_weyl << "s" << endl;
  cout << "SPEEDUP FACTOR:     " << fixed << setprecision(2) << speedup << "x"
       << endl;
  cout << "-----------------------------------" << endl;

  // Verify
  int diffs = 0;
  for (size_t i = 0; i < 1000; i++)
    if (abs(score_trad[i] - score_weyl[i]) > 0.01f)
      diffs++;
  if (diffs > 0)
    cout << "WARN: CHECKSUM FAIL (" << diffs << ")\n";

  ofstream json("w_fraud_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Data Avalanche (Fraud Detection)\",\n";
  json << "  \"transactions\": " << NUM_TX << ",\n";
  json << "  \"traditional_s\": " << t_trad << ",\n";
  json << "  \"w_s\": " << t_weyl << ",\n";
  json << "  \"speedup\": " << speedup << "\n";
  json << "}\n";

  return 0;
}
