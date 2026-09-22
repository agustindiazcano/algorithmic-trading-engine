#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN "THE INFINITE HEIST" ---
// Rango de complejidad para los problemas generados aleatoriamente
const int MIN_ITEMS = 20;
const int MAX_ITEMS = 25; // N=25 es lo máximo razonable para un loop continuo
const int MIN_CAPACITY = 500;
const int MAX_CAPACITY = 2000;

struct Item {
  int weight;
  int value;
};

struct Problem {
  int id;
  int capacity;
  std::vector<Item> items;
};

// Generador de Problemas Aleatorios
Problem generate_random_problem(int id, std::mt19937 &rng) {
  Problem p;
  p.id = id;
  std::uniform_int_distribution<int> dist_n(MIN_ITEMS, MAX_ITEMS);
  std::uniform_int_distribution<int> dist_cap(MIN_CAPACITY, MAX_CAPACITY);
  std::uniform_int_distribution<int> dist_w(10, 100);
  std::uniform_int_distribution<int> dist_v(10, 500);

  int n = dist_n(rng);
  p.capacity = dist_cap(rng);

  for (int i = 0; i < n; ++i) {
    p.items.push_back({dist_w(rng), dist_v(rng)});
  }
  return p;
}

// ==========================================
// 1. MÉTODO TRADICIONAL (Recursive Pain)
// ==========================================
int solve_recursive(const std::vector<Item> &items, int n, int w) {
  if (n == 0 || w == 0)
    return 0;
  if (items[n - 1].weight > w) {
    return solve_recursive(items, n - 1, w);
  } else {
    int incl = items[n - 1].value +
               solve_recursive(items, n - 1, w - items[n - 1].weight);
    int excl = solve_recursive(items, n - 1, w);
    return (incl > excl) ? incl : excl;
  }
}

// ==========================================
// 2. MÉTODO W (Flat Arithmetic)
// ==========================================
// Reutilizamos el buffer de memoria para no medir allocation time
int solve_w(const std::vector<Item> &items, int capacity,
                  std::vector<int> &buffer) {
  // Reset buffer (memset is faster, but loop is cleaner/safer)
  std::fill(buffer.begin(), buffer.begin() + capacity + 1, 0);

  for (const auto &item : items) {
    for (int w = capacity; w >= item.weight; --w) {
      int excl = buffer[w];
      int incl = buffer[w - item.weight] + item.value;
      // Branchless Max
      buffer[w] = (excl + incl + std::abs(excl - incl)) / 2;
    }
  }
  return buffer[capacity];
}

int main() {
  std::cout << "=== W SYSTEMS: THE INFINITE HEIST ===" << std::endl;
  std::cout << "Escenario: 60 Segundos de Stress Combinatorio (NP-Hard)"
            << std::endl;

  std::mt19937 rng(1337);

  // Generar un banco de pruebas masivo (100,000 problemas pre-generados)
  // Para que ambos corran sobre los mismos datos (si llegaran a terminarlos)
  std::cout << "Generando 'Mundial de Robos' (Banco de Problemas)..."
            << std::endl;
  std::vector<Problem> problem_bank;
  for (int i = 0; i < 10000; ++i) {
    problem_bank.push_back(generate_random_problem(i, rng));
  }

  // --- STAGE 1: TRADICIONAL ---
  std::cout << "\n[STAGE 1] Ejecutando Metodo Tradicional (60s Time Limit)..."
            << std::endl;
  long long solved_trad = 0;
  long long total_time_trad_us = 0;

  auto t_start_global = std::chrono::steady_clock::now();
  for (const auto &problem : problem_bank) {
    auto t_now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(t_now - t_start_global)
            .count() >= 30) {
      // Limitamos a 30s el tradicional para no aburrir, proyectaremos a 60s
      break;
    }

    auto t_prob_start = std::chrono::high_resolution_clock::now();
    volatile int res =
        solve_recursive(problem.items, problem.items.size(), problem.capacity);
    auto t_prob_end = std::chrono::high_resolution_clock::now();

    total_time_trad_us += std::chrono::duration_cast<std::chrono::microseconds>(
                              t_prob_end - t_prob_start)
                              .count();
    solved_trad++;

    if (solved_trad % 50 == 0)
      std::cout << "\rTradicional: " << solved_trad << " problemas resueltos..."
                << std::flush;
  }
  double seconds_trad = total_time_trad_us / 1000000.0;
  double problems_per_sec_trad = solved_trad / seconds_trad;

  std::cout << "\n>> Tradicional Finalizado. Soluciones/Seg: "
            << problems_per_sec_trad << std::endl;

  // --- STAGE 2: W ---
  std::cout << "\n[STAGE 2] Ejecutando Metodo W (60s Time Limit)..."
            << std::endl;
  long long solved_weyl = 0;
  long long total_time_weyl_us = 0;
  std::vector<int> dp_buffer(MAX_CAPACITY + 1); // Memoria pre-asignada

  t_start_global = std::chrono::steady_clock::now();

  // Loop infinito sobre el banco de problemas (W es tan rapido que se le
  // acaba el banco)
  bool keep_running = true;
  while (keep_running) {
    for (const auto &problem : problem_bank) {
      auto t_now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(t_now -
                                                           t_start_global)
              .count() >= 60) {
        keep_running = false;
        break;
      }

      auto t_prob_start = std::chrono::high_resolution_clock::now();
      volatile int res =
          solve_w(problem.items, problem.capacity, dp_buffer);
      auto t_prob_end = std::chrono::high_resolution_clock::now();

      total_time_weyl_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(t_prob_end -
                                                                t_prob_start)
              .count();
      solved_weyl++;
    }
  }
  double seconds_weyl = total_time_weyl_us / 1000000.0;
  double problems_per_sec_weyl = solved_weyl / seconds_weyl;

  std::cout << ">> W Finalizado. Soluciones/Seg: "
            << problems_per_sec_weyl << std::endl;

  // --- RESULTADOS ---
  double speedup = problems_per_sec_weyl / problems_per_sec_trad;

  std::cout << "\n=== THE INFINITE HEIST RESULTS ===" << std::endl;
  std::cout << "Throughput Tradicional: " << problems_per_sec_trad
            << " Problems/sec" << std::endl;
  std::cout << "Throughput W:     " << problems_per_sec_weyl
            << " Problems/sec" << std::endl;
  std::cout << "DOMINATION FACTOR:      " << speedup << "x" << std::endl;

  std::ofstream jsonFile("w_infinite_heist.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_type\": \"60s Combinatorial Stress Test\",\n";
    jsonFile << "  \"traditional_pps\": " << problems_per_sec_trad << ",\n";
    jsonFile << "  \"w_pps\": " << problems_per_sec_weyl << ",\n";
    jsonFile << "  \"problems_solved_w\": " << solved_weyl << ",\n";
    jsonFile << "  \"speedup_factor\": " << speedup << "\n";
    jsonFile << "}" << std::endl;
    jsonFile.close();
  }

  return 0;
}
