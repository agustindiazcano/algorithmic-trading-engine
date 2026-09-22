/*
═══════════════════════════════════════════════════════════════════════
🔥 EXPERIMENTO 1: EL LABERINTO DEL CAOS (Pathfinding Denso) 🔥
Escenario: 1 Millón de Robots vs Obstáculos Densos
Contexto: Navegación Autónoma Masiva
Target: Comparar Branching Logic vs Algebraic Forces (SIMD)
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


// OpenMP Support
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

using namespace std;
using namespace chrono;

// ===================================
// CONFIGURACIÓN ESCENARIO
// ===================================
const int NUM_ROBOTS = 1'000'000;
// Simulamos que el sistema de partición espacial ya nos dio los
// 128 obstáculos más cercanos para procesar (Hot Loop).
const int OBSTACLES_PER_ROBOT = 128;

// Estructuras simples (SoA sería mejor para SIMD, pero AoS es más común en OOP)
struct Vec2 {
  float x, y;
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
// IMPLEMENTACIÓN
// ===================================
int main() {
  cout << "╔════════════════════════════════════════════════════╗\n";
  cout << "║ EXPERIMENTO 1: EL LABERINTO DEL CAOS (W)     ║\n";
  cout << "╚════════════════════════════════════════════════════╝\n";
  cout << "Robots: " << NUM_ROBOTS << "\n";
  cout << "Obstáculos locales por robot: " << OBSTACLES_PER_ROBOT << "\n";
  cout << "Total Interacciones: " << (long long)NUM_ROBOTS * OBSTACLES_PER_ROBOT
       << "\n";

  // 1. Setup Data
  // Posiciones aleatorias.
  // Usamos vector de obstáculos "cacheado" para simular que son los locales
  // En un caso real, esto vendría de un Grid/Quadtree.
  // Para el benchmark de CPU, lo que importa es el bucle de proceso.
  vector<Vec2> obstacles(OBSTACLES_PER_ROBOT);
  vector<Vec2> robots(NUM_ROBOTS);
  vector<float> turns_trad(NUM_ROBOTS);
  vector<float> turns_weyl(NUM_ROBOTS);

  mt19937 gen(42);
  uniform_real_distribution<float> pos_dist(-1000.0f, 1000.0f);

  for (auto &o : obstacles)
    o = {pos_dist(gen), pos_dist(gen)};
  for (auto &r : robots)
    r = {pos_dist(gen), pos_dist(gen)};

  // Radio de "Peligro Inminente" (threshold para el IF)
  const float SAFETY_RADIUS = 50.0f;
  const float SAFETY_RADIUS_SQ = SAFETY_RADIUS * SAFETY_RADIUS;

  // ==============================================================
  // 1. TRADICIONAL (Branching Logic)
  // ==============================================================
  // Lógica: Si está cerca -> Decidir dirección evasiva (If/Else)
  double t_trad;
  {
    Timer t;
// Parallel for para ser justos (W tambien usará hilos)
#pragma omp parallel for
    for (int i = 0; i < NUM_ROBOTS; i++) {
      float turn_decision = 0;
      Vec2 r = robots[i];

      for (int j = 0; j < OBSTACLES_PER_ROBOT; j++) {
        float dx = r.x - obstacles[j].x;
        float dy = r.y - obstacles[j].y;
        float d2 = dx * dx + dy * dy;

        // BRANCH PREDICTION KILLER:
        // El obstáculo está cerca? (~50% prob en este escenario denso)
        if (d2 < SAFETY_RADIUS_SQ) {
          // Lógica compleja de decisión (Simulada)
          // Si está a la izquierda, girar derecha.
          // Esto añade MÁS branches anidados.
          if (dx > 0) {
            turn_decision += 1.0f; // Turn Right
          } else {
            turn_decision -= 1.0f; // Turn Left
          }
        }
      }
      turns_trad[i] = turn_decision;
    }
    t_trad = t.elapsed();
    cout << ">> Tradicional (Branching): " << t_trad << " s" << endl;
  }

  // ==============================================================
  // 2. W (Algebraic Force Field)
  // ==============================================================
  // Lógica: Sumar fuerzas repulsivas.
  // force = 1 / (d2 + epsilon) * direction_sign
  // Sin IFs. Vectorizable AVX.
  double t_weyl;
  {
    Timer t;
#pragma omp parallel for
    for (int i = 0; i < NUM_ROBOTS; i++) {
      float force_x = 0;
      Vec2 r = robots[i];

      // Compiler Hint: This loop is safe for SIMD
      // Clang/GCC vectorize this automatically because logic is linear.
      for (int j = 0; j < OBSTACLES_PER_ROBOT; j++) {
        float dx = r.x - obstacles[j].x;
        float dy = r.y - obstacles[j].y;
        float d2 = dx * dx + dy * dy + 0.1f; // Epsilon to avoid div/0

        // Algebraic "Decision"
        // Force strength falls off with distance.
        // dx acts as the "sign" (left/right awareness implicit in vector)

        // Repulsion logic:
        // push = strength / distance^2
        // turn = push * (dx / dist) roughly maps to "turn away"

        // Optimized Math:
        float strength = 1000.0f / (d2 * d2); // Steep falloff (Soft wall)

        // Accumulate turn force directly
        force_x += dx * strength;
      }
      turns_weyl[i] = force_x;
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

  // JSON Export
  ofstream json("w_maze_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Chaos Maze (Dense Pathfinding)\",\n";
  json << "  \"robots\": " << NUM_ROBOTS << ",\n";
  json << "  \"obstacles_local\": " << OBSTACLES_PER_ROBOT << ",\n";
  json << "  \"traditional_s\": " << t_trad << ",\n";
  json << "  \"w_s\": " << t_weyl << ",\n";
  json << "  \"speedup\": " << speedup << "\n";
  json << "}\n";

  return 0;
}
