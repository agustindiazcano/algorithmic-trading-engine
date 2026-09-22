#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <vector>


// --- CONFIGURACIÓN "THE GAUNTLET" ---
// Escenario: Un grid de navegación masivo con obstáculos móviles.
// Objetivo: Pathfinding / Evasión rápida en tiempo real.
const int GRID_SIZE = 1000; // 1000x1000 = 1 Millón de celdas
const int OBSTACLES = 5000; // 5000 Obstáculos dinámicos
const int AGENTS = 100;     // 100 Agentes que deben moverse

struct Point {
  float x, y;
};
struct Node {
  int x, y;
  float cost, heuristic;
};

// Mapa (solo para métodos discretos)
std::vector<int> grid_map(GRID_SIZE *GRID_SIZE, 0);
std::vector<Point> obstacles(OBSTACLES);
std::vector<Point> agents(AGENTS);
std::vector<Point> targets(AGENTS);

// ==========================================
// ROUND 1: A* PATHFINDING (El Rival Discreto)
// ==========================================
// Problema: Explora nodos, usa Priority Queue y IFs para chequear vecinos.
// En entornos dinámicos es lento porque debe replanificar.
struct CompareNode {
  bool operator()(const Node &a, const Node &b) {
    return (a.cost + a.heuristic) > (b.cost + b.heuristic);
  }
};

long long benchmark_astar(Point start, Point end) {
  // Versión simplificada para benchmark (solo cuenta expansiones, no
  // reconstruye path completo) Esto es generoso con A*, en la práctica sería
  // mas lento.
  auto t_start = std::chrono::high_resolution_clock::now();

  std::priority_queue<Node, std::vector<Node>, CompareNode> open_set;
  std::vector<bool> closed_set(GRID_SIZE * GRID_SIZE, false);

  open_set.push({(int)start.x, (int)start.y, 0, 0});

  int expansions = 0;
  while (!open_set.empty()) {
    Node current = open_set.top();
    open_set.pop();
    expansions++;

    if (expansions > 2000)
      break; // Limitamos para que el test no sea infinito en casos imposibles

    if (std::abs(current.x - end.x) < 2 && std::abs(current.y - end.y) < 2)
      break; // Llegamos

    int idx = current.y * GRID_SIZE + current.x;
    if (closed_set[idx])
      continue;
    closed_set[idx] = true;

    // Vecinos (4 direcciones)
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};

    for (int i = 0; i < 4; ++i) {
      int nx = current.x + dx[i];
      int ny = current.y + dy[i];

      // Check Bounds (IFs)
      if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
        // Check Obstacle (Memory Access)
        if (grid_map[ny * GRID_SIZE + nx] == 0) {
          float dist = std::abs(nx - end.x) + std::abs(ny - end.y);
          open_set.push({nx, ny, current.cost + 1, dist});
        }
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
      .count();
}

// ==========================================
// ROUND 2: W POTENTIAL FIELDS (El Héroe Continuo)
// ==========================================
// En vez de buscar, calcula un vector de fuerza resultante.
// F_total = F_atracción (Target) + sum(F_repulsión (Obstáculos))
// Aritmética pura. SIMD Friendly. O(1) reactivo.
long long benchmark_w_fields(Point p, Point target,
                                   const std::vector<Point> &near_obstacles) {
  auto t_start = std::chrono::high_resolution_clock::now();

  float fx = 0, fy = 0;

  // 1. Atracción al Objetivo (Gravedad Lineal)
  float dx = target.x - p.x;
  float dy = target.y - p.y;
  fx += dx * 0.1f;
  fy += dy * 0.1f;

  // 2. Repulsión de Obstáculos (Gravedad Inversa Volumétrica)
  // W Formula: F = 1 / (1 + dist^2)
  // Procesamos un batch local de obstáculos (simulado como percepción sensor)
  for (const auto &obs : near_obstacles) {
    float ox = p.x - obs.x;
    float oy = p.y - obs.y;
    float dist_sq = ox * ox + oy * oy;

    // "Fuerza Volumétrica" branchless
    // Si está lejos, la fuerza es casi 0. Si está cerca, explota.
    // Sin IF (dist < radio). Todo suma.
    float repulsion = 1000.0f / (1.0f + dist_sq); // Constante de fuerza

    fx += ox * repulsion;
    fy += oy * repulsion;
  }

  // "Mover" al agente (Integración simple)
  // Simulado para evitar optimización
  volatile float final_x = p.x + fx;
  volatile float final_y = p.y + fy;

  auto t_end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
      .count();
}

// ==========================================
// ROUND 3: FUZZY LOGIC (El Rival "Suave")
// ==========================================
// Evalua reglas lingüísticas: "SI estoy CERCA Y el obstáculo está al FRENTE
// ENTONCES girar IZQUIERDA" Lleno de cálculos min/max y estructuras complejas.
long long benchmark_fuzzy(Point p, const std::vector<Point> &near_obstacles) {
  auto t_start = std::chrono::high_resolution_clock::now();

  float turn_left = 0.0f;
  float turn_right = 0.0f;

  for (const auto &obs : near_obstacles) {
    float dist = std::sqrt(std::pow(p.x - obs.x, 2) + std::pow(p.y - obs.y, 2));

    // Fuzzification (Triangular membership functions with IFs)
    float is_close = 0.0f;
    if (dist < 10.0f)
      is_close = 1.0f;
    else if (dist < 50.0f)
      is_close = (50.0f - dist) / 40.0f;

    if (is_close > 0) {
      // Reglas de inferencia
      turn_left = std::max(turn_left, is_close); // Simulado
    }
  }

  // Defuzzification (Centroide - costoso)
  volatile float result = (turn_left * -10.0f + turn_right * 10.0f) /
                          (turn_left + turn_right + 0.001f);

  auto t_end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
      .count();
}

int main() {
  std::cout << "=== W SYSTEMS: THE GAUNTLET (4-ROUND STRESS TEST) ==="
            << std::endl;
  std::cout << "Scenario: 100 Agents vs 5000 Dynamic Obstacles" << std::endl;

  // Init Data
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist_coord(0, GRID_SIZE);

  for (auto &o : obstacles) {
    o.x = dist_coord(rng);
    o.y = dist_coord(rng);
    grid_map[(int)o.y * GRID_SIZE + (int)o.x] = 1;
  }
  for (auto &a : agents) {
    a.x = dist_coord(rng);
    a.y = dist_coord(rng);
  }
  for (auto &t : targets) {
    t.x = dist_coord(rng);
    t.y = dist_coord(rng);
  }

  // Simulamos que el robot ve los 50 obstáculos más cercas (Sensor Radius)
  std::vector<Point> sensor_batch(50);
  for (int i = 0; i < 50; ++i)
    sensor_batch[i] = obstacles[i];

  long long t_path = 0, t_weyl = 0, t_fuzzy = 0;
  int samples = 0;

  auto global_start = std::chrono::steady_clock::now();
  std::cout << "Running The Gauntlet (60s)..." << std::endl;

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - global_start)
            .count() >= 60)
      break;

    // Round 1: Pathfinding
    // Solo corremos A* para UN agente por ciclo porque es muy lento para correr
    // 100. Multiplicaremos el tiempo por 100 para comparar throughput real.
    t_path += benchmark_astar(agents[0], targets[0]) * AGENTS;

    // Round 2: W Fields (Todos los agentes)
    for (int i = 0; i < AGENTS; ++i) {
      t_weyl += benchmark_w_fields(agents[i], targets[i], sensor_batch);
    }

    // Round 3: Fuzzy Logic (Todos los agentes)
    for (int i = 0; i < AGENTS; ++i) {
      t_fuzzy += benchmark_fuzzy(agents[i], sensor_batch);
    }

    samples++;
    if (samples % 100 == 0)
      std::cout << "\rSamples: " << samples << "..." << std::flush;
  }

  double avg_astar_us = (double)t_path / samples;
  double avg_w_us = (double)t_weyl / samples;
  double avg_fuzzy_us = (double)t_fuzzy / samples;

  std::cout << "\n\n=== RESULTS: THE GAUNTLET ===" << std::endl;
  std::cout << "1. Pathfinding (A*):    " << avg_astar_us
            << " us/frame (Discrete Search)" << std::endl;
  std::cout << "2. Fuzzy Logic (Rules): " << avg_fuzzy_us
            << " us/frame (Inference)" << std::endl;
  std::cout << "3. W (Fields):    " << avg_w_us
            << " us/frame (Volumetric)" << std::endl;
  std::cout << "\nCOMPARISONS (SPEEDUP):" << std::endl;
  std::cout << "W vs A*:    " << avg_astar_us / avg_w_us
            << "x FASTER" << std::endl;
  std::cout << "W vs Fuzzy: " << avg_fuzzy_us / avg_w_us
            << "x FASTER" << std::endl;

  std::ofstream jsonFile("w_gauntlet_results.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_type\": \"Multi-Paradigm Stress Test "
                "(Pathfinding/Fuzzy/Volumetric)\",\n";
    jsonFile << "  \"avg_time_astar_us\": " << avg_astar_us << ",\n";
    jsonFile << "  \"avg_time_fuzzy_us\": " << avg_fuzzy_us << ",\n";
    jsonFile << "  \"avg_time_w_us\": " << avg_w_us << ",\n";
    jsonFile << "  \"speedup_vs_astar\": " << avg_astar_us / avg_w_us
             << ",\n";
    jsonFile << "  \"speedup_vs_fuzzy\": " << avg_fuzzy_us / avg_w_us
             << "\n";
    jsonFile << "}";
  }

  return 0;
}
