#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>


// ==================================================================================
// W SYSTEMS: MASSIVE SCALE SIMULATION
// Scenario: "The Hive World"
// Scale: 40,000 Agents vs 1,000,000 Obstacles
// Optimization: Spatial Partitioning (Uniform Grid)
// ==================================================================================

const int GRID_WIDTH = 5000;
const int GRID_HEIGHT = 5000;
const int CELL_SIZE = 10; // 500x500 grid buckets
const int GRID_COLS = GRID_WIDTH / CELL_SIZE;
const int GRID_ROWS = GRID_HEIGHT / CELL_SIZE;

const int NUM_OBSTACLES = 1000000;
const int NUM_FOOD_SOURCES = 100;
const int AGENTS_PER_ALGO = 10000; // 10k agents per swarm
const int SIMULATION_STEPS =
    500; // Reduced steps for massive scale runtime sanity

struct Vec2 {
  float x, y;
};

struct Agent {
  Vec2 pos;
  Vec2 velocity;
  bool active;
  int id;
};

struct FoodSource {
  Vec2 pos;
  float amount;
};

// ============================================================
// SPATIAL PARTITIONING (THE ENGINE)
// ============================================================
class SpatialGrid {
public:
  // We store obstacle counts or indices. For 1M obstacles, we just need to know
  // "is there something here" and for W, "where is it exactly". Hybrid
  // approach:
  // 1. Boolean Grid for fast check (Ants/Bees)
  // 2. Buckets for exact position (W/Fuzzy)

  std::vector<std::vector<Vec2>> buckets;
  std::vector<bool> blocked_map;

  SpatialGrid() {
    buckets.resize(GRID_COLS * GRID_ROWS);
    blocked_map.resize(GRID_WIDTH * GRID_HEIGHT, false);
  }

  void clear_buckets() {
    for (auto &b : buckets)
      b.clear();
    std::fill(blocked_map.begin(), blocked_map.end(), false);
  }

  void add_obstacle(Vec2 p) {
    if (p.x < 0 || p.x >= GRID_WIDTH || p.y < 0 || p.y >= GRID_HEIGHT)
      return;

    // Exact spatial bucket
    int bx = (int)p.x / CELL_SIZE;
    int by = (int)p.y / CELL_SIZE;
    if (bx >= 0 && bx < GRID_COLS && by >= 0 && by < GRID_ROWS) {
      buckets[by * GRID_COLS + bx].push_back(p);
    }

    // Rasterize to block map (integer coordinates)
    int ix = (int)p.x;
    int iy = (int)p.y;
    blocked_map[iy * GRID_WIDTH + ix] = true;
    // Block 2x2 area to make it solid
    if (ix + 1 < GRID_WIDTH)
      blocked_map[iy * GRID_WIDTH + (ix + 1)] = true;
    if (iy + 1 < GRID_HEIGHT)
      blocked_map[(iy + 1) * GRID_WIDTH + ix] = true;
    if (ix + 1 < GRID_WIDTH && iy + 1 < GRID_HEIGHT)
      blocked_map[(iy + 1) * GRID_WIDTH + (ix + 1)] = true;
  }

  // Fast Neighborhood Query for W
  void get_neighbors(Vec2 p, float radius, std::vector<Vec2> &out) const {
    int cx = (int)p.x / CELL_SIZE;
    int cy = (int)p.y / CELL_SIZE;
    int rad_cells = (int)radius / CELL_SIZE + 1;

    for (int y = cy - rad_cells; y <= cy + rad_cells; y++) {
      for (int x = cx - rad_cells; x <= cx + rad_cells; x++) {
        if (x >= 0 && x < GRID_COLS && y >= 0 && y < GRID_ROWS) {
          const auto &bucket = buckets[y * GRID_COLS + x];
          for (const auto &obs : bucket) {
            float dx = p.x - obs.x;
            float dy = p.y - obs.y;
            if (dx * dx + dy * dy < radius * radius) {
              out.push_back(obs);
            }
          }
        }
      }
    }
  }

  bool is_blocked(int x, int y) const {
    if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT)
      return true;
    return blocked_map[y * GRID_WIDTH + x];
  }
};

// Global Environment
std::vector<Vec2> obstacles;
std::vector<FoodSource> foods;
SpatialGrid grid;
std::mt19937 rng(42);

// Utils
float fast_inv_sqrt(float number) {
  long i;
  float x2, y;
  const float threehalfs = 1.5F;
  x2 = number * 0.5F;
  y = number;
  i = *(long *)&y;
  i = 0x5f3759df - (i >> 1);
  y = *(float *)&i;
  y = y * (threehalfs - (x2 * y * y));
  return y;
}

// ============================================================
// 1. W ALGORITHM (Massive Volumetric Fields)
// ============================================================
class WSwarm {
public:
  std::vector<Agent> agents;
  int food_collected = 0;
  std::vector<Vec2> sensor_buffer; // Reusable buffer to avoid allocs

  void init() {
    std::uniform_real_distribution<float> dx(100, GRID_WIDTH - 100);
    std::uniform_real_distribution<float> dy(100, GRID_HEIGHT - 100);
    sensor_buffer.reserve(100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, true, i});
    }
  }

  void update() {
    for (auto &a : agents) {
      float fx = 0, fy = 0;

      // Attraction (Nearest Food)
      float min_d2 = 1e9;
      int best_f = -1;
      for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        float dx = foods[i].pos.x - a.pos.x;
        float dy = foods[i].pos.y - a.pos.y;
        float d2 = dx * dx + dy * dy;
        if (d2 < min_d2) {
          min_d2 = d2;
          best_f = i;
        }
      }

      if (best_f != -1) {
        Vec2 target = foods[best_f].pos;
        float dx = target.x - a.pos.x;
        float dy = target.y - a.pos.y;
        float inv_d = fast_inv_sqrt(min_d2 + 0.01f);
        fx += dx * inv_d * 2.0f;
        fy += dy * inv_d * 2.0f;
      }

      // Repulsion (Local Grid Query)
      sensor_buffer.clear();
      grid.get_neighbors(a.pos, 20.0f,
                         sensor_buffer); // 20.0f perception radius

      for (const auto &obs : sensor_buffer) {
        float dx = a.pos.x - obs.x;
        float dy = a.pos.y - obs.y;
        float d2 = dx * dx + dy * dy;
        // W Force: Explosive close range repulsion
        float force = 1000.0f / (1.0f + d2);
        fx += dx * force;
        fy += dy * force;
      }

      // Integration
      a.velocity.x = (a.velocity.x + fx) * 0.6f;
      a.velocity.y = (a.velocity.y + fy) * 0.6f;

      // Speed Limit
      float v2 = a.velocity.x * a.velocity.x + a.velocity.y * a.velocity.y;
      if (v2 > 25.0f) { // Max speed 5
        float iv = fast_inv_sqrt(v2);
        a.velocity.x *= (5.0f * iv * v2);
        a.velocity.y *= (5.0f * iv * v2);
      }

      // Move & Collide (Soft)
      float nx = a.pos.x + a.velocity.x;
      float ny = a.pos.y + a.velocity.y;

      // Simple bound & block check
      if (!grid.is_blocked((int)nx, (int)ny)) {
        a.pos.x = nx;
        a.pos.y = ny;
      } else {
        a.velocity.x *= -0.5f; // Bounce
        a.velocity.y *= -0.5f;
      }

      // Eat
      if (best_f != -1 && min_d2 < 100.0f) {
        food_collected++;
      }
    }
  }
};

// ============================================================
// 2. ANT COLONY (Grid Walkers)
// ============================================================
class AntColony {
  std::vector<uint8_t> pheromones; // Compact pheromone grid (0-255)
  int p_width, p_height;
  int scale = 10;

public:
  std::vector<Agent> agents;
  int food_collected = 0;

  AntColony() : p_width(GRID_WIDTH / 10), p_height(GRID_HEIGHT / 10) {
    pheromones.resize(p_width * p_height, 0);
  }

  void init() {
    std::uniform_real_distribution<float> dx(100, GRID_WIDTH - 100);
    std::uniform_real_distribution<float> dy(100, GRID_HEIGHT - 100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, true, i});
    }
  }

  void update() {
    // Evaporation (Random decay for speed instead of full sweep)
    int evaps = pheromones.size() / 100; // 1% random decay per frame
    std::uniform_int_distribution<int> dist_idx(0, pheromones.size() - 1);
    for (int i = 0; i < evaps; i++) {
      int idx = dist_idx(rng);
      if (pheromones[idx] > 0)
        pheromones[idx]--;
    }

    std::uniform_real_distribution<float> rand_dir(-1.0f, 1.0f);

    for (auto &a : agents) {
      int gx = (int)a.pos.x / scale;
      int gy = (int)a.pos.y / scale;

      // Sensor: Smell strongest neighbor
      int best_dx = 0, best_dy = 0;
      int max_ph = 0;

      for (int ny = -1; ny <= 1; ny++) {
        for (int nx = -1; nx <= 1; nx++) {
          if (nx == 0 && ny == 0)
            continue;
          int cx = gx + nx;
          int cy = gy + ny;
          if (cx >= 0 && cx < p_width && cy >= 0 && cy < p_height) {
            if (pheromones[cy * p_width + cx] > max_ph) {
              max_ph = pheromones[cy * p_width + cx];
              best_dx = nx;
              best_dy = ny;
            }
          }
        }
      }

      // Move
      float dx = (float)best_dx + rand_dir(rng);
      float dy = (float)best_dy + rand_dir(rng);

      // Normalize
      float d2 = dx * dx + dy * dy;
      if (d2 > 0.1f) {
        float inv = fast_inv_sqrt(d2);
        dx *= inv * 3.0f; // Speed 3
        dy *= inv * 3.0f;
      }

      float nx = a.pos.x + dx;
      float ny = a.pos.y + dy;

      if (!grid.is_blocked((int)nx, (int)ny)) {
        a.pos.x = nx;
        a.pos.y = ny;
      } else {
        // Bounce random
        a.pos.x += rand_dir(rng) * 5.0f;
        a.pos.y += rand_dir(rng) * 5.0f;
      }

      // Check food & Mark
      for (const auto &f : foods) {
        float dist = (a.pos.x - f.pos.x) * (a.pos.x - f.pos.x) +
                     (a.pos.y - f.pos.y) * (a.pos.y - f.pos.y);
        if (dist < 100.0f) {
          food_collected++;
          int p_idx = gy * p_width + gx;
          if (p_idx >= 0 && p_idx < pheromones.size()) {
            if (pheromones[p_idx] < 200)
              pheromones[p_idx] += 50; // Deposit
          }
        }
      }
    }
  }
};

// ============================================================
// 3. FUZZY LOGIC (Rule Based Navigation)
// ============================================================
class FuzzySwarm {
public:
  std::vector<Agent> agents;
  int food_collected = 0;
  std::vector<Vec2> sensor;

  void init() {
    std::uniform_real_distribution<float> dx(100, GRID_WIDTH - 100);
    std::uniform_real_distribution<float> dy(100, GRID_HEIGHT - 100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, true, i});
    }
  }

  void update() {
    for (auto &a : agents) {
      // 1. Fuzzify: Get inputs
      // Distance to nearest obstacle
      sensor.clear();
      grid.get_neighbors(a.pos, 30.0f, sensor); // Wider visual range

      float min_obs_d = 9999.0f;
      Vec2 obs_pos = {0, 0};
      if (!sensor.empty()) {
        for (auto &o : sensor) {
          float d = (a.pos.x - o.x) * (a.pos.x - o.x) +
                    (a.pos.y - o.y) * (a.pos.y - o.y);
          if (d < min_obs_d) {
            min_obs_d = d;
            obs_pos = o;
          }
        }
      }
      float obs_dist = std::sqrt(
          min_obs_d); // Expensive sqrt, but "Fuzzy" needs crisp values

      // Distance to Food
      float min_food_d = 1e9;
      Vec2 food_pos = {0, 0};
      for (auto &f : foods) {
        float d = (a.pos.x - f.pos.x) * (a.pos.x - f.pos.x) +
                  (a.pos.y - f.pos.y) * (a.pos.y - f.pos.y);
        if (d < min_food_d) {
          min_food_d = d;
          food_pos = f.pos;
        }
      }

      // Rules
      // IF Obstacle < 10 THEN HARD TURN
      // IF Obstacle < 30 THEN SOFT TURN
      // IF Clear THEN SEEK FOOD

      float turn_x = 0, turn_y = 0;
      float speed = 0;

      if (obs_dist < 10.0f) {
        // PANIC
        turn_x = (a.pos.x - obs_pos.x);
        turn_y = (a.pos.y - obs_pos.y);
        speed = 1.0f;
      } else if (obs_dist < 30.0f) {
        // AVOID
        turn_x = (a.pos.x - obs_pos.x) + (food_pos.x - a.pos.x) * 0.5f;
        turn_y = (a.pos.y - obs_pos.y) + (food_pos.y - a.pos.y) * 0.5f;
        speed = 2.0f;
      } else {
        // SEEK
        turn_x = (food_pos.x - a.pos.x);
        turn_y = (food_pos.y - a.pos.y);
        speed = 4.0f;
      }

      // Normalize direction
      float len = std::sqrt(turn_x * turn_x + turn_y * turn_y + 0.001f);
      a.velocity.x = (turn_x / len) * speed;
      a.velocity.y = (turn_y / len) * speed;

      float nx = a.pos.x + a.velocity.x;
      float ny = a.pos.y + a.velocity.y;

      // Check collisions
      if (!grid.is_blocked((int)nx, (int)ny)) {
        a.pos.x = nx;
        a.pos.y = ny;
      }

      if (min_food_d < 100.0f)
        food_collected++;
    }
  }
};

// ============================================================
// 4. BEES (Recruitment)
// ============================================================
class BeeSwarm {
public:
  std::vector<Agent> agents;
  int food_collected = 0;

  void init() {
    std::uniform_real_distribution<float> dx(100, GRID_WIDTH - 100);
    std::uniform_real_distribution<float> dy(100, GRID_HEIGHT - 100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, true, i});
    }
  }

  void update() {
    // Bees are simple: Browninan motion until food, then stay and signal
    // Simplified for massive scale

    std::uniform_real_distribution<float> jitter(-2.0f, 2.0f);
    Vec2 center_of_activity = {GRID_WIDTH / 2, GRID_HEIGHT / 2};
    bool activity_found = false;

    // 1. Scan results from LAST frame (simplified)
    // In this implementation, bees that found food instantly "teleport"
    // neighbors or attract them.

    for (auto &a : agents) {
      // Random Walk
      a.velocity.x += jitter(rng);
      a.velocity.y += jitter(rng);

      // Dampen
      a.velocity.x *= 0.9f;
      a.velocity.y *= 0.9f;

      float nx = a.pos.x + a.velocity.x;
      float ny = a.pos.y + a.velocity.y;

      if (!grid.is_blocked((int)nx, (int)ny)) {
        a.pos.x = nx;
        a.pos.y = ny;
      } else {
        a.velocity.x *= -1;
        a.velocity.y *= -1;
      }

      // Check food
      float min_d = 1e9;
      for (const auto &f : foods) {
        float d = (a.pos.x - f.pos.x) * (a.pos.x - f.pos.x) +
                  (a.pos.y - f.pos.y) * (a.pos.y - f.pos.y);
        if (d < min_d)
          min_d = d;
      }
      if (min_d < 100.0f) {
        food_collected++;
        // "Dance" / Signal -> Set global attractor (Cheating but efficient for
        // bees)
        center_of_activity = a.pos;
        activity_found = true;
      }
    }

    // Global recruitment (Massive swarm behavior)
    if (activity_found) {
      // Pull 1% of bees towards the activity center
      for (int i = 0; i < agents.size(); i += 100) {
        float dx = center_of_activity.x - agents[i].pos.x;
        float dy = center_of_activity.y - agents[i].pos.y;
        float mag = std::sqrt(dx * dx + dy * dy + 0.01f);
        agents[i].velocity.x += (dx / mag) * 5.0f; // Zoom to food
        agents[i].velocity.y += (dy / mag) * 5.0f;
      }
    }
  }
};

int main() {
  std::cout << "=== W SYSTEMS: MASSIVE SCALE SIMULATION ===" << std::endl;
  std::cout << "Agents: " << AGENTS_PER_ALGO * 4 << " (4 Swarms)" << std::endl;
  std::cout << "Obstacles: " << NUM_OBSTACLES << " (Spatial Grid)" << std::endl;
  std::cout << "Map Size: " << GRID_WIDTH << "x" << GRID_HEIGHT << std::endl;
  std::cout << "Init World..." << std::endl;

  // Init Grid & Obstacles
  grid.clear_buckets();
  std::uniform_real_distribution<float> coord_x(0, GRID_WIDTH);
  std::uniform_real_distribution<float> coord_y(0, GRID_HEIGHT);
  for (int i = 0; i < NUM_OBSTACLES; i++) {
    // Create clumps of obstacles for labyrinth effect
    float cx = coord_x(rng);
    float cy = coord_y(rng);
    for (int j = 0; j < 5; j++) {
      Vec2 p = {cx + (float)(j * 10), cy + (float)(j * 5)};
      obstacles.push_back(p);
      grid.add_obstacle(p);
    }
    i += 4; // Skip count
  }

  // Init Food
  for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
    foods.push_back({{coord_x(rng), coord_y(rng)}, 1000});
  }

  // Init Swarms
  WSwarm w;
  w.init();
  AntColony ants;
  ants.init();
  FuzzySwarm fuzzy;
  fuzzy.init();
  BeeSwarm bees;
  bees.init();

  std::cout << "Starting Simulation Loop..." << std::endl;

  long long t_w = 0, t_a = 0, t_f = 0, t_b = 0;

  for (int step = 0; step < SIMULATION_STEPS; step++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    w.update();
    auto t2 = std::chrono::high_resolution_clock::now();
    t_w +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    ants.update();
    t2 = std::chrono::high_resolution_clock::now();
    t_a +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    fuzzy.update();
    t2 = std::chrono::high_resolution_clock::now();
    t_f +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    bees.update();
    t2 = std::chrono::high_resolution_clock::now();
    t_b +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (step % 10 == 0)
      std::cout << "Step: " << step << " / " << SIMULATION_STEPS << "\r"
                << std::flush;
  }

  std::cout << "\nFinished." << std::endl;

  // JSON
  std::ofstream json("w_massive_results.json");
  json << "{\n";
  json << "  \"test\": \"Massive Scale (1M Obstacles, 40k Agents)\",\n";
  json << "  \"results\": {\n";
  json << "    \"w\": { \"time_ms\": " << t_w / 1000
       << ", \"food\": " << w.food_collected << " },\n";
  json << "    \"ants\": { \"time_ms\": " << t_a / 1000
       << ", \"food\": " << ants.food_collected << " },\n";
  json << "    \"fuzzy\": { \"time_ms\": " << t_f / 1000
       << ", \"food\": " << fuzzy.food_collected << " },\n";
  json << "    \"bees\": { \"time_ms\": " << t_b / 1000
       << ", \"food\": " << bees.food_collected << " }\n";
  json << "  }\n";
  json << "}";

  // Print Console Report
  std::cout << "RESULTS:\n";
  std::cout << "W: " << t_w / 1000
            << "ms || Score: " << w.food_collected << "\n";
  std::cout << "Ants:    " << t_a / 1000
            << "ms || Score: " << ants.food_collected << "\n";
  std::cout << "Fuzzy:   " << t_f / 1000
            << "ms || Score: " << fuzzy.food_collected << "\n";
  std::cout << "Bees:    " << t_b / 1000
            << "ms || Score: " << bees.food_collected << "\n";

  return 0;
}
