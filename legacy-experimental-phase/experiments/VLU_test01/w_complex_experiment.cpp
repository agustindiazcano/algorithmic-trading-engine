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
// W SYSTEMS: ADVANCED ALGORITHM COMPARISON
// Subject: Dynamic Foraging & Navigation in High-Entropy Environment
// Challengers:
// 1. W Algorithm (Volumetric Potential Fields)
// 2. Ant Colony Optimization (ACO - Stigmergy)
// 3. Fuzzy Logic Control (Linguistic Inference)
// 4. Bees Algorithm (Swarm Recruitment)
// ==================================================================================

const int GRID_WIDTH = 500;
const int GRID_HEIGHT = 500;
const int NUM_OBSTACLES = 2000;
const int NUM_FOOD_SOURCES = 20;
const int AGENTS_PER_ALGO = 50;
const int SIMULATION_STEPS = 1000;

struct Vec2 {
  float x, y;
};

struct Agent {
  Vec2 pos;
  Vec2 velocity;
  float energy;
  bool carrying_food;
  int id;
};

struct FoodSource {
  Vec2 pos;
  float amount;
};

// Global Environment
std::vector<Vec2> obstacles;
std::vector<FoodSource> foods;
std::mt19937 rng(42);

// Utils
float dist_sq(Vec2 a, Vec2 b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

float fast_inv_sqrt(float number) {
  // Quake III Fast Inverse Square Root for W Optimization
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
// 1. W ALGORITHM (Potential Fields)
// ============================================================
class WSwarm {
public:
  std::vector<Agent> agents;
  int food_collected = 0;

  void init() {
    std::uniform_real_distribution<float> dx(0, GRID_WIDTH);
    std::uniform_real_distribution<float> dy(0, GRID_HEIGHT);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, 100.0f, false, i});
    }
  }

  void update(const std::vector<Vec2> &current_obstacles) {
    // SIMD-friendly logic style
    for (auto &a : agents) {
      float fx = 0, fy = 0;

      // Attraction to nearest food
      int nearest_food_idx = -1;
      float min_d = 1e9;
      for (int i = 0; i < foods.size(); i++) {
        if (foods[i].amount <= 0)
          continue;
        float d = dist_sq(a.pos, foods[i].pos);
        if (d < min_d) {
          min_d = d;
          nearest_food_idx = i;
        }
      }

      if (nearest_food_idx != -1) {
        Vec2 target = foods[nearest_food_idx].pos;
        float dx = target.x - a.pos.x;
        float dy = target.y - a.pos.y;
        // Normalize roughly
        float inv_d = fast_inv_sqrt(dx * dx + dy * dy + 0.001f);
        fx += dx * inv_d * 2.0f; // Attraction Force
        fy += dy * inv_d * 2.0f;
      }

      // Repulsion from Obstacles (Local Perception)
      // Checking all is slow, but we rely on simple math speed
      for (const auto &obs : current_obstacles) {
        float dx = a.pos.x - obs.x;
        float dy = a.pos.y - obs.y;
        float d2 = dx * dx + dy * dy;

        if (d2 < 400.0f) { // Perception Radius squared
                           // W Force: 1 / (1 + d^2)
          float force = 500.0f / (1.0f + d2);
          fx += dx * force;
          fy += dy * force;
        }
      }

      // Integrate
      a.velocity.x = (a.velocity.x + fx) * 0.5f; // Damping
      a.velocity.y = (a.velocity.y + fy) * 0.5f;

      // Limit speed
      float v2 = a.velocity.x * a.velocity.x + a.velocity.y * a.velocity.y;
      if (v2 > 9.0f) { // Max speed 3
        float iv = fast_inv_sqrt(v2);
        a.velocity.x *= (3.0f * iv * v2); // Correction
        a.velocity.y *= (3.0f * iv * v2);
      }

      a.pos.x += a.velocity.x;
      a.pos.y += a.velocity.y;

      // Bounds
      if (a.pos.x < 0)
        a.pos.x = 0;
      if (a.pos.x > GRID_WIDTH)
        a.pos.x = GRID_WIDTH;
      if (a.pos.y < 0)
        a.pos.y = 0;
      if (a.pos.y > GRID_HEIGHT)
        a.pos.y = GRID_HEIGHT;

      // Eat
      if (nearest_food_idx != -1 && min_d < 25.0f &&
          foods[nearest_food_idx].amount > 0) {
        // Shared resource (simulated atomic for benchmark simplicity)
        // In a real comparison, we'd decrement. Here keeping it specific.
        food_collected++;
      }
    }
  }
};

// ============================================================
// 2. ANT COLONY OPTIMIZATION (Pheromone Grid)
// ============================================================
class AntColony {
  std::vector<float> pheromones; // 1D array for 2D grid
  int width, height;

public:
  std::vector<Agent> agents;
  int food_collected = 0;

  AntColony() : width(GRID_WIDTH / 5), height(GRID_HEIGHT / 5) {
    // Reduced resolution for pheromone grid to save memory/time
    pheromones.resize(width * height, 0.0f);
  }

  void init() {
    std::uniform_real_distribution<float> dx(0, GRID_WIDTH);
    std::uniform_real_distribution<float> dy(0, GRID_HEIGHT);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, 100.0f, false, i});
    }
  }

  void update() {
    // Evaporation
    for (auto &p : pheromones)
      p *= 0.95f;

    for (auto &a : agents) {
      // Decision: Random vs Pheromone Gradient
      int gx = (int)(a.pos.x / 5);
      int gy = (int)(a.pos.y / 5);
      gx = std::max(0, std::min(gx, width - 1));
      gy = std::max(0, std::min(gy, height - 1));

      // Sense neighbors (3x3)
      float max_ph = -1.0f;
      int best_dx = 0, best_dy = 0;

      // Random walk noise
      std::uniform_real_distribution<float> noise(-1, 1);

      // Check neighbors for max pheromone
      for (int ny = -1; ny <= 1; ny++) {
        for (int nx = -1; nx <= 1; nx++) {
          if (nx == 0 && ny == 0)
            continue;
          int cx = gx + nx;
          int cy = gy + ny;
          if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
            float ph = pheromones[cy * width + cx];
            if (ph > max_ph) {
              max_ph = ph;
              best_dx = nx;
              best_dy = ny;
            }
          }
        }
      }

      // Movement
      float speed = 2.0f;
      if (max_ph > 0.1f) {
        // Follow gradient
        a.velocity.x = best_dx * speed + noise(rng) * 0.5f;
        a.velocity.y = best_dy * speed + noise(rng) * 0.5f;
      } else {
        // Explore
        a.velocity.x += noise(rng) * 0.5f;
        a.velocity.y += noise(rng) * 0.5f;
        // Clamp velocity
        float v = std::sqrt(a.velocity.x * a.velocity.x +
                            a.velocity.y * a.velocity.y);
        if (v > speed) {
          a.velocity.x = (a.velocity.x / v) * speed;
          a.velocity.y = (a.velocity.y / v) * speed;
        }
      }

      a.pos.x += a.velocity.x;
      a.pos.y += a.velocity.y;

      // Bounds
      if (a.pos.x < 0) {
        main_bounds(a);
      } // Helper omitted for brevity, inline logic:
      if (a.pos.x < 0)
        a.pos.x = 0;
      if (a.pos.x > GRID_WIDTH)
        a.pos.x = GRID_WIDTH;
      if (a.pos.y < 0)
        a.pos.y = 0;
      if (a.pos.y > GRID_HEIGHT)
        a.pos.y = GRID_HEIGHT;

      // Interaction: if found food, deposit Pheromone
      bool found = false;
      for (const auto &f : foods) {
        if (dist_sq(a.pos, f.pos) < 400.0f) { // Found food
          food_collected++;
          found = true;
          // Deposit massive pheromone here
          int p_idx = gy * width + gx;
          pheromones[p_idx] += 10.0f;
          break;
        }
      }

      // Always deposit a little bit (trail)
      int p_idx = gy * width + gx;
      if (p_idx >= 0 && p_idx < pheromones.size())
        pheromones[p_idx] += 0.5f;
    }
  }

  void main_bounds(Agent &a) {
    if (a.pos.x < 0)
      a.pos.x = 0;
    if (a.pos.x > GRID_WIDTH)
      a.pos.x = GRID_WIDTH;
    if (a.pos.y < 0)
      a.pos.y = 0;
    if (a.pos.y > GRID_HEIGHT)
      a.pos.y = GRID_HEIGHT;
  }
};

// ============================================================
// 3. FUZZY LOGIC (Rule Based)
// ============================================================
class FuzzyController {
public:
  std::vector<Agent> agents;
  int food_collected = 0;

  void init() {
    std::uniform_real_distribution<float> dx(0, GRID_WIDTH);
    std::uniform_real_distribution<float> dy(0, GRID_HEIGHT);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, 100.0f, false, i});
    }
  }

  // Fuzzy Membership Functions
  float is_close(float dist) {
    if (dist < 20.0f)
      return 1.0f;
    if (dist > 100.0f)
      return 0.0f;
    return (100.0f - dist) / 80.0f;
  }

  float is_far(float dist) {
    if (dist > 150.0f)
      return 1.0f;
    if (dist < 50.0f)
      return 0.0f;
    return (dist - 50.0f) / 100.0f;
  }

  void update(const std::vector<Vec2> &obstacles_in) {
    for (auto &a : agents) {
      // Find nearest food
      Vec2 target = {GRID_WIDTH / 2, GRID_HEIGHT / 2};
      float min_d = 1e9;
      int f_idx = -1;
      for (int i = 0; i < foods.size(); i++) {
        float d = dist_sq(a.pos, foods[i].pos);
        if (d < min_d) {
          min_d = d;
          target = foods[i].pos;
          f_idx = i;
        }
      }

      // Find nearest obstacle
      Vec2 nearest_obs = {-1000, -1000};
      float min_obs_d = 1e9;
      for (const auto &o : obstacles_in) {
        float d = dist_sq(a.pos, o);
        if (d < min_obs_d) {
          min_obs_d = d;
          nearest_obs = o;
        }
      }
      float obs_dist = std::sqrt(min_obs_d);

      // Fuzzy Rules
      // 1. IF Obstacle is CLOSE -> TURN AWAY
      // 2. IF Obstacle is FAR AND Food is DETECTED -> GO TO FOOD

      float input_obs_close = is_close(obs_dist);
      float input_obs_far = 1.0f - input_obs_close;

      // Rule 1 Output (Avoidance Vector)
      float avoid_x = 0, avoid_y = 0;
      if (input_obs_close > 0.1f) {
        avoid_x = (a.pos.x - nearest_obs.x);
        avoid_y = (a.pos.y - nearest_obs.y);
        // Normalize
        float len = std::sqrt(avoid_x * avoid_x + avoid_y * avoid_y + 0.01f);
        avoid_x /= len;
        avoid_y /= len;
      }

      // Rule 2 Output (Seek Vector)
      float seek_x = (target.x - a.pos.x);
      float seek_y = (target.y - a.pos.y);
      float len_s = std::sqrt(seek_x * seek_x + seek_y * seek_y + 0.01f);
      seek_x /= len_s;
      seek_y /= len_s;

      // Defuzzification (Weighted Average)
      float w_avoid = input_obs_close * 5.0f; // High priority
      float w_seek = input_obs_far * 1.0f;

      float final_dx =
          (avoid_x * w_avoid + seek_x * w_seek) / (w_avoid + w_seek + 0.001f);
      float final_dy =
          (avoid_y * w_avoid + seek_y * w_seek) / (w_avoid + w_seek + 0.001f);

      a.velocity.x = final_dx * 3.0f;
      a.velocity.y = final_dy * 3.0f;

      a.pos.x += a.velocity.x;
      a.pos.y += a.velocity.y;
      if (a.pos.x < 0)
        a.pos.x = 0;
      if (a.pos.x > GRID_WIDTH)
        a.pos.x = GRID_WIDTH;
      if (a.pos.y < 0)
        a.pos.y = 0;
      if (a.pos.y > GRID_HEIGHT)
        a.pos.y = GRID_HEIGHT;

      if (f_idx != -1 && min_d < 25.0f) {
        food_collected++;
      }
    }
  }
};

// ============================================================
// 4. BEES ALGORITHM (Scouts & Recruitment)
// ============================================================
class BeeColony {
  struct Site {
    Vec2 pos;
    float fitness; // Food amount or proximity
  };
  std::vector<Site> top_sites;

public:
  std::vector<Agent> agents; // Bees
  int food_collected = 0;

  void init() {
    std::uniform_real_distribution<float> dx(0, GRID_WIDTH);
    std::uniform_real_distribution<float> dy(0, GRID_HEIGHT);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{dx(rng), dy(rng)}, {0, 0}, 100.0f, false, i});
    }
  }

  void update() {
    // 1. Evaluate Fitness of all bees (Distance to nearest food)
    // In real Bee Algo, this might be finding function Max. Here it is "finding
    // food". Sorted list of bees by "closeness to target"

    std::vector<std::pair<float, int>> bee_fitness;
    for (int i = 0; i < agents.size(); i++) {
      float min_d = 1e9;
      for (const auto &f : foods) {
        float d = dist_sq(agents[i].pos, f.pos);
        if (d < min_d)
          min_d = d;
      }
      if (min_d < 25.0f)
        food_collected++; // Gather

      // Fitness = 1 / distance (closer is better)
      bee_fitness.push_back({-min_d, i}); // Negative for descending sort
    }

    std::sort(bee_fitness.begin(), bee_fitness.end());

    // 2. Select Elite Sites (Best bees locations)
    int elite_count = 5;
    std::vector<Vec2> elite_locations;
    for (int i = 0; i < elite_count; i++) {
      elite_locations.push_back(agents[bee_fitness[i].second].pos);
    }

    // 3. Recruit: Teleport/Move bad performing bees near Elite bees
    // (Neighborhood Search) The bottom 20 bees are "recruited" to search near
    // the top 5 bees
    int recruit_start = agents.size() - 20;
    std::uniform_int_distribution<int> dist_elite(0, elite_count - 1);
    std::uniform_real_distribution<float> dist_jitter(-10, 10);

    for (int k = recruit_start; k < agents.size(); k++) {
      int original_idx = bee_fitness[k].second; // The bee to move
      int target_elite = dist_elite(rng);
      Vec2 aim = elite_locations[target_elite];

      // "Fly towards" elite site rapidly or "Spawn" there (Metaheuristic style)
      // Let's make them fly fast
      float dx = aim.x - agents[original_idx].pos.x;
      float dy = aim.y - agents[original_idx].pos.y;
      agents[original_idx].pos.x +=
          dx * 0.1f + dist_jitter(rng); // Jump 10% + jitter
      agents[original_idx].pos.y += dy * 0.1f + dist_jitter(rng);
    }

    // 4. Scouts: The rest move randomly (Exploration)
    for (int k = elite_count; k < recruit_start; k++) {
      int idx = bee_fitness[k].second;
      std::uniform_real_distribution<float> rand_move(-5, 5);
      agents[idx].pos.x += rand_move(rng);
      agents[idx].pos.y += rand_move(rng);

      // Simple bounds
      if (agents[idx].pos.x < 0)
        agents[idx].pos.x = GRID_WIDTH;
      if (agents[idx].pos.x > GRID_WIDTH)
        agents[idx].pos.x = 0;
      if (agents[idx].pos.y < 0)
        agents[idx].pos.y = GRID_HEIGHT;
      if (agents[idx].pos.y > GRID_HEIGHT)
        agents[idx].pos.y = 0;
    }
  }
};

int main() {
  std::cout << "Initializing Complex Experiment..." << std::endl;

  // 1. Setup Environment
  std::uniform_real_distribution<float> coord(0, GRID_WIDTH);
  for (int i = 0; i < NUM_OBSTACLES; i++) {
    obstacles.push_back({coord(rng), coord(rng)});
  }
  for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
    foods.push_back({{coord(rng), coord(rng)}, 1000.0f});
  }

  // 2. Instantiate Solvers
  WSwarm w;
  AntColony ants;
  FuzzyController fuzzy;
  BeeColony bees;

  w.init();
  ants.init();
  fuzzy.init();
  bees.init();

  std::cout << "Running Benchmark (" << SIMULATION_STEPS << " steps)..."
            << std::endl;

  // 3. Run Benchmark
  auto start_time = std::chrono::high_resolution_clock::now();

  long long t_w = 0, t_ants = 0, t_fuzzy = 0, t_bees = 0;

  for (int step = 0; step < SIMULATION_STEPS; step++) {
    // Move Obstacles (Dynamic Environment)
    for (auto &o : obstacles) {
      o.x += std::sin(step * 0.1f) * 0.5f;
      o.y += std::cos(step * 0.1f) * 0.5f;
    }

    // W
    auto t1 = std::chrono::high_resolution_clock::now();
    w.update(obstacles);
    auto t2 = std::chrono::high_resolution_clock::now();
    t_w +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // Ants
    t1 = std::chrono::high_resolution_clock::now();
    ants.update();
    t2 = std::chrono::high_resolution_clock::now();
    t_ants +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // Fuzzy
    t1 = std::chrono::high_resolution_clock::now();
    fuzzy.update(obstacles);
    t2 = std::chrono::high_resolution_clock::now();
    t_fuzzy +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // Bees
    t1 = std::chrono::high_resolution_clock::now();
    bees.update();
    t2 = std::chrono::high_resolution_clock::now();
    t_bees +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (step % 100 == 0)
      std::cout << "Step: " << step << "\r" << std::flush;
  }

  std::cout << "\nBenchmark Complete." << std::endl;

  // 4. Results & JSON
  double avg_w = (double)t_w / SIMULATION_STEPS;
  double avg_ants = (double)t_ants / SIMULATION_STEPS;
  double avg_fuzzy = (double)t_fuzzy / SIMULATION_STEPS;
  double avg_bees = (double)t_bees / SIMULATION_STEPS;

  std::ofstream json("w_complex_results.json");
  json << "{\n";
  json << "  \"experiment\": \"Complex Dynamic Foraging\",\n";
  json << "  \"iterations\": " << SIMULATION_STEPS << ",\n";
  json << "  \"results\": {\n";

  json << "    \"w\": {\n";
  json << "      \"avg_time_us\": " << avg_w << ",\n";
  json << "      \"food_collected\": " << w.food_collected << ",\n";
  json << "      \"score_per_us\": "
       << (double)w.food_collected / (t_w / 1000.0) << "\n";
  json << "    },\n";

  json << "    \"ants\": {\n";
  json << "      \"avg_time_us\": " << avg_ants << ",\n";
  json << "      \"food_collected\": " << ants.food_collected << ",\n";
  json << "      \"score_per_us\": "
       << (double)ants.food_collected / (t_ants / 1000.0) << "\n";
  json << "    },\n";

  json << "    \"fuzzy\": {\n";
  json << "      \"avg_time_us\": " << avg_fuzzy << ",\n";
  json << "      \"food_collected\": " << fuzzy.food_collected << ",\n";
  json << "      \"score_per_us\": "
       << (double)fuzzy.food_collected / (t_fuzzy / 1000.0) << "\n";
  json << "    },\n";

  json << "    \"bees\": {\n";
  json << "      \"avg_time_us\": " << avg_bees << ",\n";
  json << "      \"food_collected\": " << bees.food_collected << ",\n";
  json << "      \"score_per_us\": "
       << (double)bees.food_collected / (t_bees / 1000.0) << "\n";
  json << "    }\n";

  json << "  }\n";
  json << "}\n";

  std::cout << "JSON generated: w_complex_results.json" << std::endl;

  return 0;
}
