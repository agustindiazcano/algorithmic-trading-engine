#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// ==================================================================================
// W SYSTEMS: 3D VOXEL HIVE
// Scenario: "Deep Space Swarm"
// Scale: 40,000 Agents vs 1,000,000 Obstacles in 3D
// Engine: Voxel-based Spatial Partitioning
// ==================================================================================

const int WORLD_SIZE = 1000;                  // 1000x1000x1000 units
const int VOXEL_SIZE = 20;                    // Bucket size
const int GRID_DIM = WORLD_SIZE / VOXEL_SIZE; // 50x50x50 buckets

const int NUM_OBSTACLES = 1000000;
const int NUM_FOOD = 100;
const int AGENTS_PER_ALGO = 10000;
const int SIM_STEPS = 500;

struct Vec3 {
  float x, y, z;
};

struct Agent {
  Vec3 pos;
  Vec3 vel;
  int id;
};

struct Food {
  Vec3 pos;
};

// Utils: 3D Fast Math
inline float dist_sq(Vec3 a, Vec3 b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

inline float fast_inv_sqrt(float number) {
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
// 3D SPATIAL GRID
// ============================================================
class VoxelGrid {
public:
  std::vector<Vec3> buckets[GRID_DIM * GRID_DIM * GRID_DIM];
  // We omit the boolean blocked map for 3D to save memory (1000^3 bits is too
  // much/complex) We rely on buckets for collision in this demo or just ignore
  // strict wall collision and process obstacle avoidance forces. Actually,
  // let's implement a coarse blocked map: 200x200x200
  std::vector<bool> blocked_map;
  int map_res = 200;
  float map_scale;

  VoxelGrid() {
    map_scale = (float)map_res / WORLD_SIZE;
    blocked_map.resize(map_res * map_res * map_res, false);
  }

  void add_obstacle(Vec3 p) {
    if (p.x < 0 || p.x >= WORLD_SIZE || p.y < 0 || p.y >= WORLD_SIZE ||
        p.z < 0 || p.z >= WORLD_SIZE)
      return;

    // Add to bucket for W
    int col = (int)p.x / VOXEL_SIZE;
    int row = (int)p.y / VOXEL_SIZE;
    int dep = (int)p.z / VOXEL_SIZE;
    if (col >= 0 && col < GRID_DIM && row >= 0 && row < GRID_DIM && dep >= 0 &&
        dep < GRID_DIM) {
      buckets[dep * GRID_DIM * GRID_DIM + row * GRID_DIM + col].push_back(p);
    }

    // Mark Blocked Map
    int mx = (int)(p.x * map_scale);
    int my = (int)(p.y * map_scale);
    int mz = (int)(p.z * map_scale);
    if (mx >= 0 && mx < map_res && my >= 0 && my < map_res && mz >= 0 &&
        mz < map_res) {
      blocked_map[mz * map_res * map_res + my * map_res + mx] = true;
    }
  }

  void get_neighbors_3d(Vec3 p, float radius, std::vector<Vec3> &out) {
    int cx = (int)p.x / VOXEL_SIZE;
    int cy = (int)p.y / VOXEL_SIZE;
    int cz = (int)p.z / VOXEL_SIZE;
    int r = 1; // Check immediate 3x3x3 neighbors buckets

    for (int z = cz - r; z <= cz + r; z++) {
      for (int y = cy - r; y <= cy + r; y++) {
        for (int x = cx - r; x <= cx + r; x++) {
          if (x >= 0 && x < GRID_DIM && y >= 0 && y < GRID_DIM && z >= 0 &&
              z < GRID_DIM) {
            const auto &b = buckets[z * GRID_DIM * GRID_DIM + y * GRID_DIM + x];
            for (const auto &o : b) {
              if (dist_sq(p, o) < radius * radius) {
                out.push_back(o);
              }
            }
          }
        }
      }
    }
  }

  bool is_blocked(float x, float y, float z) {
    int mx = (int)(x * map_scale);
    int my = (int)(y * map_scale);
    int mz = (int)(z * map_scale);
    if (mx < 0 || mx >= map_res || my < 0 || my >= map_res || mz < 0 ||
        mz >= map_res)
      return true; // World bounds
    return blocked_map[mz * map_res * map_res + my * map_res + mx];
  }
};

std::vector<Food> foods;
VoxelGrid grid;
std::mt19937 rng(42);

// ===================================
// 1. W 3D (Volumetric Fields)
// ===================================
class W3D {
public:
  std::vector<Agent> agents;
  int food_collected = 0;
  std::vector<Vec3> sensor;

  void init() {
    std::uniform_real_distribution<float> c(100, WORLD_SIZE - 100);
    sensor.reserve(100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{c(rng), c(rng), c(rng)}, {0, 0, 0}, i});
    }
  }

  void update() {
    for (auto &a : agents) {
      float fx = 0, fy = 0, fz = 0;

      // Attract Food
      float min_d2 = 1e9;
      int best_f = -1;
      for (int i = 0; i < foods.size(); i++) {
        float d2 = dist_sq(a.pos, foods[i].pos);
        if (d2 < min_d2) {
          min_d2 = d2;
          best_f = i;
        }
      }

      if (best_f != -1) {
        float dx = foods[best_f].pos.x - a.pos.x;
        float dy = foods[best_f].pos.y - a.pos.y;
        float dz = foods[best_f].pos.z - a.pos.z;
        float id = fast_inv_sqrt(min_d2 + 0.01f);
        fx += dx * id * 2.0f;
        fy += dy * id * 2.0f;
        fz += dz * id * 2.0f;
      }

      // Repulse Obstacles
      sensor.clear();
      grid.get_neighbors_3d(a.pos, 25.0f, sensor);

      for (const auto &o : sensor) {
        float dx = a.pos.x - o.x;
        float dy = a.pos.y - o.y;
        float dz = a.pos.z - o.z;
        float d2 = dx * dx + dy * dy + dz * dz;

        float force = 800.0f / (1.0f + d2);
        fx += dx * force;
        fy += dy * force;
        fz += dz * force;
      }

      // Move
      a.vel.x = (a.vel.x + fx) * 0.7f;
      a.vel.y = (a.vel.y + fy) * 0.7f;
      a.vel.z = (a.vel.z + fz) * 0.7f;

      // Limit
      float v2 = a.vel.x * a.vel.x + a.vel.y * a.vel.y + a.vel.z * a.vel.z;
      if (v2 > 36.0f) {
        float im = fast_inv_sqrt(v2);
        a.vel.x *= (6.0f * im * v2);
        a.vel.y *= (6.0f * im * v2);
        a.vel.z *= (6.0f * im * v2);
      }

      // Integration
      float nx = a.pos.x + a.vel.x;
      float ny = a.pos.y + a.vel.y;
      float nz = a.pos.z + a.vel.z;

      // Collision (Soft Bounce)
      if (!grid.is_blocked(nx, ny, nz)) {
        a.pos.x = nx;
        a.pos.y = ny;
        a.pos.z = nz;
      } else {
        a.vel.x *= -0.8f;
        a.vel.y *= -0.8f;
        a.vel.z *= -0.8f;
        a.pos.x += a.vel.x;
        a.pos.y += a.vel.y;
        a.pos.z += a.vel.z;
      }

      if (best_f != -1 && min_d2 < 100.0f) {
        // Should remove food in real sim, here just count score
        food_collected++;
      }
    }
  }
};

// ===================================
// 2. 3D ANTS (Voxel Pheromones)
// ===================================
class Ants3D {
  // 3D Pheromone Grid flattened
  // Reduced resolution: 100x100x100 = 1 Million cells (Small memory footprint)
  std::vector<uint8_t> pheromones;
  int dim = 100;
  float scale;

public:
  std::vector<Agent> agents;
  int food_collected = 0;

  Ants3D() {
    scale = (float)dim / WORLD_SIZE;
    pheromones.resize(dim * dim * dim, 0);
  }

  void init() {
    std::uniform_real_distribution<float> c(100, WORLD_SIZE - 100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{c(rng), c(rng), c(rng)}, {0, 0, 0}, i});
    }
  }

  void update() {
    // Evaporation (Random sampling)
    std::uniform_int_distribution<int> d(0, pheromones.size() - 1);
    for (int k = 0; k < 10000; k++) {
      int idx = d(rng);
      if (pheromones[idx] > 0)
        pheromones[idx]--;
    }

    std::uniform_real_distribution<float> noise(-1, 1);

    for (auto &a : agents) {
      int gx = (int)(a.pos.x * scale);
      int gy = (int)(a.pos.y * scale);
      int gz = (int)(a.pos.z * scale);

      // 3D Sensor (Check 6 neighbors + center?)
      int best_x = 0, best_y = 0, best_z = 0;
      int max_ph = 0;

      // Little random walk baseline
      a.vel.x += noise(rng) * 0.5f;
      a.vel.y += noise(rng) * 0.5f;
      a.vel.z += noise(rng) * 0.5f;

      // Check neighbors
      for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0 && dz == 0)
              continue;
            int nx = gx + dx;
            int ny = gy + dy;
            int nz = gz + dz;
            if (nx >= 0 && nx < dim && ny >= 0 && ny < dim && nz >= 0 &&
                nz < dim) {
              int ph = pheromones[nz * dim * dim + ny * dim + nx];
              if (ph > max_ph) {
                max_ph = ph;
                best_x = dx;
                best_y = dy;
                best_z = dz;
              }
            }
          }
        }
      }

      if (max_ph > 10) {
        // Follow scent
        a.vel.x += (float)best_x * 1.5f;
        a.vel.y += (float)best_y * 1.5f;
        a.vel.z += (float)best_z * 1.5f;
      }

      // Normalize Speed
      float v2 = a.vel.x * a.vel.x + a.vel.y * a.vel.y + a.vel.z * a.vel.z;
      if (v2 > 16.0f) {
        float im = fast_inv_sqrt(v2);
        a.vel.x *= (4.0f * im * v2);
        a.vel.y *= (4.0f * im * v2);
        a.vel.z *= (4.0f * im * v2);
      }

      float nx = a.pos.x + a.vel.x;
      float ny = a.pos.y + a.vel.y;
      float nz = a.pos.z + a.vel.z;

      if (!grid.is_blocked(nx, ny, nz)) {
        a.pos.x = nx;
        a.pos.y = ny;
        a.pos.z = nz;
      } else {
        a.vel.x *= -1;
        a.vel.y *= -1;
        a.vel.z *= -1;
      }

      // Food?
      for (const auto &f : foods) {
        if (dist_sq(a.pos, f.pos) < 100.0f) {
          food_collected++;
          // Deposit
          if (gx >= 0 && gx < dim && gy >= 0 && gy < dim && gz >= 0 &&
              gz < dim) {
            pheromones[gz * dim * dim + gy * dim + gx] = 255;
          }
        }
      }
    }
  }
};

// ===================================
// 3. FUZZY 3D
// ===================================
class Fuzzy3D {
public:
  std::vector<Agent> agents;
  int food_collected = 0;
  std::vector<Vec3> sensor;

  void init() {
    std::uniform_real_distribution<float> c(100, WORLD_SIZE - 100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{c(rng), c(rng), c(rng)}, {0, 0, 0}, i});
    }
  }

  void update() {
    for (auto &a : agents) {
      // Find closest food
      Vec3 target = {WORLD_SIZE / 2, WORLD_SIZE / 2, WORLD_SIZE / 2};
      float min_f = 1e9;
      for (auto &f : foods) {
        float d = dist_sq(a.pos, f.pos);
        if (d < min_f) {
          min_f = d;
          target = f.pos;
        }
      }

      // Find closest obstacle
      sensor.clear();
      grid.get_neighbors_3d(a.pos, 30.0f, sensor);
      Vec3 obs = {0, 0, 0};
      float min_o = 1e9;
      if (!sensor.empty()) {
        for (auto &o : sensor) {
          float d = dist_sq(a.pos, o);
          if (d < min_o) {
            min_o = d;
            obs = o;
          }
        }
      }
      float dist_obs = std::sqrt(min_o); // Crisp value for fuzzy

      // Fuzzy Rules 3D
      // If dist < 10 -> EMERGENCY EVADE (Opposite to obstacle)
      // If dist < 30 -> SOFT AVOID
      // ELSE -> GO TO FOOD

      float tx, ty, tz, speed = 4.0f;

      if (dist_obs < 15.0f) {
        // Panic vector
        tx = a.pos.x - obs.x;
        ty = a.pos.y - obs.y;
        tz = a.pos.z - obs.z;
        speed = 2.0f;
      } else if (dist_obs < 30.0f) {
        // Blend
        float wx = (a.pos.x - obs.x) * 2.0f;
        float wy = (a.pos.y - obs.y) * 2.0f;
        float wz = (a.pos.z - obs.z) * 2.0f;
        float fx = (target.x - a.pos.x);
        float fy = (target.y - a.pos.y);
        float fz = (target.z - a.pos.z);
        tx = wx + fx;
        ty = wy + fy;
        tz = wz + fz;
      } else {
        tx = target.x - a.pos.x;
        ty = target.y - a.pos.y;
        tz = target.z - a.pos.z;
      }

      float len = std::sqrt(tx * tx + ty * ty + tz * tz + 0.001f);
      a.vel.x = (tx / len) * speed;
      a.vel.y = (ty / len) * speed;
      a.vel.z = (tz / len) * speed;

      float nx = a.pos.x + a.vel.x;
      float ny = a.pos.y + a.vel.y;
      float nz = a.pos.z + a.vel.z;

      if (!grid.is_blocked(nx, ny, nz)) {
        a.pos.x = nx;
        a.pos.y = ny;
        a.pos.z = nz;
      }

      if (min_f < 100.0f)
        food_collected++;
    }
  }
};

// ===================================
// 4. BEES 3D
// ===================================
class Bees3D {
public:
  std::vector<Agent> agents;
  int food_collected = 0;
  void init() {
    std::uniform_real_distribution<float> c(100, WORLD_SIZE - 100);
    for (int i = 0; i < AGENTS_PER_ALGO; i++) {
      agents.push_back({{c(rng), c(rng), c(rng)}, {0, 0, 0}, i});
    }
  }

  void update() {
    std::uniform_real_distribution<float> j(-2, 2);
    Vec3 hive_mind_target = {0, 0, 0};
    bool found = false;

    for (auto &a : agents) {
      a.vel.x += j(rng);
      a.vel.y += j(rng);
      a.vel.z += j(rng);
      a.vel.x *= 0.95f;
      a.vel.y *= 0.95f;
      a.vel.z *= 0.95f; // Drag

      float nx = a.pos.x + a.vel.x;
      float ny = a.pos.y + a.vel.y;
      float nz = a.pos.z + a.vel.z;

      if (!grid.is_blocked(nx, ny, nz)) {
        a.pos.x = nx;
        a.pos.y = ny;
        a.pos.z = nz;
      } else {
        a.vel.x *= -1;
        a.vel.y *= -1;
        a.vel.z *= -1;
      }

      for (const auto &f : foods) {
        if (dist_sq(a.pos, f.pos) < 100.0f) {
          food_collected++;
          hive_mind_target = f.pos;
          found = true;
        }
      }
    }

    if (found) {
      // Signal global recruit (10% of bees fly to target)
      for (int i = 0; i < agents.size(); i += 10) {
        float dx = hive_mind_target.x - agents[i].pos.x;
        float dy = hive_mind_target.y - agents[i].pos.y;
        float dz = hive_mind_target.z - agents[i].pos.z;
        float im = fast_inv_sqrt(dx * dx + dy * dy + dz * dz + 0.01f);
        agents[i].vel.x += dx * im;
        agents[i].vel.y += dy * im;
        agents[i].vel.z += dz * im;
      }
    }
  }
};

int main() {
  std::cout << "=== W 3D: DEEP SPACE SWARM ===" << std::endl;
  std::cout << "Resolution: " << WORLD_SIZE << "x" << WORLD_SIZE << "x"
            << WORLD_SIZE << std::endl;
  std::cout << "Obstacles: " << NUM_OBSTACLES << " (Voxel Octree equivalent)"
            << std::endl;

  std::uniform_real_distribution<float> c(0, WORLD_SIZE);

  // Gen obstacles
  for (int i = 0; i < NUM_OBSTACLES; i++) {
    grid.add_obstacle({c(rng), c(rng), c(rng)});
  }
  // Gen food
  for (int i = 0; i < NUM_FOOD; i++) {
    foods.push_back({{c(rng), c(rng), c(rng)}});
  }

  W3D w;
  w.init();
  Ants3D ants;
  ants.init();
  Fuzzy3D fuzzy;
  fuzzy.init();
  Bees3D bees;
  bees.init();

  long long tw = 0, ta = 0, tf = 0, tb = 0;

  std::cout << "Running 3D Simulation..." << std::endl;
  for (int s = 0; s < SIM_STEPS; s++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    w.update();
    tw += std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - t1)
              .count();

    t1 = std::chrono::high_resolution_clock::now();
    ants.update();
    ta += std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - t1)
              .count();

    t1 = std::chrono::high_resolution_clock::now();
    fuzzy.update();
    tf += std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - t1)
              .count();

    t1 = std::chrono::high_resolution_clock::now();
    bees.update();
    tb += std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - t1)
              .count();

    if (s % 50 == 0)
      std::cout << "Step " << s << "\r" << std::flush;
  }

  std::ofstream json("w_3d_results.json");
  json << "{\n \"test\": \"Deep Space 3D\",\n \"results\": {\n";
  json << "  \"w\": { \"ms\": " << tw / 1000
       << ", \"score\": " << w.food_collected << " },\n";
  json << "  \"ants\": { \"ms\": " << ta / 1000
       << ", \"score\": " << ants.food_collected << " },\n";
  json << "  \"fuzzy\": { \"ms\": " << tf / 1000
       << ", \"score\": " << fuzzy.food_collected << " },\n";
  json << "  \"bees\": { \"ms\": " << tb / 1000
       << ", \"score\": " << bees.food_collected << " }\n";
  json << " }}\n";

  std::cout << "\n\nRESULTS (3D):" << std::endl;
  std::cout << "W: " << tw / 1000
            << "ms | Score: " << w.food_collected << std::endl;
  std::cout << "Ants:    " << ta / 1000 << "ms | Score: " << ants.food_collected
            << std::endl;
  std::cout << "Fuzzy:   " << tf / 1000
            << "ms | Score: " << fuzzy.food_collected << std::endl;
  std::cout << "Bees:    " << tb / 1000 << "ms | Score: " << bees.food_collected
            << std::endl;

  return 0;
}
