#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// --- CONFIGURACIÓN ---
const int NUM_PARTICLES = 1000000;
const int NUM_SOURCES = 50;
const float DT = 0.01f;
const float RADIUS_CHECK = 25.0f; // AUMENTAMOS EL RADIO (Trampa para el IF)

struct Point3D {
  float x, y, z;
};

float rand_float(float min, float max) {
  static std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(min, max);
  return dist(rng);
}

// ==========================================
// 1. KERNEL TRADICIONAL (El que va a sufrir)
// ==========================================
void update_traditional(std::vector<Point3D> &particles,
                        const std::vector<Point3D> &sources) {
  for (auto &p : particles) {
    float force_x = 0, force_y = 0, force_z = 0;

    for (const auto &s : sources) {
      float dx = p.x - s.x;
      float dy = p.y - s.y;
      float dz = p.z - s.z;

      float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

      // AQUI ESTA LA CLAVE:
      // Al aumentar el radio a 25.0, muchas más partículas entran aquí.
      // El "Branch Predictor" se va a volver loco tratando de adivinar.
      if (dist < RADIUS_CHECK && dist > 0.001f) {
        // Matemática pesada que ahora SÍ se ejecuta
        float magnitude = std::sin(dist) * std::exp(-dist * 0.5f);
        force_x += (dx / dist) * magnitude;
        force_y += (dy / dist) * magnitude;
        force_z += (dz / dist) * magnitude;
      }
    }
    p.x += force_x * DT;
    p.y += force_y * DT;
    p.z += force_z * DT;
  }
}

// ==========================================
// 2. KERNEL W (El Inmutable)
// ==========================================
void update_w(std::vector<Point3D> &particles,
                    const std::vector<Point3D> &sources) {
  for (auto &p : particles) {
    float force_x = 0, force_y = 0, force_z = 0;

    for (const auto &s : sources) {
      float dx = p.x - s.x;
      float dy = p.y - s.y;
      float dz = p.z - s.z;

      float dist_sq = dx * dx + dy * dy + dz * dz;

      // Lógica Volumétrica:
      // No le importa si está cerca o lejos.
      // Fluye a través de los datos sin saltos.
      float denom = 1.0f + dist_sq * dist_sq * 0.05f; // Ajuste leve de curva
      float magnitude = 1.0f / denom;

      force_x += dx * magnitude;
      force_y += dy * magnitude;
      force_z += dz * magnitude;
    }

    float scale = DT * 0.1f;
    p.x += force_x * scale;
    p.y += force_y * scale;
    p.z += force_z * scale;
  }
}

int main() {
  std::cout << "=== W C++ VENGEANCE BENCHMARK ===" << std::endl;
  std::cout << "Particulas: " << NUM_PARTICLES << " | Fuentes: " << NUM_SOURCES
            << std::endl;
  std::cout << "Radio de Corte Tradicional: " << RADIUS_CHECK
            << " (Alta Densidad)" << std::endl;

  std::vector<Point3D> particles(NUM_PARTICLES);
  std::vector<Point3D> sources(NUM_SOURCES);

  // GENERACIÓN MÁS DENSA: Apretamos las partículas contra las fuentes
  for (auto &p : particles) {
    p.x = rand_float(-30, 30);
    p.y = rand_float(-30, 30);
    p.z = rand_float(-30, 30);
  }
  for (auto &s : sources) {
    s.x = rand_float(-15, 15);
    s.y = rand_float(-15, 15);
    s.z = rand_float(-15, 15);
  }

  auto particles_trad = particles;
  auto particles_weyl = particles;

  // --- TEST TRADICIONAL ---
  std::cout << "\nEjecutando Tradicional (Branch Misprediction Hell)..."
            << std::endl;
  auto start_trad = std::chrono::high_resolution_clock::now();
  update_traditional(particles_trad, sources);
  auto end_trad = std::chrono::high_resolution_clock::now();
  double time_trad =
      std::chrono::duration<double, std::milli>(end_trad - start_trad).count();
  std::cout << "Tiempo: " << time_trad << " ms" << std::endl;

  // --- TEST W ---
  std::cout << "Ejecutando W (SIMD Highway)..." << std::endl;
  auto start_weyl = std::chrono::high_resolution_clock::now();
  update_w(particles_weyl, sources);
  auto end_weyl = std::chrono::high_resolution_clock::now();
  double time_weyl =
      std::chrono::duration<double, std::milli>(end_weyl - start_weyl).count();
  std::cout << "Tiempo: " << time_weyl << " ms" << std::endl;

  // --- RESULTADOS ---
  double speedup = time_trad / time_weyl;
  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "SPEEDUP FACTOR: " << std::fixed << std::setprecision(2)
            << speedup << "x" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  // Export Results
  std::ofstream jsonFile("w_vengeance_results.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_name\": \"W Vengeance Benchmark (Branch "
                "Misprediction)\",\n";
    jsonFile << "  \"particle_count\": " << NUM_PARTICLES << ",\n";
    jsonFile << "  \"radius_check\": " << RADIUS_CHECK << ",\n";
    jsonFile << "  \"traditional_ms\": " << time_trad << ",\n";
    jsonFile << "  \"w_ms\": " << time_weyl << ",\n";
    jsonFile << "  \"speedup_factor\": " << speedup << "\n";
    jsonFile << "}\n";
  }

  return 0;
}
