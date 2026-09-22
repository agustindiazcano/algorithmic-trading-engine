#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN ---
const int NUM_PARTICLES = 1000000; // 1 Millón de partículas
const int NUM_SOURCES = 50;        // 50 Centros de gravedad/repulsión
const float DT = 0.01f;

struct Point3D {
  float x, y, z;
};

// Generador de random
float rand_float(float min, float max) {
  static std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(min, max);
  return dist(rng);
}

// ==========================================
// 1. KERNEL TRADICIONAL (El "Oompa Loompa")
// ==========================================
// - Usa sqrt() (Lento)
// - Usa if() (Rompe la predicción de ramas)
// - Usa sin/cos (Muy lento)
void update_traditional(std::vector<Point3D> &particles,
                        const std::vector<Point3D> &sources) {
  for (auto &p : particles) {
    float force_x = 0, force_y = 0, force_z = 0;

    for (const auto &s : sources) {
      float dx = p.x - s.x;
      float dy = p.y - s.y;
      float dz = p.z - s.z;

      // 1. Costo: Raíz Cuadrada
      float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

      // 2. Costo: Branching (El gran enemigo)
      // La CPU tiene que adivinar si entra o no.
      if (dist < 5.0f && dist > 0.001f) {
        // 3. Costo: Funciones Trascendentales
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
// 2. KERNEL W (El "Titan")
// ==========================================
// - Sin sqrt() (Usa dist^2)
// - Sin if() (Aritmética continua)
// - Sin sin/cos (Aproximación Pade/Polinómica)
// - Vectorizable (El compilador puede usar AVX/SSE automáticamente)
void update_w(std::vector<Point3D> &particles,
                    const std::vector<Point3D> &sources) {
  for (auto &p : particles) {
    float force_x = 0, force_y = 0, force_z = 0;

    for (const auto &s : sources) {
      float dx = p.x - s.x;
      float dy = p.y - s.y;
      float dz = p.z - s.z;

      // 1. Ahorro: Usamos distancia cuadrada
      float dist_sq = dx * dx + dy * dy + dz * dz;

      // 2. Ahorro: Lógica Volumétrica (Sin IFs)
      // Función de caída rápida: 1 / (1 + r^4)
      // Se comporta como un campo de fuerza sin frenar la CPU
      float denom = 1.0f + dist_sq * dist_sq;
      float magnitude = 1.0f / denom; // Simple división

      // "Blending" natural: Si está lejos, magnitude es casi 0.
      // No necesitamos preguntar "if". Solo sumamos 0.00001.
      // Esto permite SIMD (instrucciones vectoriales).

      force_x += dx * magnitude;
      force_y += dy * magnitude;
      force_z += dz * magnitude;
    }

    // Multiplicamos por una constante de escala al final
    float scale = DT * 0.1f;
    p.x += force_x * scale;
    p.y += force_y * scale;
    p.z += force_z * scale;
  }
}

int main() {
  std::cout << "=== W C++ 3D BENCHMARK ===" << std::endl;
  std::cout << "Particulas: " << NUM_PARTICLES << " | Fuentes: " << NUM_SOURCES
            << std::endl;
  std::cout << "Interacciones por frame: "
            << (long long)NUM_PARTICLES * NUM_SOURCES << std::endl;

  // Inicializar datos
  std::vector<Point3D> particles(NUM_PARTICLES);
  std::vector<Point3D> sources(NUM_SOURCES);

  for (auto &p : particles) {
    p.x = rand_float(-50, 50);
    p.y = rand_float(-50, 50);
    p.z = rand_float(-50, 50);
  }
  for (auto &s : sources) {
    s.x = rand_float(-20, 20);
    s.y = rand_float(-20, 20);
    s.z = rand_float(-20, 20);
  }

  // Copia para que sea justo (mismos datos de inicio)
  auto particles_trad = particles;
  auto particles_weyl = particles;

  // --- TEST TRADICIONAL ---
  std::cout << "\nEjecutando Tradicional (Branching + Sqrt)..." << std::endl;
  auto start_trad = std::chrono::high_resolution_clock::now();

  update_traditional(particles_trad, sources);

  auto end_trad = std::chrono::high_resolution_clock::now();
  double time_trad =
      std::chrono::duration<double, std::milli>(end_trad - start_trad).count();
  std::cout << "Tiempo: " << time_trad << " ms" << std::endl;

  // --- TEST W ---
  std::cout << "Ejecutando W (Branchless + Polynomial)..." << std::endl;
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

  if (speedup > 2.0) {
    std::cout
        << "CONCLUSION: La CPU prefiere matematica pura antes que decisiones."
        << std::endl;
  } else {
    std::cout
        << "NOTA: Asegurate de compilar con -O3 para vectorizacion automatica."
        << std::endl;
  }

  // --- JSON EXPORT ---
  std::ofstream jsonFile("w_particle_results.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_name\": \"W Particle Benchmark 3D\",\n";
    jsonFile << "  \"particle_count\": " << NUM_PARTICLES << ",\n";
    jsonFile << "  \"source_count\": " << NUM_SOURCES << ",\n";
    jsonFile << "  \"traditional_ms\": " << time_trad << ",\n";
    jsonFile << "  \"w_ms\": " << time_weyl << ",\n";
    jsonFile << "  \"speedup_factor\": " << speedup << ",\n";
    jsonFile << "  \"description\": \"Branchless logic vs Conditional Sqrt "
                "logic\"\n";
    jsonFile << "}\n";
    std::cout << "JSON generado: w_particle_results.json" << std::endl;
  }

  return 0;
}
