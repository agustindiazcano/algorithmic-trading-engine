#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>


// ==========================================
// ESCENARIO REALISTA: "SENSOR FUSION LOOP"
// Contexto: Un robot/cohete procesando datos de LIDAR/Radar a 100Hz.
// Problema: Los datos reales son RUIDOSOS. A veces el sensor ve 'aire' (vacío),
// a veces ve 'pared' (denso).
//
// Meta: Estabilidad (Jitter bajo). En sistemas críticos, no importa si eres
// rápido en promedio, importa que NUNCA seas lento (Worst Case Execution Time).
// ==========================================

const int FRAMES_TO_SIMULATE = 1000;
const int POINTS_PER_FRAME = 10000;
const int SOURCES = 20;

struct Point3D {
  float x, y, z;
};

// Simulamos una carga de trabajo variable (Ruido del mundo real)
// 0.0 = Aire limpio (Puntos dispersos)
// 1.0 = Tormenta de arena / Colisión (Puntos muy densos cerca del sensor)
void generate_sensor_frame(std::vector<Point3D> &points, float density_factor) {
  static std::mt19937 rng(42);
  // Si density es alta, el rango es pequeño (todo cerca)
  // Si density es baja, el rango es amplio (todo lejos)
  float range = 100.0f - (90.0f * density_factor);
  std::uniform_real_distribution<float> dist(-range, range);

  for (auto &p : points) {
    p.x = dist(rng);
    p.y = dist(rng);
    p.z = dist(rng);
  }
}

// 1. Enfoque Tradicional (Optimista)
// Rápido cuando no pasa nada. Lento cuando hay problemas.
void process_traditional(const std::vector<Point3D> &points,
                         const std::vector<Point3D> &dangerous_objects,
                         volatile float *output_checksum) {
  float local_sum = 0;
  for (const auto &p : points) {
    for (const auto &obj : dangerous_objects) {
      float dx = p.x - obj.x;
      float dy = p.y - obj.y;
      float dz = p.z - obj.z;
      float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

      // Branch Check: ¿Es amenaza?
      if (dist < 15.0f) {
        // Cálculos de evasión costosos
        local_sum += std::sin(dist) * std::cos(dist);
      }
    }
  }
  *output_checksum = local_sum;
}

// 2. Enfoque W (Determinista / Real-Time)
// Velocidad constante sin importar el pánico.
void process_w(const std::vector<Point3D> &points,
                     const std::vector<Point3D> &dangerous_objects,
                     volatile float *output_checksum) {
  float local_sum = 0;
  for (const auto &p : points) {
    for (const auto &obj : dangerous_objects) {
      float dx = p.x - obj.x;
      float dy = p.y - obj.y;
      float dz = p.z - obj.z;
      float dist_sq = dx * dx + dy * dy + dz * dz;

      // Branchless Logic
      // Evaluamos la "importancia" matemáticamente
      // 1.0 si dist_sq es 0, decae suavemente.
      float weight = 225.0f / (225.0f + dist_sq * dist_sq + 1e-5f);

      // polinomio simple aproximado de sin*cos
      float val = (dist_sq * 0.001f) * weight;

      local_sum += val;
    }
  }
  *output_checksum = local_sum;
}

struct Stats {
  double min_ms, max_ms, avg_ms, std_dev;
};

Stats calculate_stats(const std::vector<double> &times) {
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double avg = sum / times.size();
  double sq_sum =
      std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / times.size() - avg * avg);
  auto minmax = std::minmax_element(times.begin(), times.end());
  return {*minmax.first, *minmax.second, avg, stdev};
}

int main() {
  std::cout << "=== REAL-TIME CRITICAL SYSTEMS BENCHMARK ===" << std::endl;
  std::cout << "Escenario: Loop de Control de Robot (1000 Frames)" << std::endl;
  std::cout << "Simulando densidad de datos variable (Aire vs Pared)..."
            << std::endl;

  std::vector<Point3D> sensor_data(POINTS_PER_FRAME);
  std::vector<Point3D> obstacles(SOURCES); // Objetos peligrosos fijos
  for (auto &o : obstacles) {
    o.x = 0;
    o.y = 0;
    o.z = 0;
  } // Todos en el centro para maximizar peligro

  std::vector<double> times_trad;
  std::vector<double> times_weyl;
  volatile float checksum;

  // SIMULACION LOOP
  std::mt19937 density_rng(999);
  std::uniform_real_distribution<float> density_dist(0.0f, 1.0f);

  for (int frame = 0; frame < FRAMES_TO_SIMULATE; frame++) {
    // Generar densidad aleatoria para este frame
    // 80% de las veces es baja (0.0-0.3), 20% es crítica (0.8-1.0)
    float input_density = density_dist(density_rng);
    if (input_density > 0.2f)
      input_density *= 0.2f; // Mayoría disperso
    else
      input_density = 0.9f; // Spike de densidad (Pared inesperada)

    generate_sensor_frame(sensor_data, input_density);

    // Run Traditional
    auto t1 = std::chrono::high_resolution_clock::now();
    process_traditional(sensor_data, obstacles, &checksum);
    auto t2 = std::chrono::high_resolution_clock::now();
    times_trad.push_back(
        std::chrono::duration<double, std::milli>(t2 - t1).count());

    // Run W
    t1 = std::chrono::high_resolution_clock::now();
    process_w(sensor_data, obstacles, &checksum);
    t2 = std::chrono::high_resolution_clock::now();
    times_weyl.push_back(
        std::chrono::duration<double, std::milli>(t2 - t1).count());
  }

  Stats s_trad = calculate_stats(times_trad);
  Stats s_weyl = calculate_stats(times_weyl);

  std::cout << "\nRESULTADOS DE ESTABILIDAD (Menor es mejor en Max/Jitter):"
            << std::endl;

  std::cout << "\n1. TRADICIONAL (If/Branching):" << std::endl;
  std::cout << "   Promedio: " << s_trad.avg_ms << " ms" << std::endl;
  std::cout << "   MIN Time: " << s_trad.min_ms << " ms (Mejor caso - Aire)"
            << std::endl;
  std::cout << "   MAX Time: " << s_trad.max_ms
            << " ms (PEOR CASO - MUERTE DEL ROBOT)" << std::endl;
  std::cout << "   Jitter (StdDev): " << s_trad.std_dev << " ms" << std::endl;

  std::cout << "\n2. W (Branchless/Deterministic):" << std::endl;
  std::cout << "   Promedio: " << s_weyl.avg_ms << " ms" << std::endl;
  std::cout << "   MIN Time: " << s_weyl.min_ms << " ms" << std::endl;
  std::cout << "   MAX Time: " << s_weyl.max_ms << " ms (Estable)" << std::endl;
  std::cout << "   Jitter (StdDev): " << s_weyl.std_dev << " ms" << std::endl;

  std::cout << "\nCOMPARACION DE RIESGO:" << std::endl;
  double volatility = s_trad.max_ms / s_trad.min_ms;
  std::cout << "Volatilidad Tradicional: " << volatility
            << "x (Diferencia entre mejor y peor frame)" << std::endl;
  std::cout << "Volatilidad W:     " << s_weyl.max_ms / s_weyl.min_ms
            << "x" << std::endl;

  return 0;
}
