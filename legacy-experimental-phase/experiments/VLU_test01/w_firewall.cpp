#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>


// --- CONFIGURACIÓN FIREWALL ---
const int NUM_PACKETS = 10000000; // 10 Millones de paquetes por lote
const int RULES_COUNT = 16;       // 16 Reglas de bloqueo activas

// Estructura de Paquete (Simulada)
struct Packet {
  uint32_t src_ip;
  uint32_t dst_ip;
  uint16_t src_port;
  uint16_t dst_port;
  uint8_t proto;
  uint8_t flags;
  uint16_t padding; // Alineación a 16 bytes para SIMD
};

// Estructura de Regla (Rango de bloqueo)
struct Rule {
  uint32_t ip_min, ip_max;
  uint16_t port_min, port_max;
  uint8_t proto;
};

std::vector<Rule> firewall_rules(RULES_COUNT);
volatile int g_sink = 0;

// Generar reglas aleatorias
void init_rules() {
  std::mt19937 rng(1337);
  for (int i = 0; i < RULES_COUNT; ++i) {
    firewall_rules[i].ip_min = rng();
    firewall_rules[i].ip_max = firewall_rules[i].ip_min + 10000;
    firewall_rules[i].port_min = rng() % 50000;
    firewall_rules[i].port_max = firewall_rules[i].port_min + 100;
    firewall_rules[i].proto = (rng() % 2 == 0) ? 6 : 17; // TCP or UDP
  }
}

// ==========================================
// 1. MÉTODO TRADICIONAL (Spaghetti Logic)
// ==========================================
// El firewall clásico evalúa reglas secuencialmente.
// "Si coincido con la regla 1, bloqueo. Si no, miro la 2..."
// Esto es pesadilla de saltos si hay tráfico mixto.
long long benchmark_firewall_branching(const std::vector<Packet> &traffic) {
  int dropped = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < NUM_PACKETS; ++i) {
    bool drop = false;

    // Iterar reglas (Linear scan)
    for (int r = 0; r < RULES_COUNT; ++r) {
      // Prediccion de Ramas sufre aquí:
      // "IP coincide?" -> NO. Salto.
      // "IP coincide?" -> SI. Entro. "Puerto coincide?" -> NO. Salto.
      if (traffic[i].src_ip >= firewall_rules[r].ip_min) {
        if (traffic[i].src_ip <= firewall_rules[r].ip_max) {
          if (traffic[i].dst_port >= firewall_rules[r].port_min) {
            if (traffic[i].dst_port <= firewall_rules[r].port_max) {
              if (traffic[i].proto == firewall_rules[r].proto) {
                drop = true;
                break; // Salir del bucle de reglas (Early Exit)
              }
            }
          }
        }
      }
    }

    if (drop)
      dropped++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = dropped;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

// ==========================================
// 2. MÉTODO W (Bitwise Matrix Logic)
// ==========================================
// En lugar de "preguntar", calculamos máscaras de bits.
// Transforma las condiciones lógicas en operaciones AND/OR puras.
// El compilador puede vectorizar esto masivamente.
long long benchmark_firewall_bitwise(const std::vector<Packet> &traffic) {
  int dropped = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < NUM_PACKETS; ++i) {
    int drop_mask = 0; // 0 = Clean, 1 = Drop

    // Unrolled loops y lógica sin branches
    for (int r = 0; r < RULES_COUNT; ++r) {
      // Evaluamos TODAS las condiciones aritmeticamente.
      // (a >= min) se convierte en 1 o 0.
      // Multiplicamos (AND lógico) los resultados.

      // Truco unsigned para rango: (val - min) <= (max - min) maneja ambos
      // límites con 1 resta y 1 comp
      uint32_t ip = traffic[i].src_ip;
      uint32_t r_ip_min = firewall_rules[r].ip_min;
      uint32_t r_ip_limit = firewall_rules[r].ip_max - r_ip_min;

      bool ip_match = (ip - r_ip_min) <= r_ip_limit;

      uint16_t port = traffic[i].dst_port;
      uint16_t r_port_min = firewall_rules[r].port_min;
      uint16_t r_port_limit = firewall_rules[r].port_max - r_port_min;

      bool port_match = (port - r_port_min) <= r_port_limit;

      bool proto_match = (traffic[i].proto == firewall_rules[r].proto);

      // Acumulamos el resultado con OR (Si CUALQUIER regla matchea, drop_mask
      // se vuelve 1) Sin IFs. Solo bits.
      drop_mask |= (ip_match & port_match & proto_match);
    }

    dropped += drop_mask;
  }

  auto end = std::chrono::high_resolution_clock::now();
  g_sink = dropped;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int main() {
  std::cout << "=== W SYSTEMS: FIREWALL TRAFFIC ANALYSIS ==="
            << std::endl;
  std::cout << "Escenario: 10GBit/s Packet Filtering (" << NUM_PACKETS
            << " pkts/cycle)" << std::endl;

  init_rules();

  // Generar Trafico Hostil (DDoS Simulation)
  // Mezcla de tráfico que machea y no machea reglas para confundir al predictor
  std::cout << "Generando Traffic Flood..." << std::endl;
  std::vector<Packet> traffic(NUM_PACKETS);
  std::mt19937 rng(999);

  for (auto &p : traffic) {
    // Generar paquetes "Near Miss" (cerca de las reglas pero no exactos) 50%
    // del tiempo
    if (rng() % 2 == 0) {
      int r = rng() % RULES_COUNT;
      p.src_ip = firewall_rules[r].ip_min + (rng() % 20000) -
                 5000; // Alrededor de la IPs baneadas
      p.dst_port = firewall_rules[r].port_min + (rng() % 200) - 50;
      p.proto = firewall_rules[r].proto;
    } else {
      p.src_ip = rng();
      p.dst_port = rng();
      p.proto = (rng() % 2 == 0) ? 6 : 17;
    }
  }

  long long total_trad = 0;
  long long total_weyl = 0;
  int samples = 0;

  auto global_start = std::chrono::steady_clock::now();
  std::cout << "Running Filtering Benchmarks (60s)..." << std::endl;

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - global_start)
            .count() >= 60)
      break;

    // Shuffle parcial para simular variavilidad de red
    traffic[rand() % NUM_PACKETS].src_ip = rng();

    long long t_trad = benchmark_firewall_branching(traffic);
    long long t_weyl = benchmark_firewall_bitwise(traffic);

    total_trad += t_trad;
    total_weyl += t_weyl;
    samples++;

    if (samples % 5 == 0) {
      std::cout << "Muestra " << samples << ": Trad(If-Chain)=" << t_trad
                << "us | W(Bitwise)=" << t_weyl
                << "us | Speedup: " << (double)t_trad / t_weyl << "x"
                << std::endl;
    }
  }

  double avg_trad = (double)total_trad / samples;
  double avg_weyl = (double)total_weyl / samples;
  double speedup = avg_trad / avg_weyl;
  double improvement = (avg_trad - avg_weyl) / avg_trad * 100.0;

  std::cout << "\n=== RESULTADOS FIREWALL (PACKET FILTERING) ===" << std::endl;
  std::cout << "Throughput Traditional: "
            << (NUM_PACKETS / (avg_trad / 1000000.0)) / 1000000.0
            << " Mpps (Million Packets/sec)" << std::endl;
  std::cout << "Throughput W:     "
            << (NUM_PACKETS / (avg_weyl / 1000000.0)) / 1000000.0 << " Mpps"
            << std::endl;
  std::cout << "LATENCY REDUCTION:      " << improvement << "%" << std::endl;
  std::cout << "SPEEDUP FACTOR:         " << speedup << "x" << std::endl;

  std::ofstream jsonFile("w_firewall_results.json");
  if (jsonFile.is_open()) {
    jsonFile << "{\n";
    jsonFile << "  \"test_type\": \"Network Packet Filtering (Firewall)\",\n";
    jsonFile << "  \"packets_per_batch\": " << NUM_PACKETS << ",\n";
    jsonFile << "  \"active_rules\": " << RULES_COUNT << ",\n";
    jsonFile << "  \"avg_traditional_latency_us\": " << avg_trad << ",\n";
    jsonFile << "  \"avg_w_latency_us\": " << avg_weyl << ",\n";
    jsonFile << "  \"throughput_traditional_mpps\": "
             << (NUM_PACKETS / (avg_trad / 1000000.0)) / 1000000.0 << ",\n";
    jsonFile << "  \"throughput_w_mpps\": "
             << (NUM_PACKETS / (avg_weyl / 1000000.0)) / 1000000.0 << ",\n";
    jsonFile << "  \"speedup_factor\": " << speedup << "\n";
    jsonFile << "}" << std::endl;
    jsonFile.close();
  }

  return 0;
}
