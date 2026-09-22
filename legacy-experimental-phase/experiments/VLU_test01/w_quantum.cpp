#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>


// --- CONFIGURACIÓN "QUANTUM" ---
// ¡CUIDADO! Si subís esto a 30 o 32, el método tradicional puede tardar MINUTOS
// u HORAS. 28 es el "Sweet Spot" para que duela pero termine en unos segundos.
const int NUM_ITEMS = 28;
const int CAPACITY = 1000; // Capacidad de la mochila

struct Item {
  int weight;
  int value;
};

// Generador de Items
std::vector<Item> generate_loot() {
  std::vector<Item> loot;
  // Semilla fija para repetibilidad
  srand(42);
  for (int i = 0; i < NUM_ITEMS; ++i) {
    // Pesos y valores aleatorios pero molestos
    loot.push_back({(rand() % 50) + 10, (rand() % 100) + 10});
  }
  return loot;
}

// ==========================================
// 1. MÉTODO TRADICIONAL (Recursive Branching)
// ==========================================
// Esto simula la "Explosión Combinatoria".
// Cada llamada abre 2 ramas nuevas. Es O(2^N).
// La CPU salta de un lado a otro en la memoria (Stack Thrashing).
int knapsack_recursive(const std::vector<Item> &items, int n, int w) {
  // Branching base
  if (n == 0 || w == 0)
    return 0;

  // Branching de decisión
  if (items[n - 1].weight > w) {
    return knapsack_recursive(items, n - 1, w);
  } else {
    // La gran duda existencial del procesador:
    // ¿Gano más incluyéndolo o excluyéndolo?
    // Calcula AMBOS universos paralelos secuencialmente.
    int include = items[n - 1].value +
                  knapsack_recursive(items, n - 1, w - items[n - 1].weight);
    int exclude = knapsack_recursive(items, n - 1, w);

    // Un "max" normal usa un IF interno (cmov)
    return (include > exclude) ? include : exclude;
  }
}

// ==========================================
// 2. MÉTODO W (Arithmetic Grid)
// ==========================================
// Aplanamos el árbol de decisiones en una tabla 1D.
// Usamos aritmética branchless para eliminar los IFs de decisión.
int knapsack_w(const std::vector<Item> &items, int capacity) {
  // Creamos un "universo plano" de memoria
  std::vector<int> dp(capacity + 1, 0);

  for (const auto &item : items) {
    // Iteramos hacia atrás para evitar usar el mismo item dos veces en el mismo
    // ciclo Esto es acceso lineal a memoria (Cache Friendly)
    for (int w = capacity; w >= item.weight; --w) {

      int exclude_val = dp[w];
      int include_val = dp[w - item.weight] + item.value;

      // W MAGIC: Branchless Max
      // Max(a, b) se puede calcular matemáticamente como:
      // (a + b + abs(a - b)) / 2
      // Esto elimina el salto condicional del procesador.

      // Nota: Usamos abs() de la librería estándar que suele compilarse a una
      // instrucción sin saltos.
      dp[w] =
          (exclude_val + include_val + std::abs(exclude_val - include_val)) / 2;
    }
  }
  return dp[capacity];
}

int main() {
  std::cout << "=== W SYSTEMS: QUANTUM SIMULATION ===" << std::endl;
  std::cout << "Problema: NP-Hard Knapsack (Combinatorial Stress)" << std::endl;
  std::cout << "Items: " << NUM_ITEMS << " (Espacio de busqueda: 2^"
            << NUM_ITEMS << " combinaciones)" << std::endl;

  auto loot = generate_loot();

  // --- TEST TRADICIONAL (Doloroso) ---
  std::cout
      << "\nIniciando Metodo Tradicional (Recursivo)... PREPARATE A ESPERAR..."
      << std::endl;
  auto start_trad = std::chrono::high_resolution_clock::now();

  int result_trad = knapsack_recursive(loot, NUM_ITEMS, CAPACITY);

  auto end_trad = std::chrono::high_resolution_clock::now();
  long long duration_trad =
      std::chrono::duration_cast<std::chrono::microseconds>(end_trad -
                                                            start_trad)
          .count();

  std::cout << ">> Resultado: $" << result_trad << std::endl;
  std::cout << ">> Tiempo CPU: " << duration_trad / 1000.0 << " ms"
            << std::endl;

  // --- TEST W (Instantáneo) ---
  std::cout << "\nIniciando Metodo W (Aritmetico Plano)..." << std::endl;
  auto start_weyl = std::chrono::high_resolution_clock::now();

  int result_weyl = knapsack_w(loot, CAPACITY);

  auto end_weyl = std::chrono::high_resolution_clock::now();
  long long duration_weyl =
      std::chrono::duration_cast<std::chrono::microseconds>(end_weyl -
                                                            start_weyl)
          .count();

  std::cout << ">> Resultado: $" << result_weyl << std::endl;
  std::cout << ">> Tiempo CPU: " << duration_weyl / 1000.0 << " ms"
            << std::endl;

  // --- REPORTE ---
  double speedup = (double)duration_trad / duration_weyl;

  std::cout << "\n=== INFORME FINAL ===" << std::endl;
  if (result_trad == result_weyl) {
    std::cout
        << "Verificacion: EXITOSA (Ambos metodos dieron el mismo valor optimo)"
        << std::endl;
  } else {
    std::cout << "Verificacion: FALLIDA (Algo salio mal en la logica)"
              << std::endl;
  }
  std::cout << "SPEEDUP FACTOR: " << speedup << "x" << std::endl;

  // JSON Export
  std::ofstream jsonFile("w_quantum_results.json");
  jsonFile << "{\n";
  jsonFile << "  \"test_type\": \"NP-Hard Combinatorial Stress\",\n";
  jsonFile << "  \"complexity\": \"2^" << NUM_ITEMS << "\",\n";
  jsonFile << "  \"time_traditional_ms\": " << duration_trad / 1000.0 << ",\n";
  jsonFile << "  \"time_w_ms\": " << duration_weyl / 1000.0 << ",\n";
  jsonFile << "  \"speedup\": " << speedup << "\n";
  jsonFile << "}";

  return 0;
}
