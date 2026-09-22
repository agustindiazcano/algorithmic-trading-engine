import pandas as pd
import numpy as np
from tqdm import tqdm # Para ver la barrita y no desesperar

# --- CONSTANTES DE HARDWARE VL ---
ALPHA = 0.2639  # Coeficiente de rozamiento de PEPE
K_UNI = 0.2816  # Constante de normalización del fluido
WINDOW_DF = 60  # Ventana de observación fractal

def higuchi_fd(series, k_max=10):
    """Calcula la Dimensión Fractal de Higuchi (Hardware Puro)"""
    N = len(series)
    if N < 10: return 1.5
    L = []
    x = np.array(series)
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            indices = np.arange(m, N, k)
            if len(indices) < 2: continue
            L_m_k = np.sum(np.abs(np.diff(x[indices])))
            norm = (N - 1) / (len(indices) * k)
            Lk.append(L_m_k * norm / k)
        if Lk: L.append(np.mean(Lk))
    
    if len(L) < 2: return 1.5
    # Ajuste lineal en escala logarítmica
    coeffs = np.polyfit(np.log(1/np.arange(1, len(L) + 1)), np.log(L), 1)
    return coeffs[0]

def procesar_monster_csv(input_file, output_file):
    print(f"📂 Cargando {input_file}...")
    
    # 1. Carga y Limpieza Quirúrgica
    # Forzamos nombres de columnas para que no haya dudas
    df = pd.read_csv(input_file, header=None, 
                     names=['ts', 'close', 'vol', 'dcpi', 'df_vascular'])

    # ¡EL FIX!: Convertimos close y vol a números reales (floats)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    
    # Eliminamos basura (NaNs) que rompan la resta
    df = df.dropna(subset=['close', 'vol']).reset_index(drop=True)
    print(f"✅ Hardware convertido. Procesando {len(df)} velas...")

    # 2. CÁLCULO DCPI (Presión Volumétrica)
    # Ecuación: $DCPI = \frac{| \Delta P | \cdot V^{\alpha}}{\bar{\Phi} \cdot K_{uni}}$
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['vol'] ** ALPHA)
    flujo_avg = flujo_bruto.rolling(window=100).mean()
    
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-12)) / K_UNI
    print("⚡ DCPI inyectado con éxito.")

    # 3. CÁLCULO DF (Higuchi) - LA PARTE PESADA
    print("🌀 Calculando Dimensión Fractal (Higuchi)...")
    close_array = df['close'].values
    df_results = np.full(len(df), 1.5) # Placeholder por defecto

    # Usamos un loop con tqdm para no morir de aburrimiento
    for i in tqdm(range(WINDOW_DF, len(df))):
        ventana = close_array[i - WINDOW_DF : i]
        df_results[i] = higuchi_fd(ventana)

    df['df_vascular'] = df_results

    # 4. GUARDADO FINAL
    df.to_csv(output_file, index=False)
    print(f"🏁 DATASET DOMADO: {output_file}")

if __name__ == "__main__":
    # Poné acá el nombre exacto de tu archivo
    ARCHIVO_ENTRADA = "MASTER_DATA_PEPEUSDT_500K.csv"
    ARCHIVO_SALIDA = "PEPE_MASTER_VL_500K.csv"
    
    procesar_monster_csv(ARCHIVO_ENTRADA, ARCHIVO_SALIDA)