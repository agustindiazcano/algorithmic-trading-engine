import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
ARCHIVO_DATA = "PEPE_MASTER_VL_500K.csv"
VENTANA_MEMORIA = 180  # El bot "recuerda" las últimas 3 horas para definir qué es alto y bajo

def escanear_bandas_dinamicas():
    print(f"📂 Cargando y procesando {ARCHIVO_DATA}...")
    try:
        df = pd.read_csv(ARCHIVO_DATA)
    except:
        return

    # Limpieza
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dcpi'] = pd.to_numeric(df['dcpi'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    print(f"🧠 Calculando Curvas Dinámicas (Ventana: {VENTANA_MEMORIA} velas)...")

    # --- CÁLCULO DE CURVAS VIVAS (ROLLING QUANTILES) ---
    # Esto es lo que hace que la línea se mueva con el mercado
    df['p90_dynamic'] = df['dcpi'].rolling(window=VENTANA_MEMORIA).quantile(0.90)
    df['p95_dynamic'] = df['dcpi'].rolling(window=VENTANA_MEMORIA).quantile(0.95)
    df['p99_dynamic'] = df['dcpi'].rolling(window=VENTANA_MEMORIA).quantile(0.99)
    
    # Inyectamos Fuerza para confirmar
    df['v_price'] = df['close'].diff()
    df['a_price'] = df['v_price'].diff()
    ALPHA = 0.2639
    df['fuerza_vl'] = (df['vol'] ** ALPHA) * df['a_price']

    # --- DETECTOR DE NIVELES ---
    # Nivel 3: Rompió P99 (Saturación Total)
    # Nivel 2: Rompió P95 (Flujo Fuerte)
    # Nivel 1: Rompió P90 (Actividad)
    
    # Filtramos solo donde hay cruces (Para no ver todo el log)
    # Condición: DCPI actual mayor a P90 Y Precio subiendo (para buscar Longs o Shorts de agotamiento)
    eventos = df[df['dcpi'] > df['p90_dynamic']].copy()
    
    # Clasificación Vectorizada de Niveles
    eventos['NIVEL'] = 'N1 (P90)'
    eventos.loc[eventos['dcpi'] > eventos['p95_dynamic'], 'NIVEL'] = 'N2 (P95)'
    eventos.loc[eventos['dcpi'] > eventos['p99_dynamic'], 'NIVEL'] = 'N3 (P99)'

    # Guardado de CSV
    output_filename = "PEPE_MAGIC_SIGNALS.csv"
    print(f"\n💾 Guardando {len(eventos)} eventos detectados en '{output_filename}'...")
    
    # Seleccionamos y ordenamos columnas para el reporte
    columnas_reporte = ['ts', 'close', 'dcpi', 'p90_dynamic', 'p95_dynamic', 'p99_dynamic', 'NIVEL', 'fuerza_vl']
    
    # Verificar si 'ts' existe, si no usar el índice
    if 'ts' not in eventos.columns:
        eventos = eventos.reset_index() # Recupera el índice como columna si es necesario
        if 'ts' not in eventos.columns:
             # Si no hay TS, intentamos usar el índice como referencia
             eventos.rename(columns={'index': 'vela_id'}, inplace=True)
             columnas_reporte = ['vela_id', 'close', 'dcpi', 'p90_dynamic', 'p95_dynamic', 'p99_dynamic', 'NIVEL', 'fuerza_vl']

    eventos[columnas_reporte].to_csv(output_filename, index=False)
    
    print("✅ Archivo guardado con éxito.")
    print("\n💡 INTERPRETACIÓN DE NIVELES:")
    print("🔸 N1 (P90): El mercado se despierta. Scalping rápido.")
    print("🔶 N2 (P95): Tren Bala en marcha. Entrar a favor de la fuerza.")
    print("🔥 N3 (P99): SATURACIÓN. Si Fuerza < 0 -> SHORT INMEDIATO.")

if __name__ == "__main__":
    escanear_bandas_dinamicas()