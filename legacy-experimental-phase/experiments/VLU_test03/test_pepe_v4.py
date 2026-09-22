import pandas as pd
import numpy as np
import requests
import time
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ==========================================
# 🔒 CONSTANTES DE FÍSICA VOLUMÉTRICA (VL)
# ==========================================
VALOR_MAGICO_DCPI = 12.0  # La Constante Maestra
K_UNIVERSAL = 0.2816
ALPHA_UNIVERSAL = 0.2639
VISCOSIDAD_HIGGS = 0.85   # Fricción del vacío
Z_UMBRAL_PROFUNDO = 1.20  # Valor teórico inicial (será recalibrado)

# ==========================================
# 📡 MÓDULO DE INGESTA DE DATOS
# ==========================================
def fetch_massive_history(symbol="PEPEUSDT", total_candles=15000):
    console.print(f"[bold cyan][>] Descargando {total_candles} velas de {symbol} para análisis de profundidad...[/bold cyan]")
    all_data = []
    end_time = int(time.time() * 1000)
    
    while len(all_data) < total_candles:
        limit = min(1000, total_candles - len(all_data))
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}&endTime={end_time}"
        try:
            res = requests.get(url).json()
            if not res or isinstance(res, dict): break
            all_data = res + all_data
            end_time = res[0][0] - 1
            time.sleep(0.05)
        except Exception as e:
            console.print(f"[red]Error de conexión: {e}[/red]")
            break
            
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qa', 'tr', 'tb', 'tq', 'ig'])
    df[['close', 'volume']] = df[['close', 'volume']].astype(float)
    return df

# ==========================================
# 🎯 CALIBRADOR DE PRECISIÓN "DÍAZ-CANO 12"
# ==========================================
def find_true_z(df):
    """
    Busca la equivalencia exacta entre el DCPI 12 y la Profundidad Z real.
    """
    # Filtramos solo los momentos donde DCPI toca tu "Número Mágico" (con tolerancia 0.1)
    umbral_exito = df[(df['dcpi'] >= 11.9) & (df['dcpi'] <= 12.1)]
    
    if not umbral_exito.empty:
        z_real = umbral_exito['vector_z'].mean()
        z_std = umbral_exito['vector_z'].std()
        
        console.print(f"\n[bold green]✅ CALIBRACIÓN COMPLETADA (DÍAZ-CANO 12)[/bold green]")
        console.print(f"Para PEPE, tu presión de [bold white]12.0[/] equivale a un Vector Z de: [bold cyan]{z_real:.4f}[/]")
        console.print(f"Desviación de Profundidad: ±{z_std:.4f}")
        return z_real
    else:
        console.print("[bold red]⚠️ ADVERTENCIA: No se encontraron eventos exactos de DCPI=12 en la muestra.[/bold red]")
        return 1.20 # Fallback teórico

# ==========================================
# 🔬 MOTOR DE ANÁLISIS DE PROFUNDIDAD
# ==========================================
def run_deep_calibration(df):
    # 1. Física Volumétrica (DCPI)
    delta_p = df['close'].diff().abs()
    flujo_bruto = delta_p * (df['volume'] ** ALPHA_UNIVERSAL) 
    flujo_avg = flujo_bruto.rolling(window=100).mean() 
    df['dcpi'] = (flujo_bruto / (flujo_avg + 1e-9)) / K_UNIVERSAL
    
    # 2. Navier-Stokes Z (Profundidad)
    # Z = (DCPI * Viscosidad) / log(Volumen)
    df['vector_z'] = (df['dcpi'] * VISCOSIDAD_HIGGS) / (np.log(df['volume'] + 1) + 1e-9)
    
    # Limpieza de datos
    df_clean = df.dropna().copy()
    
    # 3. Ejecutar Calibración del 12
    true_z_threshold = find_true_z(df_clean)
    
    # 4. Estadísticas Generales
    correlacion = df_clean['dcpi'].corr(df_clean['vector_z'])
    
    # Ratio de 'Deep Impact' usando el Z CALIBRADO
    pumps_presion = df_clean[df_clean['dcpi'] > VALOR_MAGICO_DCPI]
    pumps_profundos = pumps_presion[pumps_presion['vector_z'] > true_z_threshold]
    coincidencia_pct = (len(pumps_profundos) / len(pumps_presion) * 100) if len(pumps_presion) > 0 else 0

    # --- REPORTE DE CONSOLA ---
    console.print(f"\n[bold magenta]🔬 AUTOPSIA DE FASE (Muestra: {len(df_clean)} min)[/bold magenta]")
    
    table = Table(title="MÉTRICAS DE PROFUNDIDAD Z", box=box.ROUNDED, style="white")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right")
    
    table.add_row("Correlación DCPI ↔ Vector Z", f"{correlacion:.4f}")
    table.add_row("Percentil del '12' (DCPI)", f"{(df_clean['dcpi'] < VALOR_MAGICO_DCPI).mean()*100:.2f}%")
    table.add_row("Z Objetivo (Calibrado)", f"[bold green]{true_z_threshold:.4f}[/]")
    table.add_row("Coherencia Laminar (con Z Real)", f"{coincidencia_pct:.1f}%")
    
    console.print(table)

    # VEREDICTO FINAL PARA LA TESIS
    if correlacion > 0.90:
        console.print(f"[bold green]🚀 VEREDICTO: El sistema es COHERENTE. Usar Z > {true_z_threshold:.4f} como filtro de entrada.[/bold green]")
    else:
        console.print("[bold red]❌ VEREDICTO: Desacople de Fase. El volumen no acompaña a la presión.[/bold red]")

if __name__ == "__main__":
    # Usamos PEPE para mantener la línea de investigación
    df_history = fetch_massive_history("PEPEUSDT", 20000)
    if not df_history.empty:
        run_deep_calibration(df_history)