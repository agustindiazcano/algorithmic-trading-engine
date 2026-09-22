import time
import math
import random
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich import box

# --- TUS ESTILOS PERSONALIZADOS ---
MODERN_SPINNERS = {
    "grok_snake": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "pulse_block": ["░", "▒", "▓", "█", "▓", "▒", "░"],
}

def generar_tabla_trades(step, spinner_char):
    """Genera la tabla central con datos simulados"""
    table = Table(box=box.SIMPLE_HEAVY, expand=True, border_style="bright_black")
    
    # Definimos columnas
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"{spinner_char} Par", style="magenta")
    table.add_column("Precio Entrada", justify="right")
    table.add_column("Precio Actual", justify="right")
    table.add_column("PNL %", justify="right")
    table.add_column("Estado", justify="center")

    # Simulamos datos que cambian
    variacion = math.sin(step * 0.5) * 2 # Movimiento oscilatorio
    
    # Fila 1: Una operación ganadora
    pnl_1 = 12.5 + variacion
    estilo_pnl_1 = "green" if pnl_1 > 0 else "red"
    table.add_row(
        "001", "PEPEUSDT", "0.00001200", f"{0.00001350 + (variacion/100000):.8f}", 
        f"[{estilo_pnl_1}]{pnl_1:+.2f}%[/{estilo_pnl_1}]", "🚀 FLYING"
    )

    # Fila 2: Una operación perdedora/estática
    pnl_2 = -2.4 + (variacion * 0.5)
    estilo_pnl_2 = "green" if pnl_2 > 0 else "red"
    table.add_row(
        "002", "BTCUSDT", "64200.00", f"{63500 + (variacion*100):.2f}", 
        f"[{estilo_pnl_2}]{pnl_2:+.2f}%[/{estilo_pnl_2}]", "⚠️ HOLD"
    )

    return table

def generar_header(step, duracion_total):
    """Genera el panel superior con barra de progreso y efecto pulsante"""
    
    # 1. Lógica del Spinner (usando tu diccionario)
    frames = MODERN_SPINNERS["grok_snake"]
    char = frames[step % len(frames)]
    
    # 2. Lógica del Pulso de Color (Grok Style)
    opacidad = (math.sin(step * 0.2) + 1) / 2
    color_titulo = "bright_cyan" if opacidad > 0.5 else "cyan"
    
    # 3. Barra de progreso manual
    progreso = min(100, int((step / (duracion_total * 10)) * 100)) # Ajuste aprox
    barra_width = 30
    llenos = int((progreso / 100) * barra_width)
    barra_str = f"[{'━' * llenos}{' ' * (barra_width - llenos)}]"

    contenido = Text.from_markup(
        f"[{color_titulo}]Validación Algorítmica en Curso[/{color_titulo}]\n"
        f"{char} Procesando Velas... {barra_str} {progreso}%"
    )
    
    return Panel(contenido, title="🤖 ENGINE V2.0", border_style="blue")

def ejecutar_dashboard_rich(duracion=20):
    console = Console()
    console.clear()
    
    # Layout simple: Header arriba, Tabla abajo
    layout = Layout()
    layout.split(
        Layout(name="header", size=4),
        Layout(name="main", ratio=1)
    )

    # Context Manager "Live" es el corazón de Rich para cosas en vivo
    with Live(layout, refresh_per_second=10, screen=False) as live:
        
        start_time = time.time()
        step = 0
        
        while (time.time() - start_time) < duracion:
            # Actualizamos las partes del layout
            layout["header"].update(generar_header(step, duracion))
            layout["main"].update(generar_tabla_trades(step, MODERN_SPINNERS["grok_snake"][step % len(MODERN_SPINNERS["grok_snake"])]))
            
            time.sleep(0.1)
            step += 1
            
    print("[bold green]✅ PROCESO FINALIZADO CORRECTAMENTE[/bold green]")

if __name__ == "__main__":
    ejecutar_dashboard_rich(20)