"""Etapa 5 — Interfaz web (Gradio) para el supervisor de mantenimiento.

La interfaz NO carga el modelo: envía la lectura a la API por HTTP y muestra
la respuesta. Así la interfaz y el modelo se despliegan por separado.

Comando:  uv run python app/gradio_app.py   -> http://localhost:7860
Variable de entorno API_URL: dirección de la API (por defecto http://localhost:8000)
"""
import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PUERTO = int(os.getenv("PORT", "7860"))

# Orden de las variables: igual que en la API
CAMPOS = [
    # (nombre, etiqueta, mínimo, máximo, paso, valor normal)
    ("vibracion_rms_mm_s", "Vibración RMS (mm/s)", 0, 20, 0.1, 2.6),
    ("temp_rodamiento_acople_c", "Temp. rodamiento lado acople (°C)", 20, 120, 0.5, 52.3),
    ("temp_rodamiento_libre_c", "Temp. rodamiento lado libre (°C)", 20, 120, 0.5, 47.8),
    ("caudal_agua_sello_m3h", "Caudal de agua de sello (m³/h)", 0, 12, 0.1, 3.0),
    ("presion_agua_sello_kpa", "Presión de agua de sello (kPa)", 300, 700, 1, 550),
    ("presion_descarga_kpa", "Presión de descarga (kPa)", 250, 600, 1, 470),
    ("velocidad_rpm", "Velocidad del variador (rpm)", 350, 460, 0.5, 398),
    ("potencia_kw", "Potencia del motor (kW)", 400, 850, 1, 649),
    ("densidad_pulpa_tm3", "Densidad de pulpa (t/m³)", 1.40, 1.90, 0.01, 1.65),
    ("nivel_cajon_pct", "Nivel del cajón de alimentación (%)", 0, 100, 1, 58),
    ("horas_desde_cambio_impulsor", "Horas desde el último cambio de impulsor", 0, 4500, 10, 1500),
]

# Lecturas típicas de cada estado (medianas del periodo de prueba)
EJEMPLOS = [
    [2.6, 52.3, 47.8, 3.0, 550, 470, 398, 649, 1.65, 58, 1500],     # Normal
    [4.8, 52.2, 47.7, 3.0, 479, 401, 394, 528, 1.66, 18, 1290],     # Cavitación
    [6.4, 65.7, 50.1, 3.0, 547, 468, 404, 672, 1.66, 58, 2120],     # Rodamiento
    [2.5, 51.8, 47.4, 5.6, 529, 476, 393, 650, 1.66, 57, 910],      # Sello
    [3.0, 53.6, 48.5, 3.0, 536, 456, 415, 651, 1.65, 58, 2810],     # Impulsor
]

COLORES = {
    "Normal": ("#0ca30c", "🟢"),
    "Cavitacion": ("#fab219", "🟡"),
    "Riesgo sello": ("#ec835a", "🟠"),
    "Desgaste impulsor": ("#ec835a", "🟠"),
    "Riesgo rodamiento": ("#d03b3b", "🔴"),
}


def tarjeta(titulo: str, detalle: str, color: str, icono: str) -> str:
    return (f"<div style='border-left:8px solid {color};padding:14px 18px;border-radius:8px;"
            f"background:rgba(127,127,127,0.08)'>"
            f"<div style='font-size:1.5em;font-weight:600'>{icono} {titulo}</div>"
            f"<div style='margin-top:6px'>{detalle}</div></div>")


def diagnosticar(*valores):
    lectura = {campo[0]: float(v) for campo, v in zip(CAMPOS, valores)}
    try:
        r = httpx.post(f"{API_URL}/predict", json=lectura, timeout=15)
    except httpx.HTTPError:
        return (tarjeta("La API no responde",
                        f"No se pudo conectar con <code>{API_URL}</code>. "
                        "Verifica que la API esté levantada e inténtalo de nuevo.",
                        "#898781", "⚠️"), None)
    if r.status_code == 422:
        return tarjeta("Datos inválidos", str(r.json().get("detail")), "#898781", "⚠️"), None
    if r.status_code != 200:
        return (tarjeta(f"Error {r.status_code}", r.json().get("detail", "Error en la API"),
                        "#898781", "⚠️"), None)

    res = r.json()
    color, icono = COLORES.get(res["estado"], ("#898781", "⚪"))
    detalle = (f"Confianza del modelo: <b>{res['confianza']:.0%}</b><br>"
               f"<b>Qué hacer:</b> {res['recomendacion']}<br>"
               f"<small>Modelo v{res['version_modelo']}</small>")
    return tarjeta(res["estado"], detalle, color, icono), res["probabilidades"]


def estado_api() -> str:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=10).json()
        return "🟢 API conectada" if r.get("modelo_cargado") else "🟠 API sin modelo cargado"
    except httpx.HTTPError:
        return f"🔴 API no disponible en {API_URL}"


with gr.Blocks(title="Estado de bombas Warman") as demo:
    gr.Markdown(
        "# Diagnóstico de bombas Warman de alimentación a ciclones\n"
        "Ingresa la lectura actual de los sensores de la bomba. El sistema diagnostica su "
        "estado y recomienda la acción de mantenimiento.")
    conexion = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=3):
            entradas = []
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Rodamientos**")
                    for c in CAMPOS[0:3]:
                        entradas.append(gr.Slider(c[2], c[3], value=c[5], step=c[4], label=c[1]))
                    gr.Markdown("**Sello**")
                    for c in CAMPOS[3:5]:
                        entradas.append(gr.Slider(c[2], c[3], value=c[5], step=c[4], label=c[1]))
                with gr.Column():
                    gr.Markdown("**Hidráulica y proceso**")
                    for c in CAMPOS[5:10]:
                        entradas.append(gr.Slider(c[2], c[3], value=c[5], step=c[4], label=c[1]))
                    gr.Markdown("**Impulsor**")
                    for c in CAMPOS[10:]:
                        entradas.append(gr.Slider(c[2], c[3], value=c[5], step=c[4], label=c[1]))
            boton = gr.Button("Diagnosticar bomba", variant="primary")
        with gr.Column(scale=2):
            resultado = gr.HTML()
            probabilidades = gr.Label(label="Probabilidad de cada estado", num_top_classes=5)

    gr.Examples(EJEMPLOS, inputs=entradas, label="Lecturas típicas por estado (clic para cargar)",
                example_labels=["🟢 Normal", "🟡 Cavitación", "🔴 Rodamiento dañado",
                                "🟠 Sello gastado", "🟠 Impulsor desgastado"])

    boton.click(diagnosticar, inputs=entradas, outputs=[resultado, probabilidades])
    demo.load(estado_api, outputs=conexion)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PUERTO)
