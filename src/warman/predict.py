"""Etapa 3 — Predicción: carga el modelo una sola vez y diagnostica una lectura."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from warman import config

# Qué debe hacer el usuario según el estado diagnosticado
RECOMENDACIONES = {
    "Normal": "Operación normal. Continuar con el monitoreo de rutina.",
    "Cavitacion": "Evento de PROCESO, no de mantenimiento: revisar el nivel del cajón "
                  "de alimentación y la dilución. No programar intervención mecánica.",
    "Riesgo rodamiento": "Programar análisis de vibraciones y cambio de rodamientos en los "
                         "próximos días. Revisar lubricación y temperatura en cada turno.",
    "Riesgo sello": "Revisar prensaestopas y agua de sello. Programar cambio de "
                    "empaquetadura en la siguiente oportunidad.",
    "Desgaste impulsor": "Impulsor y revestimientos cerca del fin de vida. Programar el "
                         "cambio en la siguiente parada; no postergar.",
}


class ModeloNoDisponible(RuntimeError):
    """El archivo del modelo no existe: hay que ejecutar `uv run entrenar`."""


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el modelo desde disco una sola vez (luego queda en memoria)."""
    if not config.ARCHIVO_MODELO.exists():
        raise ModeloNoDisponible(
            f"No existe {config.ARCHIVO_MODELO.name}. Ejecuta `uv run entrenar`.")
    return joblib.load(config.ARCHIVO_MODELO)


def cargar_ficha() -> dict:
    """Devuelve el contenido de models/metrics.json."""
    if not config.ARCHIVO_METRICAS.exists():
        raise ModeloNoDisponible("No existe metrics.json. Ejecuta `uv run entrenar`.")
    return json.loads(config.ARCHIVO_METRICAS.read_text(encoding="utf-8"))


def predecir(lectura: dict) -> dict:
    """Diagnostica el estado de la bomba a partir de una lectura de sensores.

    `lectura` es un diccionario con las variables de config.VARIABLES.
    """
    paquete = cargar_modelo()
    modelo, variables = paquete["modelo"], paquete["variables"]
    X = pd.DataFrame([lectura])[variables]

    probabilidades = modelo.predict_proba(X)[0]
    clases = list(modelo.classes_)
    probs = {c: round(float(p), 4) for c, p in zip(clases, probabilidades)}
    estado = max(probs, key=probs.get)
    return {
        "estado": estado,
        "confianza": probs[estado],
        "probabilidades": probs,
        "recomendacion": RECOMENDACIONES[estado],
        "version_modelo": paquete["version"],
    }
