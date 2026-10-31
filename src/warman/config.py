"""Configuración central del proyecto.

Todos los parámetros (rutas, semilla, variables, rangos, umbral de calidad) viven
aquí. Ningún otro archivo debe tener "números mágicos": si hay que cambiar algo,
se cambia en este archivo y todo el proyecto lo respeta.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data"
DIR_RAW = DIR_DATOS / "raw"
DIR_PROCESADO = DIR_DATOS / "processed"
DIR_MODELOS = RAIZ / "models"

ARCHIVO_SENSORES = DIR_RAW / "sensores_bombas_warman.csv"
ARCHIVO_ORDENES = DIR_RAW / "ordenes_trabajo.csv"
ARCHIVO_DATASET = DIR_PROCESADO / "dataset.csv"
ARCHIVO_MODELO = DIR_MODELOS / "modelo.joblib"
ARCHIVO_METRICAS = DIR_MODELOS / "metrics.json"

# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
SEMILLA = 2026

# ----------------------------------------------------------------------------
# Simulación de planta: 4 bombas Warman de alimentación a ciclones
# ----------------------------------------------------------------------------
BOMBAS = ["BB-01", "BB-02", "BB-03", "BB-04"]
FECHA_INICIO = "2024-01-01 00:00"
FECHA_FIN = "2025-12-31 23:00"
PARADAS_PLANTA = ["2024-04-14", "2024-10-06", "2025-04-13", "2025-10-05"]

# ----------------------------------------------------------------------------
# Variables de entrada del modelo y su rango físico válido (mínimo, máximo)
# Se usan en tres lugares: limpieza de datos, validación de datos y la API.
# ----------------------------------------------------------------------------
RANGOS = {
    "vibracion_rms_mm_s": (0.0, 50.0),
    "temp_rodamiento_acople_c": (-10.0, 150.0),
    "temp_rodamiento_libre_c": (-10.0, 150.0),
    "caudal_agua_sello_m3h": (0.0, 20.0),
    "presion_agua_sello_kpa": (0.0, 1500.0),
    "presion_descarga_kpa": (0.0, 1500.0),
    "velocidad_rpm": (0.0, 600.0),
    "potencia_kw": (0.0, 1500.0),
    "densidad_pulpa_tm3": (1.0, 2.5),
    "nivel_cajon_pct": (0.0, 100.0),
    "horas_desde_cambio_impulsor": (0.0, 10000.0),
}
VARIABLES = list(RANGOS)
OBJETIVO = "estado"

# Clases que predice el modelo (estado de la bomba)
CLASES = [
    "Normal",
    "Cavitacion",
    "Riesgo rodamiento",
    "Riesgo sello",
    "Desgaste impulsor",
]

# Umbrales con los que se etiqueta el estado (índice de daño 0 a 1)
UMBRAL_DANO_RODAMIENTO = 0.35
UMBRAL_DANO_SELLO = 0.20
UMBRAL_DESGASTE_IMPULSOR = 0.85

# Un sensor que repite exactamente el mismo valor este número de horas se
# considera "congelado" y sus lecturas se descartan.
HORAS_SENSOR_CONGELADO = 12

# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
# División temporal: se entrena con el pasado y se evalúa con el futuro.
FIN_ENTRENAMIENTO = "2025-04-01"   # train: antes de esta fecha
FIN_VALIDACION = "2025-07-01"      # validación: abr-jun 2025 / test: jul-dic 2025

PARAMS_RANDOM_FOREST = {
    "n_estimators": 200,
    "max_depth": 14,
    "min_samples_leaf": 3,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}
PARAMS_REGRESION_LOGISTICA = {
    "max_iter": 2000,
    "class_weight": "balanced",
}

# Puerta de calidad: si el modelo no alcanza este F1-macro en test, NO se guarda.
F1_MACRO_MINIMO = 0.80

VERSION_MODELO = "1.0.0"
