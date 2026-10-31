"""Etapa 4 — API de inferencia (FastAPI).

Comando:  uv run api        -> http://localhost:8000/docs

Endpoints:
  GET  /health      ¿la API está viva y el modelo cargado?
  GET  /model-info  ficha del modelo (versión, fecha, parámetros, métricas)
  POST /predict     recibe una lectura de sensores y devuelve el estado de la bomba
"""
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from warman import config
from warman.predict import ModeloNoDisponible, cargar_ficha, cargar_modelo, predecir

R = config.RANGOS  # rangos físicos válidos, definidos una sola vez en config.py


class LecturaBomba(BaseModel):
    """Lectura de sensores de UNA bomba en un momento dado.

    Pydantic rechaza (error 422) tipos incorrectos, campos faltantes, campos de
    más y valores fuera del rango físico.
    """
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "vibracion_rms_mm_s": 7.8, "temp_rodamiento_acople_c": 71.5,
        "temp_rodamiento_libre_c": 50.2, "caudal_agua_sello_m3h": 3.0,
        "presion_agua_sello_kpa": 545.0, "presion_descarga_kpa": 470.0,
        "velocidad_rpm": 395.0, "potencia_kw": 650.0, "densidad_pulpa_tm3": 1.65,
        "nivel_cajon_pct": 58.0, "horas_desde_cambio_impulsor": 1200,
    }]})

    vibracion_rms_mm_s: float = Field(ge=R["vibracion_rms_mm_s"][0], le=R["vibracion_rms_mm_s"][1],
                                      description="Vibración global RMS en la caja de rodamientos (mm/s)")
    temp_rodamiento_acople_c: float = Field(ge=R["temp_rodamiento_acople_c"][0],
                                            le=R["temp_rodamiento_acople_c"][1],
                                            description="Temperatura del rodamiento lado acople (°C)")
    temp_rodamiento_libre_c: float = Field(ge=R["temp_rodamiento_libre_c"][0],
                                           le=R["temp_rodamiento_libre_c"][1],
                                           description="Temperatura del rodamiento lado libre (°C)")
    caudal_agua_sello_m3h: float = Field(ge=R["caudal_agua_sello_m3h"][0], le=R["caudal_agua_sello_m3h"][1],
                                         description="Consumo de agua de sello (m³/h)")
    presion_agua_sello_kpa: float = Field(ge=R["presion_agua_sello_kpa"][0], le=R["presion_agua_sello_kpa"][1],
                                          description="Presión del agua de sello (kPa)")
    presion_descarga_kpa: float = Field(ge=R["presion_descarga_kpa"][0], le=R["presion_descarga_kpa"][1],
                                        description="Presión de descarga a ciclones (kPa)")
    velocidad_rpm: float = Field(ge=R["velocidad_rpm"][0], le=R["velocidad_rpm"][1],
                                 description="Velocidad del variador (rpm)")
    potencia_kw: float = Field(ge=R["potencia_kw"][0], le=R["potencia_kw"][1],
                               description="Potencia consumida por el motor (kW)")
    densidad_pulpa_tm3: float = Field(ge=R["densidad_pulpa_tm3"][0], le=R["densidad_pulpa_tm3"][1],
                                      description="Densidad de la pulpa (t/m³)")
    nivel_cajon_pct: float = Field(ge=R["nivel_cajon_pct"][0], le=R["nivel_cajon_pct"][1],
                                   description="Nivel del cajón de alimentación (%)")
    horas_desde_cambio_impulsor: float = Field(ge=R["horas_desde_cambio_impulsor"][0],
                                               le=R["horas_desde_cambio_impulsor"][1],
                                               description="Horas de operación desde el último cambio de impulsor")


class Prediccion(BaseModel):
    estado: str = Field(description="Estado diagnosticado de la bomba")
    confianza: float = Field(description="Probabilidad del estado diagnosticado (0 a 1)")
    probabilidades: dict[str, float]
    recomendacion: str
    version_modelo: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo se carga UNA vez al arrancar, no en cada petición.
    try:
        cargar_modelo()
    except ModeloNoDisponible as e:
        print(f"ADVERTENCIA: {e}")
    yield


app = FastAPI(
    title="API — Estado de bombas Warman",
    description="Mantenimiento predictivo de bombas Warman de alimentación a ciclones.",
    version=config.VERSION_MODELO,
    lifespan=lifespan,
)


def _modelo_cargado() -> bool:
    try:
        cargar_modelo()
        return True
    except ModeloNoDisponible:
        return False


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": _modelo_cargado()}


@app.get("/model-info")
def model_info():
    try:
        return cargar_ficha()
    except ModeloNoDisponible as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=Prediccion)
def predict(lectura: LecturaBomba):
    try:
        return predecir(lectura.model_dump())
    except ModeloNoDisponible as e:
        raise HTTPException(status_code=503, detail=str(e))


def main():
    uvicorn.run("warman.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
