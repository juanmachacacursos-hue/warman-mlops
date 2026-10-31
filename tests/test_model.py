"""Pruebas de la capa del MODELO: calidad mínima, ficha y forma de la salida."""
import json

import pytest

from warman import config
from warman.predict import predecir

NORMAL = {
    "vibracion_rms_mm_s": 2.6, "temp_rodamiento_acople_c": 52.3, "temp_rodamiento_libre_c": 47.8,
    "caudal_agua_sello_m3h": 3.0, "presion_agua_sello_kpa": 550.0, "presion_descarga_kpa": 470.0,
    "velocidad_rpm": 398.0, "potencia_kw": 649.0, "densidad_pulpa_tm3": 1.65,
    "nivel_cajon_pct": 58.0, "horas_desde_cambio_impulsor": 1500.0,
}


@pytest.fixture(scope="module")
def ficha():
    return json.loads(config.ARCHIVO_METRICAS.read_text(encoding="utf-8"))


def test_ficha_tiene_campos_obligatorios(ficha):
    for campo in ["version", "fecha_entrenamiento", "parametros", "metricas_test", "puerta_calidad"]:
        assert campo in ficha


def test_modelo_supera_puerta_de_calidad(ficha):
    assert ficha["metricas_test"]["f1_macro"] >= config.F1_MACRO_MINIMO
    assert ficha["puerta_calidad"]["aprobado"] is True


def test_salida_tiene_forma_correcta():
    res = predecir(NORMAL)
    assert res["estado"] in config.CLASES
    assert set(res["probabilidades"]) == set(config.CLASES)
    assert abs(sum(res["probabilidades"].values()) - 1) < 1e-3
    assert 0 <= res["confianza"] <= 1


@pytest.mark.parametrize("cambios, esperado", [
    ({}, "Normal"),
    ({"vibracion_rms_mm_s": 6.4, "temp_rodamiento_acople_c": 65.7}, "Riesgo rodamiento"),
    ({"caudal_agua_sello_m3h": 5.6, "presion_agua_sello_kpa": 529.0}, "Riesgo sello"),
    ({"vibracion_rms_mm_s": 4.8, "nivel_cajon_pct": 18.0, "presion_descarga_kpa": 401.0,
      "presion_agua_sello_kpa": 479.0, "potencia_kw": 528.0}, "Cavitacion"),
])
def test_modelo_reconoce_casos_tipicos(cambios, esperado):
    assert predecir({**NORMAL, **cambios})["estado"] == esperado
