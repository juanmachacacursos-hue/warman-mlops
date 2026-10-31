"""Pruebas de la capa de la API con TestClient."""
import pytest
from fastapi.testclient import TestClient

from warman import api, predict
from warman.predict import ModeloNoDisponible

LECTURA = {
    "vibracion_rms_mm_s": 2.6, "temp_rodamiento_acople_c": 52.3, "temp_rodamiento_libre_c": 47.8,
    "caudal_agua_sello_m3h": 3.0, "presion_agua_sello_kpa": 550.0, "presion_descarga_kpa": 470.0,
    "velocidad_rpm": 398.0, "potencia_kw": 649.0, "densidad_pulpa_tm3": 1.65,
    "nivel_cajon_pct": 58.0, "horas_desde_cambio_impulsor": 1500.0,
}


@pytest.fixture(scope="module")
def cliente():
    with TestClient(api.app) as c:
        yield c


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "modelo_cargado": True}


def test_model_info(cliente):
    r = cliente.get("/model-info")
    assert r.status_code == 200
    assert "version" in r.json() and "metricas_test" in r.json()


def test_predict_valido(cliente):
    r = cliente.post("/predict", json=LECTURA)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["estado"] == "Normal"
    assert 0 <= cuerpo["confianza"] <= 1


@pytest.mark.parametrize("cambio", [
    {"vibracion_rms_mm_s": -5},          # fuera de rango
    {"densidad_pulpa_tm3": "alta"},      # tipo incorrecto
    {"campo_inventado": 1},              # campo que no existe
])
def test_predict_invalido_da_422(cliente, cambio):
    r = cliente.post("/predict", json={**LECTURA, **cambio})
    assert r.status_code == 422


def test_predict_campo_faltante_da_422(cliente):
    incompleto = {k: v for k, v in LECTURA.items() if k != "velocidad_rpm"}
    assert cliente.post("/predict", json=incompleto).status_code == 422


def test_sin_modelo_da_503(cliente, monkeypatch):
    def sin_modelo(_):
        raise ModeloNoDisponible("modelo ausente")
    monkeypatch.setattr(api, "predecir", sin_modelo)
    assert cliente.post("/predict", json=LECTURA).status_code == 503
    assert predict.cargar_modelo() is not None  # el modelo real sigue intacto
