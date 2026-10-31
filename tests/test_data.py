"""Pruebas de la capa de DATOS: limpieza y validación."""
import pandas as pd
import pytest

from warman import config
from warman.data import DatosInvalidos, cargar_dataset, limpiar, validar


def lectura_normal(**cambios):
    fila = {
        "fecha_hora": pd.Timestamp("2025-01-01"), "bomba_id": "BB-01", "en_operacion": 1,
        "vibracion_rms_mm_s": 2.6, "temp_rodamiento_acople_c": 52.3,
        "temp_rodamiento_libre_c": 47.8, "caudal_agua_sello_m3h": 3.0,
        "presion_agua_sello_kpa": 550.0, "presion_descarga_kpa": 470.0,
        "velocidad_rpm": 398.0, "potencia_kw": 649.0, "densidad_pulpa_tm3": 1.65,
        "nivel_cajon_pct": 58.0, "horas_desde_cambio_impulsor": 1500.0, "estado": "Normal",
    }
    fila.update(cambios)
    return fila


def test_limpiar_elimina_valores_imposibles():
    df = pd.DataFrame([lectura_normal(), lectura_normal(temp_rodamiento_acople_c=999.0),
                       lectura_normal(vibracion_rms_mm_s=-1.0)])
    limpio = limpiar(df)
    assert len(limpio) == 1
    assert limpio["temp_rodamiento_acople_c"].max() < 150


def test_limpiar_descarta_bomba_detenida():
    df = pd.DataFrame([lectura_normal(), lectura_normal(en_operacion=0)])
    assert len(limpiar(df)) == 1


def test_limpiar_detecta_sensor_congelado():
    horas = pd.date_range("2025-01-01", periods=30, freq="h")
    filas = [lectura_normal(fecha_hora=h, vibracion_rms_mm_s=2.87) for h in horas]
    assert len(limpiar(pd.DataFrame(filas))) == 0


def test_validar_rechaza_nulos_y_rangos():
    df = cargar_dataset().head(2000).copy()
    df.loc[df.index[0], "vibracion_rms_mm_s"] = None
    df.loc[df.index[1], "densidad_pulpa_tm3"] = 9.9
    with pytest.raises(DatosInvalidos):
        validar(df)


def test_validar_rechaza_columna_faltante():
    df = cargar_dataset().drop(columns=["nivel_cajon_pct"])
    with pytest.raises(DatosInvalidos, match="Faltan columnas"):
        validar(df)


def test_dataset_procesado_es_valido():
    df = cargar_dataset()
    assert set(config.VARIABLES).issubset(df.columns)
    assert set(df[config.OBJETIVO].unique()) == set(config.CLASES)
    assert df[config.VARIABLES].notna().all().all()
