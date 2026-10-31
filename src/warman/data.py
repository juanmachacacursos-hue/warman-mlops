"""Etapa 1 — Datos: generación, limpieza y validación.

Comando:  uv run preparar-datos

Flujo:
  1. Genera los datos crudos con el simulador (o los carga si ya existen).
  2. Limpia los errores de instrumentación.
  3. VALIDA el resultado en código: si algo no cumple, el proceso se detiene.
  4. Guarda data/processed/dataset.csv, que es lo que usa el entrenamiento.
"""
import pandas as pd

from warman import config
from warman.simulador import simular


class DatosInvalidos(ValueError):
    """Se lanza cuando el dataset no pasa la validación."""


def cargar_crudos():
    """Genera los datos crudos con el simulador (semilla fija) y los guarda en data/raw.

    Se regeneran siempre (toma unos segundos) para que un cambio en config.py
    nunca quede con datos viejos. Con la misma semilla salen siempre iguales.
    """
    sensores, ordenes = simular(config.SEMILLA)
    config.DIR_RAW.mkdir(parents=True, exist_ok=True)
    sensores.to_csv(config.ARCHIVO_SENSORES, index=False)
    ordenes.to_csv(config.ARCHIVO_ORDENES, index=False)
    sensores = pd.read_csv(config.ARCHIVO_SENSORES, parse_dates=["fecha_hora"])
    ordenes = pd.read_csv(config.ARCHIVO_ORDENES, parse_dates=["fecha_inicio", "fecha_fin"])
    return sensores, ordenes


def _marcar_sensor_congelado(serie: pd.Series, horas: int) -> pd.Series:
    """True en las lecturas que repiten exactamente el mismo valor >= `horas` seguidas."""
    grupo = (serie != serie.shift()).cumsum()
    largo = serie.groupby(grupo).transform("size")
    return largo >= horas


def limpiar(sensores: pd.DataFrame) -> pd.DataFrame:
    """Limpia los datos crudos y deja solo lo que el modelo necesita."""
    df = sensores.copy()

    # 1) El modelo diagnostica bombas EN OPERACIÓN. Una bomba detenida no se evalúa.
    df = df[df["en_operacion"] == 1]

    # 2) Valores fuera del rango físico (999 °C, -1 mm/s, 9999 kPa) -> vacío.
    for col, (minimo, maximo) in config.RANGOS.items():
        fuera = (df[col] < minimo) | (df[col] > maximo)
        df.loc[fuera, col] = float("nan")

    # 3) Sensor congelado: el mismo valor repetido muchas horas no es una lectura real.
    for col in ["vibracion_rms_mm_s", "temp_rodamiento_acople_c", "temp_rodamiento_libre_c"]:
        congelado = df.groupby("bomba_id")[col].transform(
            lambda s: _marcar_sensor_congelado(s, config.HORAS_SENSOR_CONGELADO))
        df.loc[congelado, col] = float("nan")

    # 4) Se descartan las filas con vacíos (son ~4 %; con 65 mil filas sobra información).
    df = df.dropna(subset=config.VARIABLES + [config.OBJETIVO])

    columnas = ["fecha_hora", "bomba_id"] + config.VARIABLES + [config.OBJETIVO]
    return df[columnas].sort_values(["bomba_id", "fecha_hora"]).reset_index(drop=True)


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Comprueba columnas, tipos, nulos, rangos y clases. Lanza DatosInvalidos si falla."""
    errores = []

    faltantes = set(config.VARIABLES + [config.OBJETIVO, "fecha_hora"]) - set(df.columns)
    if faltantes:
        raise DatosInvalidos(f"Faltan columnas: {sorted(faltantes)}")

    if len(df) < 1000:
        errores.append(f"Muy pocas filas: {len(df)}")

    nulos = df[config.VARIABLES + [config.OBJETIVO]].isna().sum()
    if nulos.any():
        errores.append(f"Hay valores nulos: {nulos[nulos > 0].to_dict()}")

    for col, (minimo, maximo) in config.RANGOS.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            errores.append(f"{col} no es numérica")
            continue
        if ((df[col] < minimo) | (df[col] > maximo)).any():
            errores.append(f"{col} fuera de rango [{minimo}, {maximo}]")

    desconocidas = set(df[config.OBJETIVO].unique()) - set(config.CLASES)
    if desconocidas:
        errores.append(f"Clases desconocidas: {sorted(desconocidas)}")
    ausentes = set(config.CLASES) - set(df[config.OBJETIVO].unique())
    if ausentes:
        errores.append(f"Clases sin ejemplos: {sorted(ausentes)}")

    if errores:
        raise DatosInvalidos("Validación de datos fallida:\n  - " + "\n  - ".join(errores))
    return df


def cargar_dataset() -> pd.DataFrame:
    """Carga el dataset procesado (lo usan el entrenamiento y las pruebas)."""
    df = pd.read_csv(config.ARCHIVO_DATASET, parse_dates=["fecha_hora"])
    return validar(df)


def main():
    print("1/3  Generando datos crudos con el simulador...")
    sensores, ordenes = cargar_crudos()
    print(f"     sensores: {sensores.shape[0]:,} filas | órdenes de trabajo: {len(ordenes)}")

    print("2/3  Limpiando...")
    df = limpiar(sensores)
    print(f"     filas útiles (bomba operando, sin errores): {len(df):,}")

    print("3/3  Validando...")
    validar(df)
    config.DIR_PROCESADO.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ARCHIVO_DATASET, index=False)

    print("\nDistribución de estados:")
    print(df[config.OBJETIVO].value_counts().to_string())
    print(f"\nDataset válido guardado en {config.ARCHIVO_DATASET.relative_to(config.RAIZ)}")


if __name__ == "__main__":
    main()
