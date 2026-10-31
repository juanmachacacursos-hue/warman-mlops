"""Etapa 2 — Entrenamiento con puerta de calidad.

Comando:  uv run entrenar

Decisiones clave (para defender en la exposición):
  * División TEMPORAL, no aleatoria. Las lecturas son horarias y una hora se
    parece mucho a la siguiente; si se mezclaran al azar, el modelo "vería" el
    futuro y la métrica saldría inflada. Se entrena con el pasado y se evalúa
    con el futuro, como se usaría en planta.
        train: ene-2024 a mar-2025 | validación: abr-jun 2025 | test: jul-dic 2025
  * Métrica F1-macro. ~85 % de las horas son "Normal". Un modelo que diga
    siempre "Normal" tendría 85 % de accuracy y no serviría para nada.
    F1-macro promedia el F1 de cada clase con el mismo peso, así que obliga
    a detectar bien las fallas, que son las que importan.
  * Se comparan dos modelos (regresión logística como línea base y Random
    Forest). Se elige con VALIDACIÓN y se reporta en TEST, que el modelo
    nunca vio durante la elección.
  * Puerta de calidad: si el F1-macro en test es menor que el mínimo
    definido en config.py, el modelo NO se guarda y el comando termina con
    error (así el CI y el Docker se detienen).
"""
import json
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from warman import config
from warman.data import cargar_dataset


def dividir(df: pd.DataFrame):
    """División temporal en train / validación / test."""
    train = df[df.fecha_hora < config.FIN_ENTRENAMIENTO]
    val = df[(df.fecha_hora >= config.FIN_ENTRENAMIENTO) & (df.fecha_hora < config.FIN_VALIDACION)]
    test = df[df.fecha_hora >= config.FIN_VALIDACION]
    return train, val, test


def candidatos():
    """Modelos a comparar. Todos usan la semilla fija."""
    return {
        "regresion_logistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=config.SEMILLA, **config.PARAMS_REGRESION_LOGISTICA),
        ),
        "random_forest": RandomForestClassifier(
            random_state=config.SEMILLA, **config.PARAMS_RANDOM_FOREST),
    }


def evaluar(modelo, X, y) -> dict:
    pred = modelo.predict(X)
    reporte = classification_report(y, pred, labels=config.CLASES, output_dict=True, zero_division=0)
    return {
        "f1_macro": round(f1_score(y, pred, labels=config.CLASES, average="macro", zero_division=0), 4),
        "accuracy": round(reporte["accuracy"], 4),
        "por_clase": {
            c: {"precision": round(reporte[c]["precision"], 4),
                "recall": round(reporte[c]["recall"], 4),
                "f1": round(reporte[c]["f1-score"], 4),
                "soporte": int(reporte[c]["support"])}
            for c in config.CLASES
        },
        "matriz_confusion": {
            "etiquetas": config.CLASES,
            "valores": confusion_matrix(y, pred, labels=config.CLASES).tolist(),
        },
    }


def main():
    df = cargar_dataset()
    train, val, test = dividir(df)
    X = lambda d: d[config.VARIABLES]          # noqa: E731
    y = lambda d: d[config.OBJETIVO]           # noqa: E731
    print(f"Filas -> train: {len(train):,} | validación: {len(val):,} | test: {len(test):,}")

    # 1) Comparar candidatos en validación
    # En validación (abr-jun 2025) no ocurrió ninguna falla de rodamiento; por eso
    # el F1-macro de validación se calcula solo con las clases presentes.
    comparacion = {}
    presentes = sorted(y(val).unique())
    for nombre, modelo in candidatos().items():
        modelo.fit(X(train), y(train))
        f1_val = f1_score(y(val), modelo.predict(X(val)), labels=presentes, average="macro", zero_division=0)
        comparacion[nombre] = round(f1_val, 4)
        print(f"  {nombre:<22} F1-macro validación = {f1_val:.4f}")
    mejor = max(comparacion, key=comparacion.get)
    print(f"Mejor modelo: {mejor}")

    # 2) Reentrenar el mejor con train + validación y evaluar en test
    modelo = candidatos()[mejor]
    train_val = pd.concat([train, val])
    modelo.fit(X(train_val), y(train_val))
    metricas_test = evaluar(modelo, X(test), y(test))
    f1 = metricas_test["f1_macro"]
    print(f"F1-macro en TEST = {f1:.4f} (mínimo exigido: {config.F1_MACRO_MINIMO})")

    # 3) Puerta de calidad
    aprobado = f1 >= config.F1_MACRO_MINIMO
    if not aprobado:
        print("PUERTA DE CALIDAD: RECHAZADO. El modelo no se guarda.")
        sys.exit(1)

    # 4) Guardar modelo y ficha del modelo
    params = config.PARAMS_RANDOM_FOREST if mejor == "random_forest" else config.PARAMS_REGRESION_LOGISTICA
    ficha = {
        "nombre": "warman-estado-bomba",
        "version": config.VERSION_MODELO,
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problema": "Clasificar el estado de una bomba Warman de alimentación a ciclones "
                    "a partir de una lectura de sus sensores.",
        "usuario": "Planner / supervisor de mantenimiento de la planta concentradora",
        "modelo": mejor,
        "parametros": {**params, "random_state": config.SEMILLA},
        "variables": config.VARIABLES,
        "rangos_validos": config.RANGOS,
        "clases": config.CLASES,
        "division": {
            "tipo": "temporal",
            "train": f"< {config.FIN_ENTRENAMIENTO}",
            "validacion": f"{config.FIN_ENTRENAMIENTO} a {config.FIN_VALIDACION}",
            "test": f">= {config.FIN_VALIDACION}",
            "filas": {"train": len(train), "validacion": len(val), "test": len(test)},
        },
        "metrica_principal": "f1_macro",
        "comparacion_validacion_f1_macro": comparacion,
        "metricas_test": metricas_test,
        "puerta_calidad": {"f1_macro_minimo": config.F1_MACRO_MINIMO, "aprobado": aprobado},
    }
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "variables": config.VARIABLES,
                 "version": config.VERSION_MODELO}, config.ARCHIVO_MODELO, compress=3)
    config.ARCHIVO_METRICAS.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PUERTA DE CALIDAD: APROBADO.")
    print(f"Modelo guardado en {config.ARCHIVO_MODELO.relative_to(config.RAIZ)}")
    print(f"Ficha guardada en  {config.ARCHIVO_METRICAS.relative_to(config.RAIZ)}")


if __name__ == "__main__":
    main()
