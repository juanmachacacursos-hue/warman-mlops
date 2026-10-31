"""Simulador de operación y mantenimiento de 4 bombas Warman.

¿Por qué simulamos?  No hay acceso a datos reales de planta (PI / SAP PM). En
lugar de usar números al azar, se construyen los datos con reglas físicas
simplificadas, de modo que cada número se pueda defender técnicamente.

Lógica física:
  - DESGASTE DE IMPULSOR / REVESTIMIENTOS: avanza con las horas de operación y
    más rápido con pulpa densa. Al desgastarse, el variador sube las rpm para
    mantener la presión, sube la potencia y baja un poco el caudal.
  - FALLA DE RODAMIENTO: vida aleatoria (Weibull, estándar en confiabilidad).
    En las últimas 150-450 h sube la vibración y la temperatura del rodamiento.
  - FALLA DE SELLO / EMPAQUETADURA: en las horas previas sube el consumo de
    agua de sello y cae su presión.
  - CAVITACIÓN (evento de PROCESO, no falla): nivel bajo del cajón, la presión
    cae y la vibración sube. El modelo debe aprender a no confundirla.
  - Paradas de planta, paradas operativas cortas y errores de instrumentación
    (vacíos, picos imposibles, sensor congelado) para practicar la limpieza.

Salida: un DataFrame de sensores (lectura horaria) con la columna `estado`
(el diagnóstico de condición de la bomba en esa hora) y un DataFrame de
órdenes de trabajo.
"""
import numpy as np
import pandas as pd

from warman import config

# Parámetros nominales de diseño de la bomba
Q0 = 3500.0        # caudal nominal (m3/h)
H0 = 30.0          # altura dinámica nominal (m de pulpa)
N0 = 380.0         # velocidad nominal (rpm)
N_MAX = 460.0      # límite del variador (rpm)
ETA0 = 0.74        # eficiencia hidráulica nominal
V_MOTOR = 4160.0   # tensión del motor (V)
FP = 0.88          # factor de potencia
HORAS_PM_IMPULSOR = 3000  # intervalo planificado de cambio de impulsor (h)

COLUMNAS = [
    "fecha_hora", "bomba_id", "en_operacion", "caudal_m3h", "presion_descarga_kpa",
    "densidad_pulpa_tm3", "velocidad_rpm", "corriente_motor_a", "potencia_kw",
    "temp_rodamiento_acople_c", "temp_rodamiento_libre_c", "vibracion_rms_mm_s",
    "presion_agua_sello_kpa", "caudal_agua_sello_m3h", "nivel_cajon_pct",
    "temp_ambiente_c", "horas_desde_cambio_impulsor", "estado",
]


def _densidad_proceso(rng, n):
    """Densidad de pulpa (t/m3): caminata aleatoria que vuelve a 1.65."""
    d = np.empty(n)
    d[0] = 1.65
    for i in range(1, n):
        d[i] = d[i - 1] + 0.08 * (1.65 - d[i - 1]) + rng.normal(0, 0.012)
    return np.clip(d, 1.45, 1.85)


def _etiquetar(b, s, w, cav):
    """Diagnóstico de condición de la bomba en una hora dada."""
    if b >= config.UMBRAL_DANO_RODAMIENTO:
        return "Riesgo rodamiento"
    if s >= config.UMBRAL_DANO_SELLO:
        return "Riesgo sello"
    if w >= config.UMBRAL_DESGASTE_IMPULSOR:
        return "Desgaste impulsor"
    if cav:
        return "Cavitacion"
    return "Normal"


def simular(semilla: int = config.SEMILLA):
    """Genera (sensores, ordenes) de forma reproducible con la semilla dada."""
    rng = np.random.default_rng(semilla)
    horas = pd.date_range(config.FECHA_INICIO, config.FECHA_FIN, freq="h")
    n = len(horas)
    dia = np.arange(n) / 24.0

    # Temperatura ambiente en la nave de molienda (zona altoandina)
    t_amb = (12 + 4 * np.sin(2 * np.pi * (dia - 30) / 365)
             + 3 * np.sin(2 * np.pi * (dia % 1 - 0.3)) + rng.normal(0, 0.8, n))

    # Paradas de planta programadas (~2-3 días)
    parada_planta = np.zeros(n, dtype=bool)
    for fecha in config.PARADAS_PLANTA:
        i0 = horas.get_loc(pd.Timestamp(fecha))
        parada_planta[i0:i0 + int(rng.integers(48, 72))] = True

    registros, ordenes = [], []
    contador_ot = [1000]

    def nueva_ot(bomba, i, horas_parada, tipo, componente, modo, desc, costo):
        contador_ot[0] += 1
        ordenes.append({
            "ot_id": f"OT-{contador_ot[0]}", "bomba_id": bomba,
            "fecha_inicio": horas[i], "fecha_fin": horas[min(i + horas_parada, n - 1)],
            "tipo_mantenimiento": tipo, "componente": componente, "modo_falla": modo,
            "descripcion": desc, "horas_parada": horas_parada,
            "costo_usd": round(costo, -1),
        })

    for bomba in config.BOMBAS:
        dens = _densidad_proceso(rng, n)
        demanda = 0.97 + 0.04 * np.sin(2 * np.pi * dia / 7) + rng.normal(0, 0.01, n)

        # Estado inicial de los componentes (la bomba no arranca nueva)
        vida_imp = rng.normal(3300, 400)
        w_h = rng.uniform(0, 2000)
        h_desde_pm = w_h
        vida_rod = rng.weibull(2.0) * 7000 + 1500
        h_rod = rng.uniform(0, vida_rod * 0.5)
        ventana_rod = rng.uniform(150, 450)
        lado_rod = rng.choice(["acople", "libre"])
        vida_sello = rng.weibull(1.5) * 2500 + 400
        h_sello = rng.uniform(0, vida_sello * 0.4)
        ventana_sello = rng.uniform(48, 200)

        parada_restante, cav_restante = 0, 0
        postergada, horas_limite = False, 0.0

        def cambio_impulsor():
            """Reinicia impulsor y sello (se cambia la empaquetadura en el mismo trabajo)."""
            return rng.normal(3300, 400), 0.0, 0.0, rng.weibull(1.5) * 2500 + 400, 0.0

        for i in range(n):
            operando = True
            if parada_restante > 0:
                parada_restante -= 1
                operando = False
            elif parada_planta[i]:
                operando = False
                # Mantenimiento oportunista al inicio de la parada de planta
                if (i == 0 or not parada_planta[i - 1]) and h_desde_pm > 2300:
                    dur = int(rng.integers(24, 36))
                    nueva_ot(bomba, i, dur, "Preventivo", "Impulsor y revestimientos",
                             "Desgaste (cambio programado)",
                             "Cambio de impulsor y revestimientos aprovechando parada de planta",
                             rng.uniform(55000, 70000))
                    vida_imp, w_h, h_desde_pm, vida_sello, h_sello = cambio_impulsor()
                    postergada = False
                    parada_restante = max(dur - 1, 0)
            elif rng.random() < 0.0015:  # parada operativa corta, sin OT
                parada_restante = int(rng.integers(1, 6))
                operando = False

            if not operando:
                registros.append([
                    horas[i], bomba, 0, 0.0, rng.normal(3, 2), np.nan, 0.0, 0.0, 0.0,
                    t_amb[i] + rng.normal(5, 2), t_amb[i] + rng.normal(4, 2),
                    abs(rng.normal(0.2, 0.1)), rng.normal(5, 3), 0.0,
                    np.clip(rng.normal(40, 15), 0, 100), t_amb[i], h_desde_pm, "Detenida",
                ])
                continue

            # ---- Envejecimiento (solo con horas de operación) ----
            w_h += (dens[i] / 1.65) ** 2
            h_desde_pm += 1
            h_rod += 1
            h_sello += 1
            w = min(w_h / vida_imp, 1.1)
            b = float(np.clip((h_rod - (vida_rod - ventana_rod)) / ventana_rod, 0, 1))
            s = float(np.clip((h_sello - (vida_sello - ventana_sello)) / ventana_sello, 0, 1))

            # ---- Cavitación (evento de proceso) ----
            if cav_restante == 0 and rng.random() < 0.003:
                cav_restante = int(rng.integers(2, 10))
            cav = 1 if cav_restante > 0 else 0
            cav_restante = max(cav_restante - 1, 0)

            # ---- Física simplificada ----
            q = Q0 * (1 - 0.06 * w) * demanda[i] * (1 - 0.12 * cav) * rng.normal(1, 0.015)
            h = H0 * (1 - 0.07 * w) * (1 - 0.15 * cav)
            presion = dens[i] * 9.81 * h * rng.normal(1, 0.02 + 0.06 * cav)
            rpm = min(N0 * (1 + 0.10 * w) * (dens[i] / 1.65) ** 0.3 * rng.normal(1, 0.006), N_MAX)
            eta = ETA0 * (1 - 0.12 * w) - 0.05 * cav - 0.03 * b
            p_hid = dens[i] * 9.81 * (q / 3600) * h                  # potencia hidráulica (kW)
            potencia = p_hid / eta / 0.95 * rng.normal(1, 0.01)      # 0.95 = eficiencia del motor
            corriente = potencia * 1000 / (np.sqrt(3) * V_MOTOR * FP)

            dano_ac = b if lado_rod == "acople" else 0.25 * b
            dano_li = b if lado_rod == "libre" else 0.25 * b
            t_ac = 52 + 0.5 * (t_amb[i] - 12) + 3 * w + 38 * dano_ac ** 2 + rng.normal(0, 1.0)
            t_li = 48 + 0.5 * (t_amb[i] - 12) + 2 * w + 32 * dano_li ** 2 + rng.normal(0, 1.0)
            vib = (2.2 + 0.9 * w + 9.5 * b ** 2.5 + cav * rng.uniform(1.5, 3.0)) * rng.lognormal(0, 0.08)
            caudal_sello = 3.0 + 5.5 * s ** 1.5 + rng.normal(0, 0.15)
            presion_sello = presion + 80 - 45 * s + rng.normal(0, 4)
            nivel = rng.uniform(10, 25) if cav else np.clip(rng.normal(58, 8), 30, 90)

            registros.append([
                horas[i], bomba, 1, q, presion, dens[i], rpm, corriente, potencia,
                t_ac, t_li, vib, presion_sello, caudal_sello, nivel, t_amb[i],
                h_desde_pm, _etiquetar(b, s, w, cav),
            ])

            # ---- ¿Ocurre una falla o toca un preventivo? ----
            if b >= 1:
                dur = int(rng.integers(24, 60))
                nueva_ot(bomba, i, dur, "Correctivo", f"Rodamiento lado {lado_rod}",
                         "Falla de rodamiento",
                         f"Alta vibración y temperatura, rodamiento lado {lado_rod} dañado",
                         rng.uniform(18000, 30000))
                parada_restante = dur
                vida_rod, h_rod = rng.weibull(2.0) * 7000 + 1500, 0.0
                ventana_rod, lado_rod = rng.uniform(150, 450), rng.choice(["acople", "libre"])
            elif s >= 1:
                dur = int(rng.integers(6, 16))
                nueva_ot(bomba, i, dur, "Correctivo", "Sello / empaquetadura", "Falla de sello",
                         "Fuga excesiva por prensaestopas, cambio de empaquetadura",
                         rng.uniform(2500, 6000))
                parada_restante = dur
                vida_sello, h_sello = rng.weibull(1.5) * 2500 + 400, 0.0
                ventana_sello = rng.uniform(48, 200)
            elif w >= 1.0:
                dur = int(rng.integers(30, 48))
                nueva_ot(bomba, i, dur, "Correctivo", "Impulsor y revestimientos",
                         "Desgaste severo de impulsor",
                         "Impulsor y revestimiento perforados, cambio de emergencia",
                         rng.uniform(70000, 90000))
                parada_restante = dur
                vida_imp, w_h, h_desde_pm, vida_sello, h_sello = cambio_impulsor()
                postergada = False
            elif h_desde_pm >= HORAS_PM_IMPULSOR and not postergada:
                if rng.random() < 0.35:  # producción posterga el preventivo
                    postergada = True
                    horas_limite = h_desde_pm + rng.uniform(400, 900)
                else:
                    dur = int(rng.integers(20, 36))
                    nueva_ot(bomba, i, dur, "Preventivo", "Impulsor y revestimientos",
                             "Desgaste (cambio programado)",
                             "Cambio programado de impulsor y revestimientos",
                             rng.uniform(55000, 70000))
                    parada_restante = dur
                    vida_imp, w_h, h_desde_pm, vida_sello, h_sello = cambio_impulsor()
            elif postergada and h_desde_pm >= horas_limite:
                dur = int(rng.integers(20, 36))
                nueva_ot(bomba, i, dur, "Preventivo", "Impulsor y revestimientos",
                         "Desgaste (cambio programado)",
                         "Cambio de impulsor (preventivo postergado por producción)",
                         rng.uniform(55000, 70000))
                parada_restante = dur
                vida_imp, w_h, h_desde_pm, vida_sello, h_sello = cambio_impulsor()
                postergada = False

    sensores = pd.DataFrame(registros, columns=COLUMNAS)

    # ---- Errores de instrumentación (para practicar la limpieza) ----
    numericas = COLUMNAS[3:16]
    vacios = rng.random((len(sensores), len(numericas))) < 0.004
    sensores[numericas] = sensores[numericas].mask(vacios)
    for col, valor in [("temp_rodamiento_acople_c", 999.0), ("vibracion_rms_mm_s", -1.0),
                       ("presion_descarga_kpa", 9999.0)]:
        idx = rng.choice(len(sensores), 25, replace=False)
        sensores.loc[idx, col] = valor
    congelado = ((sensores.bomba_id == "BB-03") & (sensores.fecha_hora >= "2025-02-10")
                 & (sensores.fecha_hora < "2025-02-15") & (sensores.en_operacion == 1))
    sensores.loc[congelado, "vibracion_rms_mm_s"] = 2.87

    sensores = sensores.round({c: 2 for c in numericas}).round({"horas_desde_cambio_impulsor": 0})
    ordenes_df = pd.DataFrame(ordenes).sort_values(["bomba_id", "fecha_inicio"]).reset_index(drop=True)
    return sensores, ordenes_df
