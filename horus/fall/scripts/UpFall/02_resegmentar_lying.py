import numpy as np
import glob
import os

CARPETA = "../../data/keypoints/upfall"
BACKUP_ORIGINALES = os.path.join(CARPETA, "_originales_lying")
BACKUP_DESCARTADOS = os.path.join(CARPETA, "_descartados_mala_deteccion")

IDX_CADERA_IZQ, IDX_CADERA_DER = 23, 24
IDX_HOMBRO_IZQ, IDX_HOMBRO_DER = 11, 12

# El CSV de omnifall para UP-Fall marca cada trial de "Activity11" (Laying) como
# UN SOLO segmento de ~60s con label_omnifall=6 (lying), sin separar la transición
# real de acostarse (omnifall label 5, "lie_down") del resto del tiempo quieto en
# el piso. Con VENTANA=32 frames en el dataset de entrenamiento, el clasificador
# termina viendo casi siempre una pose estática indistinguible de "fallen".
#
# Además, MediaPipe detecta mal la pose de una persona acostada: en varios trials
# la mayoría de los frames no tienen detección real y quedan rellenados por
# interpolación lineal (trayectoria fabricada, no pose real). Esos trials se
# descartan acá en vez de usarse.
#
# Este script:
#  1. Descarta trials con menos de UMBRAL_VALIDEZ de frames con detección real.
#  2. En los que quedan, ubica los 1-2 picos de velocidad (acostarse / pararse)
#     y separa cada trial en sub-clips de transición ("lie_down") y sub-clips
#     estáticos ("lying"), en vez de un único blob de ~1000 frames.

UMBRAL_VALIDEZ = 0.5

VENTANA_SUAVIZADO = 21
MIN_SEPARACION_PICOS = 150
FACTOR_PICO = 3.0
FACTOR_BORDE = 1.5
MAX_MEDIO_TRAMO = 45
MIN_FRAMES_TRANSICION = 8
PADDING_TRANSICION = 10
MAX_PICOS = 2

DURACION_ESTATICO = 90
MARGEN_EXCLUSION = 30
MAX_CLIPS_ESTATICOS = 2


def centro_cuerpo(kp):
    cadera = (kp[:, IDX_CADERA_IZQ, :2] + kp[:, IDX_CADERA_DER, :2]) / 2
    hombro = (kp[:, IDX_HOMBRO_IZQ, :2] + kp[:, IDX_HOMBRO_DER, :2]) / 2
    return (cadera + hombro) / 2


def ratio_validez(kp):
    return (kp.reshape(len(kp), -1).sum(axis=1) != 0).mean()


def interpolar_ceros(kp):
    valido = kp.reshape(len(kp), -1).sum(axis=1) != 0
    if valido.sum() == 0:
        return kp
    idx_validos = np.where(valido)[0]
    out = kp.copy()
    for i in range(len(kp)):
        if valido[i]:
            continue
        antes = idx_validos[idx_validos < i]
        despues = idx_validos[idx_validos > i]
        if len(antes) and len(despues):
            a, d = antes[-1], despues[0]
            peso = (i - a) / (d - a)
            out[i] = kp[a] * (1 - peso) + kp[d] * peso
        elif len(antes):
            out[i] = kp[antes[-1]]
        elif len(despues):
            out[i] = kp[despues[0]]
    return out


def velocidad_suavizada(kp):
    centro = centro_cuerpo(kp)
    vel = np.linalg.norm(np.diff(centro, axis=0), axis=1)
    vel = np.concatenate([[vel[0]], vel])
    kernel = np.ones(VENTANA_SUAVIZADO) / VENTANA_SUAVIZADO
    return np.convolve(vel, kernel, mode="same")


def detectar_transiciones(vel):
    baseline = np.median(vel)
    umbral = baseline * FACTOR_PICO
    candidatos = np.where(vel > umbral)[0]
    if len(candidatos) == 0:
        return []

    orden = candidatos[np.argsort(-vel[candidatos])]
    picos = []
    for idx in orden:
        idx = int(idx)
        if all(abs(idx - p) >= MIN_SEPARACION_PICOS for p in picos):
            picos.append(idx)
        if len(picos) >= MAX_PICOS:
            break

    borde = baseline * FACTOR_BORDE
    tramos = []
    for pico in picos:
        a = pico
        while a > 0 and vel[a] > borde and pico - a < MAX_MEDIO_TRAMO:
            a -= 1
        b = pico
        while b < len(vel) - 1 and vel[b] > borde and b - pico < MAX_MEDIO_TRAMO:
            b += 1
        if b - a >= MIN_FRAMES_TRANSICION:
            tramos.append((a, b))
    tramos.sort()
    return tramos


def tramos_libres_de(ocupado):
    libres = np.where(~ocupado)[0]
    if len(libres) == 0:
        return []
    tramos = []
    inicio = anterior = libres[0]
    for idx in libres[1:]:
        if idx != anterior + 1:
            tramos.append((inicio, anterior + 1))
            inicio = idx
        anterior = idx
    tramos.append((inicio, anterior + 1))
    return tramos


def procesar_archivo(ruta):
    npz = np.load(ruta, allow_pickle=True)
    kp_crudo = npz["keypoints"]
    sujeto = str(npz["sujeto"])
    actividad = str(npz["actividad"])
    label = str(npz["label"])

    validez = ratio_validez(kp_crudo)
    if validez < UMBRAL_VALIDEZ:
        return "descartado", validez, None

    kp = interpolar_ceros(kp_crudo)
    vel = velocidad_suavizada(kp)
    tramos = detectar_transiciones(vel)

    nuevos = []
    for a, b in tramos:
        ini = max(0, a - PADDING_TRANSICION)
        fin = min(len(kp), b + PADDING_TRANSICION)
        nuevos.append({
            "keypoints": kp[ini:fin],
            "label": label,
            "label_omnifall": 5,
            "sujeto": sujeto,
            "actividad": actividad,
        })

    ocupado = np.zeros(len(kp), dtype=bool)
    for a, b in tramos:
        ocupado[max(0, a - MARGEN_EXCLUSION): min(len(kp), b + MARGEN_EXCLUSION)] = True

    libres = sorted(tramos_libres_de(ocupado), key=lambda t: t[1] - t[0], reverse=True)

    generados = 0
    for a, b in libres:
        if generados >= MAX_CLIPS_ESTATICOS:
            break
        if (b - a) < DURACION_ESTATICO:
            continue
        centro = (a + b) // 2
        ini = max(a, centro - DURACION_ESTATICO // 2)
        fin = min(b, ini + DURACION_ESTATICO)
        nuevos.append({
            "keypoints": kp[ini:fin],
            "label": label,
            "label_omnifall": 6,
            "sujeto": sujeto,
            "actividad": actividad,
        })
        generados += 1

    return "procesado", validez, (nuevos, tramos)


if __name__ == "__main__":
    os.makedirs(BACKUP_ORIGINALES, exist_ok=True)
    os.makedirs(BACKUP_DESCARTADOS, exist_ok=True)

    archivos = sorted(glob.glob(os.path.join(CARPETA, "*Activity11*_adl.npz")))
    print(f"{len(archivos)} trials de Activity11 (lying) encontrados\n")

    total_transicion = 0
    total_estatico = 0
    total_descartados = 0

    for ruta in archivos:
        nombre = os.path.basename(ruta)
        estado, validez, resultado = procesar_archivo(ruta)

        if estado == "descartado":
            os.rename(ruta, os.path.join(BACKUP_DESCARTADOS, nombre))
            total_descartados += 1
            print(f"  {nombre}: {validez:.1%} frames válidos -> DESCARTADO (mala detección)")
            continue

        nuevos, tramos = resultado
        base = nombre.replace("_seg0_adl.npz", "")
        n_trans = n_est = 0
        for item in nuevos:
            if item["label_omnifall"] == 5:
                salida = os.path.join(CARPETA, f"{base}_seg{n_trans}_transicion_adl.npz")
                n_trans += 1
                total_transicion += 1
            else:
                salida = os.path.join(CARPETA, f"{base}_seg{n_est}_estatico_adl.npz")
                n_est += 1
                total_estatico += 1
            np.savez(salida, **item)

        os.rename(ruta, os.path.join(BACKUP_ORIGINALES, nombre))
        largos = [f"{b - a}f" for a, b in tramos] or ["ninguna"]
        print(f"  {nombre}: {validez:.1%} frames válidos, transiciones={largos} "
              f"-> {n_trans} clip(s) lie_down, {n_est} clip(s) lying")

    print(f"\nTotal: {total_transicion} clips lie_down (transición), "
          f"{total_estatico} clips lying (estático), {total_descartados} trials descartados")
