"""Re-ensambla los .npz por segmento (le2i_omnifall, upfall) en streams de video
completo para 17_evaluar_evento.py. Los videos originales no hacen falta: los
segmentos de OMNIFALL son contiguos (gap mediano 0.2 ms) y el seg{i} del nombre
de archivo coincide con el orden de las filas del csv para ese video.

Salida: ../data/keypoints/streams/{origen}/{video}.npz con
  kp                 (n,33,4) crudo de MediaPipe, ceros = sin deteccion
  t                  (n,) segundos, i/fps
  fps                float, deducido por video de len(segmento)/(end-start)
  onset_caida        start del primer segmento con label in {1,2}; nan si no hay caida
  fin_fallen         end del ultimo segmento con label in {1,2}; nan si no hay caida
  subclase_por_frame (n,) codigo OMNIFALL activo en cada frame; -1 sin cobertura
  origen, sujeto, video

Mas ../data/keypoints/streams/indice.csv con una fila por stream.
"""
import os
import re

import numpy as np
import pandas as pd

SALIDA = "../data/keypoints/streams"
LABELS_CAIDA = {1, 2}   # 1 = fall (transicion), 2 = fallen (en el piso)


TASAS_ESTANDAR = (24.0, 25.0, 29.97, 30.0)


def deducir_fps(segmentos, fps_default):
    """segmentos: lista de (start, end, n_frames). Los extractores cortaron cada
    segmento como kp[round(start*fps):round(end*fps)], asi que la tasa correcta
    es la que reproduce exactamente esos largos. Estimar n/(end-start) hereda el
    sesgo del redondeo (da 24.8 en vez de 25) y deriva ~0.4s en un video de 60s.
    Si ninguna tasa estandar reproduce los largos (UP-Fall: timestamps irregulares
    de PNGs), caigo a la mediana de n/(end-start) sobre los segmentos largos."""
    def aciertos(fps):
        # el ultimo segmento suele estar truncado por el fin del video, tolero 1 fallo
        return sum(int(round(e * fps)) - int(round(s * fps)) == n for s, e, n in segmentos)

    mejor = max(TASAS_ESTANDAR, key=aciertos)
    if aciertos(mejor) >= max(1, len(segmentos) - 1):
        return mejor
    est = [n / (e - s) for s, e, n in segmentos if (e - s) >= 1.0]
    return float(np.median(est)) if est else fps_default


def armar_stream(filas, buscar_npz, fps_default):
    """filas: DataFrame con las filas del csv de UN video, en orden original.
    buscar_npz(i, label_bin) -> ruta o None. Devuelve dict del stream o None."""
    segmentos = []
    for i, seg in enumerate(filas.itertuples()):
        label_bin = "fall" if seg.label in LABELS_CAIDA else "adl"
        ruta = buscar_npz(i, label_bin)
        if ruta is None:
            continue
        kp = np.load(ruta, allow_pickle=True)["keypoints"]
        if len(kp) == 0:
            continue
        segmentos.append((seg.start, seg.end, int(seg.label), kp))
    if not segmentos:
        return None

    fps = deducir_fps([(s, e, len(kp)) for s, e, _, kp in segmentos], fps_default)

    # primero calculo donde va cada segmento, despues reservo el largo total
    ubicaciones = [(int(round(s * fps)), lab, kp) for s, e, lab, kp in segmentos]
    n_total = max(f_ini + len(kp) for f_ini, _, kp in ubicaciones)

    kp_total = np.zeros((n_total, 33, 4), dtype=np.float64)
    sub = np.full(n_total, -1, dtype=np.int16)
    for f_ini, lab, kp in ubicaciones:
        kp_total[f_ini:f_ini + len(kp)] = kp
        sub[f_ini:f_ini + len(kp)] = lab

    caidas = [(s, e) for s, e, lab, _ in segmentos if lab in LABELS_CAIDA]
    onset = min(s for s, _ in caidas) if caidas else np.nan
    fin = max(e for _, e in caidas) if caidas else np.nan

    return dict(kp=kp_total, t=np.arange(n_total) / fps, fps=fps,
                onset_caida=onset, fin_fallen=fin, subclase_por_frame=sub)


def guardar(stream, origen, sujeto, video, filas_indice):
    carpeta = os.path.join(SALIDA, origen)
    os.makedirs(carpeta, exist_ok=True)
    np.savez(os.path.join(carpeta, f"{video}.npz"), origen=origen, sujeto=sujeto, video=video, **stream)
    filas_indice.append(dict(origen=origen, sujeto=sujeto, video=video,
                             n_frames=len(stream["kp"]), fps=round(stream["fps"], 2),
                             duracion_seg=round(stream["t"][-1], 2),
                             tiene_caida=not np.isnan(stream["onset_caida"]),
                             onset_caida=stream["onset_caida"], fin_fallen=stream["fin_fallen"],
                             frac_con_deteccion=round(float((stream["kp"].reshape(len(stream["kp"]), -1).sum(1) != 0).mean()), 3)))


def armar_le2i(filas_indice):
    df = pd.read_csv("le2i/le2i_omnifall.csv")
    df["escenario"] = df["path"].str.split("/").str[0]
    df["num"] = df["path"].str.extract(r"video_(\d+)").astype(int)
    carpeta = "../data/keypoints/le2i_omnifall"

    n = 0
    for (esc, num), g in df.groupby(["escenario", "num"], sort=False):
        def buscar(i, lab, esc=esc, num=num):
            r = os.path.join(carpeta, f"{esc}_video_{num}_seg{i}_{lab}.npz")
            return r if os.path.exists(r) else None
        stream = armar_stream(g, buscar, fps_default=25.0)
        if stream is None:
            continue
        guardar(stream, "le2i", f"le2i_s{g.subject.iloc[0]}", f"{esc}_video_{num}", filas_indice)
        n += 1
    print(f"le2i: {n} streams de {df.groupby(['escenario', 'num']).ngroups} videos del csv")


def armar_upfall(filas_indice):
    df = pd.read_csv("UpFall/up_fall_omnifall.csv")
    df["carpeta"] = df["path"].str.split("/").str[-1]
    base = "../data/keypoints/upfall"
    # 02_resegmentar_lying.py movio los trials de "lying" originales a subcarpetas
    # (los reemplazo por versiones re-cortadas). Para el stream quiero la linea de
    # tiempo original completa, asi que los busco tambien ahi.
    carpetas = [base, os.path.join(base, "_originales_lying"), os.path.join(base, "_descartados_mala_deteccion")]

    n = 0
    for carp, g in df.groupby("carpeta", sort=False):
        def buscar(i, lab, carp=carp):
            for c in carpetas:
                r = os.path.join(c, f"{carp}_seg{i}_{lab}.npz")
                if os.path.exists(r):
                    return r
            return None
        stream = armar_stream(g, buscar, fps_default=18.0)
        if stream is None:
            continue
        guardar(stream, "upfall", f"upfall_s{g.subject.iloc[0]}", carp, filas_indice)
        n += 1
    print(f"upfall: {n} streams de {df.carpeta.nunique()} trials del csv (solo los que tienen keypoints extraidos)")


if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)
    filas_indice = []
    armar_le2i(filas_indice)
    armar_upfall(filas_indice)

    indice = pd.DataFrame(filas_indice)
    indice.to_csv(os.path.join(SALIDA, "indice.csv"), index=False)

    print(f"\nTotal: {len(indice)} streams -> {SALIDA}/indice.csv")
    for origen, g in indice.groupby("origen"):
        print(f"  {origen}: {len(g)} streams, {g.tiene_caida.sum()} con caida, "
              f"{g.duracion_seg.sum() / 60:.1f} min totales, fps mediano {g.fps.median():.1f}, "
              f"deteccion mediana {g.frac_con_deteccion.median():.0%}")
        print(f"    sujetos: {sorted(g.sujeto.unique())}")
