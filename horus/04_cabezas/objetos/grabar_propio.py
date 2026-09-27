# -*- coding: utf-8 -*-
"""
grabar_propio.py · HORUS — negativos difíciles de TU cámara, para reentrenar.

El 25/09, con la cámara del cuarto: 13 alertas de "pistola" en 7 minutos sin
ninguna pistola en cuadro, y personas donde no había nadie. Ningún dataset
público tiene tu cuarto. Esto graba frames de tu cámara diciendo qué clases
seguro NO están, y el entrenamiento aprende que eso no es una pistola.

Dos escenas
-----------
    vacia     cuarto vacío, sin nada.       Seguro NO hay: ninguna de las 7 clases
              (dejá afuera celular, cajas y paquetes; ver abajo)
    conmigo   estás vos, sin armas.         Seguro NO hay: pistola, cuchillo

Todo lo demás (humo, llama, celular, paquete, y persona en "conmigo") queda
como "no sé": el entrenamiento no lo toca. Así no se le enseña que tu celular
no es un celular.

Minado
------
Si encuentra el modelo actual (modelos/head_best_solo.pt), lo corre sobre cada
frame y **guarda enseguida los que disparan** una clase que no está: son justo
los errores que hay que corregir. Además guarda uno cada `--cada` segundos
aunque no dispare nada, para que haya variedad. Al final te dice con qué score
disparó cada clase — sirve también para ajustar umbrales.

Uso (con HORUS CERRADO: en Windows la cámara la abre un solo programa)
---
    python grabar_propio.py --escena vacia   --minutos 3
    python grabar_propio.py --escena conmigo --minutos 5

Qué hacer mientras graba, para que sirva:
  - vacia:   salí del cuadro. Prendé y apagá la luz, abrí la cortina, dejá
             sobre la mesa lo que tengas (mouse, control, celular, taza).
  - conmigo: hacé lo de siempre. Agarrá el celular, el mouse, el control, una
             botella; lo que en el panel salga como "pistola". Movete, sentate,
             acercate y alejate.

Sale en datasets/propio/. Después: zip de esa carpeta -> dataset de Kaggle ->
adjuntalo al notebook. sumar_propio.py lo mete en el entrenamiento.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parent.parent

CLASES: Tuple[str, ...] = ("humo", "llama", "persona", "pistola", "cuchillo",
                           "celular", "paquete")

# Qué clases seguro NO están en cada escena. Lo que no figura acá es "no sé".
ESCENAS: Dict[str, Tuple[str, ...]] = {
    # 26/09: vacía = negativo PURO. Antes solo declaraba persona/pistola/
    # cuchillo, así que el paquete falso que salía sobre la silla quedaba como
    # "no sé" y nunca se corregía. En un cuarto vacío no hay ninguna de las 7.
    "vacia": CLASES,
    "conmigo": ("pistola", "cuchillo"),
}

MANIFIESTO = "horus_propio.json"


# --------------------------------------------------------------------------- #
def abrir_camara(fuente: str):
    import cv2
    src = int(fuente) if str(fuente).isdigit() else fuente
    cap = None
    # Igual que el servicio: en Windows DirectShow abre webcams que MSMF no.
    if isinstance(src, int) and os.name == "nt":
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"No abre la cámara {fuente!r}.\n"
                 "  ¿Está HORUS corriendo? En Windows la cámara la toma un solo "
                 "programa:\n  cerrá la ventana 'Horus modelos' (y cualquier "
                 "app de videollamada) y probá de nuevo.")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def cargar_motor(pesos: Optional[str]):
    """El modelo actual, en CPU, con umbral bajo: queremos ver los disparos
    débiles también, porque el tracker los usa para sostener tracks."""
    if not pesos:
        return None
    if not Path(pesos).exists():
        print(f"[minado] no encuentro {pesos}: grabo sin minar "
              f"(igual sirve, pero menos apuntado)")
        return None
    for p in (_RAIZ / "03_backbone", _AQUI):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from objects_engine import EngineConfig, ObjectsEngine
    import torch
    cfg = EngineConfig(pesos=pesos, score_thresh=0.15, max_batch=1, warmup=1,
                       verboso=False, calcular_novedad=False)
    if not torch.cuda.is_available():
        cfg.device, cfg.precision, cfg.cuda_graphs = "cpu", "fp32", False
    print("[minado] cargando el modelo actual (en CPU tarda ~20 s)…")
    return ObjectsEngine(cfg)


def distinto(a: Optional[np.ndarray], b: np.ndarray, umbral: float) -> bool:
    """¿Cambió algo desde el último guardado? Con la cámara fija y el cuarto
    vacío, sin esto se guardan 200 copias del mismo cuadro."""
    if a is None:
        return True
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) >= umbral


def miniatura(frame: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA)


def actualizar_manifiesto(salida: Path, escena: str,
                          ausentes: Sequence[str]) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / MANIFIESTO
    datos = {"version": 1, "clases": list(CLASES), "escenas": {}}
    if ruta.exists():
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    prev = datos.setdefault("escenas", {}).get(escena)
    if prev and sorted(prev.get("ausentes", [])) != sorted(ausentes):
        sys.exit(f"La escena '{escena}' ya existe en {ruta} con ausentes "
                 f"{prev.get('ausentes')} y ahora pedís {list(ausentes)}.\n"
                 "Mezclarlas le mentiría al entrenamiento sobre los frames "
                 "viejos. Usá otro nombre de escena (--escena) o borrá la vieja.")
    datos["escenas"][escena] = {"ausentes": list(ausentes)}
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False),
                    encoding="utf-8")


# --------------------------------------------------------------------------- #
def grabar(a: argparse.Namespace) -> int:
    import cv2

    ausentes = tuple(a.ausentes.split(",")) if a.ausentes else ESCENAS.get(a.escena)
    if not ausentes:
        sys.exit(f"Escena '{a.escena}' desconocida. Usá {list(ESCENAS)} o "
                 "decí qué clases seguro no están con --ausentes pistola,cuchillo")
    malas = [c for c in ausentes if c not in CLASES]
    if malas:
        sys.exit(f"--ausentes: {malas} no son clases. Son {list(CLASES)}.")
    idx_aus = {CLASES.index(c) for c in ausentes}

    salida = Path(a.salida)
    dir_img = salida / a.escena / "images"
    dir_img.mkdir(parents=True, exist_ok=True)
    actualizar_manifiesto(salida, a.escena, ausentes)

    motor = None if a.sin_minar else cargar_motor(a.pesos)
    cap = abrir_camara(a.camara)

    print()
    print(f"Escena '{a.escena}': seguro NO hay {', '.join(ausentes)}.")
    print("Todo lo demás queda como 'no sé' y no se le enseña nada.")
    if a.escena == "vacia":
        print("SALÍ DEL CUADRO antes de que termine la cuenta.")
    for s in range(a.espera, 0, -1):
        print(f"  arranco en {s}…", end="\r", flush=True)
        # Seguir leyendo para que el búfer de la cámara no quede viejo.
        t_fin = time.time() + 1.0
        while time.time() < t_fin:
            cap.read()
    print(" " * 30, end="\r")
    print(f"Grabando {a.minutos:g} min. {'q en la ventana o ' if a.ver else ''}"
          f"Ctrl-C para cortar antes.\n")

    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    registro_path = salida / a.escena / "registro.csv"
    nuevo = not registro_path.exists()
    fh = open(registro_path, "a", newline="", encoding="utf-8")
    reg = csv.writer(fh)
    if nuevo:
        reg.writerow(["archivo", "motivo", "clase", "score", "x1", "y1", "x2", "y2"])

    fin = time.time() + a.minutos * 60.0
    ultimo_periodico = 0.0
    ultimo_disparo = 0.0
    ultima_min: Optional[np.ndarray] = None
    n_guardados = n_disparos = n = 0
    scores: Dict[str, List[float]] = {c: [] for c in ausentes}

    try:
        while time.time() < fin:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            ahora = time.time()

            malos = []
            if motor is not None:
                res = motor.infer(frame, camera_id="propio")
                malos = [d for d in res.detections if d.class_id in idx_aus]

            motivo = None
            if malos and ahora - ultimo_disparo >= a.min_entre_disparos:
                motivo = "disparo"
            elif ahora - ultimo_periodico >= a.cada:
                mini = miniatura(frame)
                if distinto(ultima_min, mini, a.umbral_cambio):
                    motivo = "periodico"
                else:
                    ultimo_periodico = ahora     # igual al anterior: esperar otro

            if motivo:
                nombre = f"{a.escena}_{sello}_{n:05d}.jpg"
                n += 1
                cv2.imwrite(str(dir_img / nombre), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                n_guardados += 1
                ultima_min = miniatura(frame)
                if motivo == "disparo":
                    ultimo_disparo = ahora
                    n_disparos += 1
                else:
                    ultimo_periodico = ahora
                if malos:
                    for d in malos:
                        reg.writerow([nombre, motivo, d.label, f"{d.score:.3f}",
                                      *(f"{v:.0f}" for v in d.bbox_xyxy)])
                        scores[d.label].append(float(d.score))
                else:
                    reg.writerow([nombre, motivo, "", "", "", "", "", ""])

            if a.ver:
                vista = frame.copy()
                for d in malos:
                    x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
                    cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(vista, f"{d.label} {d.score:.2f} (falso)",
                                (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)
                resta = max(0, int(fin - ahora))
                cv2.putText(vista, f"{a.escena} · {n_guardados} guardados "
                                   f"({n_disparos} disparos) · {resta // 60}:{resta % 60:02d}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("HORUS · grabar_propio (q = cortar)", vista)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\ncortado a mano")
    finally:
        fh.close()
        cap.release()
        if a.ver:
            cv2.destroyAllWindows()

    total = len(list(dir_img.glob("*.jpg")))
    print(f"\n{n_guardados} frames nuevos ({n_disparos} por disparo del modelo) "
          f"-> {dir_img}   (total en la escena: {total})")
    if motor is not None:
        print("\nLo que el modelo actual vio y NO estaba:")
        for c in ausentes:
            v = np.array(scores[c])
            if len(v):
                print(f"  {c:<9} {len(v):>4} detecciones   score mediana "
                      f"{np.median(v):.2f} · p90 {np.percentile(v, 90):.2f} · "
                      f"máx {v.max():.2f}")
            else:
                print(f"  {c:<9}    0 detecciones")
        print("\n(El umbral para nacer un track es 0,25 en persona y 0,35 en "
              "pistola/cuchillo:\n lo que esté arriba de eso es lo que llegaba "
              "a alerta.)")
    if total < 150:
        print(f"\nConviene juntar más: con {total} frames de esta escena pesa "
              "poco contra ~13.000 fotos.\nApuntá a 300+ por escena, cambiando "
              "luz y objetos entre tandas.")
    print("\nSiguiente: grabá la otra escena, después comprimí "
          f"{salida} en un .zip y subilo\na Kaggle como dataset (ver el "
          "paso 5b del notebook horus_objetos_kaggle.ipynb).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Grabar negativos difíciles de la cámara propia")
    ap.add_argument("--escena", required=True,
                    help=f"{' | '.join(ESCENAS)} (u otro nombre con --ausentes)")
    ap.add_argument("--ausentes", default=None,
                    help="clases que seguro NO están, separadas por coma. "
                         "Por defecto salen de la escena")
    ap.add_argument("--camara", default="0", help="índice o URL (rtsp://…)")
    ap.add_argument("--minutos", type=float, default=3.0)
    ap.add_argument("--cada", type=float, default=2.0,
                    help="segundos entre frames 'de variedad' (sin disparo)")
    ap.add_argument("--min-entre-disparos", type=float, default=0.5)
    ap.add_argument("--umbral-cambio", type=float, default=3.0,
                    help="diferencia media mínima (0-255) para guardar un "
                         "frame de variedad; evita 200 copias del mismo cuadro")
    ap.add_argument("--espera", type=int, default=8,
                    help="segundos de cuenta regresiva (para salir del cuadro)")
    ap.add_argument("--pesos", default=str(_AQUI / "modelos" / "head_best_solo.pt"))
    ap.add_argument("--sin-minar", action="store_true",
                    help="no cargar el modelo: solo frames periódicos")
    ap.add_argument("--salida", default=str(_AQUI / "datasets" / "propio"))
    ap.add_argument("--sin-ver", dest="ver", action="store_false", default=True)
    return grabar(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
