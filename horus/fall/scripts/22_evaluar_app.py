"""Evaluacion a nivel evento de LA CADENA QUE CORRE LA APP.

17_evaluar_evento.py mide MaquinaCaida (umbral 0.70, su propia persistencia),
que es la logica de los scripts de webcam. El .exe corre otra cosa:

    DetectorCaidas (detector_caidas.py) -> MotorFusion -> ReglaCaida -> Evento

con la etiqueta "caida" desde 0.50 y los 2 s de "tumbado" que pide la regla.
Optimizar sobre el harness viejo es optimizar otra cosa. Aca se reproduce cada
stream de Le2i causalmente por esa cadena, con el codigo de la app tal cual.

Que se emula y que no:
  - La pose sale de los keypoints del stream (los mismos que mide 17), por un
    backend de pose falso. El clasificador es el real (LOSO o un checkpoint).
  - La persona se declara QUIETA siempre: los streams no tienen cajas en
    pixeles del frame, asi que el filtro de quietud del tracker no se puede
    medir. Es el caso pesimista para falsas alarmas: la app real filtra
    ademas por quietud, asi que sus falsas son <= que las de aca.

Modelo:
  --loso [--semillas N]   un modelo (o ensamble) por sujeto dejado afuera, de
                          ../checkpoints/loso (los entrena 17_evaluar_evento.py
                          --loso --semillas N). Numero honesto.
  --checkpoint RUTA       uno fijo (optimista si vio a los sujetos).

Salida: ../resultados/app_{tag}.json (+ fila en resultados/README.md) y el
diagnostico por frame en ../data/processed/app_diag/{tag}/ para analizar.

    python 22_evaluar_app.py --loso --semillas 3 --tag app_ens3
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
FALL = os.path.dirname(AQUI)
HORUS = os.path.dirname(FALL)
for _p in (os.path.join(HORUS, "05_tracking", "local"), os.path.join(HORUS, "06_fusion_decision"),
           os.path.join(FALL, "src"), FALL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import CABEZA_ACCION, CABEZA_OBJETOS, CABEZA_POSE, ObservacionCamara  # noqa: E402
from motor_fusion import ConfigFusion, MotorFusion  # noqa: E402
import reglas  # noqa: E402
from tracker_local import TrackLocal  # noqa: E402
from detector_caidas import ConfigCaidas, DetectorCaidas  # noqa: E402
from ensamble import Ensamble, cargar_checkpoint  # noqa: E402


def _cargar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, os.path.join(AQUI, archivo))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


h17 = _cargar("h17", "17_evaluar_evento.py")

RESULTADOS = os.path.join(FALL, "resultados")
DIAG = os.path.join(FALL, "data", "processed", "app_diag")
CAJA = (0.0, 0.0, 200.0, 400.0)


# ------------------------------------------------------------------ modelo

class Clasificador:
    """Mismo contrato que ClasificadorSTGCN: clips (B,32,33,4) -> P(caida).
    `espejo`: promedia con el clip espejado (test-time augmentation; el
    entrenamiento ya espeja la mitad de las veces)."""

    def __init__(self, modelo, device, espejo=False):
        self.modelo, self.device, self.espejo = modelo, device, espejo

    def __call__(self, clips):
        x = torch.tensor(np.ascontiguousarray(clips), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p = torch.softmax(self.modelo(x), 1)[:, 1]
            if self.espejo:
                from dataset_stgcn import espejar
                xe = torch.tensor(np.stack([espejar(c) for c in clips]), dtype=torch.float32,
                                  device=self.device)
                p = (p + torch.softmax(self.modelo(xe), 1)[:, 1]) / 2
        return p.cpu().numpy()


def modelos_loso(grupos, device, loso_dir, semillas):
    out = {}
    for g in grupos:
        miembros = []
        for k in range(semillas):
            ruta = os.path.join(loso_dir, f"{g}.pt" if k == 0 else f"{g}_s{k}.pt")
            if not os.path.exists(ruta):
                raise SystemExit(f"falta {ruta}: entrenalo con 17_evaluar_evento.py --loso "
                                 f"--loso-dir {loso_dir} --semillas {semillas}")
            miembros.append(cargar_checkpoint(ruta, device)[0])
        out[g] = miembros[0] if semillas == 1 else Ensamble(miembros).to(device).eval()
    return out


# ------------------------------------------------------------------ cadena

class PoseDelStream:
    kp = None

    def __call__(self, recorte):
        if self.kp is None or self.kp.reshape(-1).sum() == 0:
            return None
        return self.kp.astype(np.float32)


def extender(stream, seg):
    """Le2i corta casi todos los videos de caida justo donde termina "fallen"
    (mediana 0,0 s despues), y la caida + el piso duran 4,7 s de mediana.
    La app pide 1,24 s de ventana + 2 s tumbado: muchas caidas se 'perdian'
    porque el VIDEO se terminaba con la persona todavia en el piso. En la vida
    real sigue ahi. Aca se repite el ultimo frame `seg` segundos (inmovil,
    con la pose que tenga -- si MediaPipe ya no la veia, sigue sin verla) y
    ese tramo cuenta como "fallen". Solo en videos que terminan en el piso."""
    if seg <= 0 or np.isnan(stream["onset"]) or stream["t"][-1] - stream["fin"] > 0.5:
        return stream
    s = dict(stream)
    fps = stream["fps"]
    n = int(round(seg * fps))
    s["t"] = np.concatenate([stream["t"], stream["t"][-1] + np.arange(1, n + 1) / fps])
    s["kp"] = np.concatenate([stream["kp"], np.repeat(stream["kp"][-1:], n, axis=0)])
    s["sub"] = np.concatenate([stream["sub"], np.full(n, 2)])
    s["fin"] = float(s["t"][-1])
    s["extendido"] = True
    return s


def reproducir(stream, clf, fps, seed, cfg_kw):
    """Un stream por la cadena de la app. Devuelve el registro por frame."""
    pose = PoseDelStream()
    det = DetectorCaidas(ConfigCaidas(**cfg_kw), backend_pose=pose, clasificador=clf, verboso=False)
    motor = MotorFusion(cfg=ConfigFusion(), fps=10.0)
    t, kp, sub = stream["t"], stream["kp"], stream["sub"]
    idx = np.arange(len(t)) if fps == "nativo" else h17.indices_submuestreo(t, float(fps), seed)
    reg = {k: [] for k in ("t", "prob", "etiqueta", "firma", "vista", "evento", "activo", "sub", "vert")}
    vistos = set()
    for n, i in enumerate(idx):
        ts = float(t[i]) + 1000.0
        pose.kp = kp[i]
        tr = TrackLocal(track_id=1, clase="persona", class_id=0, bbox_xyxy=CAJA, score=0.8,
                        camera_id="cam", frame_idx=n, ts=ts, estado="confirmado", hits=100,
                        edad=100, frames_sin_ver=0, ts_nacimiento=ts - 10, visto_s=10.0,
                        quieto=True, quieto_s=30.0)
        acs = det.procesar("cam", np.zeros((400, 200, 3), np.uint8), [tr], ts=ts)
        obs = ObservacionCamara(
            camera_id="cam", frame_idx=n, ts=ts, tracks=[tr], acciones=acs, tam_frame=(400, 200),
            cabezas=frozenset({CABEZA_OBJETOS, CABEZA_ACCION, CABEZA_POSE} | {a.fuente for a in acs}))
        nuevo = activo = False
        for e in motor.procesar([obs], ts=ts):
            if e.tipo == "caida":
                activo = True
                if e.evento_id not in vistos:
                    vistos.add(e.evento_id)
                    nuevo = True
        est = det._estados.get(("cam", 1))
        reg["t"].append(t[i])
        reg["prob"].append(est.prob if est else 0.0)
        reg["etiqueta"].append(any(a.accion == "caida" for a in acs))
        reg["firma"].append(bool(est and est.puerta.firma_reciente(ts, det.cfg.ventana_post_pico_s)))
        reg["vista"].append(bool(est and est.ultima_vista_ts == ts))
        reg["evento"].append(nuevo)
        reg["activo"].append(activo)
        reg["sub"].append(int(sub[i]))
        v = est.puerta.ultima_verticalidad if est else None
        reg["vert"].append(np.nan if v is None else v)
    return {k: np.asarray(v) for k, v in reg.items()}


# ------------------------------------------------------------------ medir

def evaluar(stream, reg):
    r = dict(t=reg["t"], alarma_nueva=reg["evento"].astype(bool), sub=reg["sub"])
    return h17.evaluar_stream(stream, r, 0.0)


def por_que_se_perdio(stream, reg, umbral):
    """La primera causa de la cadena que explica una caida no detectada."""
    v_ini, v_fin = stream["onset"], stream["fin"] + h17.MARGEN_POST_FALLEN_SEG
    m = (reg["t"] >= v_ini) & (reg["t"] <= v_fin)
    if m.sum() == 0 or reg["vista"][m].mean() < 0.3:
        return "sin_pose"                 # MediaPipe casi no la vio en el piso
    if reg["prob"][m].max() < umbral:
        return "modelo_no_la_ve"
    if not reg["etiqueta"][m].any():
        return "puerta_sin_firma"         # el modelo si, pero no hubo pico + torso
    if reg["evento"].any() and reg["t"][reg["evento"]].min() < v_ini:
        return "evento_previo_enganchado" # una falsa alarma antes se comio la real
    return "regla_no_sostuvo_2s"          # la etiqueta parpadeo menos de 2 s


def resumir(evals, regs, streams, umbral, n_boot=2000, seed=0):
    con = [e for e in evals if e["tiene_caida"]]
    det = [e for e in con if e["detectada"]]
    horas = sum(e["tiempo_normal"] for e in evals) / 3600
    falsas = sum(e["falsas"] for e in evals)
    lat = np.array([e["latencia"] for e in det])
    # bootstrap por video: con 130 caidas y ~40 min normales, una diferencia
    # de 2-3 eventos es ruido, y el intervalo lo dice
    rng = np.random.default_rng(seed)
    n = len(evals)
    rec_b, fh_b = [], []
    for _ in range(n_boot):
        sel = [evals[i] for i in rng.integers(0, n, n)]
        c = [e for e in sel if e["tiene_caida"]]
        h = sum(e["tiempo_normal"] for e in sel) / 3600
        rec_b.append(sum(1 for e in c if e["detectada"]) / max(len(c), 1))
        fh_b.append(sum(e["falsas"] for e in sel) / max(h, 1e-9))
    # Recall "con alerta activa": la caida tiene una alerta de caida ABIERTA
    # en algun momento de su ventana, aunque la haya abierto un disparo de
    # antes (ej. al levantarse justo antes de caer). Para el que mira el
    # panel es una alerta valida; el recall estricto no la cuenta.
    activas = 0
    for s, e in zip(streams, evals):
        if e["tiene_caida"]:
            r = regs[s["video"]]
            m = (r["t"] >= s["onset"]) & (r["t"] <= s["fin"] + h17.MARGEN_POST_FALLEN_SEG)
            activas += bool(e["detectada"] or r["activo"][m].any())
    causas = {}
    for s, e in zip(streams, evals):
        if e["tiene_caida"] and not e["detectada"]:
            k = por_que_se_perdio(s, regs[s["video"]], umbral)
            causas[k] = causas.get(k, 0) + 1
    atrib = {}
    for e in evals:
        for x in e["sub_falsas"]:
            nm = h17.NOMBRES.get(x, str(x))
            atrib[nm] = atrib.get(nm, 0) + 1
    return dict(eventos=len(con), detectadas=len(det), recall=len(det) / max(len(con), 1),
                recall_con_alerta_activa=activas / max(len(con), 1),
                recall_ic90=[float(np.percentile(rec_b, 5)), float(np.percentile(rec_b, 95))],
                falsas=falsas, horas_normales=horas, falsas_por_hora=falsas / max(horas, 1e-9),
                falsas_ic90=[float(np.percentile(fh_b, 5)), float(np.percentile(fh_b, 95))],
                latencia_mediana=float(np.median(lat)) if len(lat) else None,
                latencia_p90=float(np.percentile(lat, 90)) if len(lat) else None,
                causas_no_detectadas=causas, falsas_por_subclase=atrib)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--loso", action="store_true")
    g.add_argument("--checkpoint")
    ap.add_argument("--loso-dir", default=os.path.join(FALL, "checkpoints", "loso"))
    ap.add_argument("--semillas", type=int, default=1)
    ap.add_argument("--origen", default="le2i", choices=["le2i", "upfall"],
                    help="upfall trae 'laying': es lo unico que mide a alguien acostandose tranquilo")
    ap.add_argument("--fps", default="10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--espejo", action="store_true", help="test-time augmentation con el clip espejado")
    ap.add_argument("--umbral", type=float, help="umbral_reporte de la cabeza (default 0.50)")
    ap.add_argument("--quieto-s", type=float, help="ReglaCaida.quieto_s (default 2.0)")
    ap.add_argument("--set", nargs="*", default=[], metavar="CAMPO=VALOR",
                    help="cualquier campo de ConfigCaidas, ej. factor_pico_velocidad=3.0")
    ap.add_argument("--extender", type=float, default=0.0, metavar="SEG",
                    help="videos que terminan con la persona en el piso: repetir el ultimo "
                         "frame SEG segundos (ver extender()). 0 = estricto, como Le2i")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_kw = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if not hasattr(ConfigCaidas(), k):
            raise SystemExit(f"ConfigCaidas no tiene el campo '{k}'")
        if v.lower() in ("true", "false", "si", "no"):
            cfg_kw[k] = v.lower() in ("true", "si")
        else:
            cfg_kw[k] = float(v)
    if args.umbral is not None:
        cfg_kw["umbral_reporte"] = args.umbral
    if args.quieto_s is not None:
        reglas.ReglaCaida.quieto_s = args.quieto_s
    umbral = cfg_kw.get("umbral_reporte", ConfigCaidas().umbral_reporte)

    streams = [extender(s, args.extender) for s in h17.cargar_streams(args.origen)]
    grupos = sorted({s["sujeto"] for s in streams})
    if args.loso:
        mods = modelos_loso(grupos, device, args.loso_dir, args.semillas)
        clf_de = {gr: Clasificador(m, device, args.espejo) for gr, m in mods.items()}
        desc = f"app LOSO x{args.semillas}" + (" +espejo" if args.espejo else "")
    else:
        m = cargar_checkpoint(args.checkpoint, device)[0]
        fijo = Clasificador(m, device, args.espejo)
        clf_de = {gr: fijo for gr in grupos}
        desc = "app " + os.path.basename(args.checkpoint) + (" +espejo" if args.espejo else "")

    print(f"{len(streams)} streams de {args.origen} | {desc} | fps={args.fps} | umbral {umbral} | "
          f"quieto_s {reglas.ReglaCaida.quieto_s} | {cfg_kw or 'config por defecto'}")
    t0 = time.perf_counter()
    regs, evals = {}, []
    dd = os.path.join(DIAG, args.tag)
    os.makedirs(dd, exist_ok=True)
    for i, s in enumerate(streams):
        r = reproducir(s, clf_de[s["sujeto"]], args.fps, args.seed, cfg_kw)
        regs[s["video"]] = r
        evals.append(evaluar(s, r))
        np.savez_compressed(os.path.join(dd, f"{s['video']}.npz"), **r)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(streams)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    res = resumir(evals, regs, streams, umbral)
    print(f"\n=== {args.tag} | {desc} ===")
    print(f"caidas: {res['detectadas']}/{res['eventos']} = {res['recall']:.1%} "
          f"(IC90 {res['recall_ic90'][0]:.1%}-{res['recall_ic90'][1]:.1%})")
    print(f"        con alerta activa en la ventana: {res['recall_con_alerta_activa']:.1%}"
          + (f" | {sum(1 for s in streams if s.get('extendido'))} videos extendidos {args.extender:.0f}s"
             if args.extender else ""))
    print(f"falsas: {res['falsas']} en {res['horas_normales']:.2f} h = {res['falsas_por_hora']:.1f}/h "
          f"(IC90 {res['falsas_ic90'][0]:.1f}-{res['falsas_ic90'][1]:.1f})")
    print(f"latencia: mediana {res['latencia_mediana']:.2f}s, p90 {res['latencia_p90']:.2f}s")
    print(f"por que se pierden: {res['causas_no_detectadas']}")
    print(f"falsas por actividad: {res['falsas_por_subclase']}")

    with open(os.path.join(RESULTADOS, f"app_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(tag=args.tag, git=git_hash(), modelo=desc, fps=args.fps, umbral=umbral,
                       quieto_s=reglas.ReglaCaida.quieto_s, config=cfg_kw, espejo=args.espejo,
                       resumen=res, streams=evals), f, indent=1, default=float)
    fila = (f"| {args.tag} | {git_hash()} | {args.origen} | {args.fps} | {desc} | "
            f"{res['detectadas']}/{res['eventos']} | {res['recall']:.1%} | "
            f"{res['latencia_mediana']:.2f} | {res['latencia_p90']:.2f} | "
            f"{res['falsas_por_hora']:.2f} | - | cadena de la app; {args.notas} |")
    with open(os.path.join(RESULTADOS, "README.md"), "a", encoding="utf-8") as f:
        f.write(fila + "\n")
    print(f"\nguardado: resultados/app_{args.tag}.json (+ fila en resultados/README.md)")
