"""Evaluacion a nivel EVENTO en condiciones de despliegue.

Reproduce cada stream (16_armar_streams.py) causalmente por DetectorCaidas -- la
misma clase que corre en 12_webcam.py -- y mide lo que importa en produccion:
recall de eventos, latencia, falsas alarmas por hora y a que subclase OMNIFALL
se atribuyen, mas AUC a nivel ventana.

Modelo:
  --checkpoint RUTA  uno fijo (ej. el desplegado). OJO: modelo_demo_todo.pt entreno
                     con todos los sujetos, incluidos los de los streams -> numero
                     optimista, es "que hace hoy el sistema", no generalizacion.
  --loso             un modelo por sujeto dejandolo afuera (receta de 14_entrenar_final,
                     seed fija por fold), cacheado en ../checkpoints/loso/. Numero honesto.

fps:
  --fps nativo       cada frame del stream (24/25 le2i, ~18 upfall)
  --fps 10           submuestrea a ~10 fps con jitter (simula el despliegue medido)

Salida: ../resultados/evento_{tag}.json + fila en ../resultados/README.md.
Cache por stream en ../data/processed/streams_probs/{tag}/ (t, prob, pico_reciente,
clasifico, verticalidad, subclase) para barrer umbral/persistencia sin correr el modelo.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("../src")
from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia
from preproceso import normalizar_frame, visibilidad_confiable
from detector_caidas import DetectorCaidas, ConfigDetector
from dataset_stgcn import DatasetLista

STREAMS = "../data/keypoints/streams"
TODOS = "../data/processed/todos.npy"
CACHE_PROBS = "../data/processed/streams_probs"
LOSO_DIR = "../checkpoints/loso"
RESULTADOS = "../resultados"

MARGEN_POST_FALLEN_SEG = 1.0   # una alarma hasta 1s despues de que termina "fallen" cuenta como deteccion
NOMBRES = {0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting", 5: "lie_down",
           6: "lying", 7: "stand_up", 8: "standing", 9: "other", -1: "sin_anotar"}

# receta de 14_entrenar_final.py, para que el LOSO mida el mismo tipo de modelo que se despliega
EPOCAS, BATCH, LR = 40, 8, 1e-3


# ------------------------------------------------------------------ modelos

def cargar_modelo(ruta, A, device):
    m = STGCN(A, canales_entrada=4).to(device)
    m.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
    return m.eval()


def entrenar_sin(grupo, todos, A, device, seed):
    """Misma receta que 14_entrenar_final.py (Adam 1e-3, batch 8, 40 epocas, CE
    ponderada, guarda la ULTIMA epoca) pero sin el grupo dejado afuera."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    items = [d for d in todos if d["grupo"] != grupo]
    loader = DataLoader(DatasetLista(items, entrenamiento=True), batch_size=BATCH, shuffle=True)

    modelo = STGCN(A, canales_entrada=4).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)
    n_adl = sum(1 for d in items if d["label"] == "adl")
    n_fall = sum(1 for d in items if d["label"] == "fall")
    pesos = torch.tensor([1.0 / max(n_adl, 1), 1.0 / max(n_fall, 1)], dtype=torch.float32)
    pesos = pesos / pesos.sum() * 2
    criterio = nn.CrossEntropyLoss(weight=pesos.to(device))

    for _ in range(EPOCAS):
        modelo.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            criterio(modelo(x), y).backward()
            opt.step()
    return modelo.eval()


def modelos_loso(grupos, A, device):
    os.makedirs(LOSO_DIR, exist_ok=True)
    todos = None
    modelos = {}
    for i, g in enumerate(sorted(grupos)):
        ruta = os.path.join(LOSO_DIR, f"{g}.pt")
        if os.path.exists(ruta):
            modelos[g] = cargar_modelo(ruta, A, device)
            continue
        if todos is None:
            todos = list(np.load(TODOS, allow_pickle=True))
        print(f"  entrenando LOSO sin {g} ({i + 1}/{len(grupos)})...", flush=True)
        t0 = time.perf_counter()
        m = entrenar_sin(g, todos, A, device, seed=i)
        torch.save(m.state_dict(), ruta)
        print(f"    {time.perf_counter() - t0:.0f}s -> {ruta}")
        modelos[g] = m
    return modelos


# ------------------------------------------------------------------ streams

def cargar_streams(origen, max_streams=None):
    indice = pd.read_csv(os.path.join(STREAMS, "indice.csv"))
    indice = indice[indice.origen == origen]
    if max_streams:
        indice = indice.head(max_streams)
    streams = []
    for fila in indice.itertuples():
        z = np.load(os.path.join(STREAMS, origen, f"{fila.video}.npz"), allow_pickle=True)
        streams.append(dict(video=fila.video, sujeto=fila.sujeto, kp=z["kp"], t=z["t"],
                            sub=z["subclase_por_frame"], onset=float(z["onset_caida"]),
                            fin=float(z["fin_fallen"]), fps=float(z["fps"])))
    return streams


def indices_submuestreo(t, fps_objetivo, seed):
    """Elige que frames del stream veria una camara a ~fps_objetivo: instantes
    k/fps con jitter uniforme de +-20% del periodo, y para cada uno el frame
    fuente mas cercano. Simula la variabilidad real de un webcam."""
    rng = np.random.default_rng(seed)
    periodo = 1.0 / fps_objetivo
    objetivo = np.arange(0, t[-1], periodo)
    objetivo = objetivo + rng.uniform(-0.2, 0.2, len(objetivo)) * periodo
    objetivo = np.clip(objetivo, 0, t[-1])
    idx = np.searchsorted(t, objetivo)
    idx = np.clip(idx, 0, len(t) - 1)
    # el mas cercano, no el siguiente
    atras = np.clip(idx - 1, 0, len(t) - 1)
    idx = np.where(np.abs(t[atras] - objetivo) < np.abs(t[idx] - objetivo), atras, idx)
    return np.unique(idx)


def reproducir(stream, modelo, device, cfg, fps, seed):
    """Pasa el stream por el detector, causalmente. Devuelve un registro por
    frame procesado."""
    t, kp, sub = stream["t"], stream["kp"], stream["sub"]
    idx = np.arange(len(t)) if fps == "nativo" else indices_submuestreo(t, float(fps), seed)

    det = DetectorCaidas(modelo, device, cfg)
    reg = {k: [] for k in ("t", "prob", "alarma_nueva", "pico_reciente", "clasifico", "verticalidad", "sub")}
    for i in idx:
        fr = kp[i]
        if fr.reshape(-1).sum() == 0 or not visibilidad_confiable(fr):
            kp_norm = None
        else:
            kp_norm = normalizar_frame(fr)
        e = det.actualizar(kp_norm, float(t[i]))
        reg["t"].append(float(t[i]))
        reg["prob"].append(e.prob)
        reg["alarma_nueva"].append(e.alarma_nueva)
        reg["pico_reciente"].append(e.pico_reciente)
        reg["clasifico"].append(e.clasifico)
        reg["verticalidad"].append(np.nan if e.verticalidad is None else e.verticalidad)
        reg["sub"].append(int(sub[i]))
    return {k: np.array(v) for k, v in reg.items()}


# ------------------------------------------------------------------ metricas

def auc(scores, labels):
    """AUC = P(score_pos > score_neg) + 0.5 P(empate), via rangos promedio."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    orden = np.argsort(scores, kind="mergesort")
    rangos = np.empty(len(scores), float)
    s = scores[orden]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rangos[orden[i:j + 1]] = (i + j) / 2 + 1   # rango promedio, base 1
        i = j + 1
    u = rangos[labels].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def evaluar_stream(stream, reg, descarte_inicial):
    onset, fin = stream["onset"], stream["fin"]
    tiene_caida = not np.isnan(onset)
    t_al = reg["t"][reg["alarma_nueva"] & (reg["t"] >= descarte_inicial)]
    duracion = float(stream["t"][-1])

    if tiene_caida:
        v_ini, v_fin = onset, fin + MARGEN_POST_FALLEN_SEG
        validas = t_al[(t_al >= v_ini) & (t_al <= v_fin)]
        falsas = t_al[(t_al < v_ini) | (t_al > v_fin)]
        tiempo_normal = max(duracion - (min(v_fin, duracion) - v_ini), 0.0)
    else:
        validas, falsas = np.array([]), t_al
        tiempo_normal = duracion

    # a que subclase se atribuye cada falsa alarma: la activa en ese instante
    sub_falsas = [int(reg["sub"][np.searchsorted(reg["t"], ta)]) for ta in falsas]

    return dict(
        video=stream["video"], sujeto=stream["sujeto"], tiene_caida=tiene_caida,
        onset=onset, fin_fallen=fin, duracion=duracion,
        alarmas=t_al.tolist(),
        detectada=bool(len(validas)) if tiene_caida else None,
        latencia=float(validas[0] - onset) if len(validas) else None,
        falsas=len(falsas), sub_falsas=sub_falsas, tiempo_normal=tiempo_normal,
    )


def agregar(evals, regs):
    con_caida = [e for e in evals if e["tiene_caida"]]
    detectadas = [e for e in con_caida if e["detectada"]]
    lat = np.array([e["latencia"] for e in detectadas])
    falsas = sum(e["falsas"] for e in evals)
    horas = sum(e["tiempo_normal"] for e in evals) / 3600
    # todas las clasificaciones con prob fresca: positivo si la subclase activa es fall/fallen
    sc = np.concatenate([r["prob"][r["clasifico"]] for r in regs]) if regs else np.array([])
    lb = np.concatenate([np.isin(r["sub"][r["clasifico"]], [1, 2]) for r in regs]) if regs else np.array([])
    atrib = {}
    for e in evals:
        for s in e["sub_falsas"]:
            atrib[NOMBRES.get(s, str(s))] = atrib.get(NOMBRES.get(s, str(s)), 0) + 1
    return dict(
        streams=len(evals), eventos=len(con_caida), detectados=len(detectadas),
        recall=len(detectadas) / len(con_caida) if con_caida else None,
        latencia_mediana=float(np.median(lat)) if len(lat) else None,
        latencia_p90=float(np.percentile(lat, 90)) if len(lat) else None,
        falsas=int(falsas), horas_normales=round(horas, 3),
        falsas_por_hora=falsas / horas if horas > 0 else None,
        auc_ventana=auc(sc, lb) if len(sc) else None,
        n_ventanas=int(len(sc)),
        falsas_por_subclase=dict(sorted(atrib.items(), key=lambda kv: -kv[1])),
    )


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def fmt(v, p=""):
    return "-" if v is None else (f"{v:.1%}" if p == "%" else f"{v:.2f}")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="checkpoint fijo para todos los streams")
    g.add_argument("--loso", action="store_true", help="un modelo por sujeto dejado afuera")
    ap.add_argument("--origen", default="le2i", choices=["le2i", "upfall"])
    ap.add_argument("--fps", default="nativo", help="'nativo' o un numero (ej. 10)")
    ap.add_argument("--tag", required=True, help="nombre de la corrida (archivo de resultados)")
    ap.add_argument("--umbral", type=float)
    ap.add_argument("--persistencia", type=float)
    ap.add_argument("--tolerancia", type=float,
                    help="seg sin deteccion antes de resetear el buffer (default 0.6)")
    ap.add_argument("--factor-pico", type=float,
                    help="cuantas veces el baseline de velocidad cuenta como pico (default 3.0)")
    ap.add_argument("--sin-bypass", action="store_true",
                    help="no aceptar picos sin historial de verticalidad (bypass_historial_corto=False)")
    ap.add_argument("--sin-pico", action="store_true", help="alarma solo con modelo + persistencia")
    ap.add_argument("--descarte-inicial", type=float, default=0.0,
                    help="ignora alarmas en los primeros N seg (regla_persistencia usaba 1.5; el despliegue no descarta)")
    ap.add_argument("--max-streams", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    cfg = ConfigDetector()
    if args.umbral is not None:
        cfg.umbral = args.umbral
    if args.persistencia is not None:
        cfg.persistencia = args.persistencia
    if args.tolerancia is not None:
        cfg.tolerancia_sin_deteccion_seg = args.tolerancia
    if args.factor_pico is not None:
        cfg.factor_pico_velocidad = args.factor_pico
    if args.sin_bypass:
        cfg.bypass_historial_corto = False
    if args.sin_pico:
        cfg.exigir_pico = False

    streams = cargar_streams(args.origen, args.max_streams)
    grupos = sorted({s["sujeto"] for s in streams})
    print(f"{len(streams)} streams de {args.origen}, sujetos: {grupos}")

    A = construir_matriz_adyacencia()
    if args.loso:
        modelos = modelos_loso(grupos, A, device)
        modelo_de = lambda s: modelos[s["sujeto"]]
        desc_modelo = f"LOSO ({LOSO_DIR})"
    else:
        fijo = cargar_modelo(args.checkpoint, A, device)
        modelo_de = lambda s: fijo
        desc_modelo = args.checkpoint

    cache_dir = os.path.join(CACHE_PROBS, args.tag)
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.perf_counter()
    regs, evals = [], []
    for k, s in enumerate(streams, 1):
        reg = reproducir(s, modelo_de(s), device, cfg, args.fps, args.seed)
        np.savez(os.path.join(cache_dir, f"{s['video']}.npz"), **reg)
        regs.append(reg)
        evals.append(evaluar_stream(s, reg, args.descarte_inicial))
        if k % 40 == 0 or k == len(streams):
            print(f"  {k}/{len(streams)} streams ({time.perf_counter() - t0:.0f}s)", flush=True)

    global_ = agregar(evals, regs)
    por_sujeto = {}
    for g in grupos:
        ii = [i for i, e in enumerate(evals) if e["sujeto"] == g]
        por_sujeto[g] = agregar([evals[i] for i in ii], [regs[i] for i in ii])

    print(f"\n=== {args.tag} | {args.origen} | fps={args.fps} | modelo={desc_modelo} ===")
    print(f"eventos: {global_['detectados']}/{global_['eventos']} detectados (recall {fmt(global_['recall'], '%')})")
    print(f"latencia: mediana {fmt(global_['latencia_mediana'])}s, p90 {fmt(global_['latencia_p90'])}s")
    print(f"falsas alarmas: {global_['falsas']} en {global_['horas_normales']:.2f} h normales "
          f"-> {fmt(global_['falsas_por_hora'])} / hora")
    print(f"AUC a nivel ventana: {fmt(global_['auc_ventana'])} ({global_['n_ventanas']} ventanas)")
    print(f"falsas por subclase activa: {global_['falsas_por_subclase']}")
    print("\npor sujeto:")
    print(f"  {'sujeto':>10} {'eventos':>9} {'recall':>7} {'lat_med':>8} {'falsas/h':>9} {'auc':>6}")
    for g, m in por_sujeto.items():
        print(f"  {g:>10} {m['detectados']:>4}/{m['eventos']:<4} {fmt(m['recall'], '%'):>7} "
              f"{fmt(m['latencia_mediana']):>8} {fmt(m['falsas_por_hora']):>9} {fmt(m['auc_ventana']):>6}")

    os.makedirs(RESULTADOS, exist_ok=True)
    salida = dict(
        tag=args.tag, git=git_hash(), fecha=time.strftime("%Y-%m-%d %H:%M"),
        args=vars(args), config_detector=asdict(cfg), modelo=desc_modelo,
        margen_post_fallen_seg=MARGEN_POST_FALLEN_SEG,
        global_=global_, por_sujeto=por_sujeto, streams=evals,
    )
    ruta_json = os.path.join(RESULTADOS, f"evento_{args.tag}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1, ensure_ascii=False, default=float)

    readme = os.path.join(RESULTADOS, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write("# Resultados a nivel evento (17_evaluar_evento.py)\n\n"
                    "Una fila por corrida. Detalle completo en `evento_{tag}.json`.\n\n"
                    "| tag | git | origen | fps | modelo | eventos | recall | lat. med (s) | lat. p90 (s) | falsas/h | AUC vent. | notas |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    notas = []
    if args.sin_pico:
        notas.append("sin pico")
    if args.descarte_inicial:
        notas.append(f"descarte {args.descarte_inicial}s")
    if args.umbral is not None or args.persistencia is not None:
        notas.append(f"umbral={cfg.umbral} pers={cfg.persistencia}")
    if args.tolerancia is not None:
        notas.append(f"tolerancia={cfg.tolerancia_sin_deteccion_seg}")
    if args.factor_pico is not None:
        notas.append(f"factor_pico={cfg.factor_pico_velocidad}")
    if args.sin_bypass:
        notas.append("sin bypass")
    with open(readme, "a", encoding="utf-8") as f:
        f.write(f"| {args.tag} | {salida['git']} | {args.origen} | {args.fps} | "
                f"{'LOSO' if args.loso else os.path.basename(args.checkpoint)} | "
                f"{global_['detectados']}/{global_['eventos']} | {fmt(global_['recall'], '%')} | "
                f"{fmt(global_['latencia_mediana'])} | {fmt(global_['latencia_p90'])} | "
                f"{fmt(global_['falsas_por_hora'])} | {fmt(global_['auc_ventana'])} | {', '.join(notas)} |\n")
    print(f"\nguardado: {ruta_json}  (+ fila en {readme})")
