# -*- coding: utf-8 -*-
"""
benchmark_objetos.py · HORUS — cuánto gana cada optimización, medido.

Hermano de 03_backbone/medir_fps.py, pero para el camino completo de objetos:
preproceso + backbone + cabeza + decode + NMS. O sea, lo que realmente tarda
un frame desde que sale de la cámara hasta que hay una caja.

Todas las variantes usan LOS MISMOS PESOS, así la columna "dets" es
comparable: si una variante detecta distinto, algo rompió.

Uso
---
    python benchmark_objetos.py
    python benchmark_objetos.py --pesos checkpoints/head_best.pt --frames 200
    python benchmark_objetos.py --trt engines/objetos_fp16.plan \\
                                --trt engines/objetos_int8.plan
    python benchmark_objetos.py --camaras 8 --target-fps 5 --csv resultados.csv

Variantes (--variantes para elegir un subconjunto):
    baseline      el camino viejo: backbone.encode() + head.predict()
    fp32          motor nuevo en fp32
    fp16          motor + autocast fp16              <- el salto grande
    bf16          motor + autocast bf16
    fp16-full     motor con el modelo .half() entero
    fp16-graphs   fp16 + CUDA Graphs                 <- suele ser el mejor
    fp16-compile  fp16 + torch.compile (tarda en arrancar)
    trt-*         los engines .plan que le pases con --trt
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_AQUI = os.path.dirname(os.path.abspath(__file__))
if _AQUI not in sys.path:
    sys.path.insert(0, _AQUI)

from objects_engine import EngineConfig, ObjectsEngine  # noqa: E402

_VARIANTES_DEF = ["baseline", "fp32", "fp16", "fp16-graphs"]
_TODAS = ["baseline", "fp32", "fp16", "bf16", "fp16-full",
          "fp16-graphs", "fp16-compile"]


@dataclass
class Medicion:
    nombre: str
    media_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    fps: float = 0.0
    vram_mb: float = 0.0
    dets: float = 0.0
    error: str = ""
    speedup: float = 1.0


def _percentil(v: Sequence[float], p: float) -> float:
    s = sorted(v)
    if not s:
        return 0.0
    i = min(len(s) - 1, max(0, int(round(p / 100.0 * len(s))) - 1))
    return s[i]


def _sync(es_cuda: bool) -> None:
    if es_cuda:
        torch.cuda.synchronize()


def _limpiar() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
def _cfg_variante(nombre: str, base: Dict) -> EngineConfig:
    c = EngineConfig(**base)
    if nombre == "fp32":
        c.precision, c.cuda_graphs, c.compile = "fp32", False, False
    elif nombre == "fp16":
        c.precision, c.cuda_graphs, c.compile = "fp16", False, False
    elif nombre == "bf16":
        c.precision, c.cuda_graphs, c.compile = "bf16", False, False
    elif nombre == "fp16-full":
        c.precision, c.cuda_graphs, c.compile = "fp16-full", False, False
    elif nombre == "fp16-graphs":
        c.precision, c.cuda_graphs, c.compile = "fp16", True, False
    elif nombre == "fp16-compile":
        c.precision, c.cuda_graphs, c.compile = "fp16", False, True
    elif nombre.startswith("trt:"):
        c.precision, c.cuda_graphs, c.compile = "fp32", False, False
        c.trt_engine = nombre.split(":", 1)[1]
    return c


def medir_baseline(base: Dict, frames: List[np.ndarray], warmup: int,
                   n: int) -> Medicion:
    """El camino de probar_objetos.py, tal cual está hoy."""
    import cv2
    from shared_backbone import BackboneConfig, SharedBackbone
    from objects_head import ObjectsHead, ObjectsHeadConfig

    _limpiar()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    es_cuda = dev.type == "cuda"

    # Geometría: del checkpoint si lo hay, si no la que venga en `base`.
    ck = {}
    if base.get("pesos"):
        ck = torch.load(base["pesos"], map_location=dev)
        if not isinstance(ck, dict) or "head" not in ck:
            ck = {"head": ck}
    tam = int(ck.get("tam") or base.get("input_size") or 384)
    niveles = tuple(ck.get("fpn_levels") or base.get("fpn_levels")
                    or ("p3", "p4", "p5"))
    anclas = tuple(tuple(x) for x in (ck.get("anchor_sizes")
                                      or base.get("anchor_sizes")
                                      or ((32,), (64,), (128,))))

    bb = SharedBackbone(BackboneConfig(pretrained=False, input_size=tam,
                                       freeze_encoder=True)).to(dev).eval()
    if base.get("pesos_backbone"):
        bb.load_state_dict(torch.load(base["pesos_backbone"],
                                      map_location=dev), strict=False)
    cfg = ObjectsHeadConfig(fpn_levels=niveles, anchor_sizes=anclas,
                            score_thresh=base["score_thresh"])
    head = ObjectsHead(cfg).to(dev).eval()
    if ck:
        head.load_state_dict(ck["head"])

    def un_frame(f: np.ndarray) -> int:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        feats = bb.encode(rgb, camera_id="bench")
        h, w = feats.input_size
        dets = head.predict(feats.pyramid, image_sizes=[(h, w)],
                            camera_ids=["bench"], frame_idxs=[0])[0]
        return len(dets)

    try:
        with torch.no_grad():
            for i in range(warmup):
                un_frame(frames[i % len(frames)])
            _sync(es_cuda)

            lat, dets = [], []
            for i in range(n):
                t0 = time.perf_counter()
                d = un_frame(frames[i % len(frames)])
                _sync(es_cuda)
                lat.append((time.perf_counter() - t0) * 1000.0)
                dets.append(d)
    except Exception as e:
        return Medicion("baseline", error=str(e)[:60])

    media = sum(lat) / len(lat)
    return Medicion("baseline", media, _percentil(lat, 50), _percentil(lat, 95),
                    _percentil(lat, 99), 1000.0 / media,
                    torch.cuda.max_memory_allocated() / 1e6 if es_cuda else 0.0,
                    float(np.mean(dets)))


def medir_variante(nombre: str, base: Dict, frames: List[np.ndarray],
                   warmup: int, n: int) -> Medicion:
    _limpiar()
    try:
        cfg = _cfg_variante(nombre, base)
        eng = ObjectsEngine(cfg)
    except Exception as e:
        return Medicion(nombre, error=str(e)[:60])

    es_cuda = eng.es_cuda
    try:
        for i in range(warmup):
            eng.infer(frames[i % len(frames)], camera_id="bench")
        _sync(es_cuda)

        lat, dets = [], []
        for i in range(n):
            t0 = time.perf_counter()
            r = eng.infer(frames[i % len(frames)], camera_id="bench")
            _sync(es_cuda)
            lat.append((time.perf_counter() - t0) * 1000.0)
            dets.append(len(r.detections))
    except Exception as e:
        return Medicion(nombre, error=str(e)[:60])

    media = sum(lat) / len(lat)
    m = Medicion(nombre, media, _percentil(lat, 50), _percentil(lat, 95),
                 _percentil(lat, 99), 1000.0 / media,
                 eng.vram_mb(), float(np.mean(dets)))
    del eng
    _limpiar()
    return m


def medir_batch(base: Dict, frames: List[np.ndarray], batches: Sequence[int],
                n: int) -> List[Tuple[int, float, float]]:
    """Throughput con varias cámaras en el mismo forward."""
    salida = []
    for b in batches:
        _limpiar()
        try:
            cfg = _cfg_variante("fp16-graphs", base)
            cfg.max_batch = b
            eng = ObjectsEngine(cfg)
            lote = [frames[i % len(frames)] for i in range(b)]
            cams = [f"c{i}" for i in range(b)]
            for _ in range(5):
                eng.infer_batch(lote, cams)
            _sync(eng.es_cuda)
            t0 = time.perf_counter()
            for _ in range(n):
                eng.infer_batch(lote, cams)
            _sync(eng.es_cuda)
            dt = time.perf_counter() - t0
            salida.append((b, (b * n) / dt, (dt / n) * 1000.0))
            del eng
        except Exception as e:
            print(f"  batch {b}: falló ({str(e)[:60]})")
    return salida



def _barrer(args) -> int:
    """Mide el mismo motor a varios input_size. Subir la resolución ayuda a los
    objetos chicos (humo lejano) pero el cómputo escala con el AREA: pasar de
    384 a 640 es 2,8x más caro. Esta tabla es para decidir con números."""
    es_cuda = torch.cuda.is_available()
    print("=" * 78)
    print("BARRIDO DE RESOLUCIÓN · cuánto cuesta ver objetos más chicos")
    print("=" * 78)
    if es_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ sin CUDA: los números no sirven para dimensionar")
    print(f"  objetivo: {args.camaras} cámaras a {args.target_fps:g} FPS "
          f"= {args.camaras * args.target_fps:g} frames/seg\n")

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (args.alto, args.ancho, 3), dtype=np.uint8)
              for _ in range(8)]

    print(f"  {'input':>7} {'media':>9} {'p95':>8} {'FPS':>8} {'VRAM':>8} "
          f"{'cámaras':>9}  {'costo':>7}")
    print("  " + "-" * 66)
    ref = None
    for size in args.barrer_tamanos:
        base = dict(pesos=args.pesos, pesos_backbone=args.pesos_backbone,
                    input_size=size, score_thresh=args.umbral, max_batch=1,
                    warmup=5, verboso=False)
        if not args.pesos:      # sin pesos hay que fijar la geometría a mano
            base["fpn_levels"] = ("p3", "p4", "p5")
            base["anchor_sizes"] = ((32,), (64,), (128,))
        nombre = "fp16-graphs" if es_cuda else "fp32"
        m = medir_variante(nombre, base, frames, args.warmup, args.frames)
        if m.error:
            print(f"  {size:>5}px  ⚠ {m.error}")
            continue
        if ref is None:
            ref = m.media_ms
        cams = m.fps / args.target_fps
        marca = "  <- alcanza" if cams >= args.camaras else ""
        print(f"  {size:>5}px {m.media_ms:>8.1f}m {m.p95_ms:>7.1f}m "
              f"{m.fps:>8.1f} {m.vram_mb:>7.0f}M {cams:>8.1f}  "
              f"{m.media_ms/ref:>6.2f}x{marca}")
    print("  " + "-" * 66)
    print("\n  Subir la resolución ayuda al humo lejano y a la pistola chica,")
    print("  pero el costo escala con el área. Elegí el más grande que todavía")
    print("  te dé las cámaras que necesitás.")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark del camino de objetos")
    ap.add_argument("--pesos", default=None)
    ap.add_argument("--pesos-backbone", default=None)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--barrer-tamanos", type=int, nargs="+", default=None,
                    metavar="PX",
                    help="medir varios input_size y comparar FPS "
                         "(ej: --barrer-tamanos 384 512 640)")
    ap.add_argument("--frames", type=int, default=100, help="frames medidos")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--umbral", type=float, default=0.30)
    ap.add_argument("--variantes", nargs="+", default=None,
                    help=f"por defecto: {' '.join(_VARIANTES_DEF)} · todas: {' '.join(_TODAS)}")
    ap.add_argument("--todas", action="store_true")
    ap.add_argument("--trt", action="append", default=[],
                    help="ruta a un .plan (se puede repetir)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--sin-batch", action="store_true")
    ap.add_argument("--camaras", type=int, default=4)
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--video", default=None,
                    help="usar frames de un video real en vez de ruido")
    args = ap.parse_args()

    if args.barrer_tamanos:
        return _barrer(args)

    es_cuda = torch.cuda.is_available()
    print("=" * 78)
    print("BENCHMARK · camino completo de objetos (preproceso + red + NMS)")
    print("=" * 78)
    if es_cuda:
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU        : {p.name}  ({p.total_memory/1e9:.0f} GB, "
              f"CC {p.major}.{p.minor})")
    else:
        print(f"  CPU        : {platform.processor() or 'desconocido'} "
              f"({torch.get_num_threads()} hilos)")
        print("  ⚠ sin CUDA: esto mide el fallback, no la GPU. Los números no "
              "sirven para dimensionar.")
    print(f"  torch      : {torch.__version__}")
    print(f"  frame      : {args.ancho}x{args.alto} -> {args.input_size}px")
    print(f"  medición   : {args.frames} frames (warmup {args.warmup})")
    print(f"  objetivo   : {args.camaras} cámaras a {args.target_fps:g} FPS\n")

    # --- frames de prueba -------------------------------------------------
    frames: List[np.ndarray] = []
    if args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        while len(frames) < 16:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(f, (args.ancho, args.alto)))
        cap.release()
        print(f"  usando {len(frames)} frames de {args.video}\n")
    if not frames:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (args.alto, args.ancho, 3), dtype=np.uint8)
                  for _ in range(8)]

    # --- pesos compartidos ------------------------------------------------
    pesos, pesos_bb = args.pesos, args.pesos_backbone
    tmp = None
    if not pesos:
        # Sin checkpoint, cada variante saldría con pesos random distintos y la
        # columna "dets" no serviría para nada. Fijamos unos pesos comunes.
        print("  sin --pesos: genero un checkpoint random compartido para que "
              "todas las variantes detecten lo mismo\n")
        torch.manual_seed(0)
        eng0 = ObjectsEngine(EngineConfig(
            input_size=args.input_size, precision="fp32", cuda_graphs=False,
            warmup=0, score_thresh=args.umbral, verboso=False))
        torch.nn.init.normal_(eng0.head.cls_logits.bias, mean=-3.0, std=1.2)
        tmp = Path(_AQUI) / ".bench_tmp"
        tmp.mkdir(exist_ok=True)
        pesos, pesos_bb = str(tmp / "head.pt"), str(tmp / "bb.pt")
        torch.save(eng0.head.state_dict(), pesos)
        torch.save(eng0.backbone.state_dict(), pesos_bb)
        del eng0
        _limpiar()

    base = dict(
        pesos=pesos, pesos_backbone=pesos_bb, input_size=args.input_size,
        score_thresh=args.umbral, max_batch=1, warmup=5,
        fpn_levels=("p3", "p4", "p5"), anchor_sizes=((32,), (64,), (128,)),
        verboso=False,
    )
    # Con checkpoint real, el motor lee del .pt las anclas y el input_size con
    # los que se entrenó. Sacamos los defaults para no pisárselos.
    if args.pesos:
        base.pop("fpn_levels", None)
        base.pop("anchor_sizes", None)
        base.pop("input_size", None)

    # --- qué variantes ----------------------------------------------------
    variantes = args.variantes or (_TODAS if args.todas else list(_VARIANTES_DEF))
    if not es_cuda:
        variantes = [v for v in variantes if v in ("baseline", "fp32")]
        print("  (sin CUDA solo corro baseline y fp32)\n")
    variantes += [f"trt:{p}" for p in args.trt]

    # --- medir ------------------------------------------------------------
    res: List[Medicion] = []
    for v in variantes:
        etiqueta = v if not v.startswith("trt:") else "trt:" + Path(v[4:]).stem
        print(f"  midiendo {etiqueta} ...", end="", flush=True)
        t0 = time.perf_counter()
        m = (medir_baseline(base, frames, args.warmup, args.frames)
             if v == "baseline"
             else medir_variante(v, base, frames, args.warmup, args.frames))
        m.nombre = etiqueta
        res.append(m)
        print(f" {time.perf_counter()-t0:.0f}s" +
              (f"  ⚠ {m.error}" if m.error else f"  {m.media_ms:.1f} ms"))

    ref = next((m for m in res if m.nombre == "baseline" and not m.error), None)
    if ref is None:
        ref = next((m for m in res if not m.error), None)
    for m in res:
        if ref and not m.error and m.media_ms > 0:
            m.speedup = ref.media_ms / m.media_ms

    # --- tabla ------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"  {'variante':<16} {'media':>8} {'p50':>7} {'p95':>7} {'p99':>7} "
          f"{'FPS':>7} {'VRAM':>8} {'dets':>6} {'vs base':>8}")
    print("-" * 78)
    for m in res:
        if m.error:
            print(f"  {m.nombre:<16} {'—':>8} {'':>7} {'':>7} {'':>7} {'':>7} "
                  f"{'':>8} {'':>6}   {m.error}")
            continue
        print(f"  {m.nombre:<16} {m.media_ms:>7.1f}m {m.p50_ms:>6.1f}m "
              f"{m.p95_ms:>6.1f}m {m.p99_ms:>6.1f}m {m.fps:>7.1f} "
              f"{m.vram_mb:>7.0f}M {m.dets:>6.1f} {m.speedup:>7.2f}x")
    print("-" * 78)

    # --- batch ------------------------------------------------------------
    if not args.sin_batch and es_cuda:
        print("\nThroughput por batch (varias cámaras en un mismo forward, fp16+graphs):")
        for b, ips, lat in medir_batch(base, frames, args.batches, max(20, args.frames // 4)):
            print(f"  batch {b}: {ips:7.1f} img/s  ({lat:5.1f} ms por lote)  "
                  f"-> {ips/args.target_fps:5.1f} cámaras a {args.target_fps:g} FPS")

    # --- veredicto --------------------------------------------------------
    mejor = max((m for m in res if not m.error), key=lambda m: m.fps, default=None)
    print("\n" + "=" * 78)
    print("VEREDICTO")
    print("=" * 78)
    if mejor is None:
        print("  No pude medir ninguna variante.")
    else:
        cams = mejor.fps / args.target_fps
        print(f"  La más rápida es '{mejor.nombre}': {mejor.media_ms:.1f} ms "
              f"por frame ({mejor.fps:.1f} FPS), p95 {mejor.p95_ms:.1f} ms.")
        if ref and mejor.nombre != ref.nombre:
            print(f"  Contra el camino actual: {mejor.speedup:.2f}x más rápido "
                  f"({ref.media_ms:.1f} -> {mejor.media_ms:.1f} ms).")
        if cams >= args.camaras:
            print(f"  ✅ Alcanza para ~{cams:.1f} cámaras a {args.target_fps:g} FPS "
                  f"(necesitás {args.camaras}).")
        else:
            print(f"  ⚠ Da para ~{cams:.1f} cámaras a {args.target_fps:g} FPS y "
                  f"necesitás {args.camaras}. Opciones:")
            print("     - procesar en batch (ver la tabla de throughput),")
            print("     - exportar a TensorRT INT8 (exportar_objetos.py --trt --int8),")
            print("     - bajar input_size a 320 o 256,")
            print("     - repartir cámaras en más equipos.")
        raros = [m for m in res if not m.error and ref and
                 abs(m.dets - ref.dets) > max(1.0, 0.05 * max(ref.dets, 1))]
        if raros:
            print("\n  ⚠ Estas variantes detectan distinto que el baseline "
                  "(revisá precisión/calibración):")
            for m in raros:
                print(f"     {m.nombre}: {m.dets:.1f} dets vs {ref.dets:.1f}")

    # --- csv ---------------------------------------------------------------
    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["variante", "media_ms", "p50_ms", "p95_ms", "p99_ms",
                        "fps", "vram_mb", "dets_prom", "speedup", "error"])
            for m in res:
                w.writerow([m.nombre, f"{m.media_ms:.2f}", f"{m.p50_ms:.2f}",
                            f"{m.p95_ms:.2f}", f"{m.p99_ms:.2f}", f"{m.fps:.2f}",
                            f"{m.vram_mb:.0f}", f"{m.dets:.2f}",
                            f"{m.speedup:.3f}", m.error])
        print(f"\n  CSV guardado en {args.csv}")

    if tmp:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
