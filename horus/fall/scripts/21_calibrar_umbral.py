"""Elige el umbral de un checkpoint para que caiga en una tasa de falsas alarmas
objetivo, y lo guarda en los metadatos del propio checkpoint.

Por que: entrenar la misma receta con distinta semilla da modelos de IGUAL calidad
(AUC 0.872 +-0.004) pero con las probabilidades en escalas distintas. Con un umbral
fijo de 0.70, el recall de eventos se mueve 13 puntos y las falsas 4.6x entre
semillas -- no porque un modelo sea mejor, sino porque 0.70 cae en un lugar
distinto de cada escala. Calibrando cada checkpoint a la misma tasa de falsas, la
dispersion baja a 3.1 puntos (medido; ver resultados/README.md).

O sea: el umbral es una propiedad DEL CHECKPOINT, no de la configuracion. Por eso
viaja adentro del archivo y DetectorCaidas lo usa si esta.

Uso:
  python 21_calibrar_umbral.py --cache var_semilla_1 --checkpoint ../checkpoints/semilla_1.pt
  python 21_calibrar_umbral.py --cache var_semilla_1 --honesto    # solo reporta, no escribe

Reusa el cache de probabilidades de una corrida de 17_evaluar_evento.py, asi no
hay que volver a pasar el modelo por los videos.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.append("../src")
import checkpoint as ckpt  # noqa: E402

_aqui = os.path.dirname(os.path.abspath(__file__))


def _cargar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, os.path.join(_aqui, archivo))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h17 = _cargar("h17", "17_evaluar_evento.py")
h19 = _cargar("h19", "19_barrer_alarma.py")


def medir(streams, regs, umbral, persistencia):
    evals = []
    for s in streams:
        r = dict(regs[s["video"]])
        r["alarma_nueva"] = h19.alarmas_desde_cache(r, umbral, persistencia)
        evals.append(h17.evaluar_stream(s, r, 0.0))
    g = h17.agregar(evals, list(regs.values()))
    return g["recall"], g["falsas_por_hora"], evals


def buscar_umbral(streams, regs, objetivo, persistencia, paso=0.01):
    """El umbral mas permisivo (mayor recall) que no pasa la tasa objetivo de
    falsas. Permisivo y no 'el mas cercano' a proposito: ante la duda prefiero
    detectar, que es el punto de un sistema de seguridad."""
    mejor = None
    for u in np.arange(0.20, 0.996, paso):
        u = round(float(u), 3)
        recall, falsas, _ = medir(streams, regs, u, persistencia)
        if falsas is not None and falsas <= objetivo:
            if mejor is None or recall > mejor["recall"]:
                mejor = dict(umbral=u, recall=recall, falsas_por_hora=falsas)
    return mejor


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="tag de una corrida de 17_evaluar_evento.py")
    ap.add_argument("--checkpoint", help="checkpoint donde escribir el umbral (si se omite, solo reporta)")
    ap.add_argument("--objetivo", type=float, default=18.0, help="falsas alarmas por hora objetivo")
    ap.add_argument("--persistencia", type=float, default=1.0)
    ap.add_argument("--origen", default="le2i")
    ap.add_argument("--honesto", action="store_true",
                    help="calibra con 8 sujetos y mide en el noveno, para saber que rinde la calibracion de verdad")
    args = ap.parse_args()

    streams = h17.cargar_streams(args.origen)
    cache_dir = os.path.join(h17.CACHE_PROBS, args.cache)
    if not os.path.isdir(cache_dir):
        raise SystemExit(f"no existe el cache {cache_dir}. Corré 17_evaluar_evento.py --tag {args.cache} primero.")
    regs = {s["video"]: dict(np.load(os.path.join(cache_dir, f"{s['video']}.npz"))) for s in streams}

    base_recall, base_falsas, _ = medir(streams, regs, 0.70, args.persistencia)
    print(f"cache: {args.cache}  |  objetivo: {args.objetivo:.0f} falsas/hora\n")
    print(f"  con umbral 0.70 (el default): recall {base_recall:.1%}, {base_falsas:.1f} falsas/h")

    mejor = buscar_umbral(streams, regs, args.objetivo, args.persistencia)
    if mejor is None:
        raise SystemExit(f"ningun umbral baja de {args.objetivo} falsas/h en este cache")
    print(f"  calibrado a {mejor['umbral']:.2f}:          recall {mejor['recall']:.1%}, "
          f"{mejor['falsas_por_hora']:.1f} falsas/h")

    if args.honesto:
        # el umbral elegido sobre los mismos videos que se reportan sale optimista.
        # Aca: elijo con 8 sujetos y mido en el que queda afuera.
        print("\n  estimacion honesta (umbral elegido con 8 sujetos, medido en el noveno):")
        grupos = sorted({s["sujeto"] for s in streams})
        num_det = num_ev = 0
        falsas_tot = horas_tot = 0.0
        for g in grupos:
            otros = [s for s in streams if s["sujeto"] != g]
            este = [s for s in streams if s["sujeto"] == g]
            m = buscar_umbral(otros, {s["video"]: regs[s["video"]] for s in otros},
                              args.objetivo, args.persistencia, paso=0.02)
            if m is None:
                continue
            _, _, evals = medir(este, {s["video"]: regs[s["video"]] for s in este},
                                m["umbral"], args.persistencia)
            con = [e for e in evals if e["tiene_caida"]]
            num_det += sum(1 for e in con if e["detectada"]); num_ev += len(con)
            falsas_tot += sum(e["falsas"] for e in evals)
            horas_tot += sum(e["tiempo_normal"] for e in evals) / 3600
            print(f"    {g:<10} umbral {m['umbral']:.2f} -> {sum(1 for e in con if e['detectada'])}/{len(con)}")
        print(f"    TOTAL: recall {num_det/max(num_ev,1):.1%}  |  {falsas_tot/max(horas_tot,1e-9):.1f} falsas/h")

    if args.checkpoint:
        modelo_sd, meta = ckpt.cargar(args.checkpoint)
        config = dict(meta.get("config", {}))
        config["umbral"] = mejor["umbral"]
        config["umbral_calibrado_con"] = dict(cache=args.cache, origen=args.origen,
                                              objetivo_falsas_hora=args.objetivo,
                                              persistencia=args.persistencia)
        metricas = dict(meta.get("metricas", {}))
        metricas.update({f"calib_{k}": v for k, v in mejor.items()})

        class _Envoltorio(torch.nn.Module):
            def __init__(self, sd): super().__init__(); self._sd = sd
            def state_dict(self, *a, **k): return self._sd

        ckpt.guardar(args.checkpoint, _Envoltorio(modelo_sd), config=config, metricas=metricas)
        print(f"\n  umbral {mejor['umbral']:.2f} escrito en {args.checkpoint}")
        print(f"  {ckpt.describir(args.checkpoint)}")
