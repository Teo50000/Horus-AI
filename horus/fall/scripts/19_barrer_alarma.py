"""Barrido de umbral x persistencia sobre la cache de una corrida de
17_evaluar_evento.py, sin volver a correr el modelo.

La cache tiene por frame: t, prob (ultima P(fall)), clasifico (True donde el
modelo corrio), pico_reciente (gate ya evaluado) y sub. Reproduzco la logica de
alarma de DetectorCaidas sobre esos datos: la racha arranca/corta en los
instantes con clasifico=True, y prob == 0.0 exacto marca un reset del detector
(el softmax nunca da 0.0). Con eso umbral y persistencia se pueden barrer
fielmente; lo que NO se puede barrer desde la cache es nada que cambie el gate
(tolerancia, factor de pico, etc.) porque pico_reciente ya viene calculado.

Uso: python 19_barrer_alarma.py --tag fase4a_defaults_loso_nativo
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("../src")
_spec = importlib.util.spec_from_file_location("h17", os.path.join(os.path.dirname(__file__), "17_evaluar_evento.py"))
h17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h17)


def alarmas_desde_cache(reg, umbral, persistencia):
    """Misma maquina de estados que DetectorCaidas.actualizar para la alarma."""
    t, prob, clasif, pico = reg["t"], reg["prob"], reg["clasifico"], reg["pico_reciente"]
    nueva = np.zeros(len(t), dtype=bool)
    inicio_racha, activa = None, False
    for i in range(len(t)):
        if prob[i] == 0.0:            # reset del detector
            inicio_racha, activa = None, False
            continue
        if not clasif[i]:
            continue
        if prob[i] >= umbral:
            if inicio_racha is None:
                inicio_racha = t[i]
            elif not activa and (t[i] - inicio_racha) >= persistencia:
                if pico[i]:
                    activa = True
                    nueva[i] = True
        else:
            inicio_racha, activa = None, False
    return nueva


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="corrida de 17 cuya cache se barre")
    ap.add_argument("--origen", default="le2i")
    ap.add_argument("--umbrales", default="0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--persistencias", default="0,0.25,0.5,0.75,1.0,1.5")
    args = ap.parse_args()

    streams = h17.cargar_streams(args.origen)
    cache_dir = os.path.join(h17.CACHE_PROBS, args.tag)
    regs = {s["video"]: dict(np.load(os.path.join(cache_dir, f"{s['video']}.npz"))) for s in streams}

    # sanity: reproducir la corrida original con su propia config
    import json
    orig = json.load(open(os.path.join(h17.RESULTADOS, f"evento_{args.tag}.json"), encoding="utf-8"))
    cfg = orig["config_detector"]
    evals = []
    for s in streams:
        r = dict(regs[s["video"]])
        r["alarma_nueva"] = alarmas_desde_cache(r, cfg["umbral"], cfg["persistencia"])
        evals.append(h17.evaluar_stream(s, r, orig["args"].get("descarte_inicial", 0.0)))
    g = h17.agregar(evals, list(regs.values()))
    print(f"reproduccion de {args.tag} (umbral={cfg['umbral']}, pers={cfg['persistencia']}): "
          f"{g['detectados']}/{g['eventos']} detectados, {g['falsas']} falsas  |  "
          f"original: {orig['global_']['detectados']}/{orig['global_']['eventos']}, {orig['global_']['falsas']} falsas")
    assert (g["detectados"], g["falsas"]) == (orig["global_"]["detectados"], orig["global_"]["falsas"]), \
        "la reproduccion desde cache no coincide con la corrida original"

    umbrales = [float(x) for x in args.umbrales.split(",")]
    pers = [float(x) for x in args.persistencias.split(",")]
    print(f"\n{'umbral':>7} {'pers(s)':>8} {'recall':>7} {'lat_med':>8} {'lat_p90':>8} {'falsas/h':>9}")
    filas = []
    for u in umbrales:
        for p in pers:
            evals = []
            for s in streams:
                r = dict(regs[s["video"]])
                r["alarma_nueva"] = alarmas_desde_cache(r, u, p)
                evals.append(h17.evaluar_stream(s, r, 0.0))
            g = h17.agregar(evals, list(regs.values()))
            filas.append(dict(umbral=u, persistencia=p, recall=g["recall"], lat_med=g["latencia_mediana"],
                              lat_p90=g["latencia_p90"], falsas_h=g["falsas_por_hora"]))
            print(f"{u:>7.2f} {p:>8.2f} {g['recall']:>7.1%} {g['latencia_mediana']:>8.2f} "
                  f"{g['latencia_p90']:>8.2f} {g['falsas_por_hora']:>9.2f}")
        print()

    salida = os.path.join(h17.RESULTADOS, f"barrido_alarma_{args.tag}.csv")
    pd.DataFrame(filas).to_csv(salida, index=False)
    print(f"guardado: {salida}")
