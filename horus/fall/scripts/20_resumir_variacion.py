"""Resume varias corridas de 17_evaluar_evento.py como una distribucion, no como
un numero.

La misma receta entrenada con distinta semilla mueve el recall de eventos varios
puntos. Entonces "este modelo da 75%" no es una medicion: es una tirada. Este
script junta las corridas que coincidan con un patron y reporta media, desvio y
rango -- lo que hay que mirar para decidir si una diferencia entre dos
configuraciones es real o es ruido.

Uso:
  python 20_resumir_variacion.py var_semilla          # todas las que empiecen asi
  python 20_resumir_variacion.py var_semilla ensamble # compara dos grupos
"""
import glob
import json
import os
import sys

import numpy as np

RESULTADOS = "../resultados"
METRICAS = [("recall", "recall de eventos", "%"),
            ("latencia_mediana", "latencia mediana", "s"),
            ("falsas_por_hora", "falsas por hora", ""),
            ("auc_ventana", "AUC a nivel ventana", "")]


def cargar_grupo(prefijo):
    corridas = []
    for f in sorted(glob.glob(os.path.join(RESULTADOS, f"evento_{prefijo}*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        corridas.append((os.path.basename(f).replace("evento_", "").replace(".json", ""), d))
    return corridas


def fmt(v, unidad):
    if v is None:
        return "-"
    return f"{v:.1%}" if unidad == "%" else (f"{v:.2f}{unidad}" if unidad else f"{v:.2f}")


def resumir(prefijo):
    corridas = cargar_grupo(prefijo)
    if not corridas:
        print(f"[{prefijo}] sin corridas en {RESULTADOS}")
        return None
    print(f"=== {prefijo} ({len(corridas)} corridas) ===")
    for tag, d in corridas:
        g = d["global_"]
        print(f"  {tag:<28} recall {fmt(g['recall'],'%'):>7}  lat {fmt(g['latencia_mediana'],'s'):>6}  "
              f"falsas/h {fmt(g['falsas_por_hora'],''):>6}")
    print()
    resumen = {}
    for clave, nombre, unidad in METRICAS:
        vals = np.array([d["global_"][clave] for _, d in corridas
                         if d["global_"].get(clave) is not None], dtype=float)
        if len(vals) == 0:
            continue
        resumen[clave] = vals
        linea = (f"  {nombre:<22} media {fmt(vals.mean(), unidad):>7}")
        if len(vals) > 1:
            linea += (f"   desvio {fmt(vals.std(ddof=1), unidad):>7}"
                      f"   rango {fmt(vals.min(), unidad)} a {fmt(vals.max(), unidad)}"
                      f"   (amplitud {fmt(vals.max()-vals.min(), unidad)})")
        print(linea)
    print()
    return resumen


if __name__ == "__main__":
    prefijos = sys.argv[1:] or ["var_semilla"]
    resumenes = {p: resumir(p) for p in prefijos}

    validos = {p: r for p, r in resumenes.items() if r}
    if len(validos) >= 2:
        print("=== comparacion entre grupos ===")
        nombres = list(validos)
        for clave, nombre, unidad in METRICAS:
            if not all(clave in validos[p] for p in nombres):
                continue
            partes = []
            for p in nombres:
                v = validos[p][clave]
                partes.append(f"{p}: {fmt(v.mean(), unidad)}" + (f" ±{fmt(v.std(ddof=1), unidad)}" if len(v) > 1 else ""))
            print(f"  {nombre:<22} " + "   vs   ".join(partes))
        # una diferencia solo cuenta si supera la dispersion del grupo con mas corridas
        ref = max(nombres, key=lambda p: len(validos[p]["recall"]))
        disp = validos[ref]["recall"]
        if len(disp) > 1:
            print(f"\n  OJO: '{ref}' tiene una amplitud de {(disp.max()-disp.min())*100:.1f} puntos de recall "
                  f"solo por cambiar la semilla.\n  Cualquier diferencia menor a eso entre dos configuraciones "
                  f"no se puede atribuir a la configuracion.")
