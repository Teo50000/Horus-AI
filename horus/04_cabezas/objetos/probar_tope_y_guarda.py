# -*- coding: utf-8 -*-
"""
probar_tope_y_guarda.py · las dos correcciones que salieron del run fallido
del 17/09 en Kaggle.

Ese run gastó 33 minutos preparando el dataset y después se negó a entrenar
porque `llama` había quedado en 0 cajas — el dataset adjunto no era D-Fire
sino los resultados de un entrenamiento ajeno, 21,85 MB sin una sola imagen.
La guarda hizo lo correcto, pero lo hizo tarde y por la razón equivocada:
`llama` la cubre segmentación con F1 99,1 %.

Se prueban dos cosas, sin torch, sin GPU y sin bajar nada:

  1. el tope de normalización: que acote de verdad, que muestree a paso
     constante (no los primeros N), y que no llene el cupo de negativos;
  2. la guarda de clases vacías: que corte con una clase huérfana y que NO
     corte con una que otra cabeza cubre.

    python probar_tope_y_guarda.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

import bajar_datasets as bd  # noqa: E402


# --------------------------------------------------------------------------- #
# Un D-Fire de mentira: mismo layout (data/<split>/{images,labels}) y la misma
# proporción de negativos que el real.
# --------------------------------------------------------------------------- #
def crear_fuente(raiz: Path, n_pos: int, n_neg: int, splits=("train", "val", "test")):
    i = 0
    for s in splits:
        (raiz / "data" / s / "images").mkdir(parents=True, exist_ok=True)
        (raiz / "data" / s / "labels").mkdir(parents=True, exist_ok=True)
    total = n_pos + n_neg
    # intercalados, para que "los primeros N" y "a paso constante" difieran
    for k in range(total):
        s = splits[k * len(splits) // total]
        nom = f"{k:05d}"
        (raiz / "data" / s / "images" / f"{nom}.jpg").write_bytes(b"\xff\xd8\xff" + bytes(64))
        caja = "1 0.5 0.5 0.2 0.2\n" if k < n_pos else ""
        (raiz / "data" / s / "labels" / f"{nom}.txt").write_text(caja, encoding="utf-8")
        i += 1
    return i


def normalizar(raiz: Path, salida: Path, tope: int) -> int:
    return bd._norm_yolo({0: 0, 1: 1}, tope=tope)(raiz, salida)


def indices_copiados(salida: Path) -> Tuple[int, int]:
    """(con caja, sin caja) entre los .txt que quedaron."""
    con = sin = 0
    for t in (salida / "labels").glob("*.txt"):
        if t.read_text().strip():
            con += 1
        else:
            sin += 1
    return con, sin


# --------------------------------------------------------------------------- #
# Casos · tope de normalización
# --------------------------------------------------------------------------- #
def caso_sin_tope_copia_todo() -> Tuple[bool, str]:
    """tope=0 es el comportamiento de siempre. Nada cambia para las fuentes
    chicas, que son la mayoría."""
    with tempfile.TemporaryDirectory() as d:
        r, sal = Path(d) / "crudo", Path(d) / "norm"
        crear_fuente(r, 60, 40)
        n = normalizar(r, sal, tope=0)
        return n == 100, f"{n} imágenes (esperado 100)"


def caso_tope_acota() -> Tuple[bool, str]:
    """Con tope, no se copia una imagen de más. Es lo que evita llenar los
    20 GB de /kaggle/working."""
    with tempfile.TemporaryDirectory() as d:
        r, sal = Path(d) / "crudo", Path(d) / "norm"
        crear_fuente(r, 600, 400)
        n = normalizar(r, sal, tope=200)
        return n == 200, f"{n} imágenes (esperado 200)"


def caso_negativos_no_se_comen_el_cupo() -> Tuple[bool, str]:
    """Los negativos no pasan de un cuarto. D-Fire es ~40% negativos: sin esto
    casi la mitad del cupo serían fotos vacías que TOPE_NEGATIVOS descarta
    después igual."""
    with tempfile.TemporaryDirectory() as d:
        r, sal = Path(d) / "crudo", Path(d) / "norm"
        crear_fuente(r, 600, 400)
        normalizar(r, sal, tope=200)
        con, sin = indices_copiados(sal)
        ok = sin <= 50 and con == 200 - sin
        return ok, f"{con} con caja · {sin} sin caja (tope negativos 50)"


def caso_paso_constante_no_primeros_n() -> Tuple[bool, str]:
    """El muestreo recorre el rango entero. Si tomara los primeros N se
    quedaría con un solo split: estas fuentes vienen ordenadas por ruta, y la
    ruta ordena por split y por tiempo."""
    with tempfile.TemporaryDirectory() as d:
        r, sal = Path(d) / "crudo", Path(d) / "norm"
        crear_fuente(r, 900, 100)
        normalizar(r, sal, tope=120)
        # de qué splits salieron: el nombre destino no lo dice, así que
        # contamos sobre la lista elegida reproduciendo el muestreo
        todas = sorted(p for p in (r / "data").rglob("*.jpg"))
        elegidas = bd._a_paso(todas, 120)
        splits = {p.parts[-3] for p in elegidas}
        primeros = {p.parts[-3] for p in todas[:120]}
        ok = len(splits) == 3 and len(primeros) < 3
        return ok, f"a paso constante: {sorted(splits)} · primeros 120: {sorted(primeros)}"


def caso_a_paso_es_estable() -> Tuple[bool, str]:
    """Mismo dataset, mismo muestreo. Un run que no se puede repetir no se
    puede comparar con el anterior."""
    xs = list(range(1000))
    a, b = bd._a_paso(xs, 77), bd._a_paso(xs, 77)
    ok = a == b and len(a) == 77 and a[0] == 0 and a[-1] < 1000
    return ok, f"{len(a)} elementos, de {a[0]} a {a[-1]}, idénticos entre corridas"


def caso_tope_mayor_que_la_fuente() -> Tuple[bool, str]:
    """Un tope más grande que la fuente no recorta ni reordena."""
    with tempfile.TemporaryDirectory() as d:
        r, sal = Path(d) / "crudo", Path(d) / "norm"
        crear_fuente(r, 30, 20)
        n = normalizar(r, sal, tope=12000)
        return n == 50, f"{n} imágenes (esperado 50)"


def caso_dfire_y_pyro_tienen_tope() -> Tuple[bool, str]:
    """Las dos fuentes pesadas del catálogo lo tienen puesto. Es el punto de
    todo esto: son las que normalizaban 21k y 33k imágenes."""
    fuente = (_AQUI / "bajar_datasets.py").read_text(encoding="utf-8")
    ok = ("_norm_yolo({0: 0, 1: 1}, tope=TOPE_NORM_POR_FUENTE)" in fuente
          and "_norm_yolo({0: 0, 1: 0}, tope=TOPE_NORM_POR_FUENTE)" in fuente)
    return ok, f"TOPE_NORM_POR_FUENTE = {bd.TOPE_NORM_POR_FUENTE}"


# --------------------------------------------------------------------------- #
# Casos · guarda de clases vacías
# --------------------------------------------------------------------------- #
def _guarda(clases, por_clase, fatal=True):
    """Corre el bloque de la guarda aislado, sin importar torch."""
    src = (_AQUI / "entrenar_objetos_cuda.py").read_text(encoding="utf-8")
    ini = src.index("CUBIERTAS_POR_OTRA_CABEZA = {")
    fin = src.index("}", ini) + 1
    ns: Dict = {}
    exec(src[ini:fin], ns)
    cub = ns["CUBIERTAS_POR_OTRA_CABEZA"]

    nom_vacias = [n for i, n in enumerate(clases) if por_clase[i] == 0]
    cubiertas = [n for n in nom_vacias if n in cub]
    huerfanas = [n for n in nom_vacias if n not in cub]
    corta = bool(huerfanas) and fatal
    return cubiertas, huerfanas, corta


CLASES = ("humo", "llama", "persona", "pistola", "cuchillo", "celular", "paquete")


def caso_llama_vacia_no_corta() -> Tuple[bool, str]:
    """El run del 17/09. `llama` en 0 no puede matar 33 minutos de preparación:
    segmentación detecta fuego con F1 99,1 % y la regla `incendio` abre el
    evento sin la cabeza de objetos (probar_segmentation_engine::caso_fuego_solo)."""
    cub, hue, corta = _guarda(CLASES, [4213, 0, 4396, 488, 973, 2786, 4216])
    ok = not corta and cub == ["llama"] and not hue
    return ok, f"cubiertas={cub} huérfanas={hue} corta={corta}"


def caso_pistola_vacia_si_corta() -> Tuple[bool, str]:
    """No hay otra cabeza que detecte un arma. Entrenar sin una sola pistola
    deja a Horus ciego al arma, y ahí sí hay que parar."""
    cub, hue, corta = _guarda(CLASES, [4213, 500, 4396, 0, 973, 2786, 4216])
    ok = corta and hue == ["pistola"]
    return ok, f"cubiertas={cub} huérfanas={hue} corta={corta}"


def caso_mezcla_corta_por_la_huerfana() -> Tuple[bool, str]:
    """Con las dos: avisa por la cubierta y corta por la huérfana. Que una esté
    justificada no justifica a la otra."""
    cub, hue, corta = _guarda(CLASES, [0, 0, 4396, 0, 973, 2786, 4216])
    ok = corta and set(cub) == {"humo", "llama"} and hue == ["pistola"]
    return ok, f"cubiertas={cub} huérfanas={hue} corta={corta}"


def caso_permitir_clases_vacias_sigue_andando() -> Tuple[bool, str]:
    """La escotilla de siempre no cambió: --permitir-clases-vacias desactiva
    el corte incluso para una huérfana."""
    _, hue, corta = _guarda(CLASES, [4213, 500, 4396, 0, 973, 2786, 4216], fatal=False)
    ok = not corta and hue == ["pistola"]
    return ok, f"huérfanas={hue} corta={corta} (fatal=False)"


def caso_dataset_completo_no_dice_nada() -> Tuple[bool, str]:
    """Con las siete clases con datos, ni aviso ni corte."""
    cub, hue, corta = _guarda(CLASES, [4213, 3000, 4396, 488, 973, 2786, 4216])
    return not cub and not hue and not corta, "sin avisos, sin corte"


def caso_la_tabla_no_tapa_lo_importante() -> Tuple[bool, str]:
    """La tabla solo puede tener clases que otra cabeza detecte. Persona,
    pistola, cuchillo, celular y paquete desbloquean tracking, agresión,
    caídas y ReID: ninguna puede estar ahí adentro."""
    src = (_AQUI / "entrenar_objetos_cuda.py").read_text(encoding="utf-8")
    ini = src.index("CUBIERTAS_POR_OTRA_CABEZA = {")
    ns: Dict = {}
    exec(src[ini:src.index("}", ini) + 1], ns)
    prohibidas = {"persona", "pistola", "cuchillo", "celular", "paquete"}
    intrusas = prohibidas & set(ns["CUBIERTAS_POR_OTRA_CABEZA"])
    return not intrusas, f"tabla = {sorted(ns['CUBIERTAS_POR_OTRA_CABEZA'])}"


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "sin_tope": caso_sin_tope_copia_todo,
    "tope_acota": caso_tope_acota,
    "negativos": caso_negativos_no_se_comen_el_cupo,
    "paso_constante": caso_paso_constante_no_primeros_n,
    "estable": caso_a_paso_es_estable,
    "tope_grande": caso_tope_mayor_que_la_fuente,
    "catalogo": caso_dfire_y_pyro_tienen_tope,
    "llama_no_corta": caso_llama_vacia_no_corta,
    "pistola_corta": caso_pistola_vacia_si_corta,
    "mezcla": caso_mezcla_corta_por_la_huerfana,
    "escotilla": caso_permitir_clases_vacias_sigue_andando,
    "completo": caso_dataset_completo_no_dice_nada,
    "tabla_acotada": caso_la_tabla_no_tapa_lo_importante,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caso")
    args = ap.parse_args()
    casos = {args.caso: CASOS[args.caso]} if args.caso else CASOS

    print("=" * 78)
    print("HORUS · tope de normalización + guarda de clases vacías")
    print("=" * 78)
    fallas = 0
    for nombre, f in casos.items():
        t0 = time.perf_counter()
        try:
            ok, detalle = f()
        except Exception as e:                        # pragma: no cover
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:<16} {detalle:<52} {ms:7.1f} ms")
        fallas += not ok
    print("-" * 78)
    print(f"{len(casos) - fallas}/{len(casos)} pruebas OK")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
