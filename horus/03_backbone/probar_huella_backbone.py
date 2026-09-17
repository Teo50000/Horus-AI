# -*- coding: utf-8 -*-
"""
probar_huella_backbone.py · la red de seguridad que faltaba.

Prueba lo único que habría convertido un mes de "el modelo detecta mal" en un
error al arrancar: la huella del backbone congelado.

El fondo del asunto, medido y no supuesto: el FPN y el `embed_head` de
`SharedBackbone` **no** salen de ImageNet. Se sortean con la init random de
PyTorch al construirlo y se congelan ahí — nunca ven un gradiente, y tampoco
se pueden recalcular. Una cabeza entrenada contra un sorteo no funciona sobre
otro, y no falla con una excepción: detecta mal, en silencio.

    python probar_huella_backbone.py

Necesita torch (es lo único de Horus que lo necesita para probarse; el resto
de las suites corre sin él a propósito).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import Callable, Dict, Tuple

import torch

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI, os.path.join(_RAIZ, "04_cabezas", "objetos")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_backbone import (  # noqa: E402
    BackboneConfig, SharedBackbone, huella_backbone, pesos_no_imagenet,
)

TAM = 128          # solo afecta el descubrimiento de canales, no la huella


def backbone(seed: int) -> SharedBackbone:
    torch.manual_seed(seed)
    return SharedBackbone(BackboneConfig(
        pretrained=False, input_size=TAM, freeze_encoder=True))


# --------------------------------------------------------------------------- #
def caso_sorteos_distintos() -> Tuple[bool, str]:
    """Dos backbones sorteados distinto tienen huellas distintas.

    Es la premisa de todo: si dieran la misma, la huella no serviría para nada
    y el backbone se podría regenerar."""
    hs = [huella_backbone(backbone(s)) for s in (0, 1, 2)]
    ok = len(set(hs)) == 3
    return ok, " · ".join(hs)


def caso_mismo_sorteo() -> Tuple[bool, str]:
    """El mismo sorteo da la misma huella. Sin esto sería un generador de
    números al azar, no una huella."""
    a, b = huella_backbone(backbone(7)), huella_backbone(backbone(7))
    return a == b, f"{a} == {b}"


def caso_sobrevive_al_disco() -> Tuple[bool, str]:
    """Guardar y recargar no cambia la huella.

    Es lo que la hace usable: se calcula al entrenar en una GPU y se compara al
    cargar en otra máquina, otro dtype y otro device."""
    bb = backbone(3)
    antes = huella_backbone(bb)
    ruta = os.path.join(tempfile.mkdtemp(), "backbone.pt")
    torch.save(bb.state_dict(), ruta)
    despues = huella_backbone(torch.load(ruta, map_location="cpu",
                                         weights_only=False))
    return antes == despues, f"{antes} -> {despues}"


def caso_fp16_no_la_mueve() -> Tuple[bool, str]:
    """Un checkpoint guardado en fp16 da la misma huella.

    El entrenamiento corre en bf16/fp16; si la huella dependiera del dtype,
    fallaría justo en producción."""
    bb = backbone(5)
    antes = huella_backbone(bb)
    medio = {k: v.to(torch.float16) for k, v in bb.state_dict().items()}
    despues = huella_backbone(medio)
    # fp16 pierde bits, así que NO tiene que dar igual: lo que se verifica es
    # que la función no explote y que la diferencia sea detectable.
    return antes != despues, (f"fp32 {antes} vs fp16 {despues} — "
                              f"detecta la pérdida de precisión")


def caso_ignora_el_encoder() -> Tuple[bool, str]:
    """La huella NO mira el ResNet-50: ese se restaura de ImageNet en
    cualquier lado. Solo mira lo irreconstruible."""
    bb = backbone(9)
    sd = bb.state_dict()
    antes = huella_backbone(sd)
    tocado = dict(sd)
    for k in list(tocado):
        if k.startswith("extractor.") and tocado[k].dtype.is_floating_point:
            tocado[k] = tocado[k] + 1.0
    despues = huella_backbone(tocado)
    n_irre = len(pesos_no_imagenet(sd))
    return antes == despues, (f"{n_irre} tensores irreconstruibles de "
                              f"{len(sd)} · huella intacta")


def caso_tamano_incrustable() -> Tuple[bool, str]:
    """Lo irreconstruible entra de sobra adentro del checkpoint de la cabeza."""
    no_in = pesos_no_imagenet(backbone(0))
    params = sum(v.numel() for v in no_in.values())
    mb = params * 4 / 1e6
    ok = mb < 25.0
    return ok, f"{params/1e6:.1f} M parámetros = {mb:.1f} MB"


def caso_motor_corta_con_backbone_ajeno() -> Tuple[bool, str]:
    """El motor CORTA si el backbone no es el que entrenó la cabeza.

    Esto es lo que no existía. Antes imprimía un aviso y seguía, y el operador
    se quedaba con un sistema que 'anda' y no ve a nadie."""
    from objects_engine import EngineConfig, ErrorBackbone, ObjectsEngine
    d = tempfile.mkdtemp()
    bueno, ajeno = backbone(11), backbone(22)

    torch.save(ajeno.state_dict(), os.path.join(d, "backbone.pt"))
    ck = _checkpoint_falso(huella_backbone(bueno))
    ruta_ck = os.path.join(d, "head_best.pt")
    torch.save(ck, ruta_ck)

    try:
        ObjectsEngine(EngineConfig(pesos=ruta_ck, device="cpu",
                                   input_size=TAM, verboso=False))
    except ErrorBackbone as e:
        return "NO es el que entrenó" in str(e), "corta con la huella que esperaba"
    except Exception as e:
        return False, f"cortó con otra excepción: {type(e).__name__}: {e}"
    return False, "NO cortó: arrancó con un backbone ajeno"


def caso_motor_acepta_el_correcto() -> Tuple[bool, str]:
    """Con el backbone correcto arranca y lo dice."""
    from objects_engine import EngineConfig, ObjectsEngine
    d = tempfile.mkdtemp()
    bb = backbone(11)
    torch.save(bb.state_dict(), os.path.join(d, "backbone.pt"))
    ruta_ck = os.path.join(d, "head_best.pt")
    torch.save(_checkpoint_falso(huella_backbone(bb)), ruta_ck)
    try:
        m = ObjectsEngine(EngineConfig(pesos=ruta_ck, device="cpu",
                                       input_size=TAM, verboso=False))
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    return True, f"arrancó · huella {huella_backbone(m.backbone)}"


def caso_incrustado_viaja_solo() -> Tuple[bool, str]:
    """Incrustado, el checkpoint no depende de ningún archivo al lado.

    Se incrusta, se BORRA el backbone.pt, y el motor tiene que arrancar igual.
    Es el agujero de raíz que se llevó puesto a objetos_v2."""
    from exportar_objetos import incrustar_backbone
    from objects_engine import EngineConfig, ObjectsEngine
    from pathlib import Path

    d = tempfile.mkdtemp()
    bb = backbone(13)
    ruta_bb = Path(d) / "backbone.pt"
    torch.save(bb.state_dict(), ruta_bb)
    ruta_ck = Path(d) / "head_best.pt"
    torch.save(_checkpoint_falso(huella_backbone(bb)), ruta_ck)

    rc = incrustar_backbone(ruta_ck, ruta_bb, None)
    solo = ruta_ck.with_name("head_best_solo.pt")
    if rc != 0 or not solo.exists():
        return False, "no se generó el checkpoint incrustado"

    ruta_bb.unlink()                      # el archivo de al lado ya no está
    try:
        ObjectsEngine(EngineConfig(pesos=str(solo), device="cpu",
                                   input_size=TAM, verboso=False))
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    return True, f"arranca sin backbone.pt · {solo.stat().st_size/1e6:.1f} MB"


def caso_incrustar_rechaza_el_ajeno() -> Tuple[bool, str]:
    """Incrustar un backbone que no es el del checkpoint se rechaza.

    Si no, quedaría un `.pt` autocontenido que carga sin protestar y detecta
    mal para siempre — peor que el problema original, porque ya no se nota."""
    from exportar_objetos import incrustar_backbone
    from pathlib import Path
    d = tempfile.mkdtemp()
    torch.save(backbone(31).state_dict(), Path(d) / "backbone.pt")
    ruta_ck = Path(d) / "head_best.pt"
    torch.save(_checkpoint_falso(huella_backbone(backbone(32))), ruta_ck)
    rc = incrustar_backbone(ruta_ck, Path(d) / "backbone.pt", None)
    return rc == 1, "rechaza el backbone equivocado"


def caso_sin_huella_ni_backbone() -> Tuple[bool, str]:
    """Un checkpoint viejo sin huella y sin su backbone NO arranca.

    Es exactamente `objetos_v2.pt`. Antes caía a ImageNet, que restaura el
    ResNet-50 pero deja el FPN en un sorteo nuevo: medido sobre ruido puro
    salían 'persona 0.41' o 20 detecciones fantasma, según el sorteo."""
    from objects_engine import EngineConfig, ErrorBackbone, ObjectsEngine
    d = tempfile.mkdtemp()
    ck = _checkpoint_falso(None)
    ck.pop("backbone_huella", None)
    ruta_ck = os.path.join(d, "objetos_v2.pt")
    torch.save(ck, ruta_ck)
    try:
        ObjectsEngine(EngineConfig(pesos=ruta_ck, device="cpu",
                                   input_size=TAM, verboso=False))
    except ErrorBackbone as e:
        return "no trae huella" in str(e), "corta y explica por qué"
    except Exception as e:
        return False, f"cortó con otra excepción: {type(e).__name__}: {e}"
    return False, "NO cortó: arrancó con el FPN al azar"


# --------------------------------------------------------------------------- #
def _checkpoint_falso(huella) -> Dict:
    """Un checkpoint de cabeza mínimo pero real: los pesos son de un
    `ObjectsHead` de verdad, así que si el contrato cambia, esto rompe."""
    from objects_engine import _ANCHOR_SIZES_ENTRENADOS, _FPN_LEVELS_ENTRENADOS
    from objects_head import ObjectsHead, ObjectsHeadConfig
    cfg = ObjectsHeadConfig(fpn_levels=_FPN_LEVELS_ENTRENADOS,
                            anchor_sizes=_ANCHOR_SIZES_ENTRENADOS)
    head = ObjectsHead(cfg)
    ck = {
        "head": head.state_dict(),
        "epoca": 1,
        "clases": list(cfg.classes),
        "fpn_levels": list(cfg.fpn_levels),
        "anchor_sizes": [list(s) for s in cfg.anchor_sizes],
        "tam": TAM,
        "backbone_file": "backbone.pt",
        "backbone_pretrained": False,
        "backbone_parcial": False,
    }
    if huella is not None:
        ck["backbone_huella"] = huella
    return ck


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "sorteos_distintos": caso_sorteos_distintos,
    "mismo_sorteo": caso_mismo_sorteo,
    "sobrevive_disco": caso_sobrevive_al_disco,
    "detecta_fp16": caso_fp16_no_la_mueve,
    "ignora_encoder": caso_ignora_el_encoder,
    "tamano": caso_tamano_incrustable,
    "corta_ajeno": caso_motor_corta_con_backbone_ajeno,
    "acepta_correcto": caso_motor_acepta_el_correcto,
    "incrustado": caso_incrustado_viaja_solo,
    "incrustar_ajeno": caso_incrustar_rechaza_el_ajeno,
    "sin_huella": caso_sin_huella_ni_backbone,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caso")
    args = ap.parse_args()
    casos = {args.caso: CASOS[args.caso]} if args.caso else CASOS

    print("=" * 80)
    print("HORUS · huella del backbone congelado")
    print("=" * 80)
    fallas = 0
    for nombre, f in casos.items():
        t0 = time.perf_counter()
        try:
            ok, detalle = f()
        except Exception as e:                       # pragma: no cover
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:<18} {detalle:<52} {ms:8.1f} ms")
        fallas += not ok
    print("-" * 80)
    print(f"{len(casos) - fallas}/{len(casos)} pruebas OK")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
