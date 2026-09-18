# -*- coding: utf-8 -*-
"""
enviar_alerta_prueba.py · una alerta real, de la fusión al panel.

No inventa un JSON a mano: arma un evento con el motor de fusión de verdad y
lo pasa por `armar_payload`, el mismo camino que va a usar el servicio cuando
corra sobre cámaras. Si esto llega al panel, llega lo real.

Con el backend arriba (HORUS.bat):

    python enviar_alerta_prueba.py                # incendio
    python enviar_alerta_prueba.py agresion
    python enviar_alerta_prueba.py --listar
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for p in (RAIZ / "horus" / "06_fusion_decision", RAIZ / "horus" / "07_alerta"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

URL = os.environ.get("HORUS_URL", "http://127.0.0.1:8000")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("escenario", nargs="?", default="incendio")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--camara", type=int, default=1)
    ap.add_argument("--repetir", action="store_true",
                    help="mandar el MISMO id otra vez, para ver la idempotencia")
    a = ap.parse_args()

    import probar_fusion as pf
    from alerta import armar_payload, validar_payload

    # pf.CASOS es {nombre: funcion}, y cada funcion devuelve (ok, detalle, eventos)
    nombres = sorted(pf.CASOS)
    if a.listar:
        print("escenarios disponibles (son los casos de probar_fusion.py):")
        for n in nombres:
            print(f"  {n}")
        return 0

    if a.escenario not in nombres:
        print(f"no conozco '{a.escenario}'. Hay {len(nombres)}, por ejemplo:")
        print(f"  {', '.join(nombres[:10])}")
        print("Todos con --listar")
        return 1

    _, _, eventos = pf.CASOS[a.escenario]()
    if not eventos:
        print(f"el escenario '{a.escenario}' no produjo ningún evento.")
        print("Es a propósito en algunos casos (escena_vacia, persona_de_paso).")
        return 1

    ev = eventos[0]
    payload = armar_payload(ev, a.camara, sitio="prueba", nodo="local")

    # Cada corrida manda un evento NUEVO. Sin esto, la segunda vez que probás
    # el backend contesta "duplicada" y no emite: la idempotencia por
    # (evento_id, secuencia) está haciendo exactamente su trabajo, pero para
    # una demo el resultado es una pantalla que no se mueve y parece rota.
    # La idempotencia se prueba en probar_alertas_backend.py, no acá.
    if not a.repetir:
        payload["id"] = f"{payload['id']}-{int(time.time() * 1000) % 1000000}"

    problemas = validar_payload(payload)
    if problemas:
        print("el payload no valida:", problemas)
        return 1

    print(f"  evento    : {ev.tipo}")
    print(f"  severidad : {ev.severidad.etiqueta}")
    print(f"  id        : {payload['id']}")
    print(f"  camera_id : {a.camara}")
    print(f"  confianza : {payload.get('confianza')}")
    print(f"  modelos   : {payload.get('modelos')}")
    print()

    req = urllib.request.Request(
        f"{URL}/alertas", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        # sin proxy: 127.0.0.1 no se sale de la máquina
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with op.open(req, timeout=10) as r:
            cuerpo = json.loads(r.read().decode("utf-8"))
        print(f"  {URL}/alertas -> {r.status}")
        print(f"  acción           : {cuerpo.get('accion')}")
        print(f"  emitido al panel : {cuerpo.get('emitido_al_panel')}")
        if cuerpo.get("emitido_al_panel"):
            print("\n  Miralo en el panel: http://localhost:1420")
        else:
            print("\n  No salió al panel. Pasa cuando el evento ya existía y no")
            print("  subió de severidad: el panel no repite una tarjeta igual.")
        return 0
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        return 1
    except urllib.error.URLError as e:
        print(f"  no pude hablar con {URL}: {e.reason}")
        print("  ¿Está corriendo el backend? Arrancalo con HORUS.bat")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
