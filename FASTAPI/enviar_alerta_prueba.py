# -*- coding: utf-8 -*-
"""
enviar_alerta_prueba.py · una alerta real, de la fusión al panel.

No inventa un JSON a mano: arma un evento con el motor de fusión de verdad y
lo pasa por `armar_payload`, el mismo camino que va a usar el servicio cuando
corra sobre cámaras. Si esto llega al panel, llega lo real.

Con el backend arriba (HORUS.bat):

    python enviar_alerta_prueba.py                # incendio, en tu primera cámara
    python enviar_alerta_prueba.py agresion --camara 2
    python enviar_alerta_prueba.py --listar

26/09 · "el nombre de la cámara no es el mismo que el registrado en la base".
Dos errores juntos:

  1. `--camara` no se usaba. Se pasaba a `armar_payload(ev, a.camara, ...)`,
     pero el segundo parámetro de `armar_payload` es la SECUENCIA. La cámara
     quedaba la del escenario de prueba ("cam-deposito"), que no existe en tu
     base, y el panel la mostraba tal cual.
  2. El backend no sabía traducir "cam-1" a la cámara registrada (ver
     `_config_de` en alerta_rutas.py).

Ahora la cámara sale de las que tenés registradas (la primera, o la que digas
con `--camara`), va con el mismo formato que usa el servicio —`cam-<id>` más
`camara_config_id`— y la alerta lleva captura y clip: del video real si el
servicio de modelos está corriendo, y si no, una imagen que dice PRUEBA.
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
SERVICIO = os.environ.get("HORUS_SERVICIO_URL", "http://127.0.0.1:8010")
_OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 127.0.0.1


def _get_json(url: str, timeout: float = 5.0):
    with _OP.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def elegir_camara(pedida):
    """La cámara registrada en el backend. (id, nombre) o sale con el motivo."""
    try:
        filas = _get_json(f"{URL}/camaras/config")
    except Exception as e:
        print(f"  no pude pedirle las cámaras a {URL}: {e}")
        print("  ¿Está corriendo el backend? Arrancalo con HORUS.bat")
        raise SystemExit(1)
    if not filas:
        print("  No hay ninguna cámara registrada en la base.")
        print("  Agregá una desde el panel (Cámaras -> +) y volvé a probar.")
        raise SystemExit(1)
    if pedida is None:
        f = filas[0]
    else:
        f = next((x for x in filas if x.get("id") == pedida), None)
        if f is None:
            print(f"  No hay ninguna cámara con id {pedida}. Las registradas son:")
            for x in filas:
                print(f"    --camara {x.get('id')}   {x.get('nombre')!r}")
            raise SystemExit(1)
    return int(f["id"]), f.get("nombre") or f"Cámara {f['id']}"


def _cuadros_del_servicio(cam: str, segundos: float = 4.0):
    """Cuadros JPEG reales del stream del servicio, si lo tiene."""
    try:
        estado = _get_json(f"{SERVICIO}/estado", timeout=2.0)
    except Exception:
        return []
    if not any(c.get("camara") == cam for c in estado.get("camaras", [])):
        return []
    out, fin = [], time.time() + segundos
    try:
        with _OP.open(f"{SERVICIO}/camaras/{cam}/stream", timeout=5) as r:
            buf = b""
            while time.time() < fin:
                buf += r.read(16384)
                while True:
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2)
                    if a < 0 or b < 0:
                        break
                    out.append((time.time(), buf[a:b + 2]))
                    buf = buf[b + 2:]
    except Exception:
        pass
    return out


def _cuadros_de_prueba(nombre: str, n: int = 12):
    """Una imagen que dice PRUEBA en grande: que nadie la confunda con una real."""
    import cv2
    import numpy as np
    out, t0 = [], time.time() - n * 0.4
    for i in range(n):
        img = np.full((360, 640, 3), (40, 30, 30), dtype=np.uint8)
        cv2.putText(img, "ALERTA DE PRUEBA", (40, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(img, nombre[:34], (40, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (230, 230, 230), 2, cv2.LINE_AA)
        x = 40 + i * 45
        cv2.rectangle(img, (x, 230), (x + 60, 330), (60, 60, 255), -1)
        ok, jpg = cv2.imencode(".jpg", img)
        out.append((t0 + i * 0.4, jpg.tobytes()))
    return out


def armar_evidencia(cam: str, nombre: str):
    from evidencia import Grabadora
    cuadros = _cuadros_del_servicio(cam)
    origen = "video real del servicio de modelos"
    if len(cuadros) < 2:
        cuadros = _cuadros_de_prueba(nombre)
        origen = "imagen de prueba (el servicio de modelos no está o no tiene la cámara)"
    g = Grabadora(segundos=30)
    for ts, jpg in cuadros:
        g.agregar(cam, jpg, ts)
    return g.adjuntos(cam), origen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("escenario", nargs="?", default="incendio")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--camara", type=int, default=None,
                    help="id de la cámara registrada (por defecto, la primera)")
    ap.add_argument("--sin-evidencia", dest="sin_evidencia", action="store_true",
                    help="mandar la alerta sin captura ni clip")
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
    cam_id, cam_nombre = elegir_camara(a.camara)
    cam = f"cam-{cam_id}"            # como la nombra el servicio de verdad
    payload = armar_payload(ev, 1, sitio="prueba", nodo="local")
    payload["sitio"]["camara"] = cam
    payload["sitio"]["camaras"] = [cam]
    payload["sitio"]["camara_config_id"] = cam_id
    # Con la hora de AHORA: los escenarios de probar_fusion tienen fechas fijas
    # (24/08) y la alerta de prueba aparecía abajo de todo en el historial.
    ahora = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ahora = ahora[:-2] + ":" + ahora[-2:]
    payload["tiempo"].update(inicio=ahora, ultimo=ahora, emitido=ahora)

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

    origen_ev = "no"
    if not a.sin_evidencia:
        adj, origen_ev = armar_evidencia(cam, cam_nombre)
        if adj:
            payload["adjuntos"] = adj

    print(f"  evento    : {ev.tipo}")
    print(f"  severidad : {ev.severidad.etiqueta}")
    print(f"  id        : {payload['id']}")
    print(f"  cámara    : {cam_nombre!r} (id {cam_id}, {cam})")
    print(f"  evidencia : {origen_ev}")
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

        # Y lo que el panel NO muestra: si el aviso salió de la casa.
        #
        # 22/09. Antes esta prueba terminaba acá y uno se quedaba pensando que
        # había andado todo. La alerta se veía en el panel, sí, pero el mail a
        # los contactos podía no haber salido nunca y no lo decía nadie.
        _contar_evidencia(payload["id"])
        _contar_mail(payload["id"])
        return 0
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        return 1
    except urllib.error.URLError as e:
        print(f"  no pude hablar con {URL}: {e.reason}")
        print("  ¿Está corriendo el backend? Arrancalo con HORUS.bat")
        return 1


def _contar_evidencia(evento_id: str) -> None:
    try:
        h = _get_json(f"{URL}/alertas/{evento_id}")
        m = (h.get("mensajes") or [{}])[-1]
    except Exception as e:
        print(f"  (no pude leer la evidencia guardada: {e})")
        return
    print(f"  cámara en la base: {m.get('nombre_camara')!r} "
          f"(camara_config_id={m.get('camara_config_id')})")
    for clave in ("captura_url", "clip_url"):
        if m.get(clave):
            print(f"  {clave:<11}: {URL}{m[clave]}")
    if not m.get("captura_url"):
        print("  captura    : NO se guardó ninguna")


def _contar_mail(evento_id: str) -> None:
    """Qué pasó con el aviso por mail de esta alerta."""
    import time
    time.sleep(0.8)                      # la tarea de fondo termina después
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(f"{URL}/alertas/{evento_id}", timeout=10) as r:
            h = json.loads(r.read().decode("utf-8"))
        m = (h.get("mensajes") or [{}])[-1]
        estado = m.get("mail_estado")
        detalle = m.get("mail_detalle") or ""
    except Exception as e:
        print(f"\n  (no pude leer el estado del mail: {e})")
        return

    print()
    if estado == "enviado":
        print(f"  MAIL: salió a {detalle}")
    elif estado == "apagado":
        print("  MAIL: APAGADO. No le llegó a nadie.")
        print(f"        {detalle}")
        print("        Se configura desde el panel: Ajustes -> Aviso por mail.")
    elif estado == "sin_destinos":
        print("  MAIL: no hay contactos con dirección cargada.")
        print("        Agregalos en el panel, en Ajustes.")
    elif estado == "no_corresponde":
        print("  MAIL: no corresponde (severidad 1). Los avisos no despiertan")
        print("        a nadie a propósito; desde 'alerta' sí.")
    elif estado in ("fallo", "parcial"):
        print(f"  MAIL: {estado.upper()} — {detalle}")
    elif estado is None:
        print("  MAIL: sin dato. Si el backend es viejo, no anota nada todavía.")
    else:
        print(f"  MAIL: {estado} — {detalle}")


if __name__ == "__main__":
    raise SystemExit(main())
