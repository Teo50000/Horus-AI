# -*- coding: utf-8 -*-
"""
probar_servicio.py · ¿el servicio AVISA, o solo se lo dice a sí mismo?

Esta suite existe por dos bugs que estuvieron ahí desde siempre y que ninguna
de las otras ocho podía ver, porque todas prueban las piezas por separado y
las dos piezas estaban bien:

  1. El bucle de `Servicio` imprimía cada evento con su severidad y NUNCA
     llamaba a `emisor.emitir()`. Había un `self.pipe.emisor = self.emisor`
     que no servía de nada: `pipeline.py` no lee ese atributo en ningún lado.

  2. `TransporteBackend` construía `TransporteHTTP(cfg_alerta)`, pasándole el
     objeto de configuración entero donde va la URL. urllib moría con
     "unknown url type: configalerta(...", el emisor atrapaba el error sin
     decir nada y mandaba la alerta a un archivo en disco.

Los dos juntos daban esto: el servicio detecta un incendio, escribe
"[ALERTA] incendio" en su consola con toda la ceremonia, y no se entera nadie.
Ni el backend, ni la base, ni el panel, ni el mail. Medido el 18/09: 6 eventos
en pantalla, 0 filas en /alertas.

Por eso esta prueba no mira funciones: levanta un backend de mentira que
ANOTA lo que le llega, corre el servicio de verdad contra él, y pregunta lo
único que importa — ¿llegó?

    python probar_servicio.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Tuple

_AQUI = os.path.dirname(os.path.abspath(__file__))
_HORUS = os.path.dirname(_AQUI)
for _p in (_AQUI, os.path.join(_HORUS, "07_alerta")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# Un backend de mentira que anota todo lo que le llega
# --------------------------------------------------------------------------- #
class BackendFalso:
    def __init__(self, camaras: List[Dict[str, Any]], responder: int = 200):
        self.recibidas: List[dict] = []
        self.camaras = camaras
        self.responder = responder
        self._lock = threading.Lock()
        anota = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silencio
                pass

            def do_GET(self):
                if self.path.startswith("/camaras/config"):
                    cuerpo = json.dumps(anota.camaras).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(cuerpo)))
                    self.end_headers()
                    self.wfile.write(cuerpo)
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                crudo = self.rfile.read(n)
                if anota.responder < 400:
                    with anota._lock:
                        anota.recibidas.append(json.loads(crudo))
                self.send_response(anota.responder)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.srv = HTTPServer(("127.0.0.1", 0), Handler)
        self.puerto = self.srv.server_address[1]
        self.hilo = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.hilo.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.puerto}"

    def cerrar(self) -> None:
        self.srv.shutdown()


def _correr_servicio(backend_url: str, segundos: float = 9.0,
                     spool: str = None) -> Tuple[Any, Any]:
    """El servicio DE VERDAD, en modo simulado, contra ese backend."""
    from servicio import ConfigServicio, Servicio
    cfg = ConfigServicio(backend=backend_url, simular=True, fps=10.0,
                         sondeo_s=1.0, verboso=False, puerto_stream=0)
    if spool:
        cfg.spool = spool
    srv = Servicio(cfg)
    srv.iniciar()
    try:
        time.sleep(segundos)
        estado = srv.estado()
    finally:
        srv.parar()
    return srv, estado


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #
def caso_el_servicio_avisa() -> Tuple[bool, str]:
    """Detecta un incendio Y el backend se entera.

    Es la prueba entera: si esto pasa a rojo, el sistema no sirve para lo
    único que existe, por más verdes que estén las otras ocho suites.
    """
    b = BackendFalso([{"id": 1, "nombre": "cam-1", "usb_index": 0}])
    try:
        srv, estado = _correr_servicio(b.url)
        recibidas = list(b.recibidas)
    finally:
        b.cerrar()

    eventos = estado["eventos"]
    entregadas = estado["alertas"]["enviadas"]
    ok = eventos > 0 and len(recibidas) > 0 and entregadas == len(recibidas)
    tipos = sorted({r.get("tipo") for r in recibidas})
    return ok, (f"{eventos} evento(s) detectados, {len(recibidas)} alerta(s) "
                f"llegaron al backend {tipos or ''}")


def caso_la_alerta_llega_entera() -> Tuple[bool, str]:
    """Lo que llega tiene los campos que el backend exige.

    No alcanza con que llegue algo: si al payload le falta un campo, FastAPI
    contesta 422, el emisor reintenta tres veces y la manda al spool — otra
    forma de quedarse callado.
    """
    b = BackendFalso([{"id": 7, "nombre": "cam-1", "usb_index": 0}])
    try:
        _correr_servicio(b.url)
        recibidas = list(b.recibidas)
    finally:
        b.cerrar()

    if not recibidas:
        return False, "no llegó ninguna alerta"
    from alerta import validar_payload
    problemas = []
    for r in recibidas:
        problemas += validar_payload(r)
    # y el id de la cámara del backend viajó adentro
    cid = recibidas[0].get("sitio", {}).get("camara_config_id")
    ok = not problemas and cid == 7
    return ok, (f"{len(recibidas)} payload(s) válidos · camara_config_id={cid} "
                f"{'· ' + '; '.join(problemas[:2]) if problemas else ''}")


def caso_backend_caido_no_pierde_la_alerta() -> Tuple[bool, str]:
    """Si el backend no contesta, la alerta va a disco, no al tacho.

    Y se avisa: el `except Exception` del emisor no decía nada, que es
    exactamente lo que dejó vivir a los dos bugs de arriba.
    """
    import io as _io
    import contextlib
    tmp = tempfile.mkdtemp()
    spool = os.path.join(tmp, "pendientes.jsonl")

    b = BackendFalso([{"id": 1, "nombre": "cam-1", "usb_index": 0}], responder=500)
    err = _io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            srv, estado = _correr_servicio(b.url, spool=spool)
    finally:
        b.cerrar()

    en_disco = estado["alertas"]["en_disco"]
    lineas = 0
    if os.path.exists(spool):
        with open(spool, encoding="utf-8") as fh:
            lineas = sum(1 for _ in fh)
    aviso = "NO se pudo entregar" in err.getvalue()
    ok = estado["eventos"] > 0 and en_disco > 0 and lineas == en_disco and aviso
    return ok, (f"{en_disco} al spool, {lineas} línea(s) en el archivo · "
                f"avisó por stderr: {aviso}")


def caso_el_transporte_recibe_una_url() -> Tuple[bool, str]:
    """`TransporteBackend` arma el HTTP con la URL, no con la config entera.

    Este es el bug #2 aislado. Se prueba aparte del de arriba porque un día
    alguien puede volver a pasar el objeto y el símbolo sigue existiendo: no
    falla al importar, falla en el primer envío, de noche.
    """
    from alerta import ConfigAlerta
    from servicio import TransporteBackend
    cfg = ConfigAlerta(url="http://127.0.0.1:9/alertas", token="t")
    tr = TransporteBackend(cfg, {})
    url = getattr(tr._http, "url", None)
    ok = isinstance(url, str) and url == cfg.url
    return ok, f"url={type(url).__name__} {str(url)[:60]}"


def caso_las_camaras_se_enganchan_solas() -> Tuple[bool, str]:
    """Una cámara dada de alta DESPUÉS del arranque entra sin recargar nada.

    Es la promesa del servicio: los modelos se cargan una vez, y enchufar una
    cámara cuesta el sondeo, no los 20 s de subir un backbone a la placa.
    """
    b = BackendFalso([])
    try:
        from servicio import ConfigServicio, Servicio
        cfg = ConfigServicio(backend=b.url, simular=True, fps=10.0,
                             sondeo_s=1.0, verboso=False, puerto_stream=0)
        srv = Servicio(cfg)
        srv.iniciar()
        try:
            time.sleep(2.0)
            sin_camaras = len(srv.fuentes)
            b.camaras.append({"id": 3, "nombre": "cam-nueva", "usb_index": 0})
            time.sleep(4.0)
            con_camara = len(srv.fuentes)
            estado = srv.estado()
        finally:
            srv.parar()
    finally:
        b.cerrar()
    ok = sin_camaras == 0 and con_camara == 1 and estado["listo_en_s"] >= 0
    return ok, (f"arrancó con {sin_camaras} cámaras, se enganchó {con_camara} "
                f"sin recargar modelos")


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "avisa": caso_el_servicio_avisa,
    "alerta_entera": caso_la_alerta_llega_entera,
    "backend_caido": caso_backend_caido_no_pierde_la_alerta,
    "transporte_url": caso_el_transporte_recibe_una_url,
    "camara_en_caliente": caso_las_camaras_se_enganchan_solas,
}


def main() -> int:
    print("=" * 78)
    print("HORUS · el servicio — ¿avisa, o solo se lo dice a sí mismo?")
    print("=" * 78)
    fallas = 0
    for nombre, f in CASOS.items():
        t0 = time.perf_counter()
        try:
            ok, detalle = f()
        except Exception as e:                           # noqa: BLE001
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:<20} {detalle:<52} {ms:7.1f} ms")
        if not ok:
            fallas += 1
    print("-" * 78)
    total = len(CASOS)
    print(f"{total - fallas}/{total} pruebas OK")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
