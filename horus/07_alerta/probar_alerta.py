# -*- coding: utf-8 -*-
"""
probar_alerta.py · autotest del emisor de alertas.

Sin red salvo el caso que levanta un backend falso en localhost, y sin GPU,
pesos ni cámara. Los eventos salen de la capa de fusión de verdad, no de
mocks: si el contrato entre las dos capas cambia, esto rompe acá y no a las
tres semanas contra el backend real.

    python probar_alerta.py              # autotest
    python probar_alerta.py --demo       # imprime mensajes reales
    python probar_alerta.py --servidor   # backend falso + POST de verdad
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Tuple

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI, os.path.join(_RAIZ, "06_fusion_decision"),
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alerta import (  # noqa: E402
    ConfigAlerta, EmisorAlertas, TransporteConsola, TransporteHTTP,
    aportes_de, armar_payload, validar_payload,
)
from contratos import Evento, Severidad  # noqa: E402

sys.path.insert(0, os.path.join(_RAIZ, "06_fusion_decision"))
import probar_fusion as pf  # noqa: E402


# --------------------------------------------------------------------------- #
def eventos_reales(caso: str) -> List[Evento]:
    """Eventos producidos por la capa de fusión de verdad."""
    ok, _, ev = pf.CASOS[caso]()
    return list(ev)


def evento(tipo: str = "merodeo", sev: Severidad = Severidad.AVISO,
           **kw: Any) -> Evento:
    base = dict(evento_id="E000042", tipo=tipo, severidad=sev,
                camera_id="cam-deposito", ts_inicio=time.time() - 20,
                ts_ultimo=time.time(), confianza=0.8, motivo="prueba",
                zona="deposito", estado="abierto", track_ids=[1, 2],
                camaras=["cam-deposito"], confirmaciones=10)
    base.update(kw)
    return Evento(**base)


class _Fake(BaseHTTPRequestHandler):
    recibidos: List[dict] = []
    fallar_veces = 0

    def do_POST(self) -> None:                           # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        cuerpo = json.loads(self.rfile.read(n) or b"{}")
        if _Fake.fallar_veces > 0:
            _Fake.fallar_veces -= 1
            self.send_response(503)
            self.end_headers()
            return
        _Fake.recibidos.append(cuerpo)
        self.send_response(202)
        self.end_headers()

    def log_message(self, *a: Any) -> None:               # silencio
        pass


def _servidor() -> Tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/alertas"


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #
def caso_contrato() -> Tuple[bool, str]:
    """Todo evento que produce la fusión arma un mensaje que valida."""
    problemas: List[str] = []
    n = 0
    for caso in ("incendio", "arma", "robo", "merodeo", "paquete",
                 "caida_pose", "pelea", "agresion_arma", "agresion_caida"):
        for ev in eventos_reales(caso):
            n += 1
            p = validar_payload(armar_payload(ev, 1, sitio="s", nodo="n"))
            problemas.extend(f"{caso}/{ev.tipo}: {x}" for x in p)
    ok = not problemas
    return ok, (f"{n} eventos de 9 escenarios validan"
                if ok else f"{len(problemas)} problema(s): {problemas[:3]}")


def caso_secuencia() -> Tuple[bool, str]:
    """`secuencia` arranca en 1 y sube por `id`, no global."""
    em = EmisorAlertas(ConfigAlerta(), transporte=TransporteConsola(open(os.devnull, "w")))
    secs_a = [em.emitir(evento(evento_id="E1"))["secuencia"] for _ in range(3)]
    secs_b = [em.emitir(evento(evento_id="E2"))["secuencia"] for _ in range(2)]
    em.cerrar()
    ok = secs_a == [1, 2, 3] and secs_b == [1, 2]
    return ok, f"E1 -> {secs_a} · E2 -> {secs_b}"


def caso_prioridad() -> Tuple[bool, str]:
    """Un crítico se adelanta a los avisos que ya estaban esperando.

    Es por qué el consumidor NO puede asumir orden cronológico: tiene que
    ordenar por `tiempo.inicio`."""
    llegada: List[str] = []

    class Lento:
        def enviar(self, p: dict) -> None:
            llegada.append(f"{p['severidad']}")
            time.sleep(0.01)

    em = EmisorAlertas(ConfigAlerta(), transporte=Lento())
    for _ in range(6):
        em.emitir(evento(evento_id=f"A{_}", sev=Severidad.AVISO))
    em.emitir(evento(evento_id="C1", tipo="incendio", sev=Severidad.CRITICO))
    em.cerrar(timeout_s=10)
    pos = llegada.index("critico") if "critico" in llegada else 99
    ok = pos < len(llegada) - 1
    return ok, f"el crítico salió {pos+1}º de {len(llegada)}"


def caso_arma_nunca_critica() -> Tuple[bool, str]:
    """El contrato rechaza un arma crítica o sin pedir VLM. No es timidez:
    esas 565 cajas son fotos de Flickr y dan 0 % mAP en objetos chicos."""
    malo = armar_payload(evento(tipo="arma", sev=Severidad.CRITICO), 1)
    p1 = validar_payload(malo)
    bueno = armar_payload(evento(tipo="arma", sev=Severidad.ALERTA,
                                 necesita_vlm=True), 1)
    p2 = validar_payload(bueno)
    ok = any("crítica" in x for x in p1) and not p2
    return ok, f"crítico rechazado={bool(p1)} · alerta+vlm acepta={not p2}"


def caso_spool() -> Tuple[bool, str]:
    """Backend caído: al disco. Backend de vuelta: se reenvía."""
    ruta = os.path.join(tempfile.mkdtemp(), "spool.jsonl")

    class Caido:
        def enviar(self, p: dict) -> None:
            raise OSError("backend caído")

    em = EmisorAlertas(ConfigAlerta(spool=ruta, reintentos=2, espera_base_s=0.0),
                       transporte=Caido())
    for i in range(3):
        em.emitir(evento(evento_id=f"S{i}"))
    em.cerrar(timeout_s=10)
    en_disco = sum(1 for _ in open(ruta, encoding="utf-8"))

    recibidos: List[dict] = []
    em2 = EmisorAlertas(ConfigAlerta(spool=ruta),
                        transporte=type("T", (), {"enviar": lambda s, p: recibidos.append(p)})())
    n = em2.reenviar_spool()
    em2.cerrar(timeout_s=10)
    ok = en_disco == 3 and n == 3 and len(recibidos) == 3
    return ok, f"{en_disco} al spool, {len(recibidos)} reenviados"


def caso_http_reintenta() -> Tuple[bool, str]:
    """Contra un backend real que devuelve 503 dos veces y después acepta."""
    _Fake.recibidos, _Fake.fallar_veces = [], 2
    srv, url = _servidor()
    try:
        em = EmisorAlertas(ConfigAlerta(url=url, reintentos=4, espera_base_s=0.01))
        em.emitir(evento(evento_id="H1", tipo="incendio", sev=Severidad.CRITICO))
        em.cerrar(timeout_s=10)
    finally:
        srv.shutdown()
    ok = (len(_Fake.recibidos) == 1
          and not validar_payload(_Fake.recibidos[0])
          and _Fake.recibidos[0]["id"] == "H1")
    return ok, f"{len(_Fake.recibidos)} entregado tras 2 fallos, valida={ok}"


def caso_aportes_incendio() -> Tuple[bool, str]:
    """Un incendio que confirman las dos cabezas lista DOS modelos.

    Es lo que separa 'lo vieron dos' de 'lo vio uno', y lo que un consumidor
    razonable muestra en la tarjeta."""
    evs = [e for e in eventos_reales("incendio") if e.tipo == "incendio"]
    p = armar_payload(evs[0], 1)
    ok = set(p["modelos"]) >= {"objetos", "segmentacion"} and len(p["aportes"]) >= 2
    return ok, f"modelos: {p['modelos']}, {len(p['aportes'])} aporte(s)"


def caso_aportes_agresion() -> Tuple[bool, str]:
    """En una correlación los aportes salen de las PIEZAS, no del evento.

    Una agresión no la ve ningún modelo: la ve la fusión cruzando dos. Si los
    aportes quedaran vacíos, la tarjeta más grave del sistema sería la única
    que no dice en qué se basa."""
    evs = [e for e in eventos_reales("agresion_arma") if e.tipo == "agresion"]
    p = armar_payload(evs[0], 1)
    modelos = set(p["modelos"])
    ok = bool(evs) and {"accion", "objetos"} <= modelos
    return ok, f"modelos: {sorted(modelos)}, aportes: {[a['aporta'] for a in p['aportes']]}"


def caso_severidad_min() -> Tuple[bool, str]:
    """`severidad_min=2` no deja salir los avisos."""
    enviados: List[dict] = []
    em = EmisorAlertas(ConfigAlerta(severidad_min=2),
                       transporte=type("T", (), {"enviar": lambda s, p: enviados.append(p)})())
    em.emitir(evento(sev=Severidad.AVISO))
    em.emitir(evento(evento_id="X", tipo="incendio", sev=Severidad.ALERTA))
    em.cerrar(timeout_s=10)
    ok = len(enviados) == 1 and enviados[0]["severidad"] == "alerta"
    return ok, f"{len(enviados)} de 2 mensajes pasaron el piso"


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "contrato": caso_contrato,
    "secuencia": caso_secuencia,
    "prioridad": caso_prioridad,
    "arma_no_critica": caso_arma_nunca_critica,
    "spool": caso_spool,
    "http_reintenta": caso_http_reintenta,
    "aportes_incendio": caso_aportes_incendio,
    "aportes_agresion": caso_aportes_agresion,
    "severidad_min": caso_severidad_min,
}


def demo() -> int:
    """Imprime los mensajes reales de un par de escenarios."""
    for caso in ("incendio", "pelea", "agresion_arma"):
        print(f"\n===== {caso} " + "=" * (60 - len(caso)))
        for ev in eventos_reales(caso):
            print(json.dumps(armar_payload(ev, 1, sitio="sucursal-centro",
                                           nodo="caja-01"),
                             ensure_ascii=False, indent=2)[:1400])
            break
    return 0


def servidor() -> int:
    _Fake.recibidos, _Fake.fallar_veces = [], 0
    srv, url = _servidor()
    print(f"backend falso escuchando en {url} — POSTeando eventos reales...")
    em = EmisorAlertas(ConfigAlerta(url=url, sitio="sucursal-centro", nodo="caja-01"))
    for caso in ("incendio", "pelea", "agresion_arma", "arma"):
        for ev in eventos_reales(caso):
            em.emitir(ev)
    em.cerrar(timeout_s=15)
    srv.shutdown()
    print(f"\n{len(_Fake.recibidos)} mensaje(s) recibidos:")
    for p in _Fake.recibidos:
        print(f"  {p['id']}.{p['secuencia']} {p['severidad']:>7} {p['tipo']:<12} "
              f"modelos={','.join(p['modelos']) or '-'}")
    malos = [x for p in _Fake.recibidos for x in validar_payload(p)]
    print(f"\nvalidación: {'todos OK' if not malos else malos[:5]}")
    return 1 if malos else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--servidor", action="store_true")
    ap.add_argument("--caso")
    args = ap.parse_args()
    if args.demo:
        return demo()
    if args.servidor:
        return servidor()

    casos = {args.caso: CASOS[args.caso]} if args.caso else CASOS
    print("=" * 76)
    print("HORUS · emisor de alertas — autotest")
    print("=" * 76)
    fallas = 0
    for nombre, f in casos.items():
        t0 = time.perf_counter()
        try:
            ok, detalle = f()
        except Exception as e:                            # pragma: no cover
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:<18} {detalle:<50} {ms:7.1f} ms")
        fallas += not ok
    print("-" * 76)
    print(f"{len(casos) - fallas}/{len(casos)} pruebas OK")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
