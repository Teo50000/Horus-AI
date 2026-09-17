"""
probar_alertas_backend.py · el endpoint de alertas, de punta a punta.

Levanta la app real con una base temporal y le manda los mensajes que produce
la capa de fusión de VERDAD (no fixtures): Horus -> emisor -> FastAPI -> SQLite
-> WebSocket. Si el contrato entre las dos mitades se rompe, rompe acá.

Es lo que faltaba la vez pasada: `probar_servicio.py` servía un backend de
mentira con el contrato viejo, la prueba daba verde, y contra el backend real
no enganchaba ni una cámara.

    python probar_alertas_backend.py
"""

import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Tuple

os.environ.setdefault("EMAIL_SENDER", "")
os.environ.setdefault("EMAIL_PASSWORD", "")

_AQUI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_AQUI)
_HORUS = os.path.join(_REPO, "horus")
for _p in (_AQUI, _REPO,
           os.path.join(_HORUS, "06_fusion_decision"),
           os.path.join(_HORUS, "07_alerta"),
           os.path.join(_HORUS, "05_tracking"),
           os.path.join(_HORUS, "05_tracking", "local")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Base temporal ANTES de importar el módulo, que crea el engine al importarse.
_TMP = tempfile.mkdtemp()
os.chdir(_TMP)

from fastapi import FastAPI                                   # noqa: E402
from fastapi.testclient import TestClient                     # noqa: E402
from sqlmodel import Session, select                          # noqa: E402

from src.database import crear_tablas, engine                 # noqa: E402
from src.models import camara_model                           # noqa: E402,F401
from src.models.alerta_model import Alerta                    # noqa: E402
from src.models.camara_model import Camara, CamaraConfig, NumeroEmergencia  # noqa: E402
from src.routers.alerta_rutas import alerta_router            # noqa: E402
from src.services import websockets as ws                     # noqa: E402

from alerta import armar_payload, validar_payload             # noqa: E402
import probar_fusion as pf                                    # noqa: E402


# --------------------------------------------------------------------------- #
EMITIDOS: List[dict] = []


async def _broadcast_espia(mensaje: str) -> None:
    EMITIDOS.append(json.loads(mensaje))


ws.manager.broadcast = _broadcast_espia

# El mail se intercepta: esta suite prueba el endpoint, no gmail. Sin esto, la
# tarea de fondo abre una conexión SMTP real y la prueba se cuelga — que es
# exactamente lo que le pasaría al backend en una máquina sin salida al 465.
MAILS: List[tuple] = []
import src.routers.alerta_rutas as ar                          # noqa: E402
ar.enviar_alerta_email = lambda *a, **k: MAILS.append(a)

app = FastAPI()
app.include_router(prefix="/alertas", router=alerta_router)
crear_tablas()
cliente = TestClient(app)

with Session(engine) as s:
    s.add(CamaraConfig(nombre="cam-deposito", usb_index=0))
    s.add(NumeroEmergencia(telefono="guardia@sucursal.test", nombre="Guardia"))
    s.commit()


def eventos(caso: str) -> List[Any]:
    _, _, ev = pf.CASOS[caso]()
    return list(ev)


def postear(payload: dict) -> dict:
    r = cliente.post("/alertas", json=payload)
    assert r.status_code == 200, f"{r.status_code}: {r.text[:300]}"
    return r.json()


def limpiar() -> None:
    EMITIDOS.clear()
    MAILS.clear()
    with Session(engine) as s:
        for tabla in (Alerta, Camara):
            for f in s.exec(select(tabla)).all():
                s.delete(f)
        s.commit()


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #
def caso_alerta_real() -> Tuple[bool, str]:
    """Un incendio real de la fusión entra, se guarda y sale al panel."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    p = armar_payload(ev, 1, sitio="sucursal-centro", nodo="caja-01")
    r = postear(p)
    with Session(engine) as s:
        fila = s.exec(select(Alerta)).first()
    ok = (r["accion"] == "nueva" and r["emitido_al_panel"]
          and fila is not None and fila.tipo == "incendio"
          and json.loads(fila.modelos) and len(EMITIDOS) == 1)
    return ok, f"accion={r['accion']}, modelos={fila.modelos if fila else '-'}"


def caso_forma_del_websocket() -> Tuple[bool, str]:
    """El mensaje del WS trae los cinco campos que lee el front, y el mensaje
    completo de Horus adentro de `alerta`.

    Los cinco quedaban `undefined` cuando se emitía el payload de Horus crudo,
    y el panel mostraba tarjetas vacías."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    postear(armar_payload(ev, 1))
    m = EMITIDOS[0]
    requeridos = ("camera_id", "event_type", "timestamp", "nombre_camara", "confidence")
    faltan = [k for k in requeridos if k not in m or m[k] is None]
    ok = not faltan and m["alerta"]["id"] == ev.evento_id and m["nombre_camara"] == "cam-deposito"
    return ok, f"faltan={faltan or 'ninguno'} · nombre={m.get('nombre_camara')}"


def caso_idempotencia() -> Tuple[bool, str]:
    """El mismo `(id, secuencia)` dos veces no duplica.

    Puede pasar de verdad: el emisor reintenta si se pierde la respuesta."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    p = armar_payload(ev, 1)
    r1, r2 = postear(p), postear(p)
    with Session(engine) as s:
        n = len(s.exec(select(Alerta)).all())
    ok = r1["accion"] == "nueva" and r2["accion"] == "duplicada" and n == 1
    return ok, f"{r1['accion']} -> {r2['accion']}, {n} fila(s)"


def caso_un_incendio_una_tarjeta() -> Tuple[bool, str]:
    """30 s de incendio = 1 tarjeta en el panel, no diez.

    Se emite al abrir y al subir de severidad; el resto queda en la base."""
    limpiar()
    ev = [e for e in eventos("dedup")]
    for i, e in enumerate(ev, start=1):
        postear(armar_payload(e, i))
    with Session(engine) as s:
        guardadas = len(s.exec(select(Alerta)).all())
        tarjetas = len(s.exec(select(Camara)).all())
    ok = guardadas == len(ev) and len(EMITIDOS) <= 2 and tarjetas == len(EMITIDOS)
    return ok, (f"{guardadas} mensaje(s) guardados, {len(EMITIDOS)} al panel, "
                f"{tarjetas} fila(s) de compatibilidad")


def caso_escalada() -> Tuple[bool, str]:
    """Subir de severidad SÍ vuelve a emitir: es lo que el operador necesita ver."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    p1 = armar_payload(ev, 1)
    p1.update(severidad="alerta", severidad_num=2)
    p2 = armar_payload(ev, 2)
    p2.update(severidad="critico", severidad_num=3)
    r1, r2 = postear(p1), postear(p2)
    ok = r1["emitido_al_panel"] and r2["emitido_al_panel"] and len(EMITIDOS) == 2
    return ok, f"{r1['accion']}/{r2['accion']}, {len(EMITIDOS)} emisiones"


def caso_fuera_de_orden() -> Tuple[bool, str]:
    """Un mensaje viejo se guarda pero NO se re-emite.

    La cola del emisor prioriza por severidad, así que llegan desordenados."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    postear(armar_payload(ev, 3))
    antes = len(EMITIDOS)
    r = postear(armar_payload(ev, 2))
    with Session(engine) as s:
        n = len(s.exec(select(Alerta)).all())
    ok = r["accion"] == "tarde" and not r["emitido_al_panel"] and n == 2 and len(EMITIDOS) == antes
    return ok, f"accion={r['accion']}, {n} guardados, {len(EMITIDOS)} emisiones"


def caso_confianza_baja() -> Tuple[bool, str]:
    """Un arma con 0.50 de confianza tiene que entrar.

    El validador de `Camara.confidence` exigía ≥ 0.90 y hacía 500 justo con
    las alertas que más importan: el arma que el VLM tiene que verificar."""
    limpiar()
    ev = [e for e in eventos("arma") if e.tipo == "arma"][0]
    p = armar_payload(ev, 1)
    p["confianza"] = 0.50
    r = postear(p)
    with Session(engine) as s:
        fila = s.exec(select(Camara)).first()
    ok = r["recibido"] and fila is not None and abs(fila.confidence - 0.50) < 1e-6
    return ok, f"guardada con confidence={getattr(fila, 'confidence', None)}"


def caso_agresion_llega_entera() -> Tuple[bool, str]:
    """La agresión llega crítica y con los aportes de las dos piezas."""
    limpiar()
    evs = [e for e in eventos("agresion_arma") if e.tipo == "agresion"]
    postear(armar_payload(evs[0], 1))
    with Session(engine) as s:
        fila = s.exec(select(Alerta).where(Alerta.tipo == "agresion")).first()
    modelos = json.loads(fila.modelos) if fila else []
    ev_json = json.loads(fila.evidencia) if fila else {}
    tracks = set(json.loads(fila.tracks)) if fila else set()
    ok = (fila is not None and fila.severidad == "critico"
          and {"accion", "objetos"} <= set(modelos)
          and ev_json.get("correlacion") == "pelea+arma"
          # los dos que peleaban, más el track del arma
          and {31, 32} <= tracks)
    return ok, (f"sev={getattr(fila,'severidad',None)} modelos={modelos} "
                f"corr={ev_json.get('correlacion')} tracks={sorted(tracks)}")


def caso_historial() -> Tuple[bool, str]:
    """`GET /alertas/{id}` devuelve todos los mensajes; el listado, uno solo."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    for i in (1, 2, 3):
        postear(armar_payload(ev, i))
    h = cliente.get(f"/alertas/{ev.evento_id}").json()
    lista = cliente.get("/alertas").json()
    ok = h["encontrado"] and len(h["mensajes"]) == 3 and len(lista) == 1
    return ok, f"{len(h['mensajes'])} mensajes en el historial, {len(lista)} en el listado"


def caso_mail_no_tumba() -> Tuple[bool, str]:
    """Con el SMTP roto la alerta igual se guarda y contesta 200.

    Antes se guardaba la fila, contestaba 500, el emisor reintentaba y se
    duplicaba todo."""
    limpiar()
    def explota(*a: Any, **k: Any) -> None:
        raise OSError("SMTP no configurado")

    original, ar.enviar_alerta_email = ar.enviar_alerta_email, explota  # noqa: F811
    try:
        ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
        r = postear(armar_payload(ev, 1))
    finally:
        ar.enviar_alerta_email = original
    with Session(engine) as s:
        n = len(s.exec(select(Alerta)).all())
    ok = r["recibido"] and n == 1
    return ok, f"200 con SMTP roto, {n} fila guardada"


def caso_campo_nuevo() -> Tuple[bool, str]:
    """Un campo que el backend no conoce no puede hacer 422.

    Si lo hiciera, el emisor reintentaría tres veces y mandaría al spool una
    alerta perfectamente válida."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    p = armar_payload(ev, 1)
    p["campo_del_futuro"] = {"algo": 1}
    r = cliente.post("/alertas", json=p)
    ok = r.status_code == 200 and r.json()["recibido"]
    return ok, f"status={r.status_code}"


def caso_mail_desde_alerta() -> Tuple[bool, str]:
    """El aviso a los contactos sale desde severidad 2, no antes.

    Un aviso (1) que despierta a alguien a las 3 de la mañana es lo que hace
    que la próxima alerta de verdad se ignore."""
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    p = armar_payload(ev, 1)
    p.update(severidad="aviso", severidad_num=1)
    postear(p)
    sin_mail = len(MAILS)
    limpiar()
    postear(armar_payload(ev, 1))               # crítico
    ok = sin_mail == 0 and len(MAILS) == 1
    return ok, f"aviso -> {sin_mail} mail(s) · crítico -> {len(MAILS)} mail(s)"


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "alerta_real": caso_alerta_real,
    "forma_ws": caso_forma_del_websocket,
    "idempotencia": caso_idempotencia,
    "una_tarjeta": caso_un_incendio_una_tarjeta,
    "escalada": caso_escalada,
    "fuera_de_orden": caso_fuera_de_orden,
    "confianza_baja": caso_confianza_baja,
    "agresion": caso_agresion_llega_entera,
    "historial": caso_historial,
    "mail_desde_2": caso_mail_desde_alerta,
    "mail_no_tumba": caso_mail_no_tumba,
    "campo_nuevo": caso_campo_nuevo,
}


def main() -> int:
    print("=" * 78)
    print("HORUS · endpoint de alertas — punta a punta (fusión -> FastAPI -> SQLite)")
    print("=" * 78)
    fallas = 0
    for nombre, f in CASOS.items():
        t0 = time.perf_counter()
        try:
            ok, detalle = f()
        except Exception as e:
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:<16} {detalle:<54} {ms:7.1f} ms")
        fallas += not ok
    print("-" * 78)
    print(f"{len(CASOS) - fallas}/{len(CASOS)} pruebas OK")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
