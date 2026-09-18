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
import re
from pathlib import Path
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

_RAIZ_REPO = Path(__file__).resolve().parent.parent

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


# El de verdad, guardado antes de pisarlo: `caso_websocket_de_verdad` lo repone
# para levantar una conexión real en vez de espiar la intención de emitir.
_MANAGER_REAL = ws.manager.broadcast
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


def caso_el_panel_conoce_los_tipos() -> Tuple[bool, str]:
    """Cada tipo que emite la fusión tiene etiqueta en el panel.

    Son dos lenguajes y dos carpetas: la fusión inventa un tipo en python y el
    panel lo traduce en un objeto de javascript. Nada los ata, así que agregar
    una regla nueva y olvidarse del panel es lo natural — y no se rompe nada,
    solo aparece "paquete_abandonado" en crudo al lado de "Incendio".

    El 18/09 el mapa conocía 5 claves y la fusión emitía 9. Coincidía una.
    """
    js = _RAIZ_REPO / "HorusAI" / "src" / "hooks" / "useWebSocketEventos.js"
    if not js.exists():
        return True, "sin HorusAI en esta rama, se saltea"

    import motor_fusion as mf
    motor = mf.MotorFusion(topologia=None, cfg=mf.ConfigFusion(), fps=10.0)
    tipos = {r.tipo for r in motor.reglas}
    for c in mf.CORRELACIONES_ESCENA:
        tipos |= {c.base, c.con, c.produce}

    texto = js.read_text(encoding="utf-8")
    i = texto.index("const TIPOS")
    claves = set(re.findall(r"^\s*([a-z_]+):", texto[i:texto.index("};", i)], re.M))

    falta = sorted(tipos - claves)
    return not falta, (f"{len(tipos)} tipos de la fusión, {len(claves)} etiquetas · "
                       f"{'sin etiqueta: ' + ', '.join(falta) if falta else 'todos cubiertos'}")


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


def _texto_js(*partes: str) -> str:
    return (_RAIZ_REPO / "HorusAI" / "src" / Path(*partes)).read_text(encoding="utf-8")


def _hay_front() -> bool:
    return (_RAIZ_REPO / "HorusAI" / "src" / "config.js").exists()


def caso_el_panel_no_tiene_la_url_a_mano() -> Tuple[bool, str]:
    """Nadie escribe la dirección del backend a mano, salvo `config.js`.

    Hasta el 18/09 `config.js` exportaba API_URL y WS_URL y no lo importaba
    NADIE: ocho archivos tenían "http://localhost:8000" adentro. Mover el
    backend a otro puerto o a otra máquina significaba editar ocho archivos y
    acordarse de los ocho, y el que se olvidaba no fallaba al compilar: fallaba
    en vivo, pidiéndole video a una máquina que no existe.
    """
    if not _hay_front():
        return True, "sin HorusAI en esta rama, se saltea"

    raiz = _RAIZ_REPO / "HorusAI" / "src"
    culpables = []
    for f in sorted(raiz.rglob("*.js")) + sorted(raiz.rglob("*.jsx")):
        if f.name == "config.js":
            continue
        for i, l in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "localhost:8000" in l or "127.0.0.1:8000" in l:
                culpables.append(f"{f.relative_to(raiz)}:{i}")
    return not culpables, ("todo sale de config.js" if not culpables
                           else f"a mano en: {', '.join(culpables[:4])}")


def _ruta_ws_del_panel() -> str:
    """La ruta del websocket que el panel va a abrir, sacada de config.js."""
    texto = _texto_js("config.js")
    linea = [l for l in texto.splitlines() if "WS_ALERTAS" in l and "=" in l][0]
    valor = linea.split("=", 1)[1].strip().strip(";").strip("`\"'")
    valor = re.sub(r"\$\{[^}]*\}", "", valor)      # se va el ${WS_URL}
    return valor.split("?", 1)[0]


def caso_la_ruta_del_panel_existe() -> Tuple[bool, str]:
    """La ruta a la que se conecta el panel es LA de alertas, no otra.

    El Dashboard abría `/camaras/ws`, que NO es el websocket de alertas.
    Andaba igual, porque `ConnectionManager.broadcast()` reparte a todas las
    conexiones sin mirar en qué bucket están. O sea: funcionaba de carambola.
    El día que broadcast filtre por cámara —que es lo razonable cuando haya
    varias— el panel se queda mudo y no hay ni un error que lo delate.

    Ojo con la trampa: los dos routers declaran `/ws`, así que mirar el final
    de la ruta no alcanza. Lo que decide es el prefijo con el que `main.py`
    monta cada router.
    """
    if not _hay_front():
        return True, "sin HorusAI en esta rama, se saltea"

    from src.routers.alerta_rutas import alerta_router as _ar

    main_py = (_RAIZ_REPO / "FASTAPI" / "src" / "main.py").read_text(encoding="utf-8")
    prefijos = dict(re.findall(r"include_router\(\s*prefix=['\"]([^'\"]+)['\"]\s*,\s*"
                               r"router=(\w+)", main_py))
    prefijo = next((p for p, r in prefijos.items() if r == "alerta_router"), None)
    if prefijo is None:
        return False, "main.py no monta alerta_router con un prefijo reconocible"

    caminos = {prefijo + x.path for x in _ar.routes
               if "WebSocket" in type(x).__name__}
    ruta = _ruta_ws_del_panel()
    return ruta in caminos, (f"el panel abre {ruta} · alertas escucha en "
                             f"{sorted(caminos)}")


def caso_websocket_de_verdad() -> Tuple[bool, str]:
    """Un websocket DE VERDAD, abierto como lo abre el panel, recibe la alerta.

    El resto de la suite espía `broadcast`, así que prueba que el backend
    quiso emitir. Acá se levanta la conexión real: si el mensaje no sale por el
    socket —porque el manager no lo registró, porque el json no serializa,
    porque la ruta no es esa— esto es lo único que se entera.
    """
    limpiar()
    espia, ws.manager.broadcast = ws.manager.broadcast, _MANAGER_REAL
    try:
        ruta = _ruta_ws_del_panel() if _hay_front() else "/alertas/ws"
        ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
        with cliente.websocket_connect(f"{ruta}?camara_config_id=0") as sock:
            postear(armar_payload(ev, 1, sitio="sucursal-centro"))
            crudo = sock.receive_text()
    finally:
        ws.manager.broadcast = espia

    m = json.loads(crudo)
    # Los cinco campos que `eventoDelSocket()` lee en useWebSocketEventos.js.
    faltan = [k for k in ("camera_id", "event_type", "timestamp",
                          "nombre_camara", "confidence") if m.get(k) is None]
    ok = not faltan and m["event_type"] == "incendio" and m["nombre_camara"] == "cam-deposito"
    return ok, (f"llegó por el socket: {m.get('event_type')} en "
                f"{m.get('nombre_camara')} · faltan={faltan or 'ninguno'}")


def caso_el_historial_dice_el_mismo_nombre() -> Tuple[bool, str]:
    """El mismo evento se llama igual en vivo que al recargar.

    La fila guarda `camara` como la nombra Horus (`cam-deposito`) y el
    websocket manda el nombre registrado en el backend. Si el listado no
    resuelve el nombre, el mismo incendio aparece como "Depósito" mientras
    pasa y como "cam-deposito" al día siguiente, y no hay forma de saber que
    son el mismo.
    """
    limpiar()
    ev = [e for e in eventos("incendio") if e.tipo == "incendio"][0]
    postear(armar_payload(ev, 1))
    en_vivo = EMITIDOS[0]["nombre_camara"]
    fila = cliente.get("/alertas").json()[0]
    ok = fila.get("nombre_camara") == en_vivo and fila["evento_id"] == ev.evento_id
    return ok, f"en vivo '{en_vivo}' · en el historial '{fila.get('nombre_camara')}'"


def caso_el_historial_se_puede_leer() -> Tuple[bool, str]:
    """Lo que devuelve `GET /alertas` tiene los campos que el panel mapea.

    `useHistorial` arranca pidiendo lo guardado y lo pasa por
    `eventoDeLaBase()`, que lee evento_id, tipo, ts_inicio y confianza. Si
    alguno viniera con otro nombre, el historial se llenaría de filas vacías
    en vez de fallar.
    """
    limpiar()
    for caso in ("incendio", "caida_pose"):
        for e in eventos(caso):
            postear(armar_payload(e, 1))
    filas = cliente.get("/alertas?limite=200").json()
    necesarios = ("evento_id", "tipo", "ts_inicio", "confianza", "nombre_camara")
    huecos = [k for f in filas for k in necesarios if f.get(k) is None]
    ok = bool(filas) and not huecos
    return ok, f"{len(filas)} fila(s) · campos vacíos: {sorted(set(huecos)) or 'ninguno'}"


def caso_los_filtros_saben_pluralizar() -> Tuple[bool, str]:
    """Cada etiqueta del panel tiene su plural escrito.

    El botón de filtro mostraba `tipo + "s"`. Con tres tipos cortos pasaba
    desapercibido; con diez da "Intrusións", "Acción externas" y "Paquete
    abandonados". No hay regla corta que acierte en castellano, así que los
    plurales se escriben — y esto verifica que no falte ninguno.
    """
    if not _hay_front():
        return True, "sin HorusAI en esta rama, se saltea"

    texto = _texto_js("hooks", "useWebSocketEventos.js")

    def valores(nombre: str) -> set:
        i = texto.index(nombre)
        cuerpo = texto[i:texto.index("};", i)]
        return set(re.findall(r':\s*"([^"]+)"', cuerpo))

    etiquetas = valores("const TIPOS = {")
    plurales = set(re.findall(r'^\s*"([^"]+)":', texto[texto.index("PLURALES = {"):
                                                       texto.index("};", texto.index("PLURALES = {"))],
                              re.M))
    falta = sorted(etiquetas - plurales)
    return not falta, (f"{len(etiquetas)} etiquetas, {len(plurales)} plurales · "
                       f"{'sin plural: ' + ', '.join(falta) if falta else 'todas'}")


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "alerta_real": caso_alerta_real,
    "forma_ws": caso_forma_del_websocket,
    "tipos_panel": caso_el_panel_conoce_los_tipos,
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
    # --- la conexión con el panel ---
    "url_centralizada": caso_el_panel_no_tiene_la_url_a_mano,
    "ruta_ws_existe": caso_la_ruta_del_panel_existe,
    "ws_de_verdad": caso_websocket_de_verdad,
    "mismo_nombre": caso_el_historial_dice_el_mismo_nombre,
    "historial_legible": caso_el_historial_se_puede_leer,
    "plurales": caso_los_filtros_saben_pluralizar,
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
