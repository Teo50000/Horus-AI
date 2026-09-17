"""
alerta_rutas.py · el endpoint que recibe las alertas de Horus.

⚠ Reescrito el 2026-09-15 desde `contrato-alertas-backend.md`. El original se
perdió y no estaba en ninguna rama; quedó solo el `.pyc`.

Las cuatro decisiones que hay que entender
------------------------------------------
**1 · Idempotente por `(id, secuencia)`.** El emisor reintenta y spoolea, así
que el mismo mensaje puede llegar dos veces si solo se perdió la respuesta. Se
contesta 200 con `accion="duplicada"` en vez de duplicar la fila.

**2 · Un mensaje viejo se descarta.** Los mensajes NO llegan en orden: la cola
del emisor prioriza por severidad, así que un crítico se adelanta a los avisos
que ya estaban esperando. Si llega una secuencia menor o igual a la última
guardada de ese evento, se guarda igual (el historial completo sirve) pero no
se vuelve a emitir al panel.

**3 · Al panel se emite solo al ABRIR y al SUBIR de severidad.** Un evento
manda apertura, subidas, re-avisos cada 20 s y cierre; el front hace
`setEventos(prev => [nuevo, ...prev])` sin deduplicar, así que emitir todo
llenaba el historial con diez tarjetas por incendio. El resto queda en la base
y se ve en `GET /alertas/{id}`.

**4 · El mail no puede tumbar la alerta.** Va en `BackgroundTasks` y envuelto:
con el SMTP mal configurado, mandarlo en línea dejaba la fila guardada pero
contestaba 500, el emisor reintentaba y se duplicaba todo.
"""

import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from src.database import get_session
from src.models.alerta_model import Alerta, AlertaEntrante, RespuestaAlerta
from src.models.camara_model import Camara, CamaraConfig, NumeroEmergencia
from src.services.email_service import enviar_alerta_email
from src.services.websockets import manager

alerta_router = APIRouter(tags=["Alertas"])

# Desde acá para arriba se avisa a los contactos de emergencia.
# 2 = alerta, 3 = crítico. Un aviso (1) no despierta a nadie a las 3 de la
# mañana, que es justamente lo que lo vuelve creíble cuando sí suena.
SEVERIDAD_MAIL = 2


# --------------------------------------------------------------------------- #
def _config_de(session: Session, nombre: str) -> Optional[CamaraConfig]:
    """La cámara de Horus (`cam-deposito`) contra la registrada en el backend.

    Horus la nombra por string y el backend la tiene por id. Se busca por
    `nombre`, y si el nombre es un número se prueba como id — así funciona con
    las dos formas de registrar que conviven hoy.
    """
    cfg = session.exec(select(CamaraConfig).where(CamaraConfig.nombre == nombre)).first()
    if cfg is None and str(nombre).isdigit():
        cfg = session.get(CamaraConfig, int(nombre))
    return cfg


def _fila_compatibilidad(session: Session, msg: AlertaEntrante,
                         cfg: Optional[CamaraConfig]) -> Optional[int]:
    """La fila de `Camara` que el panel viejo ya sabía leer.

    Se escribe con el MISMO criterio que la emisión al WebSocket (solo
    apertura y subida de severidad) para que lo que se ve al recargar sea lo
    mismo que se vio en vivo.

    Ojo: `Camara.confidence` tenía un validador que exigía ≥ 0.90. Horus
    trabaja al revés a propósito — 'persistencia antes que umbral' — y manda
    eventos reales con 0.82 (una pelea) o 0.50 (un arma que el VLM tiene que
    verificar). Con el validador puesto, la mayoría de las alertas de verdad
    explotaban acá. Ver la nota en `camara_model.py`.
    """
    try:
        fila = Camara(
            event_type=msg.tipo,
            confidence=float(msg.confianza),
            timestamp=msg.tiempo.inicio,
            camara_config_id=getattr(cfg, "id", None),
            camara_config_nombre=getattr(cfg, "nombre", None) or msg.sitio.camara,
            description=msg.motivo or None,
        )
        session.add(fila)
        session.flush()
        return fila.camera_id
    except Exception as e:                      # nunca tumbar la alerta por esto
        print(f"[alertas] no se pudo escribir la fila de compatibilidad: {e}")
        return None


def _avisar_por_mail(event_type: str, nombre_camara: str, confidence: float,
                     timestamp: str, destinos: List[str]) -> None:
    """Envuelto entero: un SMTP mal configurado no puede romper nada."""
    for destino in destinos:
        try:
            enviar_alerta_email(event_type, nombre_camara, confidence,
                                timestamp, destino)
        except Exception as e:
            print(f"[alertas] falló el mail a {destino}: {e}")


def _destinos(session: Session) -> List[str]:
    """Contactos de emergencia con mail.

    `NumeroEmergencia` no tiene columna de email: guarda `telefono`. Se acepta
    lo que parezca una dirección para no bloquear el aviso, pero la tabla
    necesita su propia columna — está anotado como pendiente.
    """
    contactos = session.exec(select(NumeroEmergencia)).all()
    return [c.telefono for c in contactos
            if c.telefono and "@" in str(c.telefono)]


# --------------------------------------------------------------------------- #
@alerta_router.post("", response_model=RespuestaAlerta)
@alerta_router.post("/", response_model=RespuestaAlerta, include_in_schema=False)
async def recibir_alerta(msg: AlertaEntrante, tareas: BackgroundTasks,
                         session: Session = Depends(get_session)) -> RespuestaAlerta:
    if msg.version != 1:
        # Subir `version` es la forma que tiene Horus de decir "cambió algo
        # incompatible". Aceptarlo a ciegas sería guardar mal en silencio.
        return RespuestaAlerta(recibido=False, accion="tarde",
                               evento_id=msg.id, secuencia=msg.secuencia,
                               detalle=f"versión {msg.version} no soportada")

    previas = session.exec(
        select(Alerta).where(Alerta.evento_id == msg.id)
        .order_by(Alerta.secuencia.desc())).all()

    if any(p.secuencia == msg.secuencia for p in previas):
        return RespuestaAlerta(recibido=True, accion="duplicada",
                               evento_id=msg.id, secuencia=msg.secuencia,
                               detalle="ya estaba guardada")

    ultima = previas[0] if previas else None
    es_nueva = ultima is None
    tarde = ultima is not None and msg.secuencia < ultima.secuencia
    subio = ultima is not None and msg.severidad_num > ultima.severidad_num

    cfg = _config_de(session, msg.sitio.camara)
    al_panel = es_nueva or (subio and not tarde)

    fila_id = _fila_compatibilidad(session, msg, cfg) if al_panel else None

    session.add(Alerta(
        evento_id=msg.id, secuencia=msg.secuencia, version=msg.version,
        tipo=msg.tipo, severidad=msg.severidad, severidad_num=msg.severidad_num,
        estado=msg.estado, confianza=msg.confianza, motivo=msg.motivo,
        camara=msg.sitio.camara, camaras=json.dumps(msg.sitio.camaras),
        zona=msg.sitio.zona, instalacion=msg.sitio.instalacion, nodo=msg.sitio.nodo,
        ts_inicio=msg.tiempo.inicio, ts_ultimo=msg.tiempo.ultimo,
        ts_emitido=msg.tiempo.emitido, epoch_inicio=msg.tiempo.epoch_inicio,
        duracion_s=msg.tiempo.duracion_s,
        global_id=msg.sujeto.global_id, tracks=json.dumps(msg.sujeto.tracks),
        aportes=json.dumps([a.model_dump() for a in msg.aportes], ensure_ascii=False),
        modelos=json.dumps(msg.modelos),
        necesita_vlm=msg.verificacion.necesita_vlm,
        verificacion_estado=msg.verificacion.estado,
        verificacion_motivo=msg.verificacion.motivo,
        confirmaciones=msg.confirmaciones,
        evidencia=json.dumps(msg.evidencia, ensure_ascii=False, default=str),
        camara_config_id=getattr(cfg, "id", None), camara_evento_id=fila_id,
        recibido_en=datetime.now().astimezone().isoformat(timespec="seconds"),
    ))
    session.commit()

    if al_panel:
        nombre = getattr(cfg, "nombre", None) or msg.sitio.camara
        # La forma vieja, que es la que `useWebSocketEventos.js` sabe leer.
        # El mensaje completo de Horus viaja adentro de `alerta`: el panel
        # nuevo lo usa y el viejo lo ignora sin enterarse.
        await manager.broadcast(json.dumps({
            "camera_id": fila_id,
            "event_type": msg.tipo,
            "timestamp": msg.tiempo.inicio,
            "nombre_camara": nombre,
            "confidence": msg.confianza,
            "alerta": msg.model_dump(),
        }, ensure_ascii=False, default=str))

        if msg.severidad_num >= SEVERIDAD_MAIL:
            destinos = _destinos(session)
            if destinos:
                tareas.add_task(_avisar_por_mail, msg.tipo, nombre,
                                msg.confianza, msg.tiempo.inicio, destinos)

    return RespuestaAlerta(
        recibido=True,
        accion="nueva" if es_nueva else ("tarde" if tarde else "actualizada"),
        evento_id=msg.id, secuencia=msg.secuencia, emitido_al_panel=al_panel,
        detalle=("abierta" if es_nueva else
                 "llegó fuera de orden, guardada sin emitir" if tarde else
                 "subió de severidad" if subio else "sostenida"))


# --------------------------------------------------------------------------- #
@alerta_router.get("")
@alerta_router.get("/", include_in_schema=False)
def listar_alertas(session: Session = Depends(get_session),
                   severidad_min: int = Query(0, ge=0, le=3),
                   tipo: Optional[str] = None,
                   camara: Optional[str] = None,
                   solo_ultimas: bool = Query(True, description="una fila por evento"),
                   limite: int = Query(100, ge=1, le=1000)):
    """El historial. Por defecto UNA fila por evento, la más reciente.

    Devolver todos los mensajes por defecto haría que el panel muestre el
    mismo incendio diez veces, que es el problema que el filtro de emisión
    resuelve del otro lado.
    """
    q = select(Alerta).where(Alerta.severidad_num >= severidad_min)
    if tipo:
        q = q.where(Alerta.tipo == tipo)
    if camara:
        q = q.where(Alerta.camara == camara)
    filas = session.exec(q.order_by(Alerta.id.desc())).all()

    if solo_ultimas:
        vistos, out = set(), []
        for f in filas:
            if f.evento_id in vistos:
                continue
            vistos.add(f.evento_id)
            out.append(f)
        filas = out
    return [_json(f) for f in filas[:limite]]


@alerta_router.get("/{evento_id}")
def historial_evento(evento_id: str = Path(...),
                     session: Session = Depends(get_session)):
    """Todos los mensajes de un evento, en orden de `secuencia`.

    Acá está lo que NO se emite al panel: los re-avisos cada 20 s, el cierre, y
    el mensaje que manda el VLM después con la verificación.
    """
    filas = session.exec(select(Alerta).where(Alerta.evento_id == evento_id)
                         .order_by(Alerta.secuencia)).all()
    if not filas:
        return {"evento_id": evento_id, "mensajes": [], "encontrado": False}
    ultima = filas[-1]
    return {
        "evento_id": evento_id,
        "encontrado": True,
        "tipo": ultima.tipo,
        "severidad": ultima.severidad,
        "estado": ultima.estado,
        "camara": ultima.camara,
        "modelos": json.loads(ultima.modelos or "[]"),
        "mensajes": [_json(f) for f in filas],
    }


@alerta_router.websocket("/ws")
async def ws_alertas(websocket: WebSocket, camara_config_id: int = Query(0)):
    """Canal del panel. Se reusa el manager que ya existía para el video."""
    await manager.connect(websocket, camara_config_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, camara_config_id)


# --------------------------------------------------------------------------- #
def _json(f: Alerta) -> dict:
    d = f.model_dump()
    for k in ("camaras", "tracks", "aportes", "modelos", "evidencia"):
        try:
            d[k] = json.loads(d.get(k) or ("{}" if k == "evidencia" else "[]"))
        except (TypeError, ValueError):
            d[k] = None
    return d
