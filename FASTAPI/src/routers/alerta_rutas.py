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
import re
from datetime import datetime
from typing import Any, List, Optional

from fastapi import (APIRouter, BackgroundTasks, Depends, HTTPException, Path,
                     Query, WebSocket, WebSocketDisconnect)
from pydantic import BaseModel
from sqlmodel import Session, select

from src.database import engine, get_session
from src.models.alerta_model import Alerta, AlertaEntrante, RespuestaAlerta
from src.models.camara_model import Camara, CamaraConfig, NumeroEmergencia
from src.services.email_service import (MailApagado, enviar_alerta_email,
                                        estado_mail)
from src.services.websockets import manager
from src.services import evidencia

alerta_router = APIRouter(tags=["Alertas"])

# Desde acá para arriba se avisa a los contactos de emergencia.
# 2 = alerta, 3 = crítico. Un aviso (1) no despierta a nadie a las 3 de la
# mañana, que es justamente lo que lo vuelve creíble cuando sí suena.
SEVERIDAD_MAIL = 2


# --------------------------------------------------------------------------- #
def _config_de(session: Session, sitio: Any) -> Optional[CamaraConfig]:
    """La cámara de Horus (`cam-1`) contra la registrada en el backend.

    26/09 — "el nombre de la cámara no es el mismo que el que está registrado
    en la base". Horus nombra las cámaras `cam-<id>` y el backend las guarda
    con el nombre que les puso el usuario ("Cocina"). Esta función buscaba
    SOLO por nombre, o por id si el nombre era un número: con "cam-1" no
    encontraba nada. Desde el 15/09 las alertas quedaban sin cámara asociada y
    el panel y el mail decían "cam-1".

    Y el dato estaba: el servicio manda `sitio.camara_config_id`, pero el
    modelo del POST lo tiraba (ver `_Sitio`). En orden:

      1. `sitio.camara_config_id`, el id exacto que manda el servicio;
      2. el nombre tal cual (una cámara registrada como "cam-deposito");
      3. `cam-<n>` -> id n, que es como las nombra el servicio;
      4. un número suelto -> id.
    """
    nombre = str(getattr(sitio, "camara", sitio) or "")
    cid = getattr(sitio, "camara_config_id", None)
    if cid is not None:
        try:
            cfg = session.get(CamaraConfig, int(cid))
        except (TypeError, ValueError):
            cfg = None
        if cfg is not None:
            return cfg
    cfg = session.exec(select(CamaraConfig).where(CamaraConfig.nombre == nombre)).first()
    if cfg is not None:
        return cfg
    m = re.fullmatch(r"cam-(\d+)", nombre)
    if m:
        cfg = session.get(CamaraConfig, int(m.group(1)))
    elif nombre.isdigit():
        cfg = session.get(CamaraConfig, int(nombre))
    return cfg


def _fila_compatibilidad(session: Session, msg: AlertaEntrante,
                         cfg: Optional[CamaraConfig],
                         captura_url: Optional[str] = None,
                         clip_url: Optional[str] = None) -> Optional[int]:
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
            # Los dos campos que los compañeros dejaron listos "para otro
            # sprint". Se llenan con la misma evidencia que la fila nueva.
            snapshot_url=captura_url,
            clip_url=clip_url,
        )
        session.add(fila)
        session.flush()
        return fila.camera_id
    except Exception as e:                      # nunca tumbar la alerta por esto
        print(f"[alertas] no se pudo escribir la fila de compatibilidad: {e}")
        return None


def _anotar_mail(evento_id: str, secuencia: int, estado: str,
                 detalle: str) -> None:
    """Deja escrito en la fila qué pasó con el aviso.

    22/09. Sin esto, una alerta que no le llegó a nadie se veía EXACTAMENTE
    igual que una que sí: misma fila, mismo 200, misma tarjeta en el panel.
    El único rastro era un `print` en una ventana que nadie mira. Para un
    sistema que existe para avisar, eso es la falla que más caro sale.

    Va en su propia sesión porque corre en `BackgroundTasks`, después de que
    la del request ya se cerró. Y va envuelto: no poder ANOTAR el fallo no
    puede ser, encima, otro motivo de caída.
    """
    try:
        with Session(engine) as s:
            fila = s.exec(select(Alerta)
                          .where(Alerta.evento_id == evento_id)
                          .where(Alerta.secuencia == secuencia)).first()
            if fila is not None:
                fila.mail_estado = estado
                fila.mail_detalle = detalle[:300]
                s.add(fila)
                s.commit()
    except Exception as e:
        print(f"[alertas] no pude anotar el estado del mail: {e}")


def _avisar_por_mail(event_type: str, nombre_camara: str, confidence: float,
                     timestamp: str, destinos: List[str],
                     evento_id: str = "", secuencia: int = 0,
                     captura: Optional[str] = None, clip: Optional[str] = None,
                     motivo: str = "") -> None:
    """Envuelto entero: un SMTP mal configurado no puede romper nada.

    Pero "no rompe nada" no puede significar "no se entera nadie": el
    resultado queda en la fila. Un mail apagado (sin credenciales) y un mail
    que falló son dos cosas distintas y se guardan distinto — el primero se
    configura una vez, el segundo se reintenta.
    """
    enviados, fallos, apagado = [], [], ""
    for destino in destinos:
        try:
            # La captura va en el cuerpo y el clip adjunto. Solo se pasan si
            # hay: así el mail de siempre sale igual cuando no hay evidencia.
            extra_mail = {k: v for k, v in (("captura", captura), ("clip", clip),
                                            ("motivo", motivo)) if v}
            enviar_alerta_email(event_type, nombre_camara, confidence,
                                timestamp, destino, **extra_mail)
            enviados.append(destino)
        except MailApagado as e:
            apagado = str(e)
            print(f"[alertas] MAIL APAGADO, nadie fue avisado: {e}")
            break                       # con el resto va a pasar lo mismo
        except Exception as e:
            fallos.append(f"{destino}: {e}")
            print(f"[alertas] falló el mail a {destino}: {e}")

    if apagado:
        estado, detalle = "apagado", apagado
    elif enviados and not fallos:
        estado, detalle = "enviado", ", ".join(enviados)
    elif enviados:
        estado, detalle = "parcial", f"salió a {', '.join(enviados)}; " \
                                     f"falló {'; '.join(fallos)}"
    else:
        estado, detalle = "fallo", "; ".join(fallos) or "sin resultado"

    if evento_id:
        _anotar_mail(evento_id, secuencia, estado, detalle)


def _destinos(session: Session) -> List[str]:
    """Contactos de emergencia a los que se les puede mandar un mail.

    22/09: la columna `email` que faltaba ya existe. `direccion_mail()` mira
    primero ahí y después en `telefono`, donde quedaron guardadas las
    direcciones de los contactos que se cargaron antes.
    """
    contactos = session.exec(select(NumeroEmergencia)).all()
    return [d for d in (c.direccion_mail() for c in contactos) if d]


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

    cfg = _config_de(session, msg.sitio)
    al_panel = es_nueva or (subio and not tarde)

    # La captura y el clip vienen en `adjuntos` (ver src/services/evidencia).
    # Se sacan del mensaje ANTES de cualquier model_dump: son uno o dos MB en
    # base64 y no pueden terminar en la columna `evidencia` ni viajar por el
    # websocket a cada panel abierto.
    extra = msg.__pydantic_extra__ if msg.__pydantic_extra__ is not None else {}
    adjuntos = extra.pop("adjuntos", None)
    ev_arch = evidencia.guardar(adjuntos, evento_id=msg.id, secuencia=msg.secuencia,
                                cfg_id=getattr(cfg, "id", None),
                                camara=msg.sitio.camara) if adjuntos else {}
    # Sin adjuntos nuevos, lo del mensaje anterior del mismo evento: el
    # historial muestra UNA fila por evento, la última, y esa suele ser un
    # re-aviso que no trae imagen.
    captura_url = ev_arch.get("captura_url") or getattr(ultima, "captura_url", None)
    clip_url = ev_arch.get("clip_url") or getattr(ultima, "clip_url", None)
    cruda_url = ev_arch.get("cruda_url") or getattr(ultima, "cruda_url", None)

    fila_id = (_fila_compatibilidad(session, msg, cfg, captura_url, clip_url)
               if al_panel else None)

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
        captura_url=captura_url, clip_url=clip_url, cruda_url=cruda_url,
        # Si el evento ya estaba revisado, el mensaje nuevo hereda la marca.
        revision=getattr(ultima, "revision", None),
        revision_nota=getattr(ultima, "revision_nota", None),
        revisado_en=getattr(ultima, "revisado_en", None),
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
            "captura_url": captura_url,
            "clip_url": clip_url,
            "alerta": msg.model_dump(),
        }, ensure_ascii=False, default=str))

        if msg.severidad_num >= SEVERIDAD_MAIL:
            destinos = _destinos(session)
            listo, motivo = estado_mail()
            if not destinos:
                # "No hay a quién avisarle" también es una forma de no avisar.
                _anotar_mail(msg.id, msg.secuencia, "sin_destinos",
                             "no hay contactos de emergencia con dirección "
                             "de mail cargada")
            elif not listo:
                _anotar_mail(msg.id, msg.secuencia, "apagado", motivo)
                print(f"[alertas] MAIL APAGADO, nadie fue avisado: {motivo}")
            else:
                tareas.add_task(_avisar_por_mail, msg.tipo, nombre,
                                msg.confianza, msg.tiempo.inicio, destinos,
                                msg.id, msg.secuencia,
                                evidencia.ruta_de(captura_url),
                                evidencia.ruta_de(clip_url), msg.motivo)
        else:
            _anotar_mail(msg.id, msg.secuencia, "no_corresponde",
                         f"severidad {msg.severidad_num} < {SEVERIDAD_MAIL}")

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
    filas = filas[:limite]
    ids = {i for i in (_id_de(f) for f in filas) if i is not None}
    nombres = {}
    if ids:
        for cfg in session.exec(select(CamaraConfig).where(CamaraConfig.id.in_(ids))).all():
            nombres[cfg.id] = cfg.nombre
    return [_json(f, nombres) for f in filas]


def _id_de(f: Alerta) -> Optional[int]:
    """El id de la cámara registrada de una fila, aunque sea de antes del
    arreglo del 26/09 (esas quedaron sin `camara_config_id` pero con
    `camara="cam-1"`, y el 1 alcanza para encontrar el nombre)."""
    if f.camara_config_id is not None:
        return f.camara_config_id
    m = re.fullmatch(r"cam-(\d+)", str(f.camara or ""))
    return int(m.group(1)) if m else None


class _Revision(BaseModel):
    veredicto: str                       # "correcta" | "falsa" | "" (sacar la marca)
    nota: Optional[str] = None


# Qué clase NO estaba en el cuadro cuando una alerta de este tipo fue falsa.
# Es lo que `sumar_propio.py` necesita para usar el cuadro como negativo sin
# enseñarle al modelo que "acá no hay NADA" (puede haber una persona de verdad
# en una falsa alarma de arma).
AUSENTES_SI_FALSA = {
    "merodeo": ["persona"], "intrusion": ["persona"],
    "arma": ["pistola", "cuchillo"], "robo": ["pistola", "cuchillo"],
    "incendio": ["humo", "llama"],
    "paquete_abandonado": ["paquete"],
}


@alerta_router.post("/{evento_id}/revision")
def revisar_alerta(revision: _Revision, evento_id: str = Path(...),
                   session: Session = Depends(get_session)):
    """Capa 12: una persona dice si la alerta fue real o falsa.

    Además de anotarlo, copia la evidencia a `FASTAPI/revision/<veredicto>/`:
    el cuadro CRUDO (sin cajas dibujadas), el clip y un `meta.json` con el
    tipo, la cámara, el motivo y las clases que seguro no estaban. De ahí sale
    el dato de la casa para el próximo reentreno.
    """
    import os
    import shutil
    veredicto = (revision.veredicto or "").strip().lower()
    if veredicto not in ("correcta", "falsa", ""):
        raise HTTPException(status_code=400,
                            detail="veredicto tiene que ser 'correcta' o 'falsa'")
    filas = session.exec(select(Alerta).where(Alerta.evento_id == evento_id)
                         .order_by(Alerta.secuencia)).all()
    if not filas:
        raise HTTPException(status_code=404, detail="no existe ese evento")

    cuando = datetime.now().astimezone().isoformat(timespec="seconds")
    for f in filas:
        f.revision = veredicto or None
        f.revision_nota = (revision.nota or None) if veredicto else None
        f.revisado_en = cuando if veredicto else None
        session.add(f)
    session.commit()

    ultima = filas[-1]
    guardado = None
    if veredicto:
        raiz = os.path.join(os.path.dirname(evidencia.CARPETA), "revision")
        destino = os.path.join(raiz, "falsas" if veredicto == "falsa" else "correctas",
                               f"{ultima.tipo}_{re.sub(r'[^A-Za-z0-9_-]', '', evento_id)}")
        os.makedirs(destino, exist_ok=True)
        copiados = []
        for pref, url in (("cruda", ultima.cruda_url), ("captura", ultima.captura_url),
                          ("clip", ultima.clip_url)):
            ruta = evidencia.ruta_de(url)
            if ruta:
                # Con prefijo: la cruda y la captura se llaman igual en sus
                # carpetas de origen y acá una pisaba a la otra.
                nombre = f"{pref}{os.path.splitext(ruta)[1]}"
                shutil.copy2(ruta, os.path.join(destino, nombre))
                copiados.append(nombre)
        meta = {
            "evento_id": evento_id, "veredicto": veredicto, "nota": revision.nota,
            "tipo": ultima.tipo, "severidad": ultima.severidad,
            "camara": ultima.camara, "camara_config_id": ultima.camara_config_id,
            "motivo": ultima.motivo, "ts_inicio": ultima.ts_inicio,
            "revisado_en": cuando,
            "ausentes": AUSENTES_SI_FALSA.get(ultima.tipo, []) if veredicto == "falsa" else [],
            "archivos": copiados,
        }
        with open(os.path.join(destino, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        guardado = destino
    return {"evento_id": evento_id, "revision": veredicto or None,
            "mensajes": len(filas), "guardado_en": guardado}


@alerta_router.delete("")
@alerta_router.delete("/", include_in_schema=False)
def borrar_historial(confirmar: bool = Query(False),
                     session: Session = Depends(get_session)):
    """Vacía el historial de alertas. 26/09, pedido desde el panel.

    No borra nada sin dejar copia: antes de tocar una fila se guarda la base
    entera (con la API de backup de SQLite, que saca una foto consistente
    aunque el backend esté recibiendo alertas) y se mueven las capturas y los
    clips a la misma carpeta:

        FASTAPI/respaldos/historial-AAAAMMDD-HHMMSS/
            horus.db          <- la base como estaba
            evidencia/...     <- las imágenes y videos

    Para volver atrás alcanza con copiar ese horus.db sobre el de FASTAPI con
    el backend cerrado. Las cámaras y los contactos NO se tocan.

    Pide `?confirmar=true` a propósito: un DELETE sin cuerpo es muy fácil de
    mandar por error.
    """
    if not confirmar:
        raise HTTPException(status_code=400,
                            detail="Para borrar el historial hace falta ?confirmar=true")
    import os
    import shutil
    import sqlite3

    ruta_db = os.path.abspath(str(engine.url.database or "horus.db"))
    destino = os.path.join(os.path.dirname(ruta_db), "respaldos",
                           "historial-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(destino, exist_ok=True)
    try:
        origen = sqlite3.connect(ruta_db)
        copia = sqlite3.connect(os.path.join(destino, "horus.db"))
        with copia:
            origen.backup(copia)
        copia.close()
        origen.close()
    except Exception as e:                               # noqa: BLE001
        # Sin copia no se borra: es el historial de un sistema de alertas.
        raise HTTPException(status_code=500,
                            detail=f"No pude guardar la copia, no borré nada: {e}")

    alertas = session.exec(select(Alerta)).all()
    viejas = session.exec(select(Camara)).all()
    for fila in (*alertas, *viejas):
        session.delete(fila)
    session.commit()

    movidos = 0
    for sub in evidencia.SUBCARPETA.values():
        carpeta = os.path.join(evidencia.CARPETA, sub)
        if not os.path.isdir(carpeta):
            continue
        for nombre in os.listdir(carpeta):
            try:
                os.makedirs(os.path.join(destino, "evidencia", sub), exist_ok=True)
                shutil.move(os.path.join(carpeta, nombre),
                            os.path.join(destino, "evidencia", sub, nombre))
                movidos += 1
            except OSError:
                pass

    print(f"[alertas] historial vaciado: {len(alertas)} mensajes, "
          f"{movidos} archivos de evidencia · copia en {destino}")
    return {"borradas": len(alertas), "filas_viejas": len(viejas),
            "evidencia_movida": movidos, "respaldo": destino}


@alerta_router.get("/{evento_id}")
def historial_evento(evento_id: str = Path(...),
                     session: Session = Depends(get_session)):
    # "ws" no es un evento: es la ruta del WebSocket, que se declara mas abajo.
    #
    # Si el servidor no sabe hacer el upgrade —falta `websockets`— el pedido
    # llega hasta aca como un GET comun y esta ruta contesta 200 con
    # {"encontrado": false}. El navegador informa "Unexpected response code:
    # 200" y el panel se queda callado. Un 501 explicito convierte esa falla
    # muda en una que se lee.
    if evento_id == "ws":
        raise HTTPException(
            status_code=501,
            detail="Este servidor no tiene soporte de WebSocket instalado "
                   "(pip install websockets). Sin eso el panel no recibe "
                   "alertas en vivo.")

    """Todos los mensajes de un evento, en orden de `secuencia`.

    Acá está lo que NO se emite al panel: los re-avisos cada 20 s, el cierre, y
    el mensaje que manda el VLM después con la verificación.
    """
    filas = session.exec(select(Alerta).where(Alerta.evento_id == evento_id)
                         .order_by(Alerta.secuencia)).all()
    if not filas:
        return {"evento_id": evento_id, "mensajes": [], "encontrado": False}
    ultima = filas[-1]
    ids = {i for i in (_id_de(f) for f in filas) if i is not None}
    nombres = {c.id: c.nombre for c in session.exec(
        select(CamaraConfig).where(CamaraConfig.id.in_(ids))).all()} if ids else {}
    return {
        "evento_id": evento_id,
        "encontrado": True,
        "tipo": ultima.tipo,
        "severidad": ultima.severidad,
        "estado": ultima.estado,
        "camara": ultima.camara,
        "nombre_camara": nombres.get(_id_de(ultima)) or ultima.camara,
        "captura_url": ultima.captura_url,
        "clip_url": ultima.clip_url,
        "modelos": json.loads(ultima.modelos or "[]"),
        "mensajes": [_json(f, nombres) for f in filas],
    }


@alerta_router.websocket("/ws")
async def ws_alertas(websocket: WebSocket, camara_config_id: int = Query(0)):
    """Canal del panel. Se reusa el manager que ya existía para el video."""
    await manager.connect(websocket, camara_config_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:                                    # noqa: BLE001
        # 26/09: solo se atrapaba WebSocketDisconnect. Una conexión que muere
        # de otra forma quedaba anotada, y cada alerta intentaba mandarle.
        pass
    finally:
        manager.disconnect(websocket, camara_config_id)


# --------------------------------------------------------------------------- #
def _json(f: Alerta, nombres: Optional[dict] = None) -> dict:
    d = f.model_dump()
    for k in ("camaras", "tracks", "aportes", "modelos", "evidencia"):
        try:
            d[k] = json.loads(d.get(k) or ("{}" if k == "evidencia" else "[]"))
        except (TypeError, ValueError):
            d[k] = None
    # El nombre que se ve en el panel.
    #
    # La fila guarda `camara`, que es como la nombra Horus (`cam-deposito`), y
    # el `camara_config_id` de la camara registrada. El websocket manda
    # `nombre_camara` resuelto contra esa config, pero el historial no lo
    # resolvia: al recargar, el mismo evento pasaba de "Deposito" a
    # "cam-deposito". Se resuelve aca para que las dos vias digan lo mismo.
    d["nombre_camara"] = (nombres or {}).get(_id_de(f)) or f.camara
    return d
