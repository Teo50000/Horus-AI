# -*- coding: utf-8 -*-
"""
alerta.py · HORUS — el emisor que le lleva los eventos al backend.

⚠ **Reescrito el 2026-09-15 desde `contrato-alertas-backend.md`.** El original
se perdió (quedó solo `__pycache__/alerta.cpython-310.pyc`) y no estaba en
ninguna rama. Esto implementa el contrato documentado, que es la especificación
que el backend y el front ya consumían; si aparece el original, `validar_payload`
es lo que hay que comparar primero.

Qué hace
--------
Toma el `Evento` que sale de `06_fusion_decision`, lo convierte al mensaje del
contrato y lo entrega. `emitir()` **no bloquea**: encola y un hilo se ocupa de
la red, porque el pipeline corre a 8-12 FPS y un backend lento no puede
frenar la inferencia.

Las tres cosas que hay que entender
-----------------------------------
**1 · `id` + `secuencia` es la clave, no el `id` solo.** Un evento manda
varios mensajes mientras dura. `secuencia` se asigna **al encolar**, no al
entregar, justamente porque la cola reordena.

**2 · La cola prioriza por severidad.** Un crítico se adelanta a los avisos
que ya estaban esperando, así que los mensajes llegan desordenados a
propósito. El consumidor ordena por `tiempo.inicio`.

**3 · Lo que no se pudo entregar va a disco.** Tres reintentos con espera
creciente y después al `spool`, para reenviarlo cuando el backend vuelva. Una
alerta que se pierde porque el backend estaba reiniciando es exactamente la
desgracia que el sistema existe para evitar.

Uso
---
    from alerta import ConfigAlerta, EmisorAlertas

    emisor = EmisorAlertas(ConfigAlerta(
        url="http://127.0.0.1:8000/alertas",
        token="...",                       # Authorization: Bearer
        severidad_min=1,                   # 1=aviso 2=alerta 3=solo crítico
        spool="alertas_pendientes.jsonl",
        sitio="sucursal-centro", nodo="caja-01"))

    for ev in motor.procesar(observaciones):
        emisor.emitir(ev)                  # no bloquea
    emisor.cerrar()                        # drena lo pendiente

Sin `url` imprime por consola, que sirve para probar el pipeline sin backend.

Verificación: `python probar_alerta.py`
"""

from __future__ import annotations

import argparse
import datetime as _dt
import heapq
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_AQUI = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.normpath(os.path.join(_AQUI, "..", "06_fusion_decision"))
if _FUSION not in sys.path:
    sys.path.insert(0, _FUSION)

VERSION_ESQUEMA = 1

SEVERIDADES = ("info", "aviso", "alerta", "critico")
ESTADOS = ("abierto", "sostenido", "cerrado")

# Modelos que pueden aparecer en `aportes`. Es la lista del contrato.
MODELOS = ("objetos", "segmentacion", "pose", "accion", "reid")


# --------------------------------------------------------------------------- #
# Mensaje
# --------------------------------------------------------------------------- #
def _iso(ts: float) -> str:
    """ISO 8601 **con offset de zona**. Sin offset, el backend interpreta la
    hora en la suya y un evento de las 03:00 aparece a las 06:00."""
    return _dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


_SIN_TILDE = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


def etiqueta_ascii(sev: Any) -> str:
    """La severidad como la espera el backend: minúsculas y **sin tildes**.

    `Severidad.etiqueta` devuelve 'crítico' con tilde porque está pensada para
    mostrarse; el contrato dice `critico`. La diferencia no rompe nada visible
    —el JSON viaja igual— pero el consumidor compara strings, así que un
    `severidad == "critico"` del lado del backend daría False para siempre en
    el único caso que no puede fallar. La normalización va acá, en la frontera,
    y no en `contratos.py`: ahí la etiqueta es para los ojos.
    """
    txt = getattr(sev, "etiqueta", None) or str(sev)
    return txt.translate(_SIN_TILDE).lower()


def _num_severidad(etiqueta: str) -> int:
    try:
        return SEVERIDADES.index(etiqueta)
    except ValueError:
        return 0


def _aporte(modelo: str, aporta: str, score: float,
            **detalle: Any) -> Dict[str, Any]:
    d = {k: v for k, v in detalle.items() if v is not None}
    return {"modelo": modelo, "aporta": aporta, "score": round(float(score), 3),
            "detalle": d or None}


def aportes_de(ev: Any) -> List[Dict[str, Any]]:
    """Qué puso cada modelo en este evento.

    Es lo que separa "un incendio que dos modelos confirman" de "un incendio
    que vio uno solo": un consumidor razonable muestra `len(modelos)` en la
    tarjeta. Se reconstruye desde `evidencia`, que es lo que cada regla dejó
    escrito — por eso el mapeo vive acá y no en las reglas: `evidencia` es
    libre y cambia, este contrato no.
    """
    e = dict(getattr(ev, "evidencia", {}) or {})
    tipo = getattr(ev, "tipo", "")
    out: List[Dict[str, Any]] = []

    # Correlaciones: el aporte real está en las piezas, una capa más adentro.
    for pieza in ("arma", "intrusion", "pelea", "caida"):
        sub = e.get(pieza)
        if isinstance(sub, dict):
            falso = type("E", (), {"tipo": pieza, "evidencia": sub,
                                   "confianza": getattr(ev, "confianza", 0.0)})()
            out.extend(aportes_de(falso))
    if out:
        return _unicos(out)

    if tipo == "incendio":
        if e.get("seg_area_fuego"):
            out.append(_aporte("segmentacion", "fuego", 0.99,
                               area=e.get("seg_area_fuego"),
                               crecimiento=e.get("crecimiento")))
        if e.get("seg_area_humo"):
            out.append(_aporte("segmentacion", "humo", 0.956,
                               area=e.get("seg_area_humo")))
        for score in (e.get("objetos_llama") or []):
            out.append(_aporte("objetos", "llama", score,
                               persistencia_s=e.get("persistencia_s")))
        for score in (e.get("objetos_humo") or []):
            out.append(_aporte("objetos", "humo", score,
                               persistencia_s=e.get("persistencia_s")))
    elif tipo == "arma":
        out.append(_aporte("objetos", e.get("clase", "arma"), e.get("score", 0.0),
                           persistencia_s=e.get("persistencia_s"),
                           nota=e.get("nota")))
    elif tipo in ("intrusion", "merodeo"):
        out.append(_aporte("objetos", "persona", e.get("score", 0.0),
                           permanencia_s=e.get("permanencia_s"),
                           radio_px=e.get("radio_px"),
                           horario=e.get("horario")))
    elif tipo == "paquete_abandonado":
        out.append(_aporte("objetos", "paquete", e.get("score", 0.0),
                           quieto_s=e.get("quieto_s"),
                           sin_dueno_s=e.get("sin_dueno_s")))
    elif tipo == "caida":
        out.append(_aporte("pose", "caida", e.get("score", 0.0),
                           tumbado_s=e.get("tumbado_s"),
                           quieto_s=e.get("quieto_s")))
    elif e.get("accion"):
        out.append(_aporte("accion", e["accion"], e.get("score", 0.0),
                           persistencia_s=e.get("persistencia_s"),
                           participantes=e.get("participantes")))

    if getattr(ev, "global_id", None) is not None:
        out.append(_aporte("reid", "identidad", 1.0, global_id=ev.global_id))
    return _unicos(out)


def _unicos(aportes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    vistos, out = set(), []
    for a in aportes:
        k = (a["modelo"], a["aporta"], a["score"])
        if k not in vistos:
            vistos.add(k)
            out.append(a)
    return out


def armar_payload(ev: Any, secuencia: int, *, sitio: Optional[str] = None,
                  nodo: Optional[str] = None,
                  version_horus: Optional[str] = None) -> Dict[str, Any]:
    """`Evento` -> el mensaje del contrato."""
    sev = etiqueta_ascii(ev.severidad)
    aportes = aportes_de(ev)
    camaras = list(getattr(ev, "camaras", None) or [ev.camera_id])
    necesita = bool(getattr(ev, "necesita_vlm", False))
    return {
        "version": VERSION_ESQUEMA,
        "id": ev.evento_id,
        "secuencia": int(secuencia),
        "tipo": ev.tipo,
        "severidad": sev,
        "severidad_num": _num_severidad(sev),
        "estado": getattr(ev, "estado", "abierto"),
        "confianza": round(float(ev.confianza), 3),
        "motivo": ev.motivo,
        "sitio": {
            "camara": ev.camera_id,
            "camaras": camaras,
            "zona": getattr(ev, "zona", None),
            "instalacion": sitio,
            "nodo": nodo,
        },
        "tiempo": {
            "inicio": _iso(ev.ts_inicio),
            "ultimo": _iso(ev.ts_ultimo),
            "emitido": _iso(time.time()),
            "duracion_s": round(max(0.0, ev.ts_ultimo - ev.ts_inicio), 2),
            "epoch_inicio": round(float(ev.ts_inicio), 3),
        },
        "sujeto": {
            "global_id": getattr(ev, "global_id", None),
            "tracks": list(getattr(ev, "track_ids", []) or []),
        },
        "aportes": aportes,
        "modelos": sorted({a["modelo"] for a in aportes}),
        "verificacion": {
            "necesita_vlm": necesita,
            "motivo": getattr(ev, "vlm_motivo", "") or None,
            "estado": "pendiente" if necesita else "no_requiere",
        },
        "confirmaciones": int(getattr(ev, "confirmaciones", 0)),
        "evidencia": dict(getattr(ev, "evidencia", {}) or {}),
        "origen": {"sistema": "horus", "version": version_horus},
    }


def validar_payload(d: Any) -> List[str]:
    """El contrato escrito como código. Devuelve la lista de problemas.

    Conviene copiar esta función del lado del backend: una discrepancia se ve
    al instante en vez de a las tres semanas.
    """
    p: List[str] = []
    if not isinstance(d, dict):
        return ["el mensaje no es un objeto JSON"]

    def falta(k: str, tipo: type, dentro: Optional[dict] = None,
              donde: str = "") -> Any:
        fuente = d if dentro is None else dentro
        if k not in fuente:
            p.append(f"falta {donde}{k}")
            return None
        if not isinstance(fuente[k], tipo):
            p.append(f"{donde}{k} debería ser {tipo.__name__}")
            return None
        return fuente[k]

    if d.get("version") != VERSION_ESQUEMA:
        p.append(f"version {d.get('version')!r} (esperada {VERSION_ESQUEMA})")
    falta("id", str)
    sec = falta("secuencia", int)
    if isinstance(sec, int) and sec < 1:
        p.append("secuencia tiene que arrancar en 1")
    falta("tipo", str)
    sev = d.get("severidad")
    if sev not in SEVERIDADES:
        p.append(f"severidad {sev!r} fuera de {SEVERIDADES}")
    elif d.get("severidad_num") != SEVERIDADES.index(sev):
        p.append("severidad_num no coincide con severidad")
    if d.get("estado") not in ESTADOS:
        p.append(f"estado {d.get('estado')!r} fuera de {ESTADOS}")
    conf = d.get("confianza")
    if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        p.append("confianza fuera de 0..1")

    sitio = falta("sitio", dict) or {}
    falta("camara", str, sitio, "sitio.")
    if not isinstance(sitio.get("camaras"), list):
        p.append("sitio.camaras debería ser lista")

    tiempo = falta("tiempo", dict) or {}
    for k in ("inicio", "ultimo", "emitido"):
        v = tiempo.get(k)
        if not isinstance(v, str):
            p.append(f"falta tiempo.{k}")
        elif not _tiene_offset(v):
            p.append(f"tiempo.{k} sin offset de zona ({v!r})")
    if not isinstance(tiempo.get("epoch_inicio"), (int, float)):
        p.append("falta tiempo.epoch_inicio")

    sujeto = falta("sujeto", dict) or {}
    if not isinstance(sujeto.get("tracks"), list):
        p.append("sujeto.tracks debería ser lista")

    aportes = falta("aportes", list)
    for i, a in enumerate(aportes or []):
        if not isinstance(a, dict):
            p.append(f"aportes[{i}] no es objeto")
            continue
        if a.get("modelo") not in MODELOS:
            p.append(f"aportes[{i}].modelo {a.get('modelo')!r} fuera de {MODELOS}")
        if not isinstance(a.get("aporta"), str):
            p.append(f"aportes[{i}].aporta debería ser str")
        s = a.get("score")
        if not isinstance(s, (int, float)) or not 0.0 <= float(s) <= 1.0:
            p.append(f"aportes[{i}].score fuera de 0..1")

    ver = falta("verificacion", dict) or {}
    if not isinstance(ver.get("necesita_vlm"), bool):
        p.append("verificacion.necesita_vlm debería ser bool")
    if ver.get("estado") not in ("pendiente", "no_requiere", "confirmado", "descartado"):
        p.append(f"verificacion.estado {ver.get('estado')!r} desconocido")

    # Regla del dominio, no del esquema: un arma sin verificar no puede
    # disparar nada irreversible, así que nunca sale sola como crítica.
    if d.get("tipo") == "arma":
        if d.get("severidad") == "critico":
            p.append("un arma sola nunca puede ser crítica")
        if not ver.get("necesita_vlm"):
            p.append("un arma siempre tiene que pedir verificación del VLM")
    return p


def _tiene_offset(s: str) -> bool:
    cola = s[10:]
    return s.endswith("Z") or "+" in cola or cola.count("-") > 0


# --------------------------------------------------------------------------- #
# Transportes
# --------------------------------------------------------------------------- #
class TransporteConsola:
    """Sin backend: imprime. Sirve para probar el pipeline entero sin red."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stdout
        self.enviados: List[dict] = []

    def enviar(self, payload: dict) -> None:
        self.enviados.append(payload)
        t = payload["tipo"]
        sev = payload["severidad"].upper()
        modelos = ",".join(payload["modelos"]) or "-"
        print(f"[{sev:>7}] {payload['id']}.{payload['secuencia']} {t} · "
              f"{payload['sitio']['camara']} · {payload['motivo']} "
              f"· modelos: {modelos}", file=self.stream)


class TransporteJSONL:
    """Todo a un archivo, una línea por mensaje."""

    def __init__(self, ruta: str) -> None:
        self.ruta = ruta

    def enviar(self, payload: dict) -> None:
        with open(self.ruta, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class TransporteHTTP:
    """POST con `urllib`, sin dependencias. Cualquier 2xx cuenta como entregado."""

    def __init__(self, url: str, token: Optional[str] = None,
                 timeout_s: float = 5.0, usar_proxy: bool = False) -> None:
        self.url, self.token, self.timeout_s = url, token, timeout_s
        # Sin proxy por defecto. El backend vive en la misma caja o en la LAN,
        # y un `HTTP_PROXY` en el entorno —de una VPN, de un antivirus, del
        # equipo de sistemas— mandaría las alertas a un proxy que no puede
        # resolver 127.0.0.1. Falla silenciosa y difícil de ver desde acá.
        self._opener = (urllib.request.build_opener()
                        if usar_proxy else
                        urllib.request.build_opener(urllib.request.ProxyHandler({})))

    def enviar(self, payload: dict) -> None:
        cuerpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=cuerpo, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Horus-Evento", f"{payload['id']}.{payload['secuencia']}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with self._opener.open(req, timeout=self.timeout_s) as r:
            if not 200 <= r.status < 300:                # pragma: no cover
                raise urllib.error.HTTPError(self.url, r.status, "no 2xx", r.headers, None)


# --------------------------------------------------------------------------- #
# Emisor
# --------------------------------------------------------------------------- #
@dataclass
class ConfigAlerta:
    url: Optional[str] = None
    token: Optional[str] = None
    severidad_min: int = 1                 # 1=aviso 2=alerta 3=solo crítico
    spool: Optional[str] = None
    sitio: Optional[str] = None
    nodo: Optional[str] = None
    version_horus: Optional[str] = None
    reintentos: int = 3
    espera_base_s: float = 0.5
    timeout_s: float = 5.0
    usar_proxy: bool = False
    cola_max: int = 2048


class EmisorAlertas:
    """Cola por severidad + hilo de entrega + spool."""

    def __init__(self, cfg: Optional[ConfigAlerta] = None,
                 transporte: Any = None) -> None:
        self.cfg = cfg or ConfigAlerta()
        if transporte is not None:
            self.transporte = transporte
        elif self.cfg.url:
            self.transporte = TransporteHTTP(self.cfg.url, self.cfg.token,
                                             self.cfg.timeout_s,
                                             self.cfg.usar_proxy)
        else:
            self.transporte = TransporteConsola()

        self._cola: List[Tuple[int, int, dict]] = []
        self._en_vuelo = 0
        self._orden = 0
        self._secuencias: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._hay = threading.Condition(self._lock)
        self._parar = False
        self.stats = {"emitidos": 0, "entregados": 0, "spool": 0,
                      "descartados": 0, "reintentos": 0}
        self._hilo = threading.Thread(target=self._bucle, name="horus-alertas",
                                      daemon=True)
        self._hilo.start()

    # ------------------------------------------------------------------ #
    def emitir(self, ev: Any) -> Optional[dict]:
        """Encola el evento. No bloquea y no toca la red."""
        sev = etiqueta_ascii(ev.severidad)
        if _num_severidad(sev) < self.cfg.severidad_min:
            return None
        with self._lock:
            sec = self._secuencias.get(ev.evento_id, 0) + 1
            self._secuencias[ev.evento_id] = sec
        payload = armar_payload(ev, sec, sitio=self.cfg.sitio, nodo=self.cfg.nodo,
                                version_horus=self.cfg.version_horus)
        self.encolar(payload)
        return payload

    def encolar(self, payload: dict) -> None:
        with self._hay:
            if len(self._cola) >= self.cfg.cola_max:
                # Se tira lo MENOS grave, nunca lo más nuevo: en un pico, el
                # mensaje que sobra es un aviso repetido, no el crítico que
                # acaba de entrar.
                peor = max(range(len(self._cola)), key=lambda i: self._cola[i][0])
                self._cola.pop(peor)
                heapq.heapify(self._cola)
                self.stats["descartados"] += 1
            self._orden += 1
            heapq.heappush(self._cola,
                           (-payload["severidad_num"], self._orden, payload))
            self.stats["emitidos"] += 1
            self._hay.notify()

    # ------------------------------------------------------------------ #
    def _bucle(self) -> None:
        while True:
            with self._hay:
                while not self._cola and not self._parar:
                    self._hay.wait(0.2)
                if self._parar and not self._cola:
                    return
                _, _, payload = heapq.heappop(self._cola)
                self._en_vuelo += 1
            try:
                self._entregar(payload)
            finally:
                with self._hay:
                    self._en_vuelo -= 1
                    self._hay.notify_all()

    def _entregar(self, payload: dict) -> None:
        espera = self.cfg.espera_base_s
        for intento in range(1, self.cfg.reintentos + 1):
            try:
                self.transporte.enviar(payload)
                self.stats["entregados"] += 1
                return
            except Exception:
                self.stats["reintentos"] += 1
                if intento == self.cfg.reintentos:
                    break
                time.sleep(espera)
                espera *= 2
        self._al_spool(payload)

    def _al_spool(self, payload: dict) -> None:
        if not self.cfg.spool:
            self.stats["descartados"] += 1
            return
        try:
            with open(self.cfg.spool, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.stats["spool"] += 1
        except OSError:                                  # pragma: no cover
            self.stats["descartados"] += 1

    # ------------------------------------------------------------------ #
    def reenviar_spool(self, borrar: bool = True) -> int:
        """Reencola lo que quedó en disco. Devuelve cuántos mensajes salieron.

        El endpoint tiene que ser **idempotente** por `(id, secuencia)`: un
        mensaje puede haber llegado y haberse perdido solo la respuesta.
        """
        ruta = self.cfg.spool
        if not ruta or not os.path.exists(ruta):
            return 0
        with open(ruta, encoding="utf-8") as fh:
            lineas = [l for l in fh.read().splitlines() if l.strip()]
        n = 0
        for l in lineas:
            try:
                self.encolar(json.loads(l))
                n += 1
            except json.JSONDecodeError:                 # pragma: no cover
                continue
        if borrar:
            os.remove(ruta)
        self.stats["spool"] = max(0, self.stats["spool"] - n)
        return n

    def pendientes(self) -> int:
        """Lo que falta entregar: la cola **más lo que está en vuelo**.

        Contar solo la cola pierde el último mensaje al cerrar: el hilo ya lo
        sacó de la cola pero todavía no lo entregó, `drenar()` ve cero y el
        proceso termina. Un emisor de alertas que pierde la última alerta al
        apagarse pierde justo la que motivó el apagado.
        """
        with self._lock:
            return len(self._cola) + self._en_vuelo

    def drenar(self, timeout_s: float = 5.0) -> bool:
        fin = time.time() + timeout_s
        with self._hay:
            while (self._cola or self._en_vuelo) and time.time() < fin:
                self._hay.wait(min(0.05, max(0.0, fin - time.time())))
            return not self._cola and not self._en_vuelo

    def cerrar(self, timeout_s: float = 5.0) -> None:
        self.drenar(timeout_s)
        with self._hay:
            self._parar = True
            self._hay.notify_all()
        self._hilo.join(timeout=timeout_s)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Emisor de alertas de Horus")
    ap.add_argument("--url", help="endpoint del backend")
    ap.add_argument("--token")
    ap.add_argument("--spool", default=None)
    ap.add_argument("--sitio"), ap.add_argument("--nodo")
    ap.add_argument("--severidad-min", type=int, default=1)
    args = ap.parse_args()
    cfg = ConfigAlerta(url=args.url, token=args.token, spool=args.spool,
                       sitio=args.sitio, nodo=args.nodo,
                       severidad_min=args.severidad_min)
    em = EmisorAlertas(cfg)
    n = em.reenviar_spool()
    print(f"spool reenviado: {n} mensaje(s)")
    em.cerrar()
    print(json.dumps(em.stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
