# -*- coding: utf-8 -*-
"""
topologia.py · HORUS — qué cámara ve qué, y qué recorridos son posibles.

Es la mitad no-visual de la fusión. El diagrama dice "reglas + topología de
cámaras" y esta es la topología: el mapa que convierte detecciones sueltas en
una escena con lugares y tiempos.

Sirve para tres cosas concretas:

1. **Zonas.** Una persona no está "en la cámara 2", está en el depósito. Sin
   zonas no hay intrusión posible, porque intrusión es entrar A ALGÚN LADO.
   El punto que decide es el de los pies, no el centro de la caja.

2. **Horarios.** La misma persona en el mismo lugar es normal a las 3 de la
   tarde y es un evento a las 3 de la mañana. Sin esto, o el sistema alerta
   todo el día o no alerta nunca.

3. **Transiciones.** Si alguien estaba en el portón hace 400 ms no puede
   estar ahora en el depósito que queda a 40 metros. `tracker_global` usa
   esto para podar candidatos ANTES de mirar el parecido de los vectores, que
   baja los falsos matches mucho más que subir el umbral de coseno.

Formato
-------
JSON, editable a mano. Ver `topologia.example.json`. Los polígonos van en
píxeles del frame ORIGINAL de cada cámara (el mismo espacio en el que el motor
devuelve las cajas con `coords_originales=True`), igual que `zonas.json` de
`02_preproceso_roi`.

Sin dependencias fuera de numpy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DIAS = ("lun", "mar", "mie", "jue", "vie", "sab", "dom")


# --------------------------------------------------------------------------- #
# Horarios
# --------------------------------------------------------------------------- #
@dataclass
class Horario:
    """Franja horaria semanal. `desde`/`hasta` en "HH:MM", hora local.

    Una franja que cruza la medianoche se escribe tal cual ("22:00"-"06:00")
    y se interpreta bien: es el caso más común en vigilancia y el que más se
    rompe cuando se implementa a la ligera.
    """
    dias: Tuple[str, ...] = DIAS
    desde: str = "00:00"
    hasta: str = "23:59"
    tz_offset_horas: Optional[float] = None    # None = hora local de la máquina

    def _minutos(self, txt: str) -> int:
        h, m = (txt.split(":") + ["0"])[:2]
        return int(h) * 60 + int(m)

    def activo(self, ts: Optional[float] = None) -> bool:
        ts = time.time() if ts is None else float(ts)
        if self.tz_offset_horas is None:
            t = time.localtime(ts)
            dia_idx, minuto = t.tm_wday, t.tm_hour * 60 + t.tm_min
        else:
            t = time.gmtime(ts + self.tz_offset_horas * 3600.0)
            dia_idx, minuto = t.tm_wday, t.tm_hour * 60 + t.tm_min

        a, b = self._minutos(self.desde), self._minutos(self.hasta)

        if a <= b:                                  # franja dentro del día
            return DIAS[dia_idx] in self.dias and a <= minuto <= b

        # Cruza la medianoche. La parte después de las 00:00 pertenece al día
        # ANTERIOR de la franja: "sáb 22:00-06:00" incluye el domingo a las 2.
        if minuto >= a:
            return DIAS[dia_idx] in self.dias
        dia_previo = DIAS[(dia_idx - 1) % 7]
        return dia_previo in self.dias and minuto <= b

    @staticmethod
    def desde_json(d: Any) -> Optional["Horario"]:
        if not d:
            return None
        return Horario(
            dias=tuple(d.get("dias", DIAS)),
            desde=str(d.get("desde", "00:00")),
            hasta=str(d.get("hasta", "23:59")),
            tz_offset_horas=d.get("tz_offset_horas"))


# --------------------------------------------------------------------------- #
# Zonas
# --------------------------------------------------------------------------- #
@dataclass
class Zona:
    nombre: str
    puntos: np.ndarray                     # (N, 2) en px del frame original
    tipo: str = "transito"                 # "transito"|"restringida"|"acceso"|"privada"
    # Cuándo se PUEDE estar acá. Fuera de esta franja, estar es un evento.
    # None en una zona restringida = nunca se puede (bóveda, patio de noche).
    horario_permitido: Optional[Horario] = None
    merodeo_s: float = 20.0                # cuánto es "quedarse" en esta zona
    prioridad: int = 0                     # ante solape, gana la más alta

    def __post_init__(self) -> None:
        self.puntos = np.asarray(self.puntos, dtype=np.float64).reshape(-1, 2)

    @property
    def restringida(self) -> bool:
        return self.tipo == "restringida"

    def permitida(self, ts: Optional[float] = None) -> bool:
        """¿Estar acá ahora es normal?"""
        if not self.restringida:
            return True
        if self.horario_permitido is None:
            return False                   # restringida siempre
        return self.horario_permitido.activo(ts)

    def contiene(self, punto: Sequence[float]) -> bool:
        return bool(self.contiene_varios(np.asarray([punto], dtype=np.float64))[0])

    def contiene_varios(self, puntos: np.ndarray) -> np.ndarray:
        """Ray casting vectorizado. `puntos` es (N, 2) -> (N,) booleanos."""
        p = np.asarray(puntos, dtype=np.float64).reshape(-1, 2)
        v = self.puntos
        if v.shape[0] < 3 or p.shape[0] == 0:
            return np.zeros(p.shape[0], dtype=bool)

        x, y = p[:, 0][:, None], p[:, 1][:, None]
        x1, y1 = v[:, 0][None, :], v[:, 1][None, :]
        # vértice anterior de cada lado (i-1), que es como se recorre
        # el polígono en el ray casting clásico.
        x2, y2 = np.roll(v[:, 0], 1)[None, :], np.roll(v[:, 1], 1)[None, :]

        cruza = ((y1 > y) != (y2 > y))
        # x del lado a la altura y. El denominador no puede ser 0 donde
        # `cruza` es True, pero se protege igual para no ensuciar el warning.
        den = np.where(np.abs(y2 - y1) < 1e-12, 1e-12, y2 - y1)
        xint = x1 + (y - y1) * (x2 - x1) / den
        return (np.sum(cruza & (x < xint), axis=1) % 2).astype(bool)


# --------------------------------------------------------------------------- #
# Cámaras
# --------------------------------------------------------------------------- #
@dataclass
class Transicion:
    """Ir de una cámara a otra: cuánto tarda como mínimo y como máximo una
    persona caminando. `t_max` no es un límite físico sino de sentido común —
    pasado ese tiempo, que sean la misma persona deja de ser información."""
    destino: str
    t_min_s: float = 1.0
    t_max_s: float = 120.0


@dataclass
class Camara:
    camera_id: str
    nombre: str = ""
    zonas: List[Zona] = field(default_factory=list)
    vecinos: Dict[str, Transicion] = field(default_factory=dict)
    solapa: List[str] = field(default_factory=list)   # ven el mismo espacio
    fps: float = 10.0
    tam_frame: Tuple[int, int] = (0, 0)               # (alto, ancho)

    def zona_de(self, punto: Sequence[float]) -> Optional[Zona]:
        """La zona que contiene ese punto. Ante solape gana la de mayor
        prioridad: un cajero adentro de un hall tiene que ganarle al hall."""
        candidatas = [z for z in self.zonas if z.contiene(punto)]
        if not candidatas:
            return None
        return max(candidatas, key=lambda z: z.prioridad)


# --------------------------------------------------------------------------- #
# Topología
# --------------------------------------------------------------------------- #
class Topologia:
    """El mapa completo. Se le pasa a `TrackerGlobal` y a `MotorFusion`."""

    def __init__(self, camaras: Optional[Dict[str, Camara]] = None,
                 simetrica: bool = True) -> None:
        self.camaras: Dict[str, Camara] = dict(camaras or {})
        if simetrica:
            self._simetrizar()

    def _simetrizar(self) -> None:
        """Si A declara vecino a B con 3-40 s, B lo declara a A igual.

        Casi siempre es lo que se quiere y casi siempre se olvida escribirlo,
        y la falta hace que el matching global funcione en un sentido y no en
        el otro — un bug muy difícil de ver desde afuera.
        """
        for cid, cam in self.camaras.items():
            for vid, tr in list(cam.vecinos.items()):
                otra = self.camaras.get(vid)
                if otra is not None and cid not in otra.vecinos:
                    otra.vecinos[cid] = Transicion(cid, tr.t_min_s, tr.t_max_s)
            for vid in cam.solapa:
                otra = self.camaras.get(vid)
                if otra is not None and cid not in otra.solapa:
                    otra.solapa.append(cid)

    # ------------------------------------------------------------------ #
    def camara(self, camera_id: str) -> Optional[Camara]:
        return self.camaras.get(camera_id)

    def zona_de(self, camera_id: str, punto: Sequence[float]) -> Optional[Zona]:
        cam = self.camaras.get(camera_id)
        return cam.zona_de(punto) if cam else None

    def zona_de_track(self, track: Any) -> Optional[Zona]:
        """La zona donde está parado el track (punto de los pies)."""
        caja = getattr(track, "bbox_xyxy", None)
        if caja is None:
            return None
        x1, _, x2, y2 = (float(v) for v in caja)
        return self.zona_de(getattr(track, "camera_id", ""), ((x1 + x2) / 2.0, y2))

    def solapadas(self, a: str, b: str) -> bool:
        if a == b:
            return True
        cam = self.camaras.get(a)
        return bool(cam and b in cam.solapa)

    def transicion_posible(self, origen: str, destino: str, dt_s: float) -> bool:
        """¿Se puede haber ido de `origen` a `destino` en `dt_s` segundos?

        Sin topología cargada devuelve True: es preferible un sistema que no
        poda a uno que poda mal.
        """
        if origen == destino or not self.camaras:
            return True
        cam = self.camaras.get(origen)
        if cam is None or not cam.vecinos:
            return True                    # cámara sin vecinos declarados
        if destino in cam.solapa:
            return True
        tr = cam.vecinos.get(destino)
        if tr is None:
            # Vecinos declarados pero este destino no está: ¿hay camino
            # indirecto? Se admite si existe alguno, con el tiempo mínimo
            # sumado. Una topología en estrella no debería prohibir todo.
            t_min = self._camino_minimo(origen, destino)
            return t_min is None or dt_s >= t_min
        return tr.t_min_s <= dt_s <= tr.t_max_s

    def _camino_minimo(self, origen: str, destino: str,
                       tope: int = 6) -> Optional[float]:
        """Dijkstra chico sobre t_min. Devuelve None si no hay camino."""
        import heapq
        cola = [(0.0, origen)]
        visto: Dict[str, float] = {}
        while cola:
            d, nodo = heapq.heappop(cola)
            if nodo in visto:
                continue
            visto[nodo] = d
            if nodo == destino:
                return d
            if len(visto) > tope * 8:
                break
            cam = self.camaras.get(nodo)
            if cam is None:
                continue
            for vid, tr in cam.vecinos.items():
                if vid not in visto:
                    heapq.heappush(cola, (d + tr.t_min_s, vid))
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def cargar(ruta: str) -> "Topologia":
        with open(ruta, "r", encoding="utf-8") as fh:
            datos = json.load(fh)
        return Topologia.desde_dict(datos)

    @staticmethod
    def desde_dict(datos: Dict[str, Any]) -> "Topologia":
        camaras: Dict[str, Camara] = {}
        for cid, d in (datos.get("camaras") or {}).items():
            zonas = []
            for z in d.get("zonas", []):
                zonas.append(Zona(
                    nombre=str(z.get("nombre", "zona")),
                    puntos=z.get("puntos", []),
                    tipo=str(z.get("tipo", "transito")),
                    horario_permitido=Horario.desde_json(z.get("horario_permitido")),
                    merodeo_s=float(z.get("merodeo_s", 20.0)),
                    prioridad=int(z.get("prioridad", 0))))
            vecinos = {}
            for v in d.get("vecinos", []):
                if isinstance(v, str):
                    vecinos[v] = Transicion(v)
                else:
                    vecinos[v["destino"]] = Transicion(
                        v["destino"], float(v.get("t_min_s", 1.0)),
                        float(v.get("t_max_s", 120.0)))
            camaras[cid] = Camara(
                camera_id=cid, nombre=str(d.get("nombre", cid)),
                zonas=zonas, vecinos=vecinos,
                solapa=list(d.get("solapa", [])),
                fps=float(d.get("fps", 10.0)),
                tam_frame=tuple(d.get("tam_frame", (0, 0))))
        return Topologia(camaras, simetrica=bool(datos.get("simetrica", True)))

    def guardar(self, ruta: str) -> None:
        with open(ruta, "w", encoding="utf-8") as fh:
            json.dump(self.a_dict(), fh, ensure_ascii=False, indent=2)

    def a_dict(self) -> Dict[str, Any]:
        return {"camaras": {
            cid: {
                "nombre": c.nombre, "fps": c.fps,
                "tam_frame": list(c.tam_frame),
                "solapa": list(c.solapa),
                "vecinos": [{"destino": t.destino, "t_min_s": t.t_min_s,
                             "t_max_s": t.t_max_s} for t in c.vecinos.values()],
                "zonas": [{
                    "nombre": z.nombre, "tipo": z.tipo,
                    "puntos": z.puntos.tolist(), "merodeo_s": z.merodeo_s,
                    "prioridad": z.prioridad,
                    "horario_permitido": (
                        None if z.horario_permitido is None else {
                            "dias": list(z.horario_permitido.dias),
                            "desde": z.horario_permitido.desde,
                            "hasta": z.horario_permitido.hasta})}
                    for z in c.zonas]}
            for cid, c in self.camaras.items()}}

    # ------------------------------------------------------------------ #
    def verificar(self) -> List[str]:
        """Problemas que se pueden detectar sin correr nada. Conviene llamarlo
        al arrancar: una topología mal escrita degrada en silencio."""
        avisos: List[str] = []
        for cid, cam in self.camaras.items():
            for vid in cam.vecinos:
                if vid not in self.camaras:
                    avisos.append(f"{cid}: vecino inexistente {vid!r}")
            for vid in cam.solapa:
                if vid not in self.camaras:
                    avisos.append(f"{cid}: solapa con cámara inexistente {vid!r}")
            if not cam.zonas:
                avisos.append(f"{cid}: sin zonas — no puede haber intrusión "
                              "ni merodeo en esta cámara")
            for z in cam.zonas:
                if z.puntos.shape[0] < 3:
                    avisos.append(f"{cid}/{z.nombre}: polígono de "
                                  f"{z.puntos.shape[0]} puntos")
                if z.restringida and z.horario_permitido is None:
                    avisos.append(
                        f"{cid}/{z.nombre}: restringida sin horario permitido, "
                        "va a alertar las 24 h. Correcto para una bóveda; si "
                        "no lo es, falta el horario.")
        # El aviso que faltaba. Una topología puede estar impecable y no
        # despertar nunca a `intrusion`: alcanza con que ninguna zona sea
        # 'restringida'. `verificar()` no decía nada y el archivo parecía
        # completo, que es la confusión de siempre — "no entró nadie" y
        # "nadie estaba mirando" no pueden verse igual.
        if not any(z.restringida for c in self.camaras.values() for z in c.zonas):
            avisos.append("ninguna zona 'restringida': la regla de intrusión "
                          "queda DORMIDA. Dibujala con  python bin/dibujar_zona.py")

        if len(self.camaras) > 1 and not any(c.vecinos for c in self.camaras.values()):
            avisos.append("ninguna cámara declara vecinos: el tracking global "
                          "no va a poder podar por física")
        return avisos

    def resumen(self) -> str:
        filas = []
        for cid, c in sorted(self.camaras.items()):
            zs = ", ".join(f"{z.nombre}({z.tipo})" for z in c.zonas) or "sin zonas"
            vs = ", ".join(sorted(c.vecinos)) or "sin vecinos"
            filas.append(f"  {cid:<12} {c.nombre:<22} zonas: {zs}\n"
                         f"  {'':<12} {'':<22} vecinos: {vs}")
        return f"Topología · {len(self.camaras)} cámaras\n" + "\n".join(filas)


def ejemplo() -> Topologia:
    """Topología mínima de dos cámaras, para probar sin archivo."""
    return Topologia.desde_dict({
        "camaras": {
            "cam-porton": {
                "nombre": "Portón de entrada",
                "tam_frame": [1080, 1920],
                "vecinos": [{"destino": "cam-deposito", "t_min_s": 4.0,
                             "t_max_s": 90.0}],
                "zonas": [
                    {"nombre": "vereda", "tipo": "transito",
                     "puntos": [[0, 600], [1920, 600], [1920, 1080], [0, 1080]]},
                    {"nombre": "acceso", "tipo": "acceso", "prioridad": 1,
                     "puntos": [[700, 400], [1250, 400], [1250, 900], [700, 900]],
                     "merodeo_s": 15.0},
                ]},
            "cam-deposito": {
                "nombre": "Depósito",
                "tam_frame": [1080, 1920],
                "zonas": [
                    {"nombre": "deposito", "tipo": "restringida", "prioridad": 1,
                     "puntos": [[100, 300], [1800, 300], [1800, 1050], [100, 1050]],
                     "merodeo_s": 10.0,
                     "horario_permitido": {"dias": ["lun", "mar", "mie", "jue", "vie"],
                                           "desde": "07:00", "hasta": "19:00"}},
                ]},
        }})
