# -*- coding: utf-8 -*-
"""
tracker_global.py · HORUS — tracking entre cámaras por ReID (vector 512-d).

Qué resuelve
------------
El tracking local sabe que en la cámara 2 hay una persona con track 17. No
sabe que es la MISMA que hace ocho segundos estaba en la cámara 1. Sin eso,
la fusión no puede decir "alguien entró por el portón, cruzó el patio y ahora
está en el depósito" — que es el evento, no las tres detecciones sueltas.

Acá se le pone un `global_id` a cada persona: un número que sobrevive el
cambio de cámara.

Cómo
----
La cabeza `04_cabezas/reid_cuerpo` produce un vector 512-d normalizado por
persona. Dos recortes de la misma persona dan vectores parecidos (coseno
alto) aunque cambie la cámara. El matching es entonces una sola multiplicación
de matrices contra la galería en RAM.

Cinco decisiones que hacen que funcione, y ninguna es obvia:

1. **El filtro de calidad importa más que el umbral de matching.**
   Un vector sacado de una persona de 30 px de alto, o cortada por el borde
   del cuadro, no describe a nadie: se parece un poco a todo el mundo. Si
   entra a la galería, envenena esa identidad para siempre y empieza a
   robarle matches a las demás. Es preferible no darle `global_id` a una
   persona que dárselo mal. Ver `_calidad()`.

2. **Un banco de K vectores por identidad, no un promedio.** La misma
   persona vista de frente y de espaldas da vectores casi ortogonales; el
   promedio de los dos no se parece a ninguna de las dos vistas. Se guardan
   varios y se matchea contra el mejor. Y solo se agrega un vector si aporta
   algo distinto (`diversidad_max`), para que el banco no sean K copias del
   mismo frame.

3. **La topología poda antes que el coseno.** Si alguien estaba en la cámara
   del portón hace 400 ms, no puede estar ahora en el depósito que queda a 40
   metros. Ese candidato se descarta por física. Baja los falsos matches
   mucho más que subir el umbral, y sin costo en los verdaderos.

4. **Todas las personas del frame se resuelven juntas, con asignación
   óptima.** No es solo velocidad: resolviendo de a una, dos personas
   parecidas del mismo frame pueden llevarse el mismo `global_id`. Con el
   húngaro sobre la matriz entera eso es imposible por construcción.

5. **Un track ya ligado no vuelve a consultar la galería.** Se revisa cada
   `reconsulta_s`, y esa revisión sirve para engordar el banco. De ahí sale
   el presupuesto de <10 ms del diagrama: en régimen, casi ningún frame hace
   una búsqueda completa.

Dependencias: solo numpy. La topología entra por duck typing (cualquier
objeto con `transicion_posible()` y `solapadas()`), así que esta capa no
importa nada de `06_fusion_decision` — la flecha va al revés.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_PADRE = os.path.dirname(_AQUI)
for _p in (_AQUI, _PADRE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comun import a_array, asignar, iou_matriz  # noqa: E402


DIM = 512


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
@dataclass
class ConfigGlobal:
    dim: int = DIM

    # --- matching ---------------------------------------------------------
    umbral_coseno: float = 0.55        # de acá para abajo, es otra persona
    umbral_misma_camara: float = 0.45  # más laxo: reengancha un track cortado
    banco_k: int = 8                   # vectores por identidad
    diversidad_max: float = 0.92       # no guardar un vector casi idéntico

    # --- galería ----------------------------------------------------------
    ttl_s: float = 300.0               # identidad sin verse -> se olvida
    max_identidades: int = 512         # techo de RAM: 512*8*512*4 B ≈ 8 MB

    # --- filtro de calidad -------------------------------------------------
    alto_min_px: int = 64              # más chica no describe a nadie
    aspecto_min: float = 0.15          # w/h de una persona de pie
    aspecto_max: float = 0.90
    margen_borde_px: int = 4           # cortada por el borde = truncada
    score_min: float = 0.45
    hits_min: int = 3                  # el track ya tiene que estar asentado

    # --- física -----------------------------------------------------------
    ventana_simultanea_s: float = 1.0  # no puede estar en dos cámaras a la vez
    reconsulta_s: float = 2.0          # cada cuánto se revisa un track ligado
    olvido_binding_s: float = 10.0     # binding de un track local que ya no está

    verboso: bool = False


# --------------------------------------------------------------------------- #
# Identidad
# --------------------------------------------------------------------------- #
@dataclass
class IdentidadGlobal:
    """El banco es un array (n, dim), no una lista de vectores: la galería se
    reconstruye concatenando N arrays y no N*K, que es la diferencia entre
    3 ms y 0,2 ms con la galería llena."""

    global_id: int
    banco: np.ndarray = field(
        default_factory=lambda: np.zeros((0, DIM), dtype=np.float32))
    calidades: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32))
    ts_primera: float = 0.0
    ts_ultima: float = 0.0
    ultima_camara: str = ""
    apariciones: int = 0
    camaras: Dict[str, float] = field(default_factory=dict)  # cam -> última ts
    etiqueta: Optional[str] = None      # la pone 09_encoder_facial si confirma

    def __post_init__(self) -> None:
        self.banco = np.asarray(self.banco, dtype=np.float32).reshape(-1, DIM)
        self.calidades = np.asarray(self.calidades, dtype=np.float32).reshape(-1)

    @property
    def n_vectores(self) -> int:
        return int(self.banco.shape[0])

    @property
    def n_camaras(self) -> int:
        return len(self.camaras)

    def resumen(self) -> str:
        cams = ", ".join(sorted(self.camaras))
        et = f" [{self.etiqueta}]" if self.etiqueta else ""
        return (f"ID {self.global_id}{et}  {self.apariciones} vistas  "
                f"{self.n_vectores} vectores  cámaras: {cams}")


@dataclass
class Match:
    """Por qué esta persona recibió este `global_id`. Sirve para depurar y
    para que la revisión humana (12_revision_feedback) pueda validar."""
    global_id: int
    similitud: float
    nuevo: bool
    misma_camara: bool
    calidad: float = 0.0


# --------------------------------------------------------------------------- #
# Tracker global
# --------------------------------------------------------------------------- #
class TrackerGlobal:
    """Galería de identidades en RAM + matching por coseno.

    Uso:
        tg = TrackerGlobal(topologia=topo)          # topología opcional
        tracks = tk_local.actualizar_desde_resultado(res)
        tg.actualizar(tracks, embeddings, tam_frame=(1080, 1920))
        # cada track de persona queda con .global_id
    """

    def __init__(self, cfg: Optional[ConfigGlobal] = None,
                 topologia: Any = None) -> None:
        self.cfg = cfg or ConfigGlobal()
        self.topo = topologia
        self._ids: Dict[int, IdentidadGlobal] = {}
        self._contador = count(1)

        # Matriz compacta de la galería, agrupada por identidad. Se reconstruye
        # como mucho una vez por frame; el matching es una gemm contra ella.
        self._V: np.ndarray = np.zeros((0, self.cfg.dim), dtype=np.float32)
        self._offsets: np.ndarray = np.zeros((1,), dtype=np.int64)
        self._orden: List[int] = []
        self._sucia = True

        # (camera_id, track_id) -> (global_id, ts_última_consulta, ts_visto)
        self._binding: Dict[Tuple[str, int], Tuple[int, float, float]] = {}
        self.ultima_latencia_ms = 0.0
        self.rechazos_calidad = 0
        self.consultas_completas = 0

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def actualizar(self, tracks: Sequence[Any],
                   embeddings: Optional[Sequence[Any]] = None,
                   tam_frame: Optional[Tuple[int, int]] = None,
                   ts: Optional[float] = None) -> List[Optional[int]]:
        """Le pone `global_id` a los tracks de persona. Los modifica in-place
        y además devuelve la lista de ids alineada con `tracks`.

        `embeddings` es lo que devuelve `ReIDBodyHead.embed_people()` para
        este frame (objetos con `.vector` y `.bbox_xyxy`). Si los tracks ya
        traen `.embedding` cargado, se usa ese y `embeddings` puede ir en None.
        """
        t0 = time.perf_counter()
        ts = time.time() if ts is None else float(ts)
        personas = [t for t in tracks if getattr(t, "clase", "") == "persona"]

        if embeddings:
            self.ligar_embeddings(personas, embeddings)
        self._caducar(ts)

        # Fase 1 — resolver lo que ya está ligado (no toca la galería) y
        # juntar lo que sí necesita consultarla.
        nuevas: List[Tuple[Any, np.ndarray, float, str]] = []
        refrescos: List[Tuple[int, np.ndarray, float, str]] = []

        for t in personas:
            cam = str(getattr(t, "camera_id", "cam-0"))
            clave = (cam, int(getattr(t, "track_id", -1)))
            lig = self._binding.get(clave)

            if lig is not None:
                gid, ts_consulta, _ = lig
                t.global_id = gid
                self._tocar(gid, cam, ts)
                if ts - ts_consulta < self.cfg.reconsulta_s:
                    self._binding[clave] = (gid, ts_consulta, ts)
                    continue
                self._binding[clave] = (gid, ts, ts)
                q, cal = self._consulta(t, tam_frame)
                if q is not None:
                    refrescos.append((gid, q, cal, cam))
                continue

            q, cal = self._consulta(t, tam_frame)
            if q is None:
                t.global_id = None          # sin vector usable: no se inventa
                continue
            nuevas.append((t, q, cal, cam))

        # Fase 2 — una sola búsqueda para todas las personas nuevas del frame.
        if nuevas:
            self._buscar_lote(nuevas, ts)

        # Fase 3 — mutaciones. Van al final, después de todas las consultas,
        # para que la matriz compacta se reconstruya una vez y no N veces.
        for gid, q, cal, cam in refrescos:
            self._agregar_vector(gid, q, cal, cam, ts)

        salida: List[Optional[int]] = []
        it = iter(personas)
        for t in tracks:
            if getattr(t, "clase", "") == "persona":
                salida.append(getattr(next(it), "global_id", None))
            else:
                salida.append(None)

        self.ultima_latencia_ms = (time.perf_counter() - t0) * 1000.0
        return salida

    def ligar_embeddings(self, tracks: Sequence[Any],
                         embeddings: Sequence[Any]) -> int:
        """Pega cada vector de ReID al track cuya caja le corresponde.

        La cabeza de ReID recibe cajas y devuelve vectores en el mismo orden,
        pero esas cajas salen del detector, no del tracker (el tracker corrige
        con Kalman). Por eso hace falta emparejar por IoU y no asumir el
        orden: si se asumiera, un frame donde el detector devuelve las cajas
        en otro orden le pone el vector de una persona a la otra, y eso
        contamina la galería sin dejar rastro.
        """
        if not tracks or not embeddings:
            return 0
        cajas_t = a_array([t.bbox_xyxy for t in tracks])
        cajas_e = a_array([_caja_emb(e) for e in embeddings])
        costo = 1.0 - iou_matriz(cajas_t, cajas_e).astype(np.float64)
        pares, _, _ = asignar(costo, costo_max=1.0 - 0.50)
        for i, j in pares:
            tracks[i].embedding = _vector(embeddings[j], self.cfg.dim)
        return len(pares)

    # ------------------------------------------------------------------ #
    def _consulta(self, track: Any, tam_frame: Optional[Tuple[int, int]]
                  ) -> Tuple[Optional[np.ndarray], float]:
        emb = getattr(track, "embedding", None)
        if emb is None:
            return None, 0.0
        cal = self._calidad(track, tam_frame)
        if cal <= 0.0:
            self.rechazos_calidad += 1
            return None, 0.0
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-6:
            return None, 0.0
        return (v / n).astype(np.float32), cal

    def _calidad(self, track: Any,
                 tam_frame: Optional[Tuple[int, int]]) -> float:
        """0 = no usar este vector. >0 = calidad relativa, para decidir qué
        vectores se quedan en el banco.

        Es el filtro más importante del módulo. Un vector malo no da un match
        malo y ya: se queda en la galería y sigue dando matches malos.
        """
        c = self.cfg
        x1, y1, x2, y2 = (float(v) for v in track.bbox_xyxy)
        w, h = x2 - x1, y2 - y1
        if h < c.alto_min_px or w <= 1:
            return 0.0
        if not (c.aspecto_min <= w / h <= c.aspecto_max):
            return 0.0
        if float(getattr(track, "score", 1.0)) < c.score_min:
            return 0.0
        if int(getattr(track, "hits", 99)) < c.hits_min:
            return 0.0
        if getattr(track, "estado", "confirmado") != "confirmado":
            return 0.0
        if int(getattr(track, "frames_sin_ver", 0)) > 0:
            return 0.0                       # caja predicha, no observada

        if tam_frame is not None:
            alto_f, ancho_f = int(tam_frame[0]), int(tam_frame[1])
            m = c.margen_borde_px
            if x1 <= m or y1 <= m or x2 >= ancho_f - m or y2 >= alto_f - m:
                return 0.0                   # persona cortada por el borde

        # Más grande y más confiable = mejor. Satura en 4x el mínimo para que
        # un primer plano no aplaste a todo lo demás dentro del banco.
        return min(h / c.alto_min_px, 4.0) * float(getattr(track, "score", 1.0))

    # ------------------------------------------------------------------ #
    def _buscar_lote(self, nuevas: List[Tuple[Any, np.ndarray, float, str]],
                     ts: float) -> None:
        """Resuelve TODAS las personas nuevas del frame de una sola vez.

        Una compactación, una gemm, una máscara por cámara y una asignación
        óptima. Que sea óptima además garantiza que dos personas del mismo
        frame no puedan quedarse con el mismo `global_id`, que resolviendo de
        a una es un error posible y difícil de ver.
        """
        c = self.cfg
        self._compactar()
        self.consultas_completas += 1

        Q = np.stack([q for _, q, _, _ in nuevas])          # (n, dim)
        asignados: Dict[int, int] = {}                      # fila -> global_id

        if self._V.shape[0] and self._orden:
            sim = Q @ self._V.T                             # (n, M)
            # Mejor vector de cada identidad, sin bucles de Python.
            por_id = np.maximum.reduceat(sim, self._offsets[:-1], axis=1)

            mascaras: Dict[str, np.ndarray] = {}
            umbrales = np.empty_like(por_id)
            for i, (_, _, _, cam) in enumerate(nuevas):
                m = mascaras.get(cam)
                if m is None:
                    m = self._mascara_fisica(cam, ts)
                    mascaras[cam] = m
                misma = np.asarray(
                    [self._ids[g].ultima_camara == cam for g in self._orden])
                umbrales[i] = np.where(misma, c.umbral_misma_camara,
                                       c.umbral_coseno)
                por_id[i] = np.where(m, por_id[i], -2.0)

            # Lo que no llega al umbral queda prohibido; así el húngaro no
            # puede cerrar la cuenta con un par malo.
            por_id = np.where(por_id >= umbrales, por_id, -2.0)
            pares, _, _ = asignar(1.0 - por_id.astype(np.float64),
                                  costo_max=1.0 - float(np.min(umbrales)))
            for fila, col in pares:
                asignados[fila] = self._orden[col]

        for i, (track, q, cal, cam) in enumerate(nuevas):
            gid = asignados.get(i)
            if gid is None:
                gid = self._nueva_identidad(q, cal, cam, ts)
            else:
                self._agregar_vector(gid, q, cal, cam, ts)
            track.global_id = gid
            self._binding[(cam, int(getattr(track, "track_id", -1)))] = (gid, ts, ts)

    def _mascara_fisica(self, cam: str, ts: float) -> np.ndarray:
        """Qué identidades de la galería PUEDEN estar en esta cámara ahora.

        Dos podas, las dos anteriores al coseno:
          - transición imposible según la topología (distancia / tiempo);
          - identidad vista en otra cámara hace un instante: no puede estar
            en dos lugares al mismo tiempo, salvo cámaras que se solapen.
        """
        c = self.cfg
        ok = np.ones(len(self._orden), dtype=bool)
        for i, gid in enumerate(self._orden):
            ident = self._ids[gid]
            ultima = ident.ultima_camara
            if not ultima or ultima == cam:
                continue
            dt = ts - ident.camaras.get(ultima, ident.ts_ultima)
            if dt < c.ventana_simultanea_s and not self._solapan(ultima, cam):
                ok[i] = False
                continue
            if self.topo is not None:
                try:
                    if not self.topo.transicion_posible(ultima, cam, dt):
                        ok[i] = False
                except Exception:            # topología incompleta: no podar
                    pass
        return ok

    def _solapan(self, a: str, b: str) -> bool:
        if self.topo is None:
            return False
        try:
            return bool(self.topo.solapadas(a, b))
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def _nueva_identidad(self, q: np.ndarray, cal: float,
                         cam: str, ts: float) -> int:
        if len(self._ids) >= self.cfg.max_identidades:
            self._desalojar()
        gid = next(self._contador)
        self._ids[gid] = IdentidadGlobal(
            global_id=gid, banco=q.reshape(1, -1).copy(),
            calidades=np.asarray([cal], dtype=np.float32),
            ts_primera=ts, ts_ultima=ts, ultima_camara=cam,
            apariciones=1, camaras={cam: ts})
        self._sucia = True
        return gid

    def _agregar_vector(self, gid: int, q: np.ndarray, cal: float,
                        cam: str, ts: float) -> None:
        ident = self._ids.get(gid)
        if ident is None:
            return
        self._tocar(gid, cam, ts, contar=True)

        if ident.n_vectores:
            sims = ident.banco @ q
            j = int(np.argmax(sims))
            if float(sims[j]) >= self.cfg.diversidad_max:
                # Ya hay una vista prácticamente igual. Guardar las dos
                # llenaría el banco de K copias del mismo instante y dejaría a
                # la identidad sin la vista de espaldas. Se queda la mejor y
                # el cupo libre sigue disponible para una vista distinta.
                if cal > float(ident.calidades[j]):
                    ident.banco[j] = q
                    ident.calidades[j] = cal
                    self._sucia = True
                return

        ident.banco = np.vstack([ident.banco, q.reshape(1, -1)])
        ident.calidades = np.append(ident.calidades, np.float32(cal))
        if ident.n_vectores > self.cfg.banco_k:
            peor = int(np.argmin(ident.calidades))
            ident.banco = np.delete(ident.banco, peor, axis=0)
            ident.calidades = np.delete(ident.calidades, peor)
        self._sucia = True

    def _tocar(self, gid: int, cam: str, ts: float,
               contar: bool = False) -> None:
        """Marca la identidad como vista ahora en `cam`.

        `contar` solo en un match real: si sumara una aparición por frame,
        `apariciones` mediría FPS en vez de cuántas veces se vio a la persona.
        """
        ident = self._ids.get(gid)
        if ident is None:
            return
        ident.ts_ultima = ts
        ident.ultima_camara = cam
        ident.camaras[cam] = ts
        if contar:
            ident.apariciones += 1

    # ------------------------------------------------------------------ #
    def _caducar(self, ts: float) -> None:
        c = self.cfg
        muertas = [g for g, i in self._ids.items() if ts - i.ts_ultima > c.ttl_s]
        for g in muertas:
            del self._ids[g]
        if muertas:
            self._sucia = True

        viejos = [k for k, (_, _, visto) in self._binding.items()
                  if ts - visto > c.olvido_binding_s]
        for k in viejos:
            del self._binding[k]

    def _desalojar(self) -> None:
        """Techo alcanzado: se va la identidad vista hace más tiempo."""
        if not self._ids:
            return
        peor = min(self._ids.values(), key=lambda i: i.ts_ultima)
        del self._ids[peor.global_id]
        self._sucia = True

    def _compactar(self) -> None:
        """Arma la matriz (M, dim) con los vectores agrupados por identidad.

        Agrupados y contiguos a propósito: así el 'mejor vector de cada
        identidad' sale con un `np.maximum.reduceat` sobre el resultado de la
        gemm, sin bucles de Python.
        """
        if not self._sucia:
            return
        bancos, offsets, orden, total = [], [], [], 0
        for gid, ident in self._ids.items():
            if ident.n_vectores == 0:
                continue
            offsets.append(total)
            orden.append(gid)
            bancos.append(ident.banco)
            total += ident.n_vectores
        self._V = (np.concatenate(bancos, axis=0) if bancos
                   else np.zeros((0, self.cfg.dim), dtype=np.float32))
        offsets.append(total)
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._orden = orden
        self._sucia = False

    # ------------------------------------------------------------------ #
    # Camino asíncrono (10_galeria_global / 09_encoder_facial)
    # ------------------------------------------------------------------ #
    def reingestar(self, global_id: int, vectores: Iterable[Sequence[float]],
                   etiqueta: Optional[str] = None) -> bool:
        """Inyecta vectores desde la galería global. No bloquea la alerta: se
        llama desde el hilo asíncrono, y lo peor que puede pasar es que el
        próximo frame matchee un poco mejor.
        """
        ident = self._ids.get(global_id)
        if ident is None:
            return False
        ts = time.time()
        for v in vectores:
            q = np.asarray(v, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(q))
            if n < 1e-6:
                continue
            self._agregar_vector(global_id, q / n, 1.0, ident.ultima_camara, ts)
        if etiqueta:
            ident.etiqueta = etiqueta
        return True

    def fusionar(self, gid_a: int, gid_b: int) -> Optional[int]:
        """Une dos identidades en una. La usa `12_revision_feedback` cuando un
        humano dice "estas dos son la misma persona". Sobrevive la más
        antigua."""
        a, b = self._ids.get(gid_a), self._ids.get(gid_b)
        if a is None or b is None or gid_a == gid_b:
            return None
        if b.ts_primera < a.ts_primera:
            a, b = b, a
        for v, cal in zip(b.banco, b.calidades):
            self._agregar_vector(a.global_id, v, float(cal),
                                 b.ultima_camara, b.ts_ultima)
        for cam, t in b.camaras.items():
            a.camaras[cam] = max(a.camaras.get(cam, 0.0), t)
        a.apariciones += b.apariciones
        a.etiqueta = a.etiqueta or b.etiqueta
        del self._ids[b.global_id]
        for k, (g, tc, tv) in list(self._binding.items()):
            if g == b.global_id:
                self._binding[k] = (a.global_id, tc, tv)
        self._sucia = True
        return a.global_id

    def etiquetar(self, global_id: int, etiqueta: str) -> bool:
        """El encoder facial confirmó quién es. Solo cambia el nombre que se
        muestra; el matching sigue siendo por cuerpo."""
        ident = self._ids.get(global_id)
        if ident is None:
            return False
        ident.etiqueta = etiqueta
        return True

    # ------------------------------------------------------------------ #
    def identidad(self, global_id: int) -> Optional[IdentidadGlobal]:
        return self._ids.get(global_id)

    def activas(self, ts: Optional[float] = None,
                ventana_s: float = 30.0) -> List[IdentidadGlobal]:
        ts = time.time() if ts is None else ts
        return sorted((i for i in self._ids.values()
                       if ts - i.ts_ultima <= ventana_s),
                      key=lambda i: i.ts_ultima, reverse=True)

    def exportar(self) -> Dict[str, Any]:
        return {"identidades": [
            {"global_id": i.global_id,
             "banco": i.banco.tolist(),
             "calidades": i.calidades.tolist(),
             "ts_primera": i.ts_primera, "ts_ultima": i.ts_ultima,
             "ultima_camara": i.ultima_camara, "apariciones": i.apariciones,
             "camaras": dict(i.camaras), "etiqueta": i.etiqueta}
            for i in self._ids.values()]}

    def importar(self, datos: Dict[str, Any]) -> int:
        n = 0
        for d in datos.get("identidades", []):
            gid = int(d["global_id"])
            self._ids[gid] = IdentidadGlobal(
                global_id=gid,
                banco=np.asarray(d.get("banco", []), dtype=np.float32
                                 ).reshape(-1, self.cfg.dim),
                calidades=np.asarray(d.get("calidades", []), dtype=np.float32),
                ts_primera=float(d.get("ts_primera", 0.0)),
                ts_ultima=float(d.get("ts_ultima", 0.0)),
                ultima_camara=str(d.get("ultima_camara", "")),
                apariciones=int(d.get("apariciones", 0)),
                camaras=dict(d.get("camaras", {})),
                etiqueta=d.get("etiqueta"))
            n += 1
        if self._ids:
            self._contador = count(max(self._ids) + 1)
        self._sucia = True
        return n

    def reset(self) -> None:
        self._ids.clear()
        self._binding.clear()
        self._sucia = True

    def resumen(self) -> str:
        vec = sum(i.n_vectores for i in self._ids.values())
        mb = vec * self.cfg.dim * 4 / (1024 ** 2)
        return (f"[global] {len(self._ids)} identidades, {vec} vectores "
                f"({mb:.1f} MB)  último matching {self.ultima_latencia_ms:.2f} ms  "
                f"búsquedas completas: {self.consultas_completas}  "
                f"rechazos por calidad: {self.rechazos_calidad}")


# --------------------------------------------------------------------------- #
def _caja_emb(e: Any) -> Sequence[float]:
    if hasattr(e, "bbox_xyxy"):
        return e.bbox_xyxy
    if isinstance(e, dict):
        return e["bbox_xyxy"]
    return e[0]


def _vector(e: Any, dim: int) -> np.ndarray:
    if hasattr(e, "vector"):
        v = e.vector
    elif isinstance(e, dict):
        v = e["vector"]
    else:
        v = e[1]
    return np.asarray(v, dtype=np.float32).reshape(-1)[:dim]
