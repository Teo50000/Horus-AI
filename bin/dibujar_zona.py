#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dibujar_zona.py - dibujar la zona restringida sobre TU camara.

crear_topologia.py se niega a inventar una zona restringida, y hace bien:
"intrusion" significa "esta persona esta donde no debe, a una hora que no
debe", y eso nadie lo puede adivinar por vos. Una zona inventada da alertas
inventadas. Pero escribir el poligono a mano en el JSON, adivinando pixeles,
es la otra forma de equivocarse: el poligono queda corrido, la persona pasa
por al lado y no entra nunca. Este script es el puente que faltaba: te
muestra el cuadro real de tu camara, hacen clic las esquinas del area
prohibida, elegis el horario, y el escribe el JSON.

Por que importa: mientras topologia.json no tenga NINGUNA zona
'restringida', la regla de intrusion se declara DORMIDA (reglas.py,
necesita_topologia = True). Dormida no es lo mismo que tranquila. El panel
lo dice justamente para que "no entro nadie" y "nadie estaba mirando" no se
parezcan.

Los puntos se guardan en pixeles del cuadro ORIGINAL de la camara, que es el
mismo sistema en el que el pipeline compara las cajas del tracker
(pipeline.py toma tam_frame de frame.shape y reglas.py no reescala nada).
Por eso conviene dibujar sobre la camara de verdad y no sobre una captura de
otra resolucion.

    python bin\dibujar_zona.py
    python bin\dibujar_zona.py --indice 1 --camara cam-2
    python bin\dibujar_zona.py --imagen logs\cuadro.png     (camara ocupada)
    python bin\dibujar_zona.py --puntos "120,300 500,300 500,470 120,470"

Teclas en la ventana:
    clic izquierdo  agregar esquina        z    deshacer la ultima
    f               cuadro nuevo           r    borrar todo
    ENTER           listo                  ESC  salir sin escribir nada
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION = os.path.join(RAIZ, "horus", "06_fusion_decision")
TOPO_DEFECTO = os.path.join(FUSION, "topologia.json")
DIAS = ("lun", "mar", "mie", "jue", "vie", "sab", "dom")
ANCHO_MAX = 1280                      # solo para mostrar; los puntos se re-escalan

try:
    import cv2
except ImportError:                                        # pragma: no cover
    sys.exit("Falta opencv. Instalalo con:  pip install opencv-python")

import numpy as np


# --------------------------------------------------------------------------- #
# Camara
# --------------------------------------------------------------------------- #
def _backends() -> List[Tuple[str, int]]:
    """DirectShow primero en Windows: MSMF abre y despues falla al leer con
    HRESULT -1072875772, y OpenCV imprime un mensaje enganoso sobre el
    backend en vez de decir 'la camara esta ocupada'."""
    if sys.platform.startswith("win"):
        return [("DirectShow", cv2.CAP_DSHOW), ("por defecto", cv2.CAP_ANY)]
    return [("por defecto", cv2.CAP_ANY)]


def abrir_camara(indice: int):
    """Devuelve (cap, nombre_backend) o (None, motivo)."""
    for nombre, api in _backends():
        cap = cv2.VideoCapture(indice, api)
        if not cap.isOpened():
            cap.release()
            continue
        # isOpened() no alcanza: con la camara tomada por otro proceso abre
        # igual y despues devuelve cuadros vacios.
        ok = False
        for _ in range(10):
            ok, _f = cap.read()
            if ok:
                break
            time.sleep(0.05)
        if ok:
            return cap, nombre
        cap.release()
    return None, ("no pude leer un cuadro de la camara %d.\n"
                  "En Windows la camara es de un proceso por vez. Si estan "
                  "abiertos el panel, el backend o el servicio de modelos, "
                  "cerralos y proba de nuevo; o sacale una captura y usa "
                  "--imagen." % indice)


def leer_cuadro(cap) -> Optional[np.ndarray]:
    """Los primeros cuadros de una webcam suelen venir negros o a medio
    exponer. Tiramos unos cuantos antes de quedarnos con uno."""
    ultimo = None
    for _ in range(12):
        ok, f = cap.read()
        if ok and f is not None and f.size:
            ultimo = f
        time.sleep(0.03)
    return ultimo


# --------------------------------------------------------------------------- #
# Dibujar
# --------------------------------------------------------------------------- #
def _cerrar_ventanas() -> None:
    """En el OpenCV 'headless' hasta destroyAllWindows() explota. Y ese es
    justo el caso en el que se usa --puntos/--imagen, que no necesita
    ninguna ventana: no tiene sentido morir al salir de algo que nunca se
    abrio."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def _pintar(base: np.ndarray, pts: Sequence[Tuple[int, int]],
            esc: float) -> np.ndarray:
    img = base.copy()
    if pts:
        p = np.array([[int(x * esc), int(y * esc)] for x, y in pts], np.int32)
        if len(p) >= 3:
            capa = img.copy()
            cv2.fillPoly(capa, [p], (0, 0, 200))
            img = cv2.addWeighted(capa, 0.28, img, 0.72, 0)
            cv2.polylines(img, [p], True, (0, 0, 255), 2)
        elif len(p) == 2:
            cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 255), 2)
        for i, (x, y) in enumerate(p):
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(img, str(i + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    lineas = ["clic = esquina   z = deshacer   r = borrar   f = cuadro nuevo",
              "ENTER = listo (3 esquinas o mas)   ESC = salir sin guardar",
              "esquinas: %d" % len(pts)]
    for i, t in enumerate(lineas):
        y = 22 + i * 22
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def pedir_puntos(frame: np.ndarray, cap=None) -> Optional[List[Tuple[int, int]]]:
    """Ventana interactiva. None = el usuario cancelo."""
    alto, ancho = frame.shape[:2]
    esc = min(1.0, ANCHO_MAX / float(ancho))
    pts: List[Tuple[int, int]] = []
    estado = {"frame": frame}

    def _base():
        f = estado["frame"]
        return f if esc >= 1.0 else cv2.resize(
            f, (int(ancho * esc), int(alto * esc)), interpolation=cv2.INTER_AREA)

    def _click(evento, x, y, _flags, _param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(round(x / esc)), int(round(y / esc))))

    titulo = "HORUS - marca la zona restringida"
    try:
        cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(titulo, _click)
    except cv2.error as e:                                 # pragma: no cover
        _cerrar_ventanas()
        raise SystemExit(
            "Tu OpenCV no tiene ventanas (esta instalado el paquete "
            "'headless').\n  %s\n\nDos salidas:\n"
            "  pip uninstall opencv-python-headless\n"
            "  pip install opencv-python\n"
            "...o pasale las esquinas a mano con --puntos, leyendo los\n"
            "pixeles en el Paint sobre una captura." % e)

    while True:
        cv2.imshow(titulo, _pintar(_base(), pts, esc))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:                                   # ESC
            _cerrar_ventanas()
            return None
        if k in (13, 10):                             # ENTER
            if len(pts) >= 3:
                _cerrar_ventanas()
                return pts
            print("  Faltan esquinas: un area necesita 3 como minimo.")
        elif k == ord("z") and pts:
            pts.pop()
        elif k == ord("r"):
            pts.clear()
        elif k == ord("f") and cap is not None:
            nuevo = leer_cuadro(cap)
            if nuevo is not None:
                estado["frame"] = nuevo
        # La ventana cerrada con la X tambien es un "no quiero"
        try:
            if cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
                _cerrar_ventanas()
                return None
        except cv2.error:
            return None


def parsear_puntos(txt: str) -> List[Tuple[int, int]]:
    pts = []
    for par in txt.replace(";", " ").split():
        x, _, y = par.partition(",")
        if not y:
            raise SystemExit("--puntos va como \"x,y x,y x,y\". Fallo en %r" % par)
        pts.append((int(float(x)), int(float(y))))
    if len(pts) < 3:
        raise SystemExit("--puntos necesita 3 esquinas como minimo.")
    return pts


# --------------------------------------------------------------------------- #
# Horario
# --------------------------------------------------------------------------- #
def _hhmm(txt: str) -> Optional[str]:
    txt = txt.strip().replace(".", ":")
    if ":" not in txt and txt.isdigit() and len(txt) in (3, 4):
        txt = txt[:-2] + ":" + txt[-2:]
    try:
        h, m = txt.split(":")
        h, m = int(h), int(m)
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return "%02d:%02d" % (h, m)


def pedir_horario() -> Optional[dict]:
    print()
    print("  Cuando SI se puede estar en esa zona?")
    print("    1  Nunca. Cualquier persona ahi es alerta, a cualquier hora.")
    print("       (boveda, deposito cerrado, sala de maquinas)")
    print("    2  En un horario. Fuera de esa franja, estar ahi es alerta.")
    print("       (una oficina: de dia se trabaja, de noche no hay nadie)")
    op = input("  > ").strip() or "2"
    if op == "1":
        return None

    crudo = input("  Dias [lun,mar,mie,jue,vie] > ").strip().lower()
    if crudo:
        dias = tuple(d for d in (x.strip()[:3] for x in crudo.replace(" ", ",").split(","))
                     if d in DIAS)
        if not dias:
            print("  No reconoci ningun dia, uso lunes a viernes.")
            dias = DIAS[:5]
    else:
        dias = DIAS[:5]

    desde = _hhmm(input("  Desde [08:00] > ") or "08:00")
    hasta = _hhmm(input("  Hasta [18:00] > ") or "18:00")
    if desde is None or hasta is None:
        raise SystemExit("La hora va en HH:MM (por ejemplo 08:00). Proba de nuevo.")
    if desde > hasta:
        print("  (la franja cruza la medianoche: %s a %s. Se entiende bien.)"
              % (desde, hasta))
    return {"dias": list(dias), "desde": desde, "hasta": hasta}


# --------------------------------------------------------------------------- #
# Escribir
# --------------------------------------------------------------------------- #
_OJO = ("Editado por dibujar_zona.py. Los puntos estan en pixeles del cuadro "
        "original de la camara. Si cambias la resolucion o moves la camara, "
        "hay que volver a dibujarlos: el poligono no se reescala solo.")


def escribir(ruta: str, camara: str, nombre: str, tipo: str,
             puntos: Sequence[Tuple[int, int]], horario: Optional[dict],
             tam: Optional[Tuple[int, int]], merodeo_s: float) -> str:
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    else:
        doc = {"simetrica": True, "camaras": {}}
    doc.setdefault("camaras", {})

    cam = doc["camaras"].get(camara)
    if cam is None:
        # Trampa silenciosa: si le errás al id, la zona queda escrita bajo una
        # cámara que el servicio no usa nunca. Se ve igual de bien en el JSON.
        if doc["camaras"]:
            print("  aviso: %r no estaba en la topología (están: %s). La creo,"
                  % (camara, ", ".join(sorted(doc["camaras"]))))
            print("  pero si le erraste al id, esta zona no la va a mirar nadie.")
            print("  El servicio numera cam-1, cam-2... en el orden del panel.")
        cam = {"nombre": camara, "fps": 10.0, "solapa": [], "vecinos": [],
               "zonas": []}
        if tam:
            cam["tam_frame"] = [int(tam[0]), int(tam[1])]
            cam["zonas"].append({
                "nombre": "todo-el-cuadro", "tipo": "transito", "prioridad": 0,
                "merodeo_s": 25.0,
                "puntos": [[0, 0], [int(tam[1]), 0],
                           [int(tam[1]), int(tam[0])], [0, int(tam[0])]]})
        doc["camaras"][camara] = cam
    elif tam:
        # El tam_frame que dejo crear_topologia.py es el que vos le tipeaste;
        # este es el que la camara da de verdad.
        cam["tam_frame"] = [int(tam[0]), int(tam[1])]

    zona = {"nombre": nombre, "tipo": tipo, "prioridad": 1,
            "merodeo_s": float(merodeo_s),
            "puntos": [[int(x), int(y)] for x, y in puntos]}
    if tipo == "restringida":
        zona["horario_permitido"] = horario

    zonas = cam.setdefault("zonas", [])
    for i, z in enumerate(zonas):
        if z.get("nombre") == nombre:
            zonas[i] = zona
            break
    else:
        zonas.append(zona)

    doc["_ojo"] = _OJO
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, ruta)
    return ruta


def _rutas_horus() -> None:
    raiz = os.path.join(RAIZ, "horus")
    for p in (FUSION, os.path.join(raiz, "05_tracking"),
              os.path.join(raiz, "05_tracking", "local")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _punto_adentro(zona) -> Optional[Tuple[float, float]]:
    """Un punto que caiga DENTRO del polígono. El centroide no sirve solo:
    en una zona con forma de L queda afuera."""
    pts = np.asarray(zona.puntos, dtype=float)
    c = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    if zona.contiene(c):
        return c
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    for i in range(1, 20):
        for j in range(1, 20):
            p = (x0 + (x1 - x0) * i / 20.0, y0 + (y1 - y0) * j / 20.0)
            if zona.contiene(p):
                return p
    return None


def _cuando_no_se_puede(zona, desde_ts: float) -> Optional[float]:
    """Primer momento de la semana que viene en el que estar ahí es alerta.
    Si no hay ninguno, el horario cubre las 168 horas y la zona no va a
    avisar nunca: eso hay que decirlo, no dejarlo pasar."""
    paso = 15 * 60
    for k in range(7 * 24 * 4):
        ts = desde_ts + k * paso
        if not zona.permitida(ts):
            return ts
    return None


def verificar(ruta: str, camara: str, nombre: str) -> bool:
    """Releer con el mismo código que usa el servicio y, sobre todo, DISPARAR
    la regla de verdad.

    Que el JSON cargue no prueba nada: una zona puede estar bien escrita y no
    alertar nunca (horario que cubre la semana entera, polígono fuera del
    cuadro, cámara con otro id que el que usa el servicio). Acá paramos una
    persona de mentira adentro de la zona, a una hora prohibida, y miramos si
    `intrusion` la ve. Si no la ve, la zona no sirve y conviene enterarse
    ahora y no dentro de tres meses.
    """
    _rutas_horus()
    try:
        from topologia import Topologia                       # type: ignore
        from reglas import Contexto, ReglaIntrusion           # type: ignore
        from contratos import CABEZA_OBJETOS, ObservacionCamara  # type: ignore
        from tracker_local import TrackLocal                  # type: ignore
    except ImportError as e:
        print("  (no pude importar el motor para probarlo: %s)" % e)
        return True

    try:
        topo = Topologia.cargar(ruta)
    except Exception as e:
        raise SystemExit("El JSON quedó escrito pero el servicio NO lo puede "
                         "cargar: %s" % e)

    print()
    if hasattr(topo, "resumen"):
        print(topo.resumen())
    for aviso in topo.verificar():
        print("  aviso: %s" % aviso)

    cam = topo.camaras.get(camara)
    zona = next((z for z in (cam.zonas if cam else []) if z.nombre == nombre), None)
    if zona is None or not zona.restringida:
        return True

    print()
    print("Probando la zona con una persona de mentira:")
    punto = _punto_adentro(zona)
    if punto is None:
        print("  NO PASO: no encontré ningún punto adentro del polígono. "
              "Quedó degenerado (todas las esquinas en línea?).")
        return False

    ts = _cuando_no_se_puede(zona, time.time())
    if ts is None:
        print("  NO PASO: el horario permitido cubre la semana entera, así que")
        print("  estar ahí nunca es alerta. La zona existe pero no sirve de nada.")
        print("  Volvé a correr esto y poné una franja más corta.")
        return False

    cx, cy = punto
    t = TrackLocal(
        track_id=1, clase="persona", class_id=0,
        bbox_xyxy=(cx - 40.0, cy - 160.0, cx + 40.0, cy),
        score=0.8, camera_id=camara, frame_idx=0, ts=ts, estado="confirmado",
        hits=50, edad=50, frames_sin_ver=0, ts_nacimiento=ts - 5.0,
        visto_s=5.0, radio_permanencia=5.0, quieto=True, quieto_s=5.0,
        global_id=None)
    obs = ObservacionCamara(
        camera_id=camara, frame_idx=0, ts=ts, tracks=[t], seg=None,
        acciones=[], novelty=0.0,
        tam_frame=tuple(cam.tam_frame), cabezas=frozenset({CABEZA_OBJETOS}))

    regla = ReglaIntrusion()
    hallazgos = regla.evaluar(obs, Contexto(topo=topo))
    cuando = time.strftime("%a %H:%M", time.localtime(ts))
    if hallazgos:
        h = hallazgos[0]
        print("  PASO: parada en (%d, %d) un %s, salta intrusion "
              "(confianza %.2f, %s)" % (cx, cy, cuando, h.confianza, h.motivo))
        return True
    else:
        falta = regla.faltantes(obs, Contexto(topo=topo))
        print("  NO PASO: un %s no salta nada. Falta: %s"
              % (cuando, ", ".join(falta) if falta else "nada evidente"))
        print("  La zona está escrita pero intrusion sigue sin ver nada.")
        return False


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dibujar la zona restringida sobre el cuadro real de la camara.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", default="cam-1",
                    help="id de la camara en la topologia (por defecto cam-1). "
                         "El servicio nombra las camaras cam-1, cam-2...")
    ap.add_argument("--indice", type=int, default=0, help="indice USB (0, 1, 2...)")
    ap.add_argument("--imagen", help="dibujar sobre una captura en vez de la "
                                     "camara en vivo (util si esta ocupada)")
    ap.add_argument("--puntos", help='esquinas a mano: "x,y x,y x,y"')
    ap.add_argument("--nombre", default="restringida", help="nombre de la zona")
    ap.add_argument("--tipo", default="restringida",
                    choices=["restringida", "transito", "acceso", "privada"])
    ap.add_argument("--merodeo-s", type=float, default=20.0,
                    help="segundos quieto dentro de la zona para que sea merodeo")
    ap.add_argument("--topologia", default=TOPO_DEFECTO)
    args = ap.parse_args()

    print("HORUS - dibujar la zona restringida")
    print("=" * 62)
    print("Mientras no haya una zona 'restringida', la regla de intrusion se")
    print("declara DORMIDA y el panel lo muestra asi. No es que no pase nada:")
    print("es que nadie esta mirando ese pedazo del cuadro.")
    print()

    cap = None
    tam: Optional[Tuple[int, int]] = None
    frame = None

    if args.imagen:
        frame = cv2.imread(args.imagen)
        if frame is None:
            raise SystemExit("No pude abrir la imagen %r" % args.imagen)
        tam = (frame.shape[0], frame.shape[1])
        print("Dibujando sobre %s (%dx%d)." % (args.imagen, tam[1], tam[0]))
        print("OJO: los pixeles tienen que ser los de la camara. Si esta")
        print("captura salio con otra resolucion, la zona va a quedar corrida.")
    else:
        cap, info = abrir_camara(args.indice)
        if cap is None:
            raise SystemExit(info)
        frame = leer_cuadro(cap)
        if frame is None:
            cap.release()
            raise SystemExit("La camara abrio pero no dio ningun cuadro.")
        tam = (frame.shape[0], frame.shape[1])
        print("Camara %d abierta por %s - %dx%d"
              % (args.indice, info, tam[1], tam[0]))

    try:
        if args.puntos:
            puntos = parsear_puntos(args.puntos)
            print("Esquinas por linea de comandos: %d" % len(puntos))
        else:
            puntos = pedir_puntos(frame, cap)
            if puntos is None:
                print("\nSaliste sin guardar. topologia.json quedo como estaba.")
                return
    finally:
        if cap is not None:
            cap.release()
        _cerrar_ventanas()

    fuera = [(x, y) for x, y in puntos
             if not (0 <= x <= tam[1] and 0 <= y <= tam[0])]
    if fuera:
        print("  aviso: %d esquina(s) caen fuera del cuadro %dx%d. "
              "La parte de afuera no existe para el tracker."
              % (len(fuera), tam[1], tam[0]))

    horario = pedir_horario() if args.tipo == "restringida" else None

    ruta = escribir(args.topologia, args.camara, args.nombre, args.tipo,
                    puntos, horario, tam, args.merodeo_s)
    print()
    print("Escrito: %s" % ruta)
    print("  camara %s - zona %r (%s), %d esquinas"
          % (args.camara, args.nombre, args.tipo, len(puntos)))
    if args.tipo == "restringida":
        if horario is None:
            print("  sin horario permitido: alerta las 24 h")
        else:
            print("  se puede estar %s de %s a %s. Fuera de eso, alerta."
                  % (",".join(horario["dias"]), horario["desde"], horario["hasta"]))

    quedo_viva = verificar(ruta, args.camara, args.nombre)
    print()
    if quedo_viva:
        print("Listo. Arranca con HORUS.bat: detecta topologia.json solo y le")
        print("pasa --topologia al servicio. En el arranque tiene que dejar de")
        print("figurar 'intrusion' entre las reglas dormidas.")
    else:
        # Decir "listo" abajo de un "NO PASO" seria la version de este script
        # del cartel tranquilo arriba de un sistema roto.
        print("NO quedo lista. El archivo esta escrito, pero con esa zona")
        print("intrusion no va a avisar nunca. Volve a correr esto y arregla")
        print("lo que dice arriba antes de confiar en el panel.")
        sys.exit(2)


if __name__ == "__main__":
    main()
