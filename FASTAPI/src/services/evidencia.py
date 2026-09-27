"""
evidencia.py · la captura y el clip que acompañan a una alerta.

26/09. Una alerta que dice "incendio en Cocina" y nada más obliga a quien la
recibe a creerle al sistema o a ir a mirar. Con la imagen del momento (y los
segundos de antes) decide en dos segundos si es de verdad.

De dónde sale el diseño
-----------------------
No es nuevo, lo habían dejado armado los compañeros:

  - La fila `Camara` (rama main) ya tenía `snapshot_url` y `clip_url`, con el
    comentario "para agregar dsp en otro sprint" y el formato de la ruta:
    `/snapshots/camera_3_2026-06-18_14-32.jpg`. Acá se llenan esos dos campos,
    con el mismo formato de nombre.
  - Mateo guardaba las capturas de cada cámara en su propia carpeta y había
    dejado anotado lo que faltaba: "que se guarden una cantidad y las viejas
    se borren". Eso es `_rotar`.

Quién hace qué
--------------
El video lo tiene el SERVICIO de modelos (es el único que abre la cámara), así
que es él quien arma la captura y el clip y los manda adentro de la misma
alerta, en `adjuntos`. Acá solo se validan, se guardan en disco y se devuelven
las rutas. Van en el mismo POST a propósito: el mail sale en el momento en que
llega la alerta, y si la imagen viniera después, el mail saldría sin ella.
"""

import base64
import binascii
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

AQUI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARPETA = os.environ.get("HORUS_EVIDENCIA", os.path.join(AQUI, "evidencia"))
PREFIJO_URL = "/evidencia"

# Qué se acepta. La extensión sale del tipo declarado, NUNCA del nombre que
# mande el cliente: el archivo se sirve después por HTTP.
TIPOS = {
    "captura": {"image/jpeg": ".jpg", "image/png": ".png"},
    "cruda": {"image/jpeg": ".jpg", "image/png": ".png"},
    "clip": {"video/mp4": ".mp4", "video/webm": ".webm", "image/gif": ".gif"},
}
MAX_BYTES = {"captura": 5 * 1024 * 1024, "cruda": 5 * 1024 * 1024,
             "clip": 25 * 1024 * 1024}
SUBCARPETA = {"captura": "capturas", "cruda": "crudas", "clip": "clips"}

# "que se guarden una cantidad y las viejas se borren" (Mateo). Por cantidad y
# por tamaño: un disco lleno también es un sistema que deja de avisar.
MAX_ARCHIVOS = {"captura": 2000, "cruda": 2000, "clip": 400}
MAX_TOTAL = {"captura": 1 * 1024 ** 3, "cruda": 1 * 1024 ** 3, "clip": 3 * 1024 ** 3}


def _decodificar(item: Any, clase: str) -> Optional[tuple]:
    """(bytes, extension) o None si el adjunto no sirve. Nunca levanta."""
    if not isinstance(item, dict):
        return None
    tipo = str(item.get("tipo") or "").lower()
    ext = TIPOS[clase].get(tipo)
    if ext is None:
        print(f"[evidencia] {clase}: tipo {tipo!r} no aceptado")
        return None
    b64 = item.get("b64")
    if not isinstance(b64, str) or not b64:
        return None
    # Antes de decodificar: el tamaño del texto ya dice si se pasa.
    if len(b64) * 3 // 4 > MAX_BYTES[clase]:
        print(f"[evidencia] {clase}: más de {MAX_BYTES[clase] // 2**20} MB, la descarto")
        return None
    try:
        datos = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        print(f"[evidencia] {clase}: base64 roto")
        return None
    if not datos:
        return None
    return datos, ext


def _nombre(clase: str, cfg_id: Optional[int], camara: str,
            evento_id: str, secuencia: int, ext: str) -> str:
    """`camera_3_2026-06-18_14-32-05_E000007-1.jpg`, el formato de la fila
    `Camara` más segundos y el evento (dos alertas en el mismo minuto no pueden
    pisarse)."""
    quien = str(cfg_id) if cfg_id is not None else re.sub(r"[^A-Za-z0-9_-]", "", camara)[:24]
    cuando = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ev = re.sub(r"[^A-Za-z0-9_-]", "", str(evento_id))[:40]
    return f"camera_{quien or 'x'}_{cuando}_{ev}-{int(secuencia)}{ext}"


def _rotar(clase: str) -> None:
    carpeta = os.path.join(CARPETA, SUBCARPETA[clase])
    try:
        archivos = [os.path.join(carpeta, n) for n in os.listdir(carpeta)]
        archivos = [(os.path.getmtime(a), os.path.getsize(a), a)
                    for a in archivos if os.path.isfile(a)]
    except OSError:
        return
    archivos.sort()                                  # el más viejo primero
    total = sum(t for _, t, _ in archivos)
    while archivos and (len(archivos) > MAX_ARCHIVOS[clase]
                        or total > MAX_TOTAL[clase]):
        _, tam, ruta = archivos.pop(0)
        try:
            os.remove(ruta)
            total -= tam
        except OSError:
            break


def guardar(adjuntos: Any, *, evento_id: str, secuencia: int,
            cfg_id: Optional[int], camara: str) -> Dict[str, str]:
    """Guarda lo que venga y devuelve {captura_url, captura_ruta, clip_url,
    clip_ruta} para lo que se pudo guardar.

    Nunca levanta: una evidencia rota no puede costar la alerta. Lo peor que
    puede pasar es que la alerta llegue sin imagen, que es como llegaba antes.
    """
    salida: Dict[str, str] = {}
    if not isinstance(adjuntos, dict):
        return salida
    for clase in ("captura", "cruda", "clip"):
        dec = _decodificar(adjuntos.get(clase), clase)
        if dec is None:
            continue
        datos, ext = dec
        try:
            carpeta = os.path.join(CARPETA, SUBCARPETA[clase])
            os.makedirs(carpeta, exist_ok=True)
            nombre = _nombre(clase, cfg_id, camara, evento_id, secuencia, ext)
            ruta = os.path.join(carpeta, nombre)
            tmp = ruta + ".parcial"
            with open(tmp, "wb") as fh:
                fh.write(datos)
            os.replace(tmp, ruta)
            salida[f"{clase}_url"] = f"{PREFIJO_URL}/{SUBCARPETA[clase]}/{nombre}"
            salida[f"{clase}_ruta"] = ruta
            _rotar(clase)
        except OSError as e:
            print(f"[evidencia] no pude guardar la {clase}: {e}")
    return salida


def ruta_de(url: Optional[str]) -> Optional[str]:
    """La ruta en disco de una url `/evidencia/...`, si el archivo existe."""
    if not url or not str(url).startswith(PREFIJO_URL + "/"):
        return None
    resto = str(url)[len(PREFIJO_URL) + 1:]
    ruta = os.path.normpath(os.path.join(CARPETA, resto))
    if not ruta.startswith(os.path.normpath(CARPETA)) or not os.path.isfile(ruta):
        return None
    return ruta


def asegurar_carpeta() -> str:
    for sub in SUBCARPETA.values():
        os.makedirs(os.path.join(CARPETA, sub), exist_ok=True)
    return CARPETA
