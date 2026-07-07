from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2


_GRIS_NEUTRO: Tuple[int, int, int] = (114, 114, 114)


@dataclass
class ROIZone:
    puntos: List[Tuple[int, int]]
    modo: str = "exclude"

    def __post_init__(self) -> None:
        if self.modo not in ("exclude", "include"):
            raise ValueError(f"modo inválido: {self.modo!r} (usar 'exclude' o 'include')")
        if len(self.puntos) < 3:
            raise ValueError("Un polígono necesita al menos 3 puntos.")


@dataclass
class CameraConfig:
    camera_id: str
    zonas: List[ROIZone] = field(default_factory=list)
    clahe: bool = False
    denoise: bool = False

    fill: Optional[Tuple[int, int, int]] = None


def cargar_config_zonas(ruta_json: str | Path) -> Dict[str, CameraConfig]:
    ruta = Path(ruta_json)
    if not ruta.exists():


        return {}

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    camaras: Dict[str, CameraConfig] = {}
    for cam_id, conf in datos.items():


        if cam_id.startswith("_") or not isinstance(conf, dict):
            continue
        zonas = [
            ROIZone(
                puntos=[tuple(p) for p in z["puntos"]],
                modo=z.get("modo", "exclude"),
            )
            for z in conf.get("zonas", [])
        ]
        fill = conf.get("fill")
        camaras[cam_id] = CameraConfig(
            camera_id=cam_id,
            zonas=zonas,
            clahe=bool(conf.get("clahe", False)),
            denoise=bool(conf.get("denoise", False)),
            fill=tuple(fill) if fill else None,
        )
    return camaras


def construir_roi_mask(alto: int, ancho: int, zonas: Sequence[ROIZone]) -> np.ndarray:
    includes = [z for z in zonas if z.modo == "include"]
    excludes = [z for z in zonas if z.modo == "exclude"]

    if includes:
        mask = np.zeros((alto, ancho), dtype=np.uint8)
        for z in includes:
            cv2.fillPoly(mask, [np.array(z.puntos, dtype=np.int32)], color=1)
    else:
        mask = np.ones((alto, ancho), dtype=np.uint8)

    for z in excludes:
        cv2.fillPoly(mask, [np.array(z.puntos, dtype=np.int32)], color=0)

    return mask


def aplicar_roi(frame: np.ndarray, mask: np.ndarray,
                fill: Optional[Tuple[int, int, int]]) -> np.ndarray:
    color = fill if fill is not None else _GRIS_NEUTRO
    out = frame.copy()
    out[mask == 0] = color
    return out


def realce_clahe(frame_bgr: np.ndarray) -> np.ndarray:

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise_suave(frame: np.ndarray) -> np.ndarray:

    return cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0.0)


class PreprocesadorROI:

    def __init__(self, camaras: Optional[Dict[str, CameraConfig]] = None) -> None:
        self.camaras: Dict[str, CameraConfig] = camaras or {}


        self._mask_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    def _get_mask(self, camera_id: str, alto: int, ancho: int) -> Optional[np.ndarray]:

        cam = self.camaras.get(camera_id)
        if cam is None or not cam.zonas:
            return None
        clave = (camera_id, alto, ancho)
        if clave not in self._mask_cache:
            self._mask_cache[clave] = construir_roi_mask(alto, ancho, cam.zonas)
        return self._mask_cache[clave]

    def procesar(self, frame: np.ndarray, camera_id: str) -> np.ndarray:

        if frame is None or frame.ndim != 3:
            raise ValueError(f"[{camera_id}] frame inválido (esperaba HxWx3).")

        cam = self.camaras.get(camera_id)
        img = frame


        if cam is not None:
            if cam.denoise:
                img = denoise_suave(img)
            if cam.clahe:
                img = realce_clahe(img)


        mask = self._get_mask(camera_id, img.shape[0], img.shape[1])
        if mask is not None:
            fill = cam.fill if cam is not None else None
            img = aplicar_roi(img, mask, fill)

        return img

    def procesar_con_mask(self, frame: np.ndarray, camera_id: str
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        salida = self.procesar(frame, camera_id)
        mask = self._get_mask(camera_id, frame.shape[0], frame.shape[1])
        return salida, mask


if __name__ == "__main__":

    H, W = 480, 640
    frame = np.full((H, W, 3), 40, dtype=np.uint8)
    frame[80:220, 420:600] = (255, 200, 200)


    camaras = {
        "camara_0": CameraConfig(
            camera_id="camara_0",
            zonas=[ROIZone(puntos=[(420, 80), (600, 80), (600, 220), (420, 220)],
                           modo="exclude")],
            clahe=True,
        )
    }

    pre = PreprocesadorROI(camaras)
    salida, mask = pre.procesar_con_mask(frame, "camara_0")

    tv_tapado = bool((salida[80:220, 420:600] == _GRIS_NEUTRO).all())
    resto_intacto = salida.shape == frame.shape

    print("=== Pre-proceso + ROI (autotest camara_0) ===")
    print(f"Frame entrada/salida : {frame.shape} -> {salida.shape}")
    print(f"Zona TV tapada       : {'sí' if tv_tapado else 'NO'}")
    print(f"Mismo formato/salida : {'sí' if resto_intacto else 'NO'}")
    print(f"Máscara válida (1)   : {int(mask.sum())} px de {H*W} totales")
    print("Listo: el frame de salida entra directo a backbone.encode() sin cambios.")
