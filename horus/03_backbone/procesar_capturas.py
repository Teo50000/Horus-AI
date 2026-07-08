from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch

from shared_backbone import BackboneConfig, SharedBackbone


sys.path.append(str(Path(__file__).resolve().parents[1] / "02_preproceso_roi"))
from preproceso_roi import PreprocesadorROI, cargar_config_zonas

sys.path.append(str(Path(__file__).resolve().parents[1] / "06_fusion_decision"))
from vlm_gate import VLMGate

GATE = VLMGate()


_RUTA_ZONAS = Path(__file__).resolve().parents[1] / "02_preproceso_roi" / "zonas.json"
PREPROCESO = PreprocesadorROI(cargar_config_zonas(_RUTA_ZONAS))


PATRON = re.compile(r"frame_(\d+)\.jpg$")


def carpeta_por_defecto() -> Path:

    raiz = Path(__file__).resolve().parent.parent
    return raiz / "01_camaras" / "Horus.Image" / "capturas_camaras"


def pendientes(base: Path, ultimo: dict) -> List[Tuple[str, int, Path]]:

    out: List[Tuple[str, int, Path]] = []
    for cam_dir in sorted(base.glob("camara_*")):
        cam = cam_dir.name
        for f in cam_dir.glob("frame_*.jpg"):
            m = PATRON.search(f.name)
            if not m:
                continue
            n = int(m.group(1))
            if n > ultimo.get(cam, -1):
                out.append((cam, n, f))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def procesar_archivo(
    backbone: SharedBackbone, ruta: Path, camera_id: str, mostrar: bool = False
) -> Optional[Tuple[float, bool]]:

    frame_bgr = cv2.imread(str(ruta))
    if frame_bgr is None:
        return None


    frame_bgr = PREPROCESO.procesar(frame_bgr, camera_id)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    features = backbone.encode(frame_rgb, camera_id=camera_id)
    correr_vlm = GATE.should_run(features)
    if mostrar:

        txt = f"{camera_id} | novedad {features.novelty:.1f}sigma | VLM {'SI' if correr_vlm else 'no'}"
        cv2.putText(frame_bgr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(camera_id, frame_bgr)
    return float(features.novelty), bool(correr_vlm)


def main() -> None:

    ap = argparse.ArgumentParser(description="Procesa las capturas de imageExtractor.py con el backbone.")
    ap.add_argument("--carpeta", default=None, help="Carpeta capturas_camaras (si no, la calcula sola).")
    ap.add_argument("--pretrained", action="store_true", help="Usar pesos ImageNet (necesita internet 1ra vez).")
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--ver", action="store_true", help="Mostrar una ventana por cámara.")
    ap.add_argument("--poll", type=float, default=0.1, help="Cada cuánto revisa las carpetas (seg).")
    ap.add_argument("--borrar-procesados", action="store_true",
                    help="Borra cada JPG tras procesarlo (evita que el disco se llene).")
    args = ap.parse_args()

    base = Path(args.carpeta) if args.carpeta else carpeta_por_defecto()
    print(f"Vigilando: {base}")
    print("Arrancá imageExtractor.py en otra terminal si todavía no lo hiciste.\n")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = SharedBackbone(
        BackboneConfig(pretrained=args.pretrained, input_size=args.input_size, freeze_encoder=True)
    ).to(device).eval()
    print(f"Backbone en {device}. Esperando frames...\n")

    ultimo: dict = {}
    try:
        while True:
            if not base.exists():
                time.sleep(0.5)
                continue
            for cam, n, ruta in pendientes(base, ultimo):
                res = procesar_archivo(backbone, ruta, cam, mostrar=args.ver)
                if res is None:
                    continue
                novedad, correr_vlm = res
                print(f"{cam} frame {n:06d} | novedad {novedad:5.1f}σ | VLM {'SI' if correr_vlm else 'no'}")
                ultimo[cam] = n
                if args.borrar_procesados:
                    try:
                        ruta.unlink()
                    except OSError:
                        pass

            if args.ver and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        if args.ver:
            cv2.destroyAllWindows()
        print("listo")


if __name__ == "__main__":
    main()
