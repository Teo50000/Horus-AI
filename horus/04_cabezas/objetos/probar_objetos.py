# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import cv2
import torch

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RAIZ, "03_backbone"))

from shared_backbone import SharedBackbone
from objects_head import ObjectsHead, ObjectsHeadConfig

COLORES = {
    "humo": (160, 160, 160),
    "llama": (0, 120, 255),
    "persona": (0, 200, 0),
    "pistola": (0, 0, 255),
    "cuchillo": (60, 20, 220),
    "celular": (255, 200, 0),
    "paquete": (30, 105, 160),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probar backbone + cabeza de objetos")
    ap.add_argument("fuente", nargs="?", default="0",
                    help="índice de webcam (0, 1, ...) o ruta a un video")
    ap.add_argument("--ver", action="store_true",
                    help="abrir ventana con las cajas dibujadas (q para salir)")
    ap.add_argument("--umbral", type=float, default=0.30,
                    help="score mínimo (default 0.30; usar 0.005 sin entrenar)")
    ap.add_argument("--pesos", default=None,
                    help="ruta a un state_dict entrenado de la cabeza (.pt)")
    args = ap.parse_args()

    fuente = int(args.fuente) if args.fuente.isdigit() else args.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        sys.exit(f"No pude abrir la fuente: {args.fuente!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device = {device}")

    backbone = SharedBackbone().to(device).eval()

    cfg = ObjectsHeadConfig(
        fpn_levels=("p3", "p4", "p5"),
        anchor_sizes=((32,), (64,), (128,)),
        score_thresh=args.umbral,
    )
    head = ObjectsHead(cfg).to(device).eval()

    if args.pesos:
        head.load_state_dict(torch.load(args.pesos, map_location=device))
        print(f"[setup] pesos cargados de {args.pesos}")
    else:
        print("[setup] SIN pesos entrenados: lo esperable es 0 detecciones "
              "(probá --umbral 0.005 para ver el post-proceso andar)")

    n_frames = 0
    t_inicio = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        feats = backbone.encode(rgb, camera_id="cam-0")

        h_in, w_in = feats.input_size
        dets = head.predict(
            feats.pyramid,
            image_sizes=[(h_in, w_in)],
            camera_ids=[feats.camera_id],
            frame_idxs=[feats.frame_idx],
        )[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        n_frames += 1

        h_org, w_org = frame.shape[:2]
        sx, sy = w_org / w_in, h_org / h_in

        for d in dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            gate = "  [needs_vlm]" if d.needs_vlm else ""
            print(f"  frame {feats.frame_idx:5d}  {d.label:8s} "
                  f"score={d.score:.3f}  caja=({x1},{y1},{x2},{y2}){gate}")
            if args.ver:
                color = COLORES.get(d.label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{d.label} {d.score:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if n_frames % 30 == 0:
            fps = n_frames / (time.perf_counter() - t_inicio)
            print(f"[frame {feats.frame_idx}] fps={fps:.1f}  "
                  f"inferencia={dt_ms:.0f} ms  novedad={feats.novelty:.2f}σ  "
                  f"detecciones={len(dets)}")

        if args.ver:
            cv2.imshow("HORUS · objetos", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    total = time.perf_counter() - t_inicio
    if n_frames:
        print(f"\n[fin] {n_frames} frames en {total:.1f}s -> {n_frames / total:.1f} fps")


if __name__ == "__main__":
    main()
