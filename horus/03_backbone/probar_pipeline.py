import sys
import time
from pathlib import Path

import cv2
import torch

from shared_backbone import BackboneConfig, SharedBackbone

sys.path.append(str(Path(__file__).resolve().parents[1] / "06_fusion_decision"))
from vlm_gate import VLMGate

gate = VLMGate()


arg = sys.argv[1] if len(sys.argv) > 1 else "0"
mostrar = "--ver" in sys.argv
fuente = int(arg) if arg.isdigit() else arg


device = "cuda" if torch.cuda.is_available() else "cpu"

backbone = SharedBackbone(
    BackboneConfig(pretrained=True, input_size=384, freeze_encoder=True)
).to(device).eval()
print(f"Backbone en {device}. Fuente: {fuente}\n")

cap = cv2.VideoCapture(fuente)
if not cap.isOpened():
    print("No pude abrir la fuente. ¿Índice de cámara o ruta correctos?")
    sys.exit(1)

idx = 0
suma_fps = 0.0
try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = backbone.encode(rgb, camera_id="prueba")
        correr_vlm = gate.should_run(features)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        suma_fps += fps
        print(f"frame {idx:5d} | novedad {features.novelty:5.1f}σ "
              f"| VLM {'SI' if correr_vlm else 'no':2} | {fps:4.1f} FPS")

        if mostrar:
            txt = f"novedad {features.novelty:.1f}sigma | VLM {'SI' if correr_vlm else 'no'}"
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("prueba", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1
finally:
    cap.release()
    if mostrar:
        cv2.destroyAllWindows()
    if idx:
        print(f"\n{idx} frames | promedio {suma_fps/idx:.1f} FPS en {device}")
    print("listo")
