import cv2
import time
import os
import sys
from pathlib import Path
import torch
from shared_backbone import SharedBackbone, BackboneConfig


sys.path.append(str(Path(__file__).resolve().parents[1] / "02_preproceso_roi"))
from preproceso_roi import PreprocesadorROI, cargar_config_zonas

sys.path.append(str(Path(__file__).resolve().parents[1] / "06_fusion_decision"))
from vlm_gate import VLMGate

GATE = VLMGate()
_RUTA_ZONAS = Path(__file__).resolve().parents[1] / "02_preproceso_roi" / "zonas.json"
PREPROCESO = PreprocesadorROI(cargar_config_zonas(_RUTA_ZONAS))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backbone = SharedBackbone(
    BackboneConfig(pretrained=True, input_size=384, freeze_encoder=True)
).to(DEVICE).eval()
print(f"Backbone cargado en {DEVICE}")


def procesar_frame(frame_bgr, camera_id):


    frame_bgr = PREPROCESO.procesar(frame_bgr, camera_id)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    features = backbone.encode(frame_rgb, camera_id=camera_id)


    incertidumbre = 0.0

    correr_vlm = GATE.should_run(features, head_uncertainty=incertidumbre)
    return features, correr_vlm


def detectar_camaras(max_probar=10):

    camaras_activas = []
    print("Buscando cámaras conectadas...")
    for i in range(max_probar):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"-> Cámara detectada en el índice: {i}")
                camaras_activas.append((i, cap))
            else:
                cap.release()
        else:
            cap.release()
    return camaras_activas


camaras = detectar_camaras()
if not camaras:
    print("No se detecto ninguna cámara")
    exit()


ruta_del_script = os.path.dirname(os.path.abspath(__file__))
CARPETA_BASE = os.path.join(ruta_del_script, "Horus.Image", "capturas_camaras")
print(f"\nCreando directorios de guardado en la carpeta '{CARPETA_BASE}'...")
for indice, _ in camaras:
    ruta_carpeta = os.path.join(CARPETA_BASE, f"camara_{indice}")
    os.makedirs(ruta_carpeta, exist_ok=True)


FPS_OBJETIVO = 5
INTERVALO = 1.0 / FPS_OBJETIVO
contador_frames = 0

print(f"\nIniciando captura a {FPS_OBJETIVO} FPS de {len(camaras)} cámara(s).")
try:
    while True:
        tiempo_inicio = time.time()


        for indice, cap in camaras:
            ret, frame = cap.read()
            if ret:
                camera_id = f"camara_{indice}"


                nombre_archivo = os.path.join(
                    CARPETA_BASE, camera_id, f"frame_{contador_frames:06d}.jpg"
                )
                cv2.imwrite(nombre_archivo, frame)


                features, correr_vlm = procesar_frame(frame, camera_id=camera_id)


                etiqueta = (
                    f"cam {indice} | novedad {features.novelty:.1f}sigma "
                    f"| VLM {'SI' if correr_vlm else 'no'}"
                )
                cv2.putText(frame, etiqueta, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow(f"Camara Indice {indice}", frame)

        contador_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        tiempo_transcurrido = time.time() - tiempo_inicio
        tiempo_restante = INTERVALO - tiempo_transcurrido
        if tiempo_restante > 0:
            time.sleep(tiempo_restante)
        elif contador_frames % 25 == 0:

            fps_real = 1.0 / max(tiempo_transcurrido, 1e-6)
            print(f"[aviso] el backbone no alcanza {FPS_OBJETIVO} FPS "
                  f"(real ~{fps_real:.1f}). Ver F0 / hilos / GPU.")
finally:
    for _, cap in camaras:
        cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")
