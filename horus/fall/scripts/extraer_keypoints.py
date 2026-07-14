import cv2
import numpy as np
import os
import re
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO   # <-- este import puede faltar

MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODELO_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

yolo_model = YOLO("yolov8n.pt")


def ordenar_por_numero(nombre_archivo):
    numeros = re.findall(r'\d+', os.path.basename(nombre_archivo)) #agarro el nombre del archivo sin el path y con r'\d+' me quedo solo con los numeros consecutivosq que hayan
    return int(numeros[-1]) if numeros else 0 #agarro el ultimo numero de la lista si devuelve algo


def detectar_y_recortar_persona(frame, margen=0.2): 
    resultados = yolo_model(frame, classes=[0], verbose=False)
    if len(resultados[0].boxes) == 0:
        return None
    boxes = resultados[0].boxes
    mejor_idx = boxes.conf.argmax().item()
    x1, y1, x2, y2 = boxes.xyxy[mejor_idx].cpu().numpy()
    h, w = frame.shape[:2]
    ancho_box = x2 - x1
    alto_box = y2 - y1
    x1 = max(0, int(x1 - ancho_box * margen))
    y1 = max(0, int(y1 - alto_box * margen))
    x2 = min(w, int(x2 + ancho_box * margen))
    y2 = min(h, int(y2 + alto_box * margen))
    return frame[y1:y2, x1:x2]


def extraer_keypoints_carpeta(carpeta_imagenes):  #script para sacar la pose de todas las imagenes de una carpeta
    imagenes = glob.glob(os.path.join(carpeta_imagenes, "*.png"))  #me quedo con todas las imagenes (que terminen con png)
    imagenes.sort(key=ordenar_por_numero) #las ordeno en base a la funcion de antes

    frames_keypoints = []
    for img_path in imagenes:
        frame = cv2.imread(img_path)
        if frame is None:
            continue #si no pudo leer la imagen pasa a la siguiente

        crop = detectar_y_recortar_persona(frame)
        if crop is None or crop.size == 0:
            frames_keypoints.append(np.zeros((33, 4))) #no hay crop = rellenar con 0
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = landmarker.detect(mp_image)

        if resultado.pose_landmarks: #hay persona
            lm = resultado.pose_landmarks[0]
            kp = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])
        else:
            kp = np.zeros((33, 4)) #deja los valores en 0
        frames_keypoints.append(kp)

    return np.array(frames_keypoints) # (n_frames, 33, 4)

#label   simplifico las clases que tiene el dataset porque no me importa la orientacion de la caida ni las que no son caidas
MAPEO_LABELS = {
    "Fall backwards": "fall",
    "Fall forward": "fall",
    "Fall left": "fall",
    "Fall right": "fall",
    "Fall sitting": "fall",
    "Hop": "adl",
    "Kneel": "adl",
    "Pick up object": "adl",
    "Sit down": "adl",
}

def procesar_dataset(raiz_caucafall, salida_dir):
    os.makedirs(salida_dir, exist_ok=True) #creo carpeta de salida

    for sujeto in os.listdir(raiz_caucafall): #recorro para cada sujeto
        ruta_sujeto = os.path.join(raiz_caucafall, sujeto)
        if not os.path.isdir(ruta_sujeto):
            continue

        for actividad in os.listdir(ruta_sujeto): #recorro para cada actividad (fall)
            ruta_actividad = os.path.join(ruta_sujeto, actividad)
            if not os.path.isdir(ruta_actividad):
                continue

            label = MAPEO_LABELS.get(actividad) #me quedo con la etiqueta
            if label is None:
                print(f"Actividad desconocida, saltando: {actividad}")
                continue

            print(f"Procesando {sujeto}/{actividad}...")
            kp = extraer_keypoints_carpeta(ruta_actividad) #extraigo los keypoints
            if len(kp) == 0: #chequeo que haya por lo menos algo
                continue

            nombre_salida = f"{sujeto}_{actividad}.npz" #creo nombre del archivo en base al sujeto y cual es la actividad
            np.savez(os.path.join(salida_dir, nombre_salida), #creo un .npz para guardar todos estos datos
                     keypoints=kp, label=label, sujeto=sujeto, actividad=actividad)

if __name__ == "__main__":
    procesar_dataset("../data/CAUCAFall", "../data/keypoints/caucafall")