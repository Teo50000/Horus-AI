import cv2
import mediapipe as mp
import numpy as np
import os
import re
import glob

mp_pose = mp.solutions.pose

def ordenar_por_numero(nombre_archivo):
    "en resumen es para que el frame 10 no este antes que el 2"
    numeros = re.findall(r'\d+', os.path.basename(nombre_archivo)) #agarro el nombre del archivo sin el path y con r'\d+' me quedo solo con los numeros consecutivosq que hayan
    return int(numeros[-1]) if numeros else 0 #agarro el ultimo numero de la lista si devuelve algo

def extraer_keypoints_carpeta(carpeta_imagenes, pose): #script para sacar la pose de todas las imagenes de una carpeta
    imagenes = glob.glob(os.path.join(carpeta_imagenes, "*.png")) #me quedo con todas las imagenes (que terminen con png)
    imagenes.sort(key=ordenar_por_numero) #las ordeno en base a la funcion de antes
    
    frames_keypoints = []
    for img_path in imagenes:
        frame = cv2.imread(img_path)
        if frame is None:
            continue #si no pudo leer la imagen pasa a la siguiente
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = pose.process(rgb) 
        
        if resultado.pose_landmarks: #hay persona
            kp = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                           for lm in resultado.pose_landmarks.landmark])
        else:
            kp = np.zeros((33, 4)) #deja los valores en 0
        frames_keypoints.append(kp)
    
    return np.array(frames_keypoints)  # (n_frames, 33, 4)


#label   simplifico las clases que tiene el dataset porque no me importa la orientacion de la caida ni las que no son caidas
MAPEO_LABELS = { 
    "Fall_forward": "fall",
    "Fall_backward": "fall",
    "Fall_left": "fall",
    "Fall_right": "fall",
    "Fall_sitting": "fall",
    "Walking": "adl",
    "Hopping": "adl",
    "PickingUpObject": "adl",
    "Sitting": "adl",
    "Kneeling": "adl",
}

def procesar_dataset(raiz_caucafall, salida_dir):
    os.makedirs(salida_dir, exist_ok=True) #creo carpeta de salida
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose: #creo mediapipe aca
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
                kp = extraer_keypoints_carpeta(ruta_actividad, pose) #extraigo los keypoints
                
                if len(kp) == 0: #chequeo que haya por lo menos algo
                    continue
                
                nombre_salida = f"{sujeto}_{actividad}.npz" #creo nombre del archivo en base al sujeto y cual es la actividad
                np.savez(os.path.join(salida_dir, nombre_salida), #creo un .npz para guardar todos estos datos
                         keypoints=kp, label=label, 
                         sujeto=sujeto, actividad=actividad)

if __name__ == "__main__":
    procesar_dataset("data/raw_videos/caucafall", "data/keypoints/caucafall")