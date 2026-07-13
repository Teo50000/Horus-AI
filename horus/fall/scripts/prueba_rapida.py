from extraer_keypoints import extraer_keypoints_carpeta
import numpy as np

kp = extraer_keypoints_carpeta("../data/CAUCAFall/Subject.1/Sit down")
print("Shape:", kp.shape)
sin_deteccion = np.where(kp.sum(axis=(1,2)) == 0)[0]
print("Frames sin detección:", len(sin_deteccion), "de", kp.shape[0])
print("Índices:", sin_deteccion)