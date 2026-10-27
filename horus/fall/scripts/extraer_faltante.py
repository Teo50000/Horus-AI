from extraer_keypoints import extraer_keypoints_carpeta
import numpy as np

kp = extraer_keypoints_carpeta("../data/CAUCAFall/Subject.7/Sit down")
print("Shape:", kp.shape)