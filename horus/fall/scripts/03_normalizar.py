import numpy as np
import glob
import os
import json

CARPETA_KEYPOINTS = "../data/keypoints/caucafall"
CARPETA_SALIDA = "../data/processed"

# División por sujeto (no por secuencia individual)
SUJETOS_TRAIN = ["Subject.1", "Subject.2", "Subject.3", "Subject.4", "Subject.5", "Subject.6"]
SUJETOS_VAL   = ["Subject.7"]
SUJETOS_TEST  = ["Subject.10"]

# de los 33 puntos que tiene el cuerpo en mediapipe hago variables del index de los puntos mas importantes
IDX_CADERA_IZQ = 23 
IDX_CADERA_DER = 24
IDX_HOMBRO_IZQ = 11
IDX_HOMBRO_DER = 12


def interpolar_frames_faltantes(kp): #Rellena frames sin detección (todo en cero) interpolando desde vecinos válidos.
    n_frames = kp.shape[0]
    valido = kp.reshape(n_frames, -1).sum(axis=1) != 0 
#En vez de tener 191,33,4, lo convierto todo en una fila de 132 numeros y hago que me quede una lista con un numero por frame, si el frame tiene numero es true else false
    if valido.sum() == 0: # secuencia entera sin detecciones
        return kp  

    indices_validos = np.where(valido)[0]  #los que son true y tienen algun valor
    kp_interpolado = kp.copy() 
#copia del array original por si al modificar frame por frame, si llego a nescesitar el original lo tengo ahi
    for i in range(n_frames): 
        if not valido[i]: #si es un frame con datos, no hace nada
            # buscar vecino válido más cercano antes y después
            antes = indices_validos[indices_validos < i]
            despues = indices_validos[indices_validos > i]

            if len(antes) > 0 and len(despues) > 0: #caso 1 tiene frame antes y despues
                a, d = antes[-1], despues[0] # a es el mas cercano por izquierda y d por derecha
                peso = (i - a) / (d - a)  #qué tan cerca está el frame vacío de cada vecino sacando el punto medio
                kp_interpolado[i] = kp[a] * (1 - peso) + kp[d] * peso #Promedio ponderado entre el valor de antes y el de después. 

            elif len(antes) > 0: #caso 2 solo hay vecino antes, no hay después 
                kp_interpolado[i] = kp[antes[-1]] #copio directamente el valor mas cercano que tengo
            elif len(despues) > 0: #caso 3 solo hay vecino después, no antes
                kp_interpolado[i] = kp[despues[0]] #copio directamente el valor mas cercano que tengo

    return kp_interpolado


def normalizar_esqueleto(kp):
#Centra en la cadera y escala por la distancia hombro-cadera.
    cadera = (kp[:, IDX_CADERA_IZQ, :2] + kp[:, IDX_CADERA_DER, :2]) / 2 #x y en todos los frames y me agarri el punto medio de las dos
    hombro = (kp[:, IDX_HOMBRO_IZQ, :2] + kp[:, IDX_HOMBRO_DER, :2]) / 2 #igual

    escala = np.linalg.norm(hombro - cadera, axis=1, keepdims=True) #saco distancia entre hombro cadera +- torso
    escala = np.where(escala < 1e-6, 1e-6, escala)  # evitar división por cero si escala = 0

    kp_norm = kp[:, :, :2] - cadera[:, None, :] 
    #le agrego una dimension mas a cadera para que encaje con las tres dimensiones de kp (frames, puntos, x/y)
    kp_norm = kp_norm / escala[:, None, :] #normalizo todos los frames

    return kp_norm  # (n_frames, 33, 2)


def procesar_split(lista_sujetos, nombre_split):
    datos = []
    archivos = glob.glob(os.path.join(CARPETA_KEYPOINTS, "*.npz"))

    for archivo in archivos:
        npz = np.load(archivo, allow_pickle=True)
        sujeto = str(npz["sujeto"])

        if sujeto not in lista_sujetos:
            continue

        kp = npz["keypoints"]  # (n_frames, 33, 4)
        label = str(npz["label"])
        actividad = str(npz["actividad"])

        if kp.shape[0] == 0:
            print(f"  Saltando {archivo}: sin frames")
            continue

        kp_interp = interpolar_frames_faltantes(kp)
        kp_norm = normalizar_esqueleto(kp_interp)

        datos.append({
            "keypoints": kp_norm,
            "label": label,
            "sujeto": sujeto,
            "actividad": actividad,
            "archivo_origen": os.path.basename(archivo)
        })

    print(f"{nombre_split}: {len(datos)} secuencias")
    np.save(os.path.join(CARPETA_SALIDA, f"{nombre_split}.npy"), datos, allow_pickle=True)
    return datos


if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    train = procesar_split(SUJETOS_TRAIN, "train")
    val = procesar_split(SUJETOS_VAL, "val")
    test = procesar_split(SUJETOS_TEST, "test")

    # Resumen de balance de clases, importante para saber si hay que ponderar la loss después
    for nombre, datos in [("train", train), ("val", val), ("test", test)]:
        labels = [d["label"] for d in datos]
        fall = labels.count("fall")
        adl = labels.count("adl")
        print(f"{nombre}: fall={fall}, adl={adl}")