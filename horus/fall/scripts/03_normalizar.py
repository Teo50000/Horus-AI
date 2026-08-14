import numpy as np
import glob
import os

CARPETA_CAUCA = "../data/keypoints/caucafall"
CARPETA_LE2I = "../data/keypoints/le2i_omnifall"
CARPETA_UPFALL = "../data/keypoints/upfall"
CARPETA_SALIDA = "../data/processed"

# División por sujeto (no por secuencia individual)
SUJETOS_TRAIN = ["Subject.1", "Subject.2", "Subject.3", "Subject.4", "Subject.5", "Subject.6"]
SUJETOS_VAL   = ["Subject.7"]
SUJETOS_TEST  = ["Subject.10"]

# En Le2i no hay ID de sujeto, así que divido por escenario/carpeta
# (cada carpeta = cámara y fondo distinto, así evito que el modelo vea el mismo fondo en train y test)
# Solo uso 3 escenarios: Office, Lecture_room y Coffee_room_02 no traen Annotation_files, así que no se pueden etiquetar
ESCENARIOS_TRAIN = ["Coffee_room_01"]
ESCENARIOS_VAL   = ["Home_01"]
ESCENARIOS_TEST  = ["Home_02"]

TODOS_ESCENARIOS = ESCENARIOS_TRAIN + ESCENARIOS_VAL + ESCENARIOS_TEST

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

def agregar_velocidades(kp_norm):
#Suma la velocidad de cada punto: (n_frames, 33, 2) -> (n_frames, 33, 4) con (x, y, dx, dy)
    vel = np.zeros_like(kp_norm)
    vel[1:] = kp_norm[1:] - kp_norm[:-1] #cuánto se movió cada punto respecto al frame anterior
    #el primer frame no tiene anterior, así que queda en cero
    return np.concatenate([kp_norm, vel], axis=2)

def detectar_escenario(nombre_archivo):
#Saca el escenario del nombre del archivo: 'Home_01_video (1)_fall.npz' -> 'Home_01'
    for esc in TODOS_ESCENARIOS:
        if nombre_archivo.startswith(esc + "_"): #el prefijo que le puse en el extractor
            return esc
    return None #si no matchea ninguno, devuelvo None y lo salteo después


def cargar_caucafall(lista_sujetos):
#CAUCAFall: el sujeto viene guardado como campo dentro del .npz
    datos = []
    for archivo in glob.glob(os.path.join(CARPETA_CAUCA, "*.npz")):
        npz = np.load(archivo, allow_pickle=True)
        sujeto = str(npz["sujeto"])

        if lista_sujetos is not None and sujeto not in lista_sujetos:
            continue

        kp = npz["keypoints"]  # (n_frames, 33, 4)
        if kp.shape[0] == 0:
            print(f"  Saltando {archivo}: sin frames")
            continue

        datos.append({
            "subclase": "caucafall",
            "keypoints": normalizar_esqueleto(interpolar_frames_faltantes(kp)),
            "label": str(npz["label"]),
            "origen": "caucafall", #para poder medir después cómo rinde en cada dataset por separado
            "grupo": sujeto, #unifico "sujeto" (cauca) y "escenario" (le2i) bajo un mismo nombre, sirve para la validación cruzada
            "archivo_origen": os.path.basename(archivo),
            "keypoints": agregar_velocidades(normalizar_esqueleto(interpolar_frames_faltantes(kp))),
        })
    return datos


def cargar_le2i(lista_sujetos=None):
#Le2i con anotaciones de Omnifall: el sujeto viene como campo dentro del .npz
    datos = []
    for archivo in glob.glob(os.path.join(CARPETA_LE2I, "*.npz")):
        npz = np.load(archivo, allow_pickle=True)
        sujeto = str(npz["sujeto"])  # ej: "le2i_s3"

        if lista_sujetos is not None and sujeto not in lista_sujetos:
            continue

        kp = npz["keypoints"]
        if kp.shape[0] == 0:
            print(f"  Saltando {os.path.basename(archivo)}: sin frames")
            continue

        datos.append({
            "subclase": int(npz["label_omnifall"]),
            "keypoints": normalizar_esqueleto(interpolar_frames_faltantes(kp)),
            "label": str(npz["label"]),
            "origen": "le2i",
            "grupo": sujeto, #ahora es el sujeto real, no el escenario
            "escenario": str(npz["escenario"]), #lo guardo por si quiero analizar por escenario después
            "archivo_origen": os.path.basename(archivo),
            "keypoints": agregar_velocidades(normalizar_esqueleto(interpolar_frames_faltantes(kp))),
        })
    return datos

def cargar_upfall(lista_sujetos=None):
    #UPFall: mismo formato de .npz que le2i, con sujeto y label_omnifall adentro
    datos = []
    for archivo in glob.glob(os.path.join(CARPETA_UPFALL, "*.npz")):
        npz = np.load(archivo, allow_pickle=True)
        sujeto = str(npz["sujeto"])  # ej: "upfall_s3"

        if lista_sujetos is not None and sujeto not in lista_sujetos:
            continue

        kp = npz["keypoints"]
        if kp.shape[0] == 0:
            continue

        sub = int(npz["label_omnifall"])
        label3 = {1: "fall", 2: "fallen"}.get(sub, "adl")

        datos.append({
            "keypoints": agregar_velocidades(normalizar_esqueleto(interpolar_frames_faltantes(kp))),
            "label": str(npz["label"]),
            "label3": label3,
            "origen": "upfall",
            "grupo": sujeto,
            "subclase": sub,
            "archivo_origen": os.path.basename(archivo),
        })
    return datos

def procesar_split(sujetos, escenarios, nombre_split):
    datos = cargar_caucafall(sujetos) + cargar_le2i(escenarios) #junto los dos datasets en una sola lista

    # Resumen de balance de clases, importante para saber si hay que ponderar la loss después
    labels = [d["label"] for d in datos]
    fall = labels.count("fall")
    adl = labels.count("adl")
    n_cauca = sum(1 for d in datos if d["origen"] == "caucafall")
    n_le2i = sum(1 for d in datos if d["origen"] == "le2i")

    print(f"{nombre_split}: {len(datos)} secuencias "
          f"(caucafall={n_cauca}, le2i={n_le2i}) | fall={fall}, adl={adl}")

    np.save(os.path.join(CARPETA_SALIDA, f"{nombre_split}.npy"), datos, allow_pickle=True)
    return datos


if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Ya no armo splits fijos, la validación cruzada por sujeto usa todo el dataset
    # y va rotando qué sujeto queda afuera en cada fold.
    todos = cargar_caucafall(None) + cargar_le2i(None) + cargar_upfall(None)
    labels = [d["label"] for d in todos]
    grupos = sorted({d["grupo"] for d in todos})

    print(f"Total: {len(todos)} secuencias")
    print(f"fall={labels.count('fall')}, adl={labels.count('adl')}")
    print(f"{len(grupos)} grupos (sujetos): {grupos}")

    np.save(os.path.join(CARPETA_SALIDA, "todos.npy"), todos, allow_pickle=True)