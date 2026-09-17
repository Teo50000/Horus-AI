import numpy as np
import glob
import os
import sys
sys.path.append("../src")

from preproceso import interpolar_frames_faltantes, normalizar_esqueleto, agregar_velocidades

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

# interpolar_frames_faltantes / normalizar_esqueleto / agregar_velocidades viven en
# src/preproceso.py, compartidas con el despliegue para que train y serve no diverjan.

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