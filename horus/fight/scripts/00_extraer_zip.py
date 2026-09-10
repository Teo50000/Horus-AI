import os
import sys
import zipfile

# Extrae el zip de RWF-2000 (bajado de Kaggle, suele venir como archive.zip) a
# horus/fight/data/raw/RWF-2000/. Uso: python 00_extraer_zip.py <ruta_al_zip>
#
# Renombro cada clip de forma deterministica (train_Fight_0001.avi, etc.) en vez
# de conservar el nombre original: algunos nombres del zip vienen en un encoding
# que rompe tanto zipfile como unzip (mojibake). El nombre no importa para nada,
# solo el contenido y a que split/clase pertenece -- y esas dos partes del path
# ("train"/"val", "Fight"/"NonFight") son ascii puro, nunca se corrompen.

ZIP = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\48792583\Downloads\archive.zip"
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

contadores = {}
z = zipfile.ZipFile(ZIP)
extraidos = 0
for info in z.infolist():
    partes = info.filename.split("/")
    if info.is_dir() or len(partes) != 4:
        continue
    _, split, clase, _nombre_original = partes
    if split not in ("train", "val") or clase not in ("Fight", "NonFight"):
        continue

    carpeta = os.path.join(SALIDA, "RWF-2000", split, clase)
    os.makedirs(carpeta, exist_ok=True)

    clave = (split, clase)
    contadores[clave] = contadores.get(clave, 0) + 1
    nombre_limpio = f"{split}_{clase}_{contadores[clave]:04d}.avi"

    with z.open(info) as origen, open(os.path.join(carpeta, nombre_limpio), "wb") as destino:
        destino.write(origen.read())
    extraidos += 1

print(f"Extraidos: {extraidos}")
for clave, n in sorted(contadores.items()):
    print(clave, n)
