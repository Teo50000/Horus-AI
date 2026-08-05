import os
import re
import glob
import pandas as pd

RAIZ = r"D:\download\FallDataset"

# el nombre de la carpeta raíz de cada escenario tal como aparece en el disco
CARPETAS = {
    "Coffee_room_01": "Coffee_room_01",
    "Coffee_room_02": "Coffee_room_02",
    "Home_01": "Home_01",
    "Home_02": "Home_02",
    "Lecture_room": "Lecture_room",
    "Office": "Office",
}


def numero_de(nombre_archivo):
    """'video (31).avi' -> 31 ; 'video_31.avi' -> 31"""
    nums = re.findall(r"\d+", os.path.basename(nombre_archivo))
    return int(nums[-1]) if nums else None


def indexar_videos(escenario):
    """Busca todos los .avi bajo la carpeta del escenario, sin importar cuán anidados estén."""
    raiz_esc = os.path.join(RAIZ, CARPETAS[escenario])
    patron = os.path.join(raiz_esc, "**", "*.avi")
    encontrados = glob.glob(patron, recursive=True)

    indice = {}
    for ruta in encontrados:
        n = numero_de(ruta)
        if n is not None:
            indice[n] = ruta
    return indice


if __name__ == "__main__":
    df = pd.read_csv("le2i_omnifall.csv")
    df["escenario"] = df["path"].str.split("/").str[0]
    df["num_video"] = df["path"].str.extract(r"video_(\d+)").astype(int)

    # una fila por video (las anotaciones traen varios segmentos por video)
    videos = df[["escenario", "num_video", "path"]].drop_duplicates()
    print(f"Videos únicos en las anotaciones: {len(videos)}\n")

    encontrados_total, faltantes_total = 0, 0

    for escenario in sorted(videos["escenario"].unique()):
        indice = indexar_videos(escenario)
        nums_esperados = sorted(videos[videos["escenario"] == escenario]["num_video"])

        faltan = [n for n in nums_esperados if n not in indice]
        hay = [n for n in nums_esperados if n in indice]

        encontrados_total += len(hay)
        faltantes_total += len(faltan)

        print(f"{escenario}: {len(hay)}/{len(nums_esperados)} encontrados"
              f"{'' if not faltan else f' | FALTAN: {faltan}'}")

        if hay:
            ejemplo = indice[hay[0]]
            print(f"    ejemplo: video_{hay[0]} -> {ejemplo}")

    print(f"\nTOTAL: {encontrados_total} encontrados, {faltantes_total} faltantes")