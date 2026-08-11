import os
import glob
import re
from datetime import datetime
import pandas as pd

CARPETA = r"C:\Users\48792583\Downloads\Subject1Activity1Trial1Camera1"
CSV = "up_fall_omnifall.csv"
RUTA_ANOTACION = "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1"

NOMBRES = {0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting",
           5: "lie_down", 6: "lying", 7: "stand_up", 8: "standing", 9: "other"}


def timestamp_de(nombre):
    """'2018-07-04T12_04_17.738369.png' -> datetime"""
    base = os.path.splitext(os.path.basename(nombre))[0]
    return datetime.strptime(base, "%Y-%m-%dT%H_%M_%S.%f")


archivos = glob.glob(os.path.join(CARPETA, "*.png"))
tiempos = sorted(timestamp_de(a) for a in archivos)

t0 = tiempos[0]
segundos = [(t - t0).total_seconds() for t in tiempos]

print(f"Frames: {len(archivos)}")
print(f"Duración: {segundos[-1]:.2f} s")
print(f"FPS promedio: {len(archivos) / segundos[-1]:.1f}")

print("\nAnotaciones de Omnifall para este trial:")
df = pd.read_csv(CSV)
segs = df[df["path"] == RUTA_ANOTACION]
for s in segs.itertuples():
    print(f"  {NOMBRES.get(s.label, s.label):>9}  {s.start:6.2f} - {s.end:6.2f} s")