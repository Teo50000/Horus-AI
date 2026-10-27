import os

CARPETA = r"C:\Users\48792583\Downloads\UpFallDataset"
SUJETOS = [1, 2, 3, 4]
ACTIVIDADES = [3, 11]
TRIALS = [1, 2, 3]

esperados = [
    f"Subject{s}Activity{a}Trial{t}Camera1.zip"
    for s in SUJETOS for a in ACTIVIDADES for t in TRIALS
]

presentes = set(os.listdir(CARPETA)) if os.path.isdir(CARPETA) else set()

faltan = [n for n in esperados if n not in presentes]
hay = [n for n in esperados if n in presentes]

print(f"Presentes: {len(hay)}/{len(esperados)}")
if faltan:
    print("\nFaltan:")
    for n in faltan:
        print("  " + n)

# archivos sospechosamente chicos (descargas cortadas o páginas de error)
print("\nRevisar tamaños:")
for n in hay:
    mb = os.path.getsize(os.path.join(CARPETA, n)) / 1e6
    marca = "  ⚠️ MUY CHICO" if mb < 1 else ""
    print(f"  {n}: {mb:.1f} MB{marca}")