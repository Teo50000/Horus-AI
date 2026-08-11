import re

ARCHIVO = "links.txt"

with open(ARCHIVO, encoding="utf-8") as f:
    lineas = [l.strip() for l in f if l.strip()]

# cada bloque de la cascada arranca con una entrada "DataSet"
bloques = []
actual = None
for linea in lineas:
    partes = linea.split("\t")
    if len(partes) != 2:
        continue
    nombre, url = partes[0].strip(), partes[1].strip()
    if nombre == "DataSet":
        actual = {}
        bloques.append(actual)
    if actual is not None:
        actual[nombre] = url

print(f"Bloques encontrados: {len(bloques)} (esperados: 561)")

# 11 actividades x 3 trials = 33 bloques por sujeto
entradas = []
for i, b in enumerate(bloques):
    sujeto = i // 33 + 1
    actividad = (i % 33) // 3 + 1
    trial = (i % 33) % 3 + 1
    entradas.append({
        "sujeto": sujeto, "actividad": actividad, "trial": trial,
        "camera1": b.get("Camera1"),
    })

# chequeo contra el zip que ya bajaste a mano
primero = entradas[0]
print(f"Primero: S{primero['sujeto']}/A{primero['actividad']}/T{primero['trial']}")
print(f"  Camera1: {primero['camera1']}")
print(f"  ¿Coincide con el que bajaste?",
      "1x-gpsGcP1jMAvWAZ9oM1O7MsGW8VaCma" in (primero["camera1"] or ""))

ultimo = entradas[-1]
print(f"Último: S{ultimo['sujeto']}/A{ultimo['actividad']}/T{ultimo['trial']}")

sin_camera = sum(1 for e in entradas if not e["camera1"])
print(f"Bloques sin Camera1: {sin_camera}")