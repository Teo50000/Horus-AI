import cv2
import time
import os

def detectar_camaras(max_probar=10):

    camaras_activas = []
    print("Buscando cámaras conectadas...")
    for i in range(max_probar):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"-> Cámara detectada en el índice: {i}")
                camaras_activas.append((i, cap))
            else:
                cap.release()
        else:
            cap.release()
    return camaras_activas


camaras = detectar_camaras()

if not camaras:
    print("No se detecto ninguna cámara")
    exit()


ruta_del_script = os.path.dirname(os.path.abspath(__file__))

CARPETA_BASE = os.path.join(ruta_del_script, "Horus.Image", "capturas_camaras")

print(f"\nCreando directorios de guardado en la carpeta '{CARPETA_BASE}'...")

for indice, _ in camaras:
    ruta_carpeta = os.path.join(CARPETA_BASE, f"camara_{indice}")
    os.makedirs(ruta_carpeta, exist_ok=True)


FPS_OBJETIVO = 5
INTERVALO = 1.0 / FPS_OBJETIVO
contador_frames = 0
print(f"\nIniciando captura a {FPS_OBJETIVO} FPS de {len(camaras)} cámara(s).")
try:
    while True:
        tiempo_inicio = time.time()

        for indice, cap in camaras:
            ret, frame = cap.read()
            if ret:
                nombre_archivo = os.path.join(
                    CARPETA_BASE,
                    f"camara_{indice}",
                    f"frame_{contador_frames:06d}.jpg"
                )

                cv2.imwrite(nombre_archivo, frame)

                cv2.imshow(f"Camara Indice {indice}", frame)

        contador_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        tiempo_transcurrido = time.time() - tiempo_inicio
        tiempo_restante = INTERVALO - tiempo_transcurrido
        if tiempo_restante > 0:
            time.sleep(tiempo_restante)

finally:
    for _, cap in camaras:
        cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")
