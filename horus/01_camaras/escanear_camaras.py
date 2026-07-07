import sys
import cv2

backends = [("default", None)]
if sys.platform.startswith("win"):
    backends.append(("DirectShow", cv2.CAP_DSHOW))

print("Buscando camaras (cerra OBS o cualqujier programa  si está abierto)...\n")
encontradas = []
for nombre, flag in backends:
    for i in range(6):
        cap = cv2.VideoCapture(i, flag) if flag is not None else cv2.VideoCapture(i)
        if cap.isOpened():
            frame = None
            for _ in range(3):
                ok, frame = cap.read()
            if frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  indice {i}  [{nombre}]  ->  {w}x{h}")
                cv2.imshow(f"indice {i} ({nombre})", frame)
                encontradas.append((i, nombre))
        cap.release()

if not encontradas:
    print("\nNo se abrio NINGUNA camara. Revisr:")
    print("  - OBS cerrado del todo (también en la bandeja del sistema)")
    print("  - permisos de cámara del sistema operativo")
    print("  - que la app 'Camara' del sistema si la vea (descarta problema de hardware)")
else:
    print("\nMirá las ventanas: la que muestra tu cara real es tu webcam.")
    print("Anota ese número. Apreta cualquier tecla para cerrar.")
    cv2.waitKey(0)
cv2.destroyAllWindows()
