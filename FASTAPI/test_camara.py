import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara encontrada en índice: {i}")
        cap.release()
    else:
        print(f"Índice {i}: no hay cámara")