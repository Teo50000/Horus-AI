
import cv2
import os

ruta_video = r"D:\download\FallDataset\Home_02\Home_02\Videos\video (31).avi"

print("¿Existe el archivo?", os.path.exists(ruta_video))

cap = cv2.VideoCapture(ruta_video)
print("¿Se pudo abrir?", cap.isOpened())

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total de frames: {total_frames}")
print(f"FPS: {fps}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ok, frame = cap.read()
if ok:
    cv2.imwrite("frame_200_check.png", frame)
    print("Guardé frame_200_check.png")

cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ok, frame = cap.read()
if ok:
    cv2.imwrite("frame_100_check.png", frame)
    print("Guardé frame_100_check.png")
cap.release()