from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

previous_ratio = None

while True
    ret, frame = cap.read() #se fija si la camara da frames asi los va pasando

    if not ret # si la camara no da video se termina el script
        break
    
    cv2.imshow("camera",frame)