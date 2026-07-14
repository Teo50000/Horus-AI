from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

previous_ratio = None

while True:
    ret, frame = cap.read() #se fija si la camara da frames asi los va pasando

    if not ret: # si la camara no da video se termina el script
        break
    
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if cls == 0: #persona
                x1, y1 , x2, y2 = map(int, box.xyxy[0]) #agarro las coordenadas (x1 y1 arriba izquierda y x2 y2 abajo derecha) los paso a enteros
                width = x2 -x1 # le resto a la poscicion del final la del inicio asi saco el ancho
                height = y2 -y1 # igual

                ratio = height/ width #ratio no se dea
                cv2.putText(
                    frame,
                    f"ratio: {ratio:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                if (ratio<0.8):
                    cv2.putText(
                        frame,
                        "POSSIBLE FALL",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3
                    )
    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
            
