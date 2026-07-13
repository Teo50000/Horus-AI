from ultralytics import YOLO

# Modelo liviano de detección (se descarga automático la primera vez)
yolo_model = YOLO("yolov8n.pt")  # nano, rápido, suficiente para detectar personas

def detectar_y_recortar_persona(frame, margen=0.2):
    """Detecta la persona más prominente y devuelve el crop con margen."""
    resultados = yolo_model(frame, classes=[0], verbose=False)  # clase 0 = persona en COCO
    
    if len(resultados[0].boxes) == 0:
        return None  # no se detectó ninguna persona
    
    # Tomar la detección con mayor confianza
    boxes = resultados[0].boxes
    mejor_idx = boxes.conf.argmax().item()
    x1, y1, x2, y2 = boxes.xyxy[mejor_idx].cpu().numpy()
    
    h, w = frame.shape[:2]
    ancho_box = x2 - x1
    alto_box = y2 - y1
    
    # Agregar margen alrededor del bounding box
    x1 = max(0, int(x1 - ancho_box * margen))
    y1 = max(0, int(y1 - alto_box * margen))
    x2 = min(w, int(x2 + ancho_box * margen))
    y2 = min(h, int(y2 + alto_box * margen))
    
    return frame[y1:y2, x1:x2]