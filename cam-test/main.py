import cv2 #importo el cv

camera = cv2.VideoCapture(0)
frcounter = 0 

while True:
    ret, frame = camera.read()#ret es el parametro  buleano que devuelve camera.read,este indica si el sistema pudo leer bien la camara
                              #frame el otro parametro que devuelve el cual representa el frame  en si

    if not ret:#evira que no siga el bule en caso de no leer la cam
        break

    frcounter += 1

    if frcounter % 30 == 0:
         cv2.imwrite(f"framestorage/frame_{frcounter}.jpg", frame)#me lo guarada como jpg  y ademas me lo rederige a la carpeta que quiero(en este caso framesstorage)

         print("Tienes 14 prende cam")
         cv2.imshow("Frame capturado:",frame)#con estro muestro el frame

    if cv2.waitKey(1) & 0xFF == ord("q"):#q para qitiar
                                           
        break 



