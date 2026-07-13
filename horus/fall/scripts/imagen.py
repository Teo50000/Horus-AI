import cv2, glob, re

imgs = sorted(glob.glob("../data/CAUCAFall/Subject.1/Sit down/*.png"),
              key=lambda x: int(re.findall(r'\d+', x)[-1]))

cv2.imwrite("sit_5.png", cv2.imread(imgs[5]))     # con detección
cv2.imwrite("sit_40.png", cv2.imread(imgs[40]))   # sin detección
cv2.imwrite("sit_70.png", cv2.imread(imgs[70]))   # sin detección
cv2.imwrite("sit_100.png", cv2.imread(imgs[100])) # sin detección
cv2.imwrite("sit_150.png", cv2.imread(imgs[150])) # sin detección