import cv2
import os

ruta = "dataset/dog/xoloitzcuintle_dog"

for archivo in os.listdir(ruta):
    img = cv2.imread(os.path.join(ruta, archivo))
    img = cv2.resize(img, (128,128))
    
    cv2.imshow("Imagen", img)
    cv2.waitKey(10)

cv2.destroyAllWindows()