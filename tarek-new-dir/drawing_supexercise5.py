import numpy as np
import cv2

canvas=np.zeros((300,300,3),dtype="uint8")
red, black, green = (0,0,255), (0,0,0), (0,255,0)
i, j = 0,0
for j in range(0,330,10):
    for i in range(0,330,10):
        if (i%20 == 0):
            cv2.rectangle(canvas,(i,j),(i+10,j+10),black,-1)
        else:
            cv2.rectangle(canvas,(i,j),(i+10,j+10),red,-1) 
        j+=10  

        if (j%20 == 0):
           cv2.rectangle(canvas,(i,j),(i+10,j+10),red,-1)
        else:
            cv2.rectangle(canvas,(i,j),(i+10,j+10),black,-1)
    

(centerX,centerY) = (canvas.shape[1] //2, canvas.shape[0] //2)
cv2.circle(canvas,(centerX,centerY),30,green,-1)
cv2.imshow("Canvas",canvas)
cv2.waitKey()

