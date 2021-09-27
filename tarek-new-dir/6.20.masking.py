import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,
    help="Path to the file")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("Original",image)

mask = np.zeros(image.shape[:2], dtype= "uint8")
(cX, cY) = (image.shape[1] //2, image.shape[0]//2)
cv2.rectangle(mask,(cX - 40, cY - 40),(cX+40,cY+40),255,-1)
cv2.imshow("Mask", mask)
masked = cv2.bitwise_and(image,image,mask = mask)
cv2.imshow("Mask applied to image",masked)
cv2.waitKey(0)

mask = np.zeros(image.shape[:2], dtype = "uint8")

cv2.circle(mask,(cX,cY),50,255,-1)
masked = cv2.bitwise_and(image,image,mask = mask)
cv2.imshow("Mask",mask)
cv2.imshow("Mask applied to image",masked)
cv2.waitKey(0)
