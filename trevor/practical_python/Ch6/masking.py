import numpy as np
import cv2 as cv
import argparse as arg
import imutils

ap = arg.ArgumentParser()
ap. add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cv.imshow("Original", image)

mask = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv.rectangle(mask, (cX - 75, cY - 75), (cX + 75, cY + 75), 255, -1)
cv.imshow("Mask", mask)

masked = cv.bitwise_and(image, image, mask = mask)
cv.imshow("Mask Applied to Image", masked)
cv.waitKey(0)

mask = np.zeros(image.shape[:2], dtype = "uint8")
cv.circle(mask, (cX, cY), 100, 255, -1)
masked = cv.bitwise_and(image, image, mask = mask)
cv.imshow("Mask", mask)
cv.imshow("Mask Applied to Image", masked)
cv.waitKey(0)