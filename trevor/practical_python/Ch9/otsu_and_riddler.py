import numpy as np
import argparse as arg
import mahotas
import cv2 as cv

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(image, (5, 5), 0)
cv.imshow("Image", image)

T = mahotas.thresholding.otsu(blurred)
print("Otsu's threshold: {}".format(T))
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < T] = 0
thresh = cv.bitwise_not(thresh)
cv.imshow("Otsu", thresh)

T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < T] = 0
thresh = cv.bitwise_not(thresh)
cv.imshow("Riddler-Calvard", thresh)
cv.waitKey(0)