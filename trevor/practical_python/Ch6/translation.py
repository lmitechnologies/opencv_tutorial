import numpy as np
import cv2 as cv
import argparse as arg
import imutils

ap = arg.ArgumentParser()
ap. add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cv.imshow("Original", image)

M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv.imshow("Shifted Down and Right", shifted)

M = np.float32([[1, 0, -50],[0, 1, -90]])
shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv.imshow("Shifted Up and Left", shifted)


shifted = imutils.translate(image, 0, 100)
cv.imshow("Shifted Down", shifted)
cv.waitKey(0)
