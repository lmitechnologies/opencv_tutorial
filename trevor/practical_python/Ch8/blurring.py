import numpy as np
import argparse as arg
import cv2 as cv

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cv.imshow("Original", image)

blurred = np.hstack([
    cv.blur(image, (3, 3)),
    cv.blur(image, (5, 5)),
    cv.blur(image, (7, 7))])
cv.imshow("Averaged", blurred)
cv.waitKey(0)

blurred = np.hstack([
    cv.GaussianBlur(image, (3, 3), 0),
    cv.GaussianBlur(image, (5, 5), 0),
    cv.GaussianBlur(image, (7, 7), 0)])
cv.imshow("Gaussian", blurred)
cv.waitKey(0)

blurred = np.hstack([
    cv.medianBlur(image, 3),
    cv.medianBlur(image, 5),
    cv.medianBlur(image, 7)])
cv.imshow("Median", blurred)
cv.waitKey(0)

blurred = np.hstack([
    cv.bilateralFilter(image, 5, 21, 21),
    cv.bilateralFilter(image, 7, 31, 31),
    cv.bilateralFilter(image, 9, 41, 41)])
cv.imshow("Bilateral", blurred)
cv.waitKey(0)