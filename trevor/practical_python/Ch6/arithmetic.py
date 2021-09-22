import numpy as np
import cv2 as cv
import argparse as arg
import imutils

ap = arg.ArgumentParser()
ap. add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cv.imshow("Original", image)

print("max of 255: {}".format(cv.add(np.uint8([200]), np.uint8([100]))))
print("max of 0: {}".format(cv.subtract(np.uint8([50]), np.uint8([100]))))
print("wrap around: {}".format(np.uint8([200]) + np.uint8([100])))
print("wrap around: {}".format(np.uint8([50]) - np.uint8([100])))

M = np.ones(image.shape, dtype="uint8") * 100
added = cv.add(image, M)
cv.imshow("Added", added)

M = np.ones(image.shape, dtype="uint8") * 50
subtracted = cv.subtract(image, M)
cv.imshow("Subtracted", subtracted)
cv.waitKey(0)
