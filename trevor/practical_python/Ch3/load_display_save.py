import cv2 as cv
import argparse as arg

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
print("wdith: {} pixels".format(image.shape[1]))
print("length: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))

cv.imshow("Image", image)
cv.waitKey(0)

cv.imwrite("newimage.jpg", image)