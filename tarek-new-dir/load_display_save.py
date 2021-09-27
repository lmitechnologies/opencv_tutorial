from __future__ import print_function
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, 
    help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))

#TFA - remember height is image.shape[0] and width is image.shape[1]

cv2.imshow("Image", image)
cv2.waitKey()

cv2.imwrite("newimage.jpg", image)
