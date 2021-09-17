#Trevor Dally
#Chapter 4 of Adrian Rosebrock "Practical Python and OpenCV" Walkthrough

import argparse as arg
import cv2 as cv

#Initialize argparse for command-line arguments
ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

#Using OpenCV to load image from command line and shows the imported image
image = cv.imread(args["image"])
cv.imshow("Original", image)

#Accessing the RGB values at pixel position (0,0) in BGR order and printing the values
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

#Setting pixel (0,0) to red and pring the pixel value
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

#Extracting a 100x100 pixel region from imported image and displaying it
corner = image[0:100, 0:100]
cv.imshow("Corner", corner)

#Setting corner region (100x100) of image to green and displaying it
image[0:100, 0:100] = (0, 255, 0)
cv.imshow("Updated", image)
cv.waitKey(0)