#Trevor Dally
#Chapter 5 of Adrian Rosebrock "Practical Python and OpenCV" Walkthrough

import numpy as np
import cv2 as cv

#Generate a black image of size 300 x 300
canvas = np.zeros((300, 300, 3), dtype = "uint8")

green = (0, 255, 0)
cv.line(canvas, (0, 0), (300, 300), green)
cv.imshow("Canvas", canvas)
cv.waitKey(0)

red = (0, 0, 255)
cv.line(canvas, (300, 0), (0, 300), red, 3)
cv.imshow("Canvas", canvas)
cv.waitKey(0)

cv.rectangle(canvas, (10, 10), (60, 60), green)
cv.imshow("Canvas", canvas)
cv.waitKey(0)

cv.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv.imshow("Canvas", canvas)
cv.waitKey(0)

blue = (255, 0, 0)
cv.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv.imshow("Canvas", canvas)
cv.waitKey(0)

canvas = np.zeros((300, 300, 3), dtype = "uint8")

(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

for r in range(0, 175, 25):
    cv.circle(canvas, (centerX, centerY), r, white)

cv.imshow("Canvas", canvas)
cv.waitKey(0)

for i in range(0, 25):
    radius = np.random.randint(5, high = 200)
    color = np.random.randint(0, high = 255, size = (3,)).tolist()
    pt = np.random.randint(0, high = 300, size = (2,))

    cv.circle(canvas, tuple(pt), radius, color, -1)

cv.imshow("Canvas", canvas)
cv.waitKey(0)