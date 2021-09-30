import numpy as np
import argparse as arg
import cv2 as cv

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (11, 11), 0)
cv.imshow("Original", image)

edged = cv.Canny(blurred, 30, 150)
cv.imshow("Edges", edged)

(cnts, _) = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))

coins = image.copy()
cv.drawContours(coins, cnts, -1, (0, 255, 0), 2)
cv.imshow("Coins", coins)
cv.waitKey(0)

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv.boundingRect(c)

    print("Coin #{}".format(i + 1))
    coin = image[y:y + h, x:x + w]
    cv.imshow("Coin", coin)

    mask = np.zeros(image.shape[:2], dtype = "uint8")
    ((centerX, centerY), radius) = cv.minEnclosingCircle(c)
    cv.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv.imshow("Masked Coin", cv.bitwise_and(coin, coin, mask = mask))
    cv.waitKey(0)