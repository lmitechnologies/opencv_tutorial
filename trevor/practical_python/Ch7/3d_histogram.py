from matplotlib import pyplot as plt
import argparse as arg
import cv2 as cv
import numpy as np

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-s", "--size", required = False, help = "Size of the largest color bin")
ap.add_argument("-b", "--bins", required = False, help = "Number of bins per color channel")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
size = float(args["size"])
bins = int(args["bins"])

hist = cv.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

print("3D histogram shape: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ratio = size / np.max(hist)
for (x, plane) in enumerate(hist):
    for (y,row) in enumerate(plane):
        for (z, col) in enumerate(row):
            if hist[x][y][z] > 0.0:
                siz = ratio * hist[x][y][z]
                rgb = (z / (bins - 1), y /(bins - 1), x / (bins - 1))
                ax.scatter(x, y, z, s = siz, facecolors = rgb)
plt.show()