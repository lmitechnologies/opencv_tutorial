import glob
import cv2 as cv
import numpy as np

#Function to rotate image by a certain angle
def rotate(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return rotated

#function that finds contours of image and returns bounding box coordinated of largest contour
def findcontours(image):
    #Thresholding based on a yellow color
    yellowthresh = cv.inRange(image, (0, 0, 40),(255, 255, 255))
    #Canny edge detector
    edged = cv.Canny(yellowthresh, 0, 255)
    #Finds contours in images
    (cnts, _) = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #Loops through all contours and returns largest area contour
    prevarea = 0
    for (i, c) in enumerate(cnts):
        area = cv.contourArea(c)
        if area > prevarea:
            index = i
        prevarea = area
    contour = cnts[index]

    #Finds a min area rectangle to bound the contour (angled bounding box)
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    return (rect , box)

file = 0
#Loops through every image in "images" path
for filename in glob.glob("images/*png"):
    image = cv.imread(filename)
    image = cv.resize(image, (1024, 1024))


    (rect, box) = findcontours(image.copy())
    #Draws contours on image
    #boundingbox = cv.drawContours(image.copy(), [box], 0, (0, 255, 0), 1)

    rotated = rotate(image, rect[2] - 90)

    (rect, box) = findcontours(rotated)
    #Draws contours on image
    #boundingbox = cv.drawContours(rotated.copy(), [box], 0, (0, 255, 0), 1)

    #Crop rotated image based off bounding box found
    cropped = rotated[box[0][1]:box[2][1], box[0][0]:box[2][0]]
    cropped = cv.resize(cropped, (512, 512))
    cv.imshow("Cropped", cropped)
    cv.waitKey(0)

    file += 1
    filepath = "imagesCropped-1" + "\\image-" + str(file) + ".png"
    print(filepath)
    cv.imwrite(filepath, cropped)
