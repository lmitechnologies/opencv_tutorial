import glob
import cv2 as cv
import numpy as np
import math

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

    #Finds the starting point and width and height for a bounding box that bounds the contour
    (x, y, w, h) = cv.boundingRect(contour)

    return (x, y, w, h)

#function used to find the left side of the sponge and calculate the angle for reorientation
def findAngle(image):
    #Thresholding based on a yellow color
    yellowthresh = cv.inRange(image, (0, 0, 40),(255, 255, 255))
    #Initializations
    weight = 1
    TopLeft = yellowthresh.shape[1] * weight
    BottomLeft = 0
    #For loop through every pixel in the image to find top left and bottom left point of sponge
    for i in range(yellowthresh.shape[1]):
        for j in range(yellowthresh.shape[0]):
            if yellowthresh[i, j] == 255:
                pointT = i*weight + j
                pointB = i*weight - j
                if pointT < TopLeft:
                    TopLeftpoint = [j, i]
                    TopLeft = pointT
                if pointB > BottomLeft:
                    BottomLeftpoint = [j, i]
                    BottomLeft = pointB
    x = BottomLeftpoint[0] - TopLeftpoint[0]
    y = BottomLeftpoint[1] - TopLeftpoint[1]
    angle = math.atan(y/x)*180/3.14159
    return angle

file = 0
#Loops through every image in "images" path
for filename in glob.glob("images/*png"):
    image = cv.imread(filename)
    image = cv.resize(image, (1024, 1024))

    #Finds the angle of the sponge and rotates the image
    angle = findAngle(image)
    rotated = rotate(image, angle - 90)

    #finds contour of sponge after image rotation
    (x, y, w, h) = findcontours(rotated)
    boundingbox = cv.rectangle(rotated.copy(), (x,y), (x + w, y + h), (0, 255, 0), 1)

    #Crops image based on bounding box found and resizes image
    cropped = rotated[y:y + h, x:x + w]
    cropped = cv.resize(cropped, (512, 512))
    cv.imshow("Cropped", cropped)
    cv.waitKey(0)


    #Saves image to "imagesCropped-2" folder
    file += 1
    filepath = "imagesCropped-2" + "\\image-" + str(file) + ".png"
    print(filepath)
    cv.imwrite(filepath, cropped)