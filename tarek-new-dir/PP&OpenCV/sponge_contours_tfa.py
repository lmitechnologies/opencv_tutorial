import numpy as np
import cv2
import argparse
import glob

def auto_Canny(image,sigma=0.33): #sigma of 0.33 is a good starting point as per Dr. AR
    v = np.median(image) #median

    lower = int(max(0, (1.0 - sigma) * v)) #calculating lower threshold from median
    upper = int(min(255, (1.0 + sigma) * v)) #calculating upper threshold from median
    edged_image = cv2.Canny(image,lower,upper)
    return edged_image

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path to input dataset of images") 
args = vars(ap.parse_args())

#for imagePath in glob.glob(args["images"] + "*/.png"):
image=cv2.imread(args["image"])
r = 500.0/image.shape[0]
dim = (int(image.shape[1] *r), 500) #retaining aspect ratio
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #resizing as sponge image is very large
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
blurred = cv2.GaussianBlur(gray, (3,3), 0)

#wide = cv2.Canny(blurred,10,200)
#tight = cv2.Canny(blurred,225,250)
auto = auto_Canny(blurred) #returns edges

(cnts,_) = cv2.findContours(auto.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #edged image required for finding contours
print("I count {} contours in this image".format(len(cnts))) #len(cnts) will get us total number of contours

cv2.imshow("Oringal", resized)
cv2.imshow("Edges",auto)
cv2.waitKey(0)

maxArea = cv2.contourArea(cnts[0])
maxPos = 0
for i in range(len(cnts)):
    if maxArea < cv2.contourArea(cnts[i]):
        maxArea = cv2.contourArea(cnts[i])
        maxPos = i

cnt_largest = cnts[maxPos]

resizedCopy1 = cv2.cvtColor(blurred.copy(),cv2.COLOR_GRAY2BGR)
cv2.drawContours(resizedCopy1, cnt_largest, maxPos , (0,255,0),2)
cv2.imshow("Sponge with largest contour",resizedCopy1)
cv2.waitKey(0)

#cntarea = cv2.contourArea(cnts)
#print("Area is :".format(cntarea))

x,y,w,h = cv2.boundingRect(cnts[maxPos])
strBRect = resizedCopy1[y:y +h,x:x+w]
cv2.rectangle(strBRect,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Bounded Rectangle1",strBRect)
cv2.waitKey(0)

rotBRect = cv2.minAreaRect(cnts)
box = cv2.boxPoints(rotBRect)
box = np.int0(box)
cv2.drawContours(rotBRect,cnt_largest,maxPos,(0,0,255),2)
cv2.imshow("Min area box",rotBRect)
cv2.waitKey(0)



#for (i,c) in enumerate(cnts):
#    (x,y,w,h) = cv2.boundingRect(c)

   # if(max(cv2.contourArea(cnt))
   # print("Coin #{}".format(i+1))
    #coin = image[y:y +h,x:x+w]
    #cv2.imshow("Coin",coin)

    #mask = np.zeros(image.shape[:2], dtype = "uint8")
    #((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    #cv2.circle(mask, (int(centerX), int(centerY)),int(radius),int(radius),255,-1)
    #mask = mask[y:y+h,x:x+w]
    #cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask =mask))
    #cv2.waitKey(0)


