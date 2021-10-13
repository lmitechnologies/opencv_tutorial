import numpy as np
import cv2
import argparse
import glob

def auto_Canny(image,sigma=0.33):
    v = np.median(image) #median

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged_image = cv2.Canny(image,lower,upper)
    return edged_image

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path to input dataset of images")
args = vars(ap.parse_args())

#for imagePath in glob.glob(args["images"] + "*/.png"):
image=cv2.imread(args["image"])
r = 500.0/image.shape[0]
dim = (int(image.shape[1] *r), 500)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3,3), 0)

#wide = cv2.Canny(blurred,10,200)
#tight = cv2.Canny(blurred,225,250)
auto = auto_Canny(blurred) #returns edges

(cnts,_) = cv2.findContours(auto.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count {} contours in this image".format(len(cnts)))

cnt = cnts[0]


cv2.imshow("Oringal", resized)
cv2.imshow("Edges",np.hstack([auto]))
cv2.waitKey(0)

resizedCopy1 = blurred.copy()
cv2.drawContours(resizedCopy1, cnts, -1, (0,255,0),2)
cv2.imshow("Sponge",resizedCopy1)
cv2.waitKey(0)

#cntarea = cv2.contourArea(cnts)
#print("Area is :".format(cntarea))

x,y,w,h = cv2.boundingRect(cnts)
cv2.rectangle(resizedCopy1,(x,y),(x+w,y+h),(0,255,0),2)
strBRect = resizedCopy1[y:y +h,x:x+w]
cv2.imshow("Bounded Rectangle1",strBRect)
cv2.waitKey(0)

rotBRect = cv2.minAreaRect(cnts)
box = cv2.boxPoints(rotBRect)
box = np.int0(box)
cv2.drawContours(resizedCopy1,[box],0,(0,0,255),2)
cv2.imshow("Min area box",resizedCopy1)
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


