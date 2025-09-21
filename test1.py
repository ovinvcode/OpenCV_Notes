import cv2 as cv

#to show images
#create cv.imread variable
#use imshow to show the frame
img = cv.imread("photos/cat.jpg")
cv.imshow("cat", img)

cv.waitKey(0)




#to show video
#use 0,1,2 to get live cam feeds 
# use vid file path to get videos

capture = cv.VideoCapture(0)

#using a while loop it loops everyframe and displays it
while True:
    status, frame = vid.read()

    cv.imshow("Window Title like Video", frame)

    ## waits 20 milsecs and then checks the key "q" to exit
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

#after capturing releases the vid capture
#and close all windows
capture.release()
cv.destroyAllWindows()




#To rescale frames 
# vids, images, and live vids anything
#frame.shape[],  0=height, 1=width
def RescaleFrame(frame,scale=0.75):

    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (height,width)

    resized_frame = cv.resize(frame, dimensions, inerpolation = cv.INTER_AREA)

    return resized_frame
#insert this function inside while loop for frame to resize image
#for vids
resized_frame = RescaleFrame(frame,0.2)
#for images
resized_img = RescaleFrame(img,0.5)
#then
cv.imshow("window title", resized_frame)




#To rescale live vid u can also use
def changeRes(width,height):

    #3,4 is the wid and hei properties of the capture
    capture.set(3,width)
    capture.set(4,height)



## to createshapes and texts

import cv2 as cv
import numpy as np

## create blank canvas 
blank = np.zeros((500,500,3), dtype="uint8")
# here the (500,500) is the size and 3 represent the colors
cv.imshow("Blank", blank)

#to colorize the blank (ex: RED)
blank[:] = 0,0,255
cv.imshow("Red", blank)

#to color only a section (ex:GREEN)
blank[200:300, 100:200] = 0,255,0
cv.imshow("Green", blank)


##to create shapes!!
#thickness = -1 mean fill the image
#1. Rectangle
cv.rectangle(blank, (50,50), (400,400), (255,0,0), thickness=3)
cv.imshow("Rectangle", blank)
# you can use img.shape[0]and[1] as height and width numbers

#2. Circle
cv.circle(blank, (250,250), 50, (255,255,255), thickness=3)
cv.imshow("Circle", blank)

#3. Line
cv.line(blank, (0,0), (500,500), (255,0,255), thickness=4)
cv.imshow("line", blank)

## Writing texts!!
cv.putText(blank, "Hello world", (0,250) ,cv.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), thickness=2)
cv.imshow("text", blank)


##filters

#create img then

#1. grayscale
gray = cv.cvtColor(img, cv.COLOR_BAYER_BG2GRAY)

#2. blur
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
#(3,3) < (7,7) more blur

#3. Edge Cascade
canny = cv.Canny(img, 120,120)
#if you want to detect less edge use a blurred img instead of img

#4. Dilating the Image
dilated = cv.dilate(img, (5,5), iterations=3)

#5. Eroding
eroded = cv.eroded(dilated, (5,5), iterations=3)


## image transformation

## Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_LINEAR)
cv.imshow("resized", resized)

## Cropping
cropped = img[50:250, 200:400]
cv.imshow("cropped", cropped)

## translating image
def translate(img, x, y):
    """
    x and y is the directional shift in pixels

    -x = left
    -y = up
    x = right
    y = down

    """
    transMat = np.float32([1,0,x],[0,1,y])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

translated = translate(img,100,100)
cv.imshow("translate", translated)

## rotation
def rotate(img, angle, rotPoint = None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img,rotMat, dimensions)

rotated = rotate(img, 45)
# when u rotate it will introduce black borders to cut off the image to rotate

## flip
# 0 = ver
# 1 = hor
# -1 = hor and ver
flipped = cv.flip(img, 0)
cv.imshow("flipped", flipped)


##color spaces

# opencv default uses BGR foramt
# but we usually use RGB
# we need to inverse the BGR to rGB

# RGB HSV Grayscale ... are color spaces
# array of same type color or smthng
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

