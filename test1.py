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
    status, frame = capture.read()

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

# RGB HSV Grayscale LAB ... are color spaces
# array of same type color or smthng

#BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

#BRG to RGB 
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#cant convert from gray to hsv directly so first make into bgr then to hsv..



## color channels

# basically red, green, blue channels
# images are these 3 channels

# 1 to split image into colors
img = cv.iimread("photo")

b,g,r = cv.split(img)

cv.imshow("blue", b)
#this will show the blue parts as white and non blue as black in a grayscale photo

# 2 to get merge images
merged = cv.merge(b,g,r)
cv.imshow("merged", merged)


# 3 to get the split grayscale images to colorized splits 

# first create a blank img
blank = np.zeros(cv.shape[:2], dtype="uint8") 

b,g,r = cv.split(img)
blue = cv.merge([b,blank,blank])
#this will show the blue split in blue color

## BLUR

# u ned a kernel to blur
# a kernel is a grid basically that takes the average value to get the blur
# 3,3 means a 3x3 grid
# bigger kernel grid more blur

## 1. Average blur
blur = cv.Blur(img, (3,3), cv.BORDER_DEFAULT)

# 2. Gaussian blur 
# this method takes the values and gives them weights and give more of a natural blur
g_blur = cv.GaussianBlur(img, (3,3), 0)

# 3. Median blur
# use low values 
g_blur = cv.medianBlur(img, 3)

# 4. Bilateral Blur
# uses a radius smthng to determine values
bilat = cv.bilateralFilter(img, 10, 20, 20)



## BITWISE OPERATIONS

blank = np.zeros((500,500),dtype="uint8")

rectangle = cv.rectangle(blank, (0,0),(250,250), thickness=-1)
circle = cv.circle(blank, (250,250), 50, thickness=-1)

# AND of images basically
# intersection like
bitwise_and = cv.bitwise_and(rectangle,circle)

# OR of images basically
# merge like
bitwise_or = cv.bitwise_or(rectangle,circle)

# XOR of images
# non contacting aareas
bitwise_xor = cv.bitwise_xor(rectangle,circle)

# NOT of image
bitwise_not = cv.bitwise_not(rectangle,circle)


## MASKING 

#get the image then

#create blank to put mask shape
blank = np.zeros(img.shape[:2], dtype="uint8")

#the shape you want the mask to be
mask_shape = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2), 50, 255, thickness=-1)

# merge the img and mask to get mask
masked = cv.bitwise_and(img,img,mask=mask_shape)
cv.imshow("masked image",masked)



## Histogram
# Step 1: Import necessary libraries
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the image
img = cv.imread("image1.jpg")

# Step 3: Validate the image loading
# If the path is incorrect or the file is corrupted, cv.imread() returns None.
if img is None:
    print("Error: Could not read the image. Check the file path.")
else:
    # Display the original image in a window titled "image".
    cv.imshow("image", img)

    # Step 4: Create a mask to define a Region of Interest (ROI)
    # -----------------------------------------------------------
    # A mask is a grayscale image where the white area defines the region we want to focus on, and the black area is ignored.
    # First, create a blank black image with the same dimensions as the original image.
    # img.shape[:2] gives the height and width of the image.
    # 'dtype="uint8"' specifies the data type for an 8-bit unsigned integer (0-255).
    blank = np.zeros(img.shape[:2], dtype="uint8")

    # Now, draw a white rectangle on the blank image. This white area is our mask.
    # - color: 255 (white)
    # - thickness: -1 fills the rectangle.
    # Here, we are defining a rectangle in a specific area.
    masked_shape = cv.rectangle(blank, (100, 100), (400, 400), 255, -1)
    cv.imshow("Mask", masked_shape)
    
    # Step 5: Convert the original image to grayscale
    # -----------------------------------------------
    # Histograms are often calculated on grayscale images to analyze pixel intensity.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Grayscale Image", gray)

    # Step 6: Apply the mask to the image (for visualization)
    # ----------------------------------------------------------
    # cv.bitwise_and performs a bitwise AND operation between the image and itself,
    # but only in the region defined by the mask. This effectively crops out
    # everything outside the white area of the mask.
    # This step is primarily for displaying what part of the image we are analyzing.
    masked_color = cv.bitwise_and(img, img, mask=masked_shape)
    cv.imshow("Masked Color Image", masked_color)

    # Step 7: Calculate and plot the histogram for the masked grayscale image
    # -----------------------------------------------------------------------
    # cv.calcHist([images], [channels], mask, [histSize], [ranges])
    # - [gray]: The source image (must be in a list).
    # - [0]: The channel index to compute the histogram for (0 for grayscale).
    # - masked_shape: The mask. The histogram is only computed for pixels where the mask is non-zero.
    # - [256]: The number of bins in the histogram (0 to 255).
    # - [0, 256]: The range of pixel values.
    gray_hist = cv.calcHist([gray], [0], masked_shape, [256], [0, 256])

    plt.figure() # Creates a new figure for plotting.
    plt.title("Masked Grayscale Histogram")
    plt.xlabel("Bins (Pixel Intensity)")
    plt.ylabel("# of Pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256]) # Set the x-axis limits to match the pixel value range.
    plt.show() # Display the plot.
    
    # Step 8: Calculate and plot the color histogram for the masked image
    # -------------------------------------------------------------------
    plt.figure()
    plt.title("Masked Color Histogram (BGR)")
    plt.xlabel("Bins (Pixel Intensity)")
    plt.ylabel("# of Pixels")
    
    # Define the colors for plotting. OpenCV uses BGR order by default, not RGB.
    colors = ("b", "g", "r") 
    
    # Loop through each color channel (Blue, Green, Red).
    # enumerate provides both the index (i) and the value (col).
    for i, col in enumerate(colors):
        # Calculate the histogram for the current channel (i) of the original color image.
        # We use the same mask to analyze the same region.
        hist = cv.calcHist([img], [i], masked_shape, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    
    plt.show()

    # Step 9: Clean up
    # ----------------
    # cv.waitKey(0) waits indefinitely for a key press. This keeps the
    # OpenCV windows open until you press any key.
    cv.waitKey(0)
    # cv.destroyAllWindows() closes all the windows created by OpenCV.
    cv.destroyAllWindows()