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
