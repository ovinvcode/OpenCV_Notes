# OpenCV Basics: Images, Videos, and Rescaling

This guide covers the fundamental operations in OpenCV for handling images and videos, as well as a function for rescaling frames.

## Displaying an Image

To display a static image, you first need to read the image file from its path and then use `imshow` to display it in a window.

-   `cv.imread("path/to/image.jpg")`: Loads an image from the specified file.

-   `cv.imshow("Window Title", img)`: Displays the image in a window with the given title.

-   `cv.waitKey(0)`: Waits indefinitely for a key press. This is crucial to prevent the window from closing immediately.

```
import cv2 as cv

# Read the image file
img = cv.imread("photos/cat.jpg")

# Display the image in a window named "cat"
cv.imshow("cat", img)

# Wait for any key to be pressed before closing the window
cv.waitKey(0)

```

## Displaying a Video

Displaying a video involves capturing the video source (either a file or a live camera feed) and then looping through each frame to display it.

-   `cv.VideoCapture(0)`: Creates a video capture object. Use `0`, `1`, etc., for live webcam feeds or a file path string (e.g., `"videos/dog.mp4"`) for a video file.

-   `vid.read()`: Reads the next frame from the capture. It returns a status (`True` if successful) and the frame itself.

-   `cv.waitKey(20) & 0xFF == ord('q')`: This line waits for 20 milliseconds. If the 'q' key is pressed during this time, the loop breaks.

-   `vid.release()`: Releases the video capture object.

-   `cv.destroyAllWindows()`: Closes all OpenCV windows.

```
import cv2 as cv

# Create a VideoCapture object for the default webcam
vid = cv.VideoCapture(0)

# Loop through each frame of the video
while True:
    # Read the current frame
    status, frame = vid.read()

    # Display the frame in a window
    cv.imshow("Window Title like Video", frame)

    # Exit the loop if the 'q' key is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
vid.release()
cv.destroyAllWindows()

```

## Rescaling a Frame (Image or Video)

You can resize a frame to make it smaller or larger. This function works for both individual images and video frames.

-   `frame.shape`: Returns a tuple of (height, width, channels).

-   `cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)`: Resizes the image. `cv.INTER_AREA` is generally recommended for shrinking images.

```
import cv2 as cv

def RescaleFrame(frame, scale=0.75):
    """
    Resizes a frame to a specific scale.
    """
    # Calculate the new dimensions
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height) # Note: width comes first for cv.resize

    # Resize the frame using the new dimensions
    resized_frame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    return resized_frame

# Example Usage (within a video loop):
# while True:
#     status, frame = vid.read()
#     resized_video_frame = RescaleFrame(frame, scale=0.5)
#     cv.imshow("Resized Video", resized_video_frame)
#     ...

# Example Usage (with an image):
# img = cv.imread("photos/cat.jpg")
# resized_image = RescaleFrame(img, scale=0.5)
# cv.imshow("Resized Cat", resized_image)
# cv.waitKey(0)

```
