import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("image1.jpg")

if img is None:
    print("Error: Could not read the image. Check the file path.")
else:
    cv.imshow("image", img)

    # Creating the mask shape
    # The mask must be a single-channel, grayscale image
    blank = np.zeros(img.shape[:2], dtype="uint8")
    # Corrected dimensions for the rectangle
    masked_shape = cv.rectangle(blank, (500, 500), (img.shape[1] // 4, img.shape[0] // 4), 255, -1)
    
    # Creating grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray",gray)

    # Masking the image
    masked = cv.bitwise_and(img, img, mask=masked_shape)
    cv.imshow("masked", masked)
    
    # The `mask` parameter in `cv.calcHist` must be the mask itself,
    # not the masked image. The masked image is for display purposes.
    gray_hist = cv.calcHist([gray], [0], masked_shape, [256], [0, 256])

    plt.figure()
    plt.title("Masked Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256])

    plt.show()
    
    # # --- Uncommented and Fixed Color Histogram section ---
    
    plt.figure()
    plt.title("Color Histogram (BGR)")
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")
    colors = ("b", "g", "r") # OpenCV's default color order is BGR
    for i, col in enumerate(colors):
        # We calculate the histogram on the original color image
        hist = cv.calcHist([img], [i], masked_shape, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    
    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()