import cv2
import numpy as np
from skimage import filters
from skimage import measure
from skimage.color import rgb2gray

def apply_scanner_effect(image_path, output_path):
    # Step 1: Load the image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found or unable to load.")
        return
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Use adaptive thresholding to simulate scan output (sharp black text on white)
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Step 4: Find contours to get the document's edges for correction (if required)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Apply the contour to "scan" the edges of the document
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)

    # Step 6: Invert the thresholded image to make background white and text black
    scanned_image = cv2.bitwise_not(thresholded)

    # Step 7: Save the output image
    cv2.imwrite(output_path, scanned_image)

    # Display the scanned result
    cv2.imshow('Scanned Image', scanned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
image_path = 'h.jpg'  # Replace with the path to your input image
output_path = 'scanned_output.jpg'   # Output scanned image path

apply_scanner_effect(image_path, output_path)
