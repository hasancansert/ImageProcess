import cv2
import numpy as np

# Load the image
image = cv2.imread('moliets.png')

# Preprocess the image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV
lower_blue = np.array([85, 60, 60])
upper_blue = np.array([130, 255, 255])

# Define the range of dark blue color in HSV
lower_dark_blue = np.array([85, 100, 0])
upper_dark_blue = np.array([130, 255, 200])

# Define the range of white color in HSV
lower_white = np.array([0, 0, 210])
upper_white = np.array([255, 30, 255])

# Create masks for blue, dark blue, and white colors
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
dark_blue_mask = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
white_mask = cv2.inRange(hsv, lower_white, upper_white)

# Combine the masks
combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(dark_blue_mask, white_mask))

# Find contours in the combined mask
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw contours on the image and calculate contour areas
count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area >= 20:
        count += 1
        detected_image = cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        


cv2.imwrite('PoolsDetected.png', detected_image)
# Display the resulting image
cv2.imshow('Pools Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Total number of pools:", count)
