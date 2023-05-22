"""
author: Xinchi Huang
"""
import cv2
import numpy as np
#
# def preprocess(map):
    # img = map  # Load grayscale image directly
    # img = img.astype('uint8')
    # # Threshold the image, let's consider values close to 0 as black (adjust according to your case)
    # _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)
    #
    # # Find contours in the thresholded image
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Initialize an empty list to store the centers
    # centers = []
    #
    # # Iterate over each contour
    # for contour in contours:
    #     # Calculate moments for each contour
    #     M = cv2.moments(contour)
    #
    #     # Calculate x,y coordinate of the center of the contour
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #
    #     # Draw the center point on the image
    #     cv2.circle(img, (cX, cY), 5, (255), -1)
    #
    #     # Append the center point to the list
    #     centers.append((cX, cY))
    #
    # # Display the image with the centers of black areas
    # cv2.imshow('Centers', img)
    # cv2.waitKey(0)
    #
    # # Print out the centers
    # for center in centers:
    #     print(center)


# image1=cv2.imread("4.jpg")
# image2=cv2.imread("4s.jpg")
# cv2.imshow("before", np.concatenate([image1,image2],axis=1))
# image1=preprocess(image1)
# image2=preprocess(image2)
# cv2.imwrite("image1.jpg",image1)
# cv2.imwrite("image2.jpg",image2)
# cv2.imshow("after", np.concatenate([image1,image2],axis=1))
# cv2.waitKey(0)

img = cv2.imread('4.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image, let's consider values close to 0 as black (adjust according to your case)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize an empty list to store the centers
centers = []

# Iterate over each contour
for contour in contours:
    # Calculate moments for each contour
    M = cv2.moments(contour)

    # Calculate x,y coordinate of the center of the contour
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Draw the center point on the image
    cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)

    # Append the center point to the list
    centers.append((cX, cY))

# Display the image with the centers of black areas
cv2.imshow('Centers', img)
cv2.waitKey(0)

# Print out the centers
for center in centers:
    print(center)