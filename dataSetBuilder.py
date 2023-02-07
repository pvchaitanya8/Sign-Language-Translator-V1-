# Importing required libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initializing video capture object
cap = cv2.VideoCapture(0)
# Initializing hand detector object
detector = HandDetector(maxHands=1)

# Setting the offset and image size for cropping
offset = 20
imgSize = 300

# Setting the folder path and counter for saving images
folder = "Data/"
counter = 0

# Continuously capturing video frames
while True:
    # Reading the video frame
    success, img = cap.read()
    # Detecting hands in the frame
    hands, img = detector.findHands(img)
    # If a hand is detected
    if hands:
        hand = hands[0]
        # Get the bounding box coordinates of the hand
        x, y, w, h = hand['bbox']

        # Creating a white image with the specified image size
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # Cropping the hand region from the original frame
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Getting the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Calculating the aspect ratio of the hand
        aspectRatio = h / w

        # Resizing the cropped image to the specified image size
        # Depending on the aspect ratio, either the width or height will be the limiting factor
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Displaying the cropped and resized images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Displaying the original frame
    cv2.imshow("Image", img)
    # Checking for the 'S' key press
    key = cv2.waitKey(1)
    if key == ord("S"):
        # Incrementing the counter and saving the image
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
