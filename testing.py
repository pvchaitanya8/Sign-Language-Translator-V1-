# Importing the necessary libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the hand detector and the classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Set the offset and image size variables
offset = 20
imgSize = 300

# Define a list of labels
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Start the main loop
while True:
    # Read a frame from the video capture
    success, img = cap.read()
    imgOutput = img.copy()

    # Find hands in the image
    hands, img = detector.findHands(img)

    # If hands are found
    if hands:
        # Get the first hand found
        hand = hands[0]

        # Get the bounding box coordinates
        x, y, w, h = hand['bbox']

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the image to the bounding box
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Get the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Calculate the aspect ratio of the hand
        aspectRatio = h / w

        # If the aspect ratio is greater than 1, resize the width
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

            # Get the prediction and index from the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        # If the aspect ratio is less than 1, resize the height
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get the prediction and index from the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw a rectangle around the hand
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        #displaying the cropped image
        cv2.imshow("ImageCrop", imgCrop)
        # displaying the white background window
        cv2.imshow("ImageWhite", imgWhite)

    #displaying the Output
    cv2.imshow("Image", imgOutput)
    #waitning delay 1 millisec
    cv2.waitKey(1)