# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

app = Flask(__name__)

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the hand detector and the classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("static/model/keras_model.h5", "static/model/labels.txt")

# Set the offset and image size variables
offset = 20
imgSize = 300

# Define a list of labels
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Generator function to get video frames
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Find hands in the frame
            hands, frame = detector.findHands(frame)

            if hands:
                # Get the first hand found
                hand = hands[0]

                # Get the bounding box coordinates
                x, y, w, h = hand['bbox']

                # Check if the bounding box coordinates are within the image boundaries
                if x - offset >= 0 and y - offset >= 0 and x + w + offset <= frame.shape[1] and y + h + offset <= frame.shape[0]:
                    # Crop and prepare the image for classification
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

                    # Perform classification and get the predicted label
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    
                    # Draw a rectangle around the hand
                    cv2.rectangle(frame, (x - offset, y - offset - 50),
                                  (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(frame, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (255, 0, 255), 4)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
