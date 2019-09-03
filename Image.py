import numpy as np
import cv2
import sys

#This program load an image and detects any faces in it. Basically testing different face detection algotithms within openCV

# Read image from your local file system
original_image = cv2.imread('.\Images\img4.jpg')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('.\Classifiers\haarcascade_frontalface_alt.xml')
detected_faces = faceCascade.detectMultiScale(
        grayscale_image,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

for (column, row, width, height) in detected_faces:
    cv2.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )


#resizing image so it fits nicely into the output window
scale_percent = 50 # percent of original size
width = int(original_image.shape[1] * scale_percent / 100)
height = int(original_image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(original_image, dim, interpolation = cv2.INTER_AREA) 
cv2.imshow("output", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


print('Found ' + str(len(detected_faces)) + ' faces') 