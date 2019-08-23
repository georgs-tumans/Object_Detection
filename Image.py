import numpy as np
import cv2
import sys

#Lai palaistu programmu - python Image.py/ .\haarcascade_frontalface_alt.xml

# Read image from your local file system
original_image = cv2.imread('.\Images\img4.jpg')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

cascPath = sys.argv[1]  #pasaka, ka vērtība mainīgajam tiks ņemta no komandrindā padota argumenta
faceCascade = cv2.CascadeClassifier(cascPath)
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


cv2.imshow('Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Found ' + str(len(detected_faces)) + ' faces') 