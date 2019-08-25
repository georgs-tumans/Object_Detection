import numpy as np
import cv2
import sys

#Lai palaistu programmu - python WebCam.py/ .\Classifiers\haarcascade_frontalface_alt.xml

webcam = cv2.VideoCapture(0) #noklusētā videokamera - 0, telefons caur droid cam - 1
#webcam = cv2.VideoCapture("https://192.168.8.107:8080/video") #šis ņem konkrētā ip adresē strīmotu video (šajā gadījumā izmantojam telefonu)

cascPath = sys.argv[1]  #pasaka, ka vērtība mainīgajam tiks ņemta no komandrindā padota argumenta
#faceCascade = cv2.CascadeClassifier(cascPath) #haar face detection cascade
faceCascade = cv2.CascadeClassifier('.\Classifiers\haarcascade_frontalface_alt.xml') 
eyeCascade = cv2.CascadeClassifier('.\Classifiers\haarcascade_eye_tree_eyeglasses.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = webcam.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        square = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(square, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
        
        #Tālāk pievienota acu atpazīšana
        eyes = eyeCascade.detectMultiScale(
            square, #acis meklējam tikai atrastās sejas laukumā
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x2, y2, w2, h2) in eyes:
            squareEye = cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
            cv2.putText(squareEye, 'Eye', (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    
    # Display the resulting frame
    cv2.imshow('Web Stream',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
webcam.release()
cv2.destroyAllWindows()