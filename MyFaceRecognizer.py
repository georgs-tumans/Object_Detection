import cv2
import os
import numpy as np

#Programm that utilizes a pretrained face recognizer to estimate whether the loaded test image contains the face it was trained to recognize
#Output is a rough estimation of probability (variable 'prediction'), gotta set a custom treshold to actually use this. THe smaller the number, the better  

def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('.\Classifiers\haarcascade_frontalface_alt.xml')
    
    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face.
    x, y, w, h = faces[0]
    #draw a rectangle around the found face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #resize the image for normal output
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    cv2.imshow("output", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h]


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("MyFaceModel.xml")

#Load a test image to see the results
test_img = cv2.imread('./FaceTrainingTestImg/me.jpg')
detectedFaceArea=detect_face(test_img)
if detectedFaceArea is None:
    print ('No face in the picture!')
else :
    prediction = face_recognizer.predict(detectedFaceArea)
    print (prediction)