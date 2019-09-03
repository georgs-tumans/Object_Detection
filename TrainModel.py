import cv2
import os
import numpy as np

#This program loads an existing face recognizer model and updates it (trains it further)

trainingData='.\FaceTraining\s1'  #trainingā izmantojamo bilžu mapes lokācija

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
    
    #under the assumption that there will be only one face,
    #extract the face area
    x, y, w, h = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]



def prepareData(directory):
    faces = []
    labels = []
    subjects = ["Georgs"]
    #get the images names that are inside the given subject directory
    imageNames = os.listdir(directory)
    for img in imageNames:

            #ignore system files like .DS_Store
            if img.startswith("."):
                continue;

            imgPath = directory + "/" + img
            image = cv2.imread(imgPath)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(0)
    
    return faces, labels

print("Preparing data...")
faces, labels = prepareData(trainingData)
print("Data prepared")

face_recognizer= cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("MyFaceModel.xml")
face_recognizer.update(faces, np.array(labels))
face_recognizer.write("MyFaceModel.xml")
print ('Model updated, exiting now')