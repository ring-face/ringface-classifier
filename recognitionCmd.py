import glob
import os
import sys
import numpy as np
from joblib import load
import face_recognition

from classifierRefit import helpers




def recognition():

    print ('Argument List:', str(sys.argv))
    personImageFile = sys.argv[1]

    clf = loadLatestClassifier()

    imageNp = face_recognition.load_image_file(personImageFile)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(imageNp)
    no = len(face_locations)
    print("Number of faces detected: ", no)

    # Predict all the faces in the test image using the trained classifier
    for i in range(no):
        encoding = face_recognition.face_encodings(imageNp)[i]
        name = clf.predict([encoding])
        encodingsDir="./data/images/" + name[0] + "/encodings"

        if isWithinTolerance(*name, encoding, encodingsDir):
            print("Found:", *name)
        else: 
            print(f"Unknown person in the image {personImageFile}")




def loadLatestClassifier():
    list_of_files = glob.glob('./data/classifier/*')
    latest_fitting = max(list_of_files, key=os.path.getctime)

    print(f"Loading the classifier from {latest_fitting}")
    clf = load(latest_fitting)

    return clf

def isWithinTolerance(person, encoding, encodingsDir):
    print(f"loading encodings of {person} from {encodingsDir}")
    known_face_encodings = helpers.loadPersonEncodings(encodingsDir)

    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
    if np.min(face_distances) > 0.6: # empirical tolerance
        return False
    else:
        return True

imageDir = "./data/images"

recognition()