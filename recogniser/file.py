import glob
import os
import sys
import numpy as np
import logging
import json
import PIL.Image

from joblib import load
import face_recognition

from classifierRefit import helpers



class FileRecognitionResult:
    def __init__(self, file):
        self.personImageFile = file
        self.recognisedPersons = [] # strings
        self.unknownPersons = [] # PIL.Image




def recognition(personImageFile, recogniserDir):

    result = FileRecognitionResult(personImageFile)

    clf = loadLatestClassifier()

    # imageNp = face_recognition.load_image_file(personImageFile)

    # image = PIL.Image.open(personImageFile)
    # image = image.convert('RGB')
    # imageNp = np.array(image)

    image = face_recognition.load_image_file(personImageFile)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(image)
    no = len(face_locations)
    logging.debug(f"Number of faces detected: {no}")
    encodings = face_recognition.face_encodings(image)

    # Predict all the faces in the test image using the trained classifier
    for i in range(no):
        encoding = encodings[i]
        name = clf.predict([encoding])
        encodingsDir="./data/images/" + name[0] + "/encodings"

        if isWithinTolerance(*name, encoding, encodingsDir):
            logging.info(f"Recognised: {name}")
            result.recognisedPersons.append(*name)
        else: 
            logging.info(f"Unknown person in the image {personImageFile}")
            top, right, bottom, left = face_locations[i]
            logging.debug("The unknown face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            thumbnail = image[top:bottom, left:right]
            pilThumbnail = PIL.Image.fromarray(thumbnail)
            # result.unknownPersons.append(pilThumbnail)
            if logging.getLogger().level == logging.DEBUG:
                pilThumbnail.show()
            

    saveResult(result, recogniserDir)


def loadLatestClassifier():
    list_of_files = glob.glob('./data/classifier/*')
    latest_fitting = max(list_of_files, key=os.path.getctime)

    logging.info(f"Loading the classifier from {latest_fitting}")
    clf = load(latest_fitting)

    return clf

def isWithinTolerance(person, encoding, encodingsDir):
    logging.debug(f"loading encodings of {person} from {encodingsDir}")
    known_face_encodings = helpers.loadPersonEncodings(encodingsDir)

    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
    if np.min(face_distances) > 0.6: # empirical tolerance
        return False
    else:
        return True

def saveResult(result, recogniserDir):
    resultJson = json.dumps(result.__dict__)
    logging.info(f"Saving result: {resultJson}")