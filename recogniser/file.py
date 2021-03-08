import os
import sys
import numpy as np
import logging
import json
import time
import uuid
import PIL.Image

import face_recognition

from . import helper
from classifierRefit import storage

from classifierRefit import helpers

class TextInfo:
    def __init__(self):
        self.personImageFile = ""


class FileRecognitionResult:
    def __init__(self, file):
        self.info = TextInfo()
        self.info.personImageFile = file
        self.info.recognisedPersons = []
        self.info.unknownPersons = []
        
        self.unknownPersonsImage = [] # PIL.Image


    def addPerson(self, name):
        self.info.recognisedPersons.append(name)

    def addUnknownPersonImage(self, pilImage):
        self.unknownPersonsImage.append(pilImage)

    def addUnknownPersonName(self, filename):
        self.info.unknownPersons.append(filename)


        

    def json(self):
        return json.dumps(self.info.__dict__)



def recognition(personImageFile, recogniserDir):

    result = FileRecognitionResult(personImageFile)

    clf = storage.loadLatestClassifier()

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
            result.addPerson(name[0])
        else: 
            logging.info(f"Unknown person in the image {personImageFile}")
            top, right, bottom, left = face_locations[i]
            logging.debug("The unknown face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            thumbnail = image[top:bottom, left:right]
            pilThumbnail = PIL.Image.fromarray(thumbnail)
            result.addUnknownPersonImage(pilThumbnail)
            if logging.getLogger().level == logging.DEBUG:
                pilThumbnail.show()
            

    saveResult(result, recogniserDir)



def isWithinTolerance(person, encoding, encodingsDir):
    logging.debug(f"loading encodings of {person} from {encodingsDir}")
    known_face_encodings = helpers.loadPersonEncodings(encodingsDir)

    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
    if np.min(face_distances) > 0.5: # empirical tolerance
        return False
    else:
        return True

def saveResult(result, recogniserDir):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    resultDir = recogniserDir + f"/run-{timestr}"
    os.mkdir(resultDir)

    i = 0;
    for img in result.unknownPersonsImage:
        unknownPersonName = f"unknown-{i}"
        helper.saveFaceToPerson(img, resultDir, unknownPersonName)
        result.addUnknownPersonName(unknownPersonName)
        i =+ 1


    resultJson = result.json()


    fileHandler = open(resultDir + "/data.json", "w")
    fileHandler.write(resultJson)
    fileHandler.close()

    logging.info(f"Saved result: {resultJson} to dir: {resultDir}")

