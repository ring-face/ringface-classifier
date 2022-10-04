import json
import logging
import os
import sys
import time
import uuid

import face_recognition
import numpy as np
import PIL.Image
from ringFace.classifierRefit import helpers

from ringFace.ringUtils import commons, clfStorage
from ringFace.ringUtils.dirStructure import DEFAULT_DIR_STUCTURE



class TextInfo:
    def __init__(self):
        self.inputFile = ""


class ImageRecognitionResult:
    def __init__(self, file):
        self.info = TextInfo()
        self.info.inputFile = file
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


def recognition(personImageFile, dirStructure = DEFAULT_DIR_STUCTURE, clf = None, fitClassifierData = None):

    result = ImageRecognitionResult(personImageFile)

    if clf is None:
        clf, fitClassifierData = clfStorage.loadLatestClassifier(dirStructure.classifierDir)

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

        knownFaceEncodings = findKnownFaceEncodings(name, fitClassifierData)
        if commons.isWithinToleranceToEncodings(encoding, knownFaceEncodings): 
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
            

    # saveResult(result, dirStructure.recogniserDir)

    return result


def findKnownFaceEncodings(name, fitClassifierData):
    for personImages in fitClassifierData['persons']:
        if personImages['personName'] == name:
            logging.debug(f"found {len(personImages['encodingsAsNumpyArray'])} known face encodings for {name}")
            return personImages['encodingsAsNumpyArray']

    logging.info(f"can not find known face encodings for {name}")
    return []


def saveResult(result, recogniserDir):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    resultDir = recogniserDir + f"/run-{timestr}"
    os.mkdir(resultDir)

    i = 1;
    for img in result.unknownPersonsImage:
        unknownPersonName = f"unknown-{i}"
        commons.saveFaceToPerson(img, resultDir, unknownPersonName)
        result.addUnknownPersonName(unknownPersonName)
        i += 1


    resultJson = result.json()


    fileHandler = open(resultDir + "/data.json", "w")
    fileHandler.write(resultJson)
    fileHandler.close()

    logging.info(f"Saved result: {resultJson} to dir: {resultDir}")

