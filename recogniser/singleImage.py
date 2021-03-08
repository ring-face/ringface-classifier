import json
import logging
import os
import sys
import time
import uuid

import face_recognition
import numpy as np
import PIL.Image
from classifierRefit import helpers, storage

from . import commons


def recognition(personImageFile, recogniserDir):

    result = commons.FileRecognitionResult(personImageFile)

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

    return result



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
        commons.saveFaceToPerson(img, resultDir, unknownPersonName)
        result.addUnknownPersonName(unknownPersonName)
        i =+ 1


    resultJson = result.json()


    fileHandler = open(resultDir + "/data.json", "w")
    fileHandler.write(resultJson)
    fileHandler.close()

    logging.info(f"Saved result: {resultJson} to dir: {resultDir}")

