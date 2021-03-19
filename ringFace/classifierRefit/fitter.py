import os
import json
import numpy as np
import time
import logging
import json

from sklearn import svm

from ringFace.ringUtils import clfStorage, commons
from . import helpers
from . import encoder

class FitterData:
    def __init__(self):
        self.name=time.strftime("%Y%m%d-%H%M%S")
        self.persons = {}
        self.fittedClassifierFile = ""

    def addPerson(self, person):
        self.persons[person] = []

    def addUsedEncoding(self, person, encodingFile):
        self.persons[person].append(encodingFile); 

    def json(self):
        return json.dumps(self.__dict__)


def fitEncodings(imagesDir, classifierDir):
    """
    Loads the 128 dimensional encodings of all faces of all persons, and fits a Support Vector Classifier.
    The classifier is then saved for further use outside of this module.
    """
    logging.info(f"processing the encodings in {imagesDir}")

    encodings = []
    encodingLabels = []

    fitterData = FitterData()

    logging.debug("loading all encoding files into memory")
    for personName in os.listdir(imagesDir):
        encodedingsDir=imagesDir + "/" + personName + "/encodings"

        if os.path.exists(encodedingsDir):
            fitterData.addPerson(personName)
            
            for encodingFileName in os.listdir(encodedingsDir):
                encodingFile = encodedingsDir + "/" + encodingFileName
                logging.debug(f"Loading encoding {encodingFile}")
                try:
                    encoding = commons.loadEncoding(encodingFile)
                    encodings.append(encoding)
                    encodingLabels.append(personName)
                    fitterData.addUsedEncoding(personName, encodingFile)

                except:
                    logging.error("Unexpected error:", sys.exc_info()[0])
        else:
            logging.warn(f"ignoring {encodedingsDir}")

    logging.debug(f"fitting {len(encodings)} encoded faces to {len(fitterData.persons)} persons")
    clf = svm.LinearSVC()
    # clf = svm.SVC(gamma='scale')
    clf.fit(encodings,encodingLabels)
    logging.debug(f"fitting finished")


    clfStorage.saveClassifier(clf, fitterData, classifierDir)

    

    return fitterData


def fitClassifier(fitClassifierRequest, dirStructure):
    logging.info(f"fitClassifier : {fitClassifierRequest}")

    encodings = []
    encodingLabels = []

    for personImages in fitClassifierRequest['persons']:
        personImages['encodings'] = []
        for imagePath in personImages['imagePaths']:
            try:
                encoding = encoder.encodeImage(imagePath)
                encodings.append(encoding)
                encodingLabels.append(personImages['personName'])
                personImages['encodings'].append(encoding.tolist())
            except helpers.MultiFaceError as err:
                logging.warn(f"MultiFaceError on {imagePath}")

    logging.debug(f"Starting the fitter with {len(encodings)} encodings of {len(fitClassifierRequest['persons'])} persons")
    clf = svm.LinearSVC()
    clf.fit(encodings,encodingLabels)
    logging.debug(f"fitting finished")

    clfStorage.saveClassifierWithRequest(clf, fitClassifierRequest, dirStructure.classifierDir)

    return None



