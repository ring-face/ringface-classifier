import os
import json
import numpy as np
import time
import logging
import json

from sklearn import svm

from ringFace.ringUtils import storage
from . import helpers

class FitterData:
    def __init__(self):
        self.persons = {}

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
                    encoding = helpers.loadEncoding(encodingFile)
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


    clfFile = storage.saveResult(clf, fitterData, classifierDir)

    

    return clfFile



