import os
import json
import numpy as np
import time
import logging

from sklearn import svm
from joblib import dump

def fitEncodings(imagesDir, classifierDir):
    """
    Loads the 128 dimensional encodings of all faces of all persons, and fits a Support Vector Classifier.
    The classifier is then saved for further use outside of this module.
    """
    logging.info(f"processing the encodings in {imagesDir}")

    encodings = []
    names = []
    persons = []

    logging.debug("loading all encoding files into memory")
    for personName in os.listdir(imagesDir):
        encodedingsDir=imagesDir + "/" + personName + "/encodings"

        if os.path.exists(encodedingsDir):
            persons.append(personName)
            
            for encodingFileName in os.listdir(encodedingsDir):
                encodingFile = encodedingsDir + "/" + encodingFileName
                logging.debug(f"Loading encoding {encodingFile}")
                try:
                    encoding = loadEncoding(encodingFile)
                    encodings.append(encoding)
                    names.append(personName)

                except:
                    logging.error("Unexpected error:", sys.exc_info()[0])
        else:
            logging.warn(f"ignoring {encodedingsDir}")

    logging.debug(f"fitting {len(encodings)} encoded faces to {len(persons)} persons")
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings,names)
    logging.debug(f"fitting finished")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    clfFile = classifierDir + "/fitting." + timestr
    logging.info(f"storing the fitted classifier to {clfFile}")
    storeClassifier(clf, clfFile)

    return clfFile


def loadEncoding(encodingFile):

    with open(encodingFile) as json_file:
        data = json.load(json_file)
        encoding = np.asarray(data)
        return encoding





def storeClassifier(clf, clfFile):
    dump(clf, clfFile) 


