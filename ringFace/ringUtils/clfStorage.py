from io import BytesIO
import time
import logging
import glob
import os
import json
import numpy as np

from . import gcs
import joblib


# """
# Stores the passed classifier (clf) into a binary file
# Stores the passed data (fitterData) into a json
# deprecated
# """
# def saveClassifier(clf, fitterData, classifierDir):
#     clfFile = f"classifier/fitting.{fitterData.name}.dat"
#     jsonFile = f"classifier/fitting.{fitterData.name}.json"

#     logging.info(f"storing the fitted classifier to {jsonFile}")

#     dump(clf, clfFile) 

#     fitterData.fittedClassifierFile = clfFile

#     jsonData = fitterData.json()
#     fileHandler = open(jsonFile, "w")
#     fileHandler.write(jsonData)
#     fileHandler.close()


"""
Loads the latest *.dat file from the passed or standard classifier dir
Returns a sklearn.svm.SVC instance
"""
def loadLatestClassifier():


    latestJsonPath = gcs.latest_classifier()

    logging.info(f"Loading the classifier from {latestJsonPath}")
    fitClassifierData = json.load(gcs.filelike_for_read(latestJsonPath))
    parseEncodingsAsNumpyArrays(fitClassifierData)

    clfDumpFile = fitClassifierData['fittedClassifierFile']
    logging.info(f"Loading the classifier from {clfDumpFile}")
    clf = joblib.load(gcs.filelike_for_read(clfDumpFile))

    return clf, fitClassifierData

'''
copies the list of lists from fitClassifierData.persons[].encodings
into list of numpyArray in fitClassifierData.persons[].encodingsAsNumpyArray
'''
def parseEncodingsAsNumpyArrays(fitClassifierData):
    for personImages in fitClassifierData['persons']:
        personImages['encodingsAsNumpyArray'] = []
        for encodingAsList in personImages['encodings']:
            encodingAsNumpyArray = np.asarray(encodingAsList)
            personImages['encodingsAsNumpyArray'].append(encodingAsNumpyArray)



"""
Stores the passed classifier (clf) into a binary file
Stores the passed data (fitterData) into a json
"""
def saveClassifierWithRequest(clf, fitClassifierData):
    name = time.strftime("%Y%m%d-%H%M%S")
    clfFile = f"classifier/fitting.{name}.dat"
    jsonFilePath = f"classifier/fitting.{name}.json"

    logging.info(f"storing the fitted classifier to {jsonFilePath}")

    buffer = BytesIO()
    joblib.dump(clf, buffer)
    gcs.save_binary(buffer, clfFile)

    fitClassifierData['fittedClassifierFile'] = clfFile

    gcs.save_json_to_gcs(fitClassifierData, jsonFilePath)
