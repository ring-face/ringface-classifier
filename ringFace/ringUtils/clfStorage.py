from joblib import dump
import time
import logging
import glob
import os
from joblib import load
import json
import numpy as np


# """
# Stores the passed classifier (clf) into a binary file
# Stores the passed data (fitterData) into a json
# deprecated
# """
# def saveClassifier(clf, fitterData, classifierDir):
#     clfFile = f"{classifierDir}/fitting.{fitterData.name}.dat"
#     jsonFile = f"{classifierDir}/fitting.{fitterData.name}.json"

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
def loadLatestClassifier(classifierDir):
    list_of_files = glob.glob(f"{classifierDir}/*.json")
    if not list_of_files:
        logging.warning("no classifier found")
        return None, None

    latestJsonPath = max(list_of_files, key=os.path.getctime)

    logging.info(f"Loading the classifier from {latestJsonPath}")
    with open(latestJsonPath) as json_file:
        fitClassifierData = json.load(json_file)
        parseEncodingsAsNumpyArrays(fitClassifierData)

    clfDumpFile = fitClassifierData['fittedClassifierFile']
    logging.info(f"Loading the classifier from {clfDumpFile}")
    clf = load(clfDumpFile)

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
def saveClassifierWithRequest(clf, fitClassifierData, classifierDir):
    name = time.strftime("%Y%m%d-%H%M%S")
    clfFile = f"{classifierDir}/fitting.{name}.dat"
    jsonFilePath = f"{classifierDir}/fitting.{name}.json"

    logging.info(f"storing the fitted classifier to {jsonFilePath}")

    dump(clf, clfFile) 

    fitClassifierData['fittedClassifierFile'] = clfFile

    with open(jsonFilePath, 'w') as outfile:
        json.dump(fitClassifierData, outfile)
