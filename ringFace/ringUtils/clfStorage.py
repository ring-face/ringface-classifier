from joblib import dump
import time
import logging
import glob
import os
from joblib import load


"""
Stores the passed classifier (clf) into a binary file
Stores the passed data (fitterData) into a json
"""
def saveClassifier(clf, fitterData, classifierDir):
    clfFile = f"{classifierDir}/fitting.{fitterData.name}.dat"
    jsonFile = f"{classifierDir}/fitting.{fitterData.name}.json"

    logging.info(f"storing the fitted classifier to {jsonFile}")

    dump(clf, clfFile) 

    fitterData.fittedClassifierFile = clfFile

    jsonData = fitterData.json()
    fileHandler = open(jsonFile, "w")
    fileHandler.write(jsonData)
    fileHandler.close()


"""
Loads the latest *.dat file from the passed or standard classifier dir
Returns a sklearn.svm.SVC instance
"""
def loadLatestClassifier(classifierDir):
    list_of_files = glob.glob(f"{classifierDir}/*.dat")
    if not list_of_files:
        logging.warning("no classifier found")
        return None

    latest_fitting = max(list_of_files, key=os.path.getctime)

    logging.info(f"Loading the classifier from {latest_fitting}")
    clf = load(latest_fitting)

    return clf
