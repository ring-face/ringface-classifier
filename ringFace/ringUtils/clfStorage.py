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
    timestr = time.strftime("%Y%m%d-%H%M%S")
    clfFile = f"{classifierDir}/fitting.{timestr}.dat"
    jsonFile = f"{classifierDir}/fitting.{timestr}.json"

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
    latest_fitting = max(list_of_files, key=os.path.getctime)

    logging.info(f"Loading the classifier from {latest_fitting}")
    clf = load(latest_fitting)

    return clf
