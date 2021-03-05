#!/usr/bin/env python3

import logging

from classifierRefit.encoder import processUnencoded
from classifierRefit.fitter import fitEncodings
from classifierRefit.qaTester import testClassifier

def main(datadir):
    imageDir = datadir + "/images"
    processUnencoded(imageDir)
    fittedClassifierFile = fitEncodings(imageDir, datadir + "/classifier")
    testClassifier(fittedClassifierFile, imageDir)

logging.getLogger().setLevel(logging.INFO)
main("./data")