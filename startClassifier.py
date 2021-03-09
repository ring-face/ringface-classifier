#!/usr/bin/env python3

import logging

from ringFace.classifierRefit.encoder import processUnencoded
from ringFace.classifierRefit.fitter import fitEncodings
from ringFace.classifierRefit.qaTester import testClassifier

def main(datadir):
    imageDir = datadir + "/images"
    processUnencoded(imageDir)
    fittedClassifierFile = fitEncodings(imageDir, datadir + "/classifier")
    testClassifier(fittedClassifierFile, imageDir)

logging.getLogger().setLevel(logging.INFO)
main("./data")