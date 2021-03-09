#!/usr/bin/env python3

import logging

from ringFace.classifierRefit.encoder import processUnencoded
from ringFace.classifierRefit.fitter import fitEncodings
from ringFace.classifierRefit.qaTester import testClassifier

from ringFace.ringUtils.dirStructure import DEFAULT_DIR_STUCTURE



logging.getLogger().setLevel(logging.INFO)

processUnencoded(DEFAULT_DIR_STUCTURE.imagesDir)
fitterData = fitEncodings(DEFAULT_DIR_STUCTURE.imagesDir, DEFAULT_DIR_STUCTURE.classifierDir)
testClassifier(fitterData.fittedClassifierFile, DEFAULT_DIR_STUCTURE.imagesDir)

print(f"Result: {fitterData.json()}")
