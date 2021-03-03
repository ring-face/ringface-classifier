import os
import sys
import numpy as np
import json


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class MultiFaceError(Error):
    """Exception raised for more than one face in the input file.

     """

    def __init__(self, filename):
        self.filename = filename

class NoDirError(Error):
    def __init__(self, dirName):
        self.dirName = dirName    

def loadPersonEncodings(encodedingsDir):
    encodings = []
    if os.path.exists(encodedingsDir):
        
        for encodingFileName in os.listdir(encodedingsDir):
            encodingFile = encodedingsDir + "/" + encodingFileName
            print(f"Loading encoding {encodingFile}")
            encoding = loadEncoding(encodingFile)
            encodings.append(encoding)

        return encodings
    else:
        raise NoDirError(encodedingsDir)

def loadEncoding(encodingFile):

    with open(encodingFile) as json_file:
        data = json.load(json_file)
        encoding = np.asarray(data)
        return encoding
