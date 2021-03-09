import json
import os
import uuid
import logging
import face_recognition
import numpy as np
import shutil

from ringFace.classifierRefit import helpers


        

"""
save the new face in a file structure as used by the classifier
person1/
/new-images/o3hfalsdcla.jpeg
if the directory person1 does not exist, it creates it with subdirs
"""
def saveFaceToPerson(faceImage, resultDir, personName):
    
    personDir = f"{resultDir}/{personName}"
    newImagesDir = f"{personDir}/new-images"

    if not os.path.isdir(personDir):
        os.mkdir(personDir)
        os.mkdir(newImagesDir)

    filename = "face-" + uuid.uuid4().hex + ".jpeg"

    faceImage.save(newImagesDir + "/" + filename, "JPEG")

    return filename


"""
returns true if some face in the encodingsDir is closer than the tolerance
uses euclidean distance
"""
def isWithinTolerance( encoding, encodingsDir):
    logging.debug(f"loading encodings of from {encodingsDir}")
    known_face_encodings = loadPersonEncodings(encodingsDir)

    return isWithinToleranceToEncodings(encoding, known_face_encodings)


def isWithinToleranceToEncodings(encoding, known_face_encodings):
    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
    if np.min(face_distances) > 0.5: # empirical tolerance
        return False
    else:
        return True


def loadPersonEncodings(encodedingsDir):
    encodings = []
    if os.path.exists(encodedingsDir):
        
        for encodingFileName in os.listdir(encodedingsDir):
            encodingFile = encodedingsDir + "/" + encodingFileName
            logging.debug(f"Loading encoding {encodingFile}")
            encoding = loadEncoding(encodingFile)
            encodings.append(encoding)

        return encodings
    else:
        raise helpers.NoDirError(encodedingsDir)

def loadEncoding(encodingFile):

    with open(encodingFile) as json_file:
        data = json.load(json_file)
        encoding = np.asarray(data)
        return encoding


def persistEncoding(encoding, encodingsDir, newPersonImage):
    if not os.path.isdir(encodingsDir):
        os.mkdir(encodingsDir)
    encodingFile=encodingsDir + "/" + newPersonImage + ".json"
    logging.debug(f"saving the encoding to {encodingFile}")

    lists = encoding.tolist()
    json_str = json.dumps(lists)
    # logging.debug(json_str)

    fileHandler = open(encodingFile, "w")
    fileHandler.write(json_str)
    fileHandler.close()


def moveFileTo(file, targetDir, comment=""):
    print (f"{comment}: moving file {file} to {targetDir}")
    if os.path.isdir(targetDir) == False:
        os.makedirs(targetDir); 
    shutil.move(file, targetDir)
