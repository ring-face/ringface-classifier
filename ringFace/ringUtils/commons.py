import json
import os
import uuid
import logging
import face_recognition
import numpy as np

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


def isWithinTolerance( encoding, encodingsDir):
    logging.debug(f"loading encodings of from {encodingsDir}")
    known_face_encodings = helpers.loadPersonEncodings(encodingsDir)

    return isWithinToleranceToEncodings(encoding, known_face_encodings)

def isWithinToleranceToEncodings(encoding, known_face_encodings):
    face_distances = face_recognition.face_distance(known_face_encodings, encoding)
    if np.min(face_distances) > 0.5: # empirical tolerance
        return False
    else:
        return True