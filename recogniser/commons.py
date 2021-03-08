import json
import os
import uuid


class TextInfo:
    def __init__(self):
        self.inputFile = ""


class FileRecognitionResult:
    def __init__(self, file):
        self.info = TextInfo()
        self.info.inputFile = file
        self.info.recognisedPersons = []
        self.info.unknownPersons = []
        
        self.unknownPersonsImage = [] # PIL.Image


    def addPerson(self, name):
        self.info.recognisedPersons.append(name)

    def addUnknownPersonImage(self, pilImage):
        self.unknownPersonsImage.append(pilImage)

    def addUnknownPersonName(self, filename):
        self.info.unknownPersons.append(filename)

    def json(self):
        return json.dumps(self.info.__dict__)
        

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
