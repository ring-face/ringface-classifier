import os
import uuid

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