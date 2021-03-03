# -*- coding: utf-8 -*-
from . import helpers
import os
import shutil
import face_recognition

import json 



def processUnencoded(imageDir):
    """
    Processes any unencoded image in the new-images subdir of a person, and moves it into encoded-images folder.
    The encoded face is stored in the same named files in the encodings subdir of the person.
    It uses the pretrained convolutional neural network to extract the 128 dimensional encodings. 
    This model http://dlib.net/python/index.html#dlib.cnn_face_detection_model_v1 is available via the face_recognition wrapping the dlib. 
    """
    for personName in os.listdir(imageDir):
        newImagesdir=imageDir + "/" + personName + "/new-images"
        encodedImagesDir=imageDir + "/" + personName + "/encoded-images"
        encodedingsDir=imageDir + "/" + personName + "/encodings"
        ignoredImagesDir=imageDir + "/" + personName + "/ignored-images"

        if os.path.exists(newImagesdir):

            print(f"encoding new faces of {personName} from dir {newImagesdir}")

            for newPersonImage in os.listdir(newImagesdir):

                print(f"processing {newPersonImage}")
                try:
                    personImageFile = newImagesdir + "/" + newPersonImage
                    encoding = encodeImage(personImageFile)

                    persistEncoding(encoding, encodedingsDir, newPersonImage)

                    moveFileTo(personImageFile, encodedImagesDir, "File processed")
                except helpers.MultiFaceError as err:
                    print(f"MultiFaceError on {newPersonImage}")
                    moveFileTo(err.filename, ignoredImagesDir, "MultiFaceError")
                except:
                    print("Unexpected error:", sys.exc_info()[0])

        else:
            print(f"ignoring person {personName}")

def encodeImage(personImageFile):
    print(f"encodeImage {personImageFile}")

    face = face_recognition.load_image_file(personImageFile)
    face_bounding_boxes = face_recognition.face_locations(face)

    #If training image contains exactly one face
    if len(face_bounding_boxes) == 1:
        face_encoding = face_recognition.face_encodings(face)[0]
        return face_encoding
    else:
        print(f"{personImageFile} was skipped and can't be used for training")
        raise helpers.MultiFaceError(personImageFile)



def persistEncoding(encoding, encodedingsDir, newPersonImage):
    encodingFile=encodedingsDir + "/" + newPersonImage + ".json"
    print(f"saving the encoding to {encodingFile}")

    lists = encoding.tolist()
    json_str = json.dumps(lists)
    # print(json_str)

    fileHandler = open(encodingFile, "w")
    fileHandler.write(json_str)
    fileHandler.close()


def moveFileTo(file, targetDir, comment=""):
    print (f"{comment}: moving file {file} to {targetDir}")
    if os.path.isdir(targetDir) == False:
        os.makedirs(targetDir); 
    shutil.move(file, targetDir)

