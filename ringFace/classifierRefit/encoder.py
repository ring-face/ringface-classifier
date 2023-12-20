# -*- coding: utf-8 -*-
import PIL.Image
import numpy as np
from . import helpers
import os
import shutil
import json 
import logging
import sys

import face_recognition

from ringFace.ringUtils import commons
from ringFace.ringUtils import gcs





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

            logging.info(f"encoding new faces of {personName} from dir {newImagesdir}")

            for newPersonImage in os.listdir(newImagesdir):

                logging.debug(f"processing {newPersonImage}")
                try:
                    personImageFile = newImagesdir + "/" + newPersonImage
                    encoding = encodeImage(personImageFile)

                    commons.persistEncoding(encoding, encodedingsDir, newPersonImage)

                    commons.moveFileTo(personImageFile, encodedImagesDir, "File processed")
                except helpers.MultiFaceError as err:
                    logging.debug(f"MultiFaceError on {newPersonImage}")
                    commons.moveFileTo(err.filename, ignoredImagesDir, "MultiFaceError")
                except:
                    logging.error("Unexpected error:", sys.exc_info()[0])

        else:
            logging.warn(f"ignoring person {personName}")

def encodeImage(personImageFile):
    """
    Takes an image file path as input, and returns the 128 long numpy array of encoding
    """
    logging.debug(f"encodeImage {personImageFile}")

    face = load_image_file(gcs.filelike_for_read(personImageFile))
    face_bounding_boxes = face_recognition.face_locations(face)

    #If training image contains exactly one face
    if len(face_bounding_boxes) == 1:
        face_encoding = face_recognition.face_encodings(face)[0]
        return face_encoding
    else:
        logging.debug(f"{personImageFile} was skipped and can't be used for training")
        raise helpers.MultiFaceError(personImageFile)




def load_image_file(filelike, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    logging.debug(f"Loading into PIL")

    im = PIL.Image.open(filelike)
    if mode:
        logging.debug(f"Converting")
        im = im.convert(mode)
    return np.array(im)


