import logging
import os
import sys
import cv2
from multiprocessing import Pool
import face_recognition
import PIL.Image



from classifierRefit import helpers, storage

from . import commons


class PersonData:
    def __init__(self):
        self.encodings = []
        self.images = []
        self.recognisedPersons = []

class VideoRecognitionData:
    def __init__(self, videoFile):
        self.persons = {}
        self.videoFile = videoFile

    def addToPerson(self, name, image, encoding):
        if name not in self.persons:
            self.persons[name] = PersonData()

        self.persons[name].encodings.append(encoding)
        self.persons[name].images.append(image)

    def getPersons(self):
        return self.persons.keys

    def addRecognisedPerson(self, name):
        self.addRecognisedPerson.append(name)

    def findSimilarPerson(self, encoding):
        for name, personData in self.persons.items():
            if commons.isWithinToleranceToEncodings(encoding, personData.encodings):
                return name

        return None


def recognition(videoFile, recogniserDir):

    personCounter = 1

    logging.info(f"processing input video {videoFile}")
    result = VideoRecognitionData(videoFile)

    clf = storage.loadLatestClassifier()

    input_movie = cv2.VideoCapture(videoFile)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counter = 0
    while True:
        frame_got, frame = input_movie.read()
        frame_counter += 1
        
        if not frame_got:
            break
        
        logging.debug(f"recognition on frame {frame_counter}")


        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        image = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image)

        no = len(face_locations)
        logging.debug(f"Number of faces detected: {no}")

        for i in range(no):
            encoding = encodings[i]
            name = clf.predict([encoding])
            encodingsDir="./data/images/" + name[0] + "/encodings"

            if commons.isWithinTolerance(encoding, encodingsDir):
                logging.info(f"Recognised: {name}")
                result.addRecognisedPerson(name[0])
            else: 
                top, right, bottom, left = face_locations[i]
                logging.debug("The unknown face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                thumbnail = image[top:bottom, left:right]
                pilThumbnail = PIL.Image.fromarray(thumbnail)
                if logging.getLogger().level == logging.DEBUG:
                    pilThumbnail.show()

                similarPerson = result.findSimilarPerson(encoding)
                if similarPerson is not None:
                    logging.debug(f"{similarPerson} in the frame")
                    result.addToPerson(similarPerson, pilThumbnail, encoding)
                else: 
                    newPersonName = f"unknown-{personCounter}"
                    personCounter += 1
                    logging.info(f"New {newPersonName} in the frame")
                    result.addToPerson(newPersonName, pilThumbnail, encoding)





    

