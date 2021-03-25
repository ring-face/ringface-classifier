import logging
import os
import sys
import time
import json
import cv2
import uuid
from multiprocessing import Pool
import face_recognition
import PIL.Image

from ringFace.classifierRefit import helpers

from ringFace.ringUtils import commons, clfStorage
from ringFace.ringUtils.dirStructure import DEFAULT_DIR_STUCTURE



EACH_FRAME=3
MAX_FRAMES=100
STOP_AFTER_EMPTY_FREAMES=10
MIN_TRUMBNAIL_SIZE_IN_PX=120

class PersonData:
    def __init__(self):
        self.encodings = []
        self.images = []
        self.imagePaths = []

class VideoRecognitionData:
    def __init__(self, videoFile):
        self.persons = {}
        self.videoFile = videoFile
        self.recognisedPersons = set()
        self.eventName = None

    def addImageFilePathToPerson(self, filePath, personName):
        self.persons[personName].imagePaths.append(filePath)


    def addToPerson(self, name, image, encoding):
        if name not in self.persons:
            self.persons[name] = PersonData()

        self.persons[name].encodings.append(encoding)
        self.persons[name].images.append(image)

    def getPersons(self):
        return self.persons.keys

    def addRecognisedPerson(self, name):
        self.recognisedPersons.add(name)

    def findSimilarPerson(self, encoding):
        for name, personData in self.persons.items():
            if commons.isWithinToleranceToEncodings(encoding, personData.encodings):
                return name

        return None

    def json(self):
        export = {}
        export['videoFile'] = self.videoFile
        export['eventName'] = self.eventName
        export['recognisedPersons'] = list(self.recognisedPersons)
        export['unknownPersons'] = []
        for unknownPerson, personData in self.persons.items():
            export['unknownPersons'].append(
                {"name": unknownPerson, 
                "images": len(personData.images),
                "imagePaths": personData.imagePaths
            })

        return json.dumps(export)



def recognition(videoFile, dirStructure = DEFAULT_DIR_STUCTURE, clf = None, fitClassifierData = None, ringEvent= None):

    personCounter = 1

    logging.info(f"processing input video {videoFile}")
    result = VideoRecognitionData(videoFile)

    if clf is None:
        clf, fitClassifierData = clfStorage.loadLatestClassifier(dirStructure.classifierDir)

    input_movie = cv2.VideoCapture(videoFile)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Total frames: {length}")

    frame_counter = 0
    noFaceFrameCounter = 0

    while True:
        
        frame_got, frame = input_movie.read()
        frame_counter += 1
        
        if not frame_got:
            break
        
        if frame_counter > MAX_FRAMES:
            logging.warn(f"will not consider more than firt {MAX_FRAMES} frames")
            break

        if frame_counter % EACH_FRAME != 0:
            logging.debug(f"skippping frame {frame_counter}")
            continue


        logging.debug(f"recognition on frame {frame_counter}")


        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        image = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(image)
        facesCount = len(face_locations)
        logging.debug(f"Number of faces detected: {facesCount}")

        # stop after couple of empty frames
        if facesCount == 0:
            noFaceFrameCounter += 1
            if noFaceFrameCounter >= STOP_AFTER_EMPTY_FREAMES:
                logging.warn(f"noFaceFrameCounter: {noFaceFrameCounter}. Stopping")
                break
            else:
                continue

        encodings = face_recognition.face_encodings(image)
        noFaceFrameCounter = 0


        for i in range(facesCount):
            encoding = encodings[i]
            
            #process the recognised face
            if clf is not None:
                name = clf.predict([encoding])
                # encodingsDir=dirStructure.imagesDir + "/" + name[0] + "/encodings"

                # if commons.isWithinTolerance(encoding, encodingsDir):
                knownFaceEncodings = findKnownFaceEncodings(name, fitClassifierData)
                if commons.isWithinToleranceToEncodings(encoding, knownFaceEncodings):    
                    logging.info(f"Recognised: {name} in frame {frame_counter}")
                    result.addRecognisedPerson(name[0])
                    continue

            # unknown face processing
            # do not process too small faces
            if faceTooSmall(face_locations[i]):
                continue

            top, right, bottom, left = face_locations[i]
            logging.debug("The unknown face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            thumbnail = image[top:bottom, left:right]
            pilThumbnail = PIL.Image.fromarray(thumbnail)
            if logging.getLogger().level == logging.DEBUG:
                pilThumbnail.show()

            similarPerson = result.findSimilarPerson(encoding)
            if similarPerson is not None:
                logging.debug(f"{similarPerson} in the frame {frame_counter}")
                result.addToPerson(similarPerson, pilThumbnail, encoding)
            else: 
                newPersonName = f"unknown-{personCounter}"
                personCounter += 1
                logging.info(f"New {newPersonName} in the frame {frame_counter}")
                result.addToPerson(newPersonName, pilThumbnail, encoding)


    # saveResultAsRun(result, dirStructure.recogniserDir)
    if ringEvent is not None:
        saveResultAsProcessedEvent(result, dirStructure, ringEvent)

    return result.json()

def findKnownFaceEncodings(name, fitClassifierData):
    for personImages in fitClassifierData['persons']:
        if personImages['personName'] == name:
            logging.debug(f"found {len(personImages['encodingsAsNumpyArray'])} known face encodings for {name}")
            return personImages['encodingsAsNumpyArray']

    logging.info(f"can not find known face encodings for {name}")
    return []

def faceTooSmall(faceLocation):
    top, right, bottom, left = faceLocation
    if bottom - top < MIN_TRUMBNAIL_SIZE_IN_PX or right - left < MIN_TRUMBNAIL_SIZE_IN_PX:
        logging.debug(f"Face too small: {bottom - top} x {right - left}")
        return True
    else:
        return False


def saveResultAsProcessedEvent(result, dirStructure, ringEvent):
    logging.debug(f"saveResultAsProcessedEvent: {ringEvent}")
    eventName = ringEvent['eventName']
    result.eventName = eventName
    resultDir = dirStructure.images + "/" + eventName
    if os.path.isdir(resultDir):
        logging.warning(f"Will replace content in {resultDir}")
    else:
        os.mkdir(resultDir)

    for unknownPersonName, personData in result.persons.items():
        for faceImage in personData.images:
            filename = "face-" + uuid.uuid4().hex + ".jpeg"
            imageFilePath = resultDir + "/" + filename
            faceImage.save(imageFilePath, "JPEG")

            result.addImageFilePathToPerson(imageFilePath, unknownPersonName)

    resultJson = result.json()


    fileHandler = open(resultDir + "/processingResult.json", "w")
    fileHandler.write(resultJson)
    fileHandler.close()

    logging.info(f"Saved result: {resultJson} to dir: {resultDir}")


