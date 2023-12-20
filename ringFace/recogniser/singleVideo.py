from io import BytesIO
import logging
import os
import sys
import time
import json
import cv2
import uuid
import multiprocessing as mp
import face_recognition
import PIL.Image
from decouple import config


from ringFace.classifierRefit import helpers

from ringFace.ringUtils import commons, clfStorage, gcs

import time




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



def recognition(videoFile, ringEvent= None):

    personCounter = 1

    logging.info(f"processing input video {videoFile}")
    result = VideoRecognitionData(videoFile)

    clf, fitClassifierData = clfStorage.loadLatestClassifier()

    input_movie = cv2.VideoCapture(gcs.tmpfile_for_read(videoFile))
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Total frames: {length}")

    frame_counter = 0
    noFaceFrameCounter = 0

    extractionResults = []

    faceFound = False

    logging.debug(f"frame 0 - {config('MAX_FRAMES')}: scheduling for extraction ")

    # process the math heavy part in parallel
    with mp.Pool(processes= config('PARALLELISM', cast=int)) as pool:
        while True:
            
            frame_got, frame = input_movie.read()

            frame_counter += 1
            
            if not frame_got:
                break
            
            if frame_counter > config('MAX_FRAMES', cast=int):
                logging.warn(f"will not consider more than first {config('MAX_FRAMES', cast=int)} frames")
                break

            if frame_counter > config('MIN_FRAMES', cast=int) and frame_counter % config('EACH_FRAME', cast=int)  != 0:
                # logging.debug(f"frame {frame_counter}: skipping ")
                continue




            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            start_time = time.time()
            image = frame[:, :, ::-1]
            extractionResults.append(pool.apply_async(extractFromImageParallel, (image, frame_counter)))


        # pool.close()


        # process the results sequentially
        for res in extractionResults:
            frame_counter, face_locations, encodings, image = res.get()
            # logging.debug(f"frame {frame_counter}: postprocessing")

            facesCount = len(face_locations)
            logging.debug(f"frame {frame_counter}: Number of faces detected: {facesCount}")

            # stop after couple of empty frames
            if facesCount == 0:
                noFaceFrameCounter += 1
                if faceFound and frame_counter > config('MIN_FRAMES', cast=int) and noFaceFrameCounter >= config('STOP_AFTER_EMPTY_FRAMES', cast=int) :
                    logging.warn(f"frame {frame_counter}: noFaceFrameCounter reached: {noFaceFrameCounter}. Stopping")
                    # at this point we do not need more 
                    pool.terminate()
                    break
                else:
                    continue

            noFaceFrameCounter = 0


            for i in range(facesCount):
                encoding = encodings[i]
                
                #process the recognised face
                if clf is not None:
                    start_time = time.time()
                    name = clf.predict([encoding])
                    logging.debug(f"frame {frame_counter}: predicted name: {name}")

                    # if commons.isWithinTolerance(encoding, encodingsDir):
                    knownFaceEncodings = findKnownFaceEncodings(name, fitClassifierData, frame_counter)
                    if commons.isWithinToleranceToEncodings(encoding, knownFaceEncodings):    
                        logging.info(f"frame {frame_counter}: Recognised: {name}")
                        faceFound = True
                        result.addRecognisedPerson(name[0])
                        continue
                    else :
                        logging.debug(f"frame {frame_counter}: Face outside of tolerance for {name}")

                # unknown face processing
                # do not process too small faces
                if faceTooSmall(face_locations[i], frame_counter):
                    continue

                top, right, bottom, left = face_locations[i]
                logging.debug(f"frame {frame_counter}: "+"The unknown face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                thumbnail = image[top:bottom, left:right]
                pilThumbnail = PIL.Image.fromarray(thumbnail)
                faceFound = True

                # if logging.getLogger().level == logging.DEBUG:
                #     pilThumbnail.show()

                similarPerson = result.findSimilarPerson(encoding)
                if similarPerson is not None:
                    logging.debug(f"frame {frame_counter}: {similarPerson} in the frame {frame_counter}")
                    result.addToPerson(similarPerson, pilThumbnail, encoding)
                else: 
                    newPersonName = f"unknown-{personCounter}"
                    personCounter += 1
                    logging.info(f"frame {frame_counter}: New {newPersonName} in the frame {frame_counter}")
                    result.addToPerson(newPersonName, pilThumbnail, encoding)

    logging.debug("wait for the workers to terminate")
    pool.join()

    # saveResultAsRun(result, dirStructure.recogniserDir)
    if ringEvent is not None:
        saveResultAsProcessedEvent(result, ringEvent)

    return result.json()

'''
This method runs async in its own process
'''
def extractFromImageParallel(image, frame_counter):
    logging.debug(f"frame {frame_counter}: extractFromImageParallel")
    face_locations = face_recognition.face_locations(image)

    encodings = face_recognition.face_encodings(image, face_locations)

    logging.debug(f"frame {frame_counter}: extracted locations and encodings")
    return frame_counter, face_locations, encodings, image



def findKnownFaceEncodings(name, fitClassifierData, frame_counter):
    for personImages in fitClassifierData['persons']:
        if personImages['personName'] == name:
            logging.debug(f"frame {frame_counter}: found {len(personImages['encodingsAsNumpyArray'])} known face encodings for {name}")
            return personImages['encodingsAsNumpyArray']

    logging.info(f"frame {frame_counter}: can not find known face encodings for {name}")
    return []

def faceTooSmall(faceLocation, frame_counter):
    top, right, bottom, left = faceLocation
    if bottom - top < config('MIN_TRUMBNAIL_SIZE_IN_PX', cast=int) or right - left < config('MIN_TRUMBNAIL_SIZE_IN_PX', cast=int):
        logging.debug(f"frame {frame_counter}: Face too small: {bottom - top} x {right - left}")
        return True
    else:
        return False


def saveResultAsProcessedEvent(result, ringEvent):
    logging.debug(f"saveResultAsProcessedEvent: {ringEvent}")
    eventName = ringEvent['eventName']
    result.eventName = eventName
    resultDir = "images/" + eventName

    for unknownPersonName, personData in result.persons.items():
        for faceImage in personData.images:
            buffer = BytesIO()
            faceImage.save(buffer, "JPEG")

            filename = "face-" + uuid.uuid4().hex + ".jpeg"
            imageFilePath = resultDir + "/" + filename
            gcs.save_binary(buffer, imageFilePath, content_type="image/jpeg")

            result.addImageFilePathToPerson(imageFilePath, unknownPersonName)

    resultJson = result.json()


    gcs.save_json_to_gcs(resultJson, resultDir + "/processingResult.json")

    logging.info(f"Saved result: {resultJson} to dir: {resultDir}")


