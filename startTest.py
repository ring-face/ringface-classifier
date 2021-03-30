#!/usr/bin/env python3
from multiprocessing import Pool
import cv2

MAX_FRAMES=100

def processVideo():

    input_movie = cv2.VideoCapture("data/videos/20210319-123713.mp4")

    extractionResults = []
    frame_counter = 0
    with Pool(processes=4) as pool:

        while True:
            frame_got, frame = input_movie.read()

            if not frame_got:
                break
            if frame_counter > MAX_FRAMES:
                print(f"will not consider more than first {MAX_FRAMES} frames")
                break

            frame_counter += 1
            image = frame[:, :, ::-1]
            extractionResults.append(pool.apply_async(extractFromImage, (image, frame_counter)))
            print(f"scheduled {frame_counter}")

        print("added all to pool")

        for res in extractionResults:
            print(f"result: {res.get()}")

        print("pool.join()")



def extractFromImage(image, frameNo):
    # # start_time = time.time()
    # face_locations = face_recognition.face_locations(image)
    # # profiler.addRuntime("locations", time.time() - start_time)

    # # start_time = time.time()
    # encodings = face_recognition.face_encodings(image)
    # # profiler.addRuntime("encodings", time.time() - start_time)

    face_locations = [frameNo]
    encodings = [frameNo]

    print(f"extracted locations and encodings from frame {frameNo}")
    return face_locations, encodings


if __name__ == '__main__':

    print("started")
    processVideo()
    print("finished")

