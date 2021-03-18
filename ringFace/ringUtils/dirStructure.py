from decouple import config

DATA_DIR = config('DATA_DIR')


# class DirStructure():
#     def __init__(self, dataDir = DATA_DIR, imagesDir = IMAGES_DIR, recogniserDir = RECOGNISER_DIR, classifierDir = CLASSIFIER_DIR):
#         self.dataDir = dataDir
#         self.imagesDir = imagesDir
#         self.recogniserDir = recogniserDir
#         self.classifierDir = classifierDir

#         self.unprocessedEvents = dataDir + "/events/unprocessed"
#         self.processedEvents = dataDir + "/events/processed"

#         self.images = dataDir + "/images"

class DirStructure():
    def __init__(self, dataDir = DATA_DIR):
        self.images = dataDir + "/images"
        self.classifierDir = dataDir + "/classifier"


DEFAULT_DIR_STUCTURE = DirStructure()
