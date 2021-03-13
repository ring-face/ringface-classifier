
DATA_DIR = './data'
IMAGES_DIR      = DATA_DIR + "/images"
CLASSIFIER_DIR  = DATA_DIR + "/classifier"
RECOGNISER_DIR  = DATA_DIR + '/recogniser'

class DirStructure():
    def __init__(self, dataDir = DATA_DIR, imagesDir = IMAGES_DIR, recogniserDir = RECOGNISER_DIR, classifierDir = CLASSIFIER_DIR):
        self.dataDir = dataDir
        self.imagesDir = imagesDir
        self.recogniserDir = recogniserDir
        self.classifierDir = classifierDir

        self.unprocessedEvents = dataDir + "/events/unprocessed"
        self.processedEvents = dataDir + "/events/processed"


DEFAULT_DIR_STUCTURE = DirStructure()