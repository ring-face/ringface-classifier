from decouple import config
from pathlib import Path


DATA_DIR = config('DATA_DIR')


class DirStructure():
    def __init__(self, dataDir = DATA_DIR):
        self.images = dataDir + "/images"
        self.classifierDir = dataDir + "/classifier"
        Path(self.images).mkdir(parents=True, exist_ok=True)
        Path(self.classifierDir).mkdir(parents=True, exist_ok=True)


DEFAULT_DIR_STUCTURE = DirStructure()
