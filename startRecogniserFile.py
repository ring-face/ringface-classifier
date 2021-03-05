#!/usr/bin/env python3

import logging
import sys
from recogniser.file import recognition

imageDir = "./data/images"
logging.getLogger().setLevel(logging.DEBUG)

if len(sys.argv) == 1:
    logging.error(f"usage: python {str(sys.argv[0])} /path/someimage.jpeg")
    sys.exit(1)

logging.debug(f"Argument List: {str(sys.argv)}")
personImageFile = sys.argv[1]

recognition(personImageFile, './data/recogniser')