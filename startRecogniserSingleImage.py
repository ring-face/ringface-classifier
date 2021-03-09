#!/usr/bin/env python3

import logging
import sys
from ringFace.recogniser.singleImage import recognition

logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) == 1:
    logging.error(f"usage: python3 {str(sys.argv[0])} /path/someimage.jpeg")
    sys.exit(1)

logging.debug(f"Argument List: {str(sys.argv)}")
personImageFile = sys.argv[1]

recognition(personImageFile)