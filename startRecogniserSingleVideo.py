#!/usr/bin/env python3

import logging
import sys
from recogniser.singleVideo import recognition

imageDir = "./data/images"
logging.getLogger().setLevel(logging.DEBUG)

if len(sys.argv) == 1:
    logging.error(f"usage: python3 {str(sys.argv[0])} /path/video.mp4")
    sys.exit(1)

logging.debug(f"Argument List: {str(sys.argv)}")
videoFile = sys.argv[1]

recognition(videoFile, './data/recogniser')