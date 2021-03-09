#!/usr/bin/env python3

import logging
import sys
from ringFace.recogniser.singleVideo import recognition


logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) == 1:
    logging.error(f"usage: python3 {str(sys.argv[0])} /path/video.mp4")
    sys.exit(1)

logging.debug(f"Argument List: {str(sys.argv)}")
videoFile = sys.argv[1]

recognition(videoFile)