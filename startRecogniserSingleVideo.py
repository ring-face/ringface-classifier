#!/usr/bin/env python3

import logging
import sys
from ringFace.recogniser.singleVideo import recognition


logging.basicConfig(format='%(message)s', level=logging.DEBUG)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        logging.error(f"usage: python3 {str(sys.argv[0])} /path/video.mp4")
        sys.exit(1)

    logging.debug(f"Argument List: {str(sys.argv)}")
    videoFile = sys.argv[1]

    ringEvent = {
        'eventName':'_commandline'
    }

    result = recognition(videoFile, ringEvent=ringEvent)

    logging.info(result)