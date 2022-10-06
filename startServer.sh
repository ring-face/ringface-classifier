#!/bin/bash

export FLASK_APP=ringFace.recognitionServer
export FLASK_ENV=development
flask run --host=0.0.0.0 --no-reload --port 5001