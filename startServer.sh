#!/bin/bash

echo "test with curl -F \"file=@./sample-data/images/barack/test-images/family.jpeg\" http://localhost:5000/recognition/singe-image"

export FLASK_APP=ringFace.recognitionServer
export FLASK_ENV=development
flask run --host=0.0.0.0