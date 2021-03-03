#!/bin/bash

echo "test with curl -v http://localhost:5000/recognition -X POST --data-binary data/images/barack/test-images/barack3.jpeg"

export FLASK_APP=recognitionServer
export FLASK_ENV=development
flask run