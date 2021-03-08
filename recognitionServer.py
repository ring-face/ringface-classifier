from flask import Flask, flash, request, redirect, url_for, abort, jsonify
import glob
import os
from joblib import load
from PIL import Image
import logging

from werkzeug.utils import secure_filename

from recogniser import singleImage
from classifierRefit import storage

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

clf = storage.loadLatestClassifier()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello():
    return 'Hello, World!'


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404


@app.route('/recognition/file', methods=["POST"])
def recognitionFile():
    fileName = saveToUploadFolder(request)
        
    logging.info(f"processing uploaded file {fileName}")

    fileRecognitionResult = singleImage.recognition(fileName, './data/recogniser')

    return fileRecognitionResult.json()

def saveToUploadFolder(request):
    # check if the post request has the file part
    if 'file' not in request.files:
        abort(404, description="No file part")
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        abort(404, description="No file specified")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filename)
        return filename
