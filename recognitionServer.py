from flask import Flask, flash, request, redirect, url_for, abort, jsonify
import glob
import os
from joblib import load
from PIL import Image
import logging

from werkzeug.utils import secure_filename

from ringFace.recogniser import singleImage, singleVideo
from ringFace.ringUtils import clfStorage

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'mp4'}

clf = clfStorage.loadLatestClassifier()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.getLogger().setLevel(logging.INFO)



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello():
    return 'Hello, World!'


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

@app.route('/recognition/singe-video', methods=["POST"])
def recognitionVideo():
    fileName = saveToUploadFolder(request)
        
    logging.info(f"processing uploaded video {fileName}")

    videoRecognitionResult = singleVideo.recognition(fileName, './data/recogniser')

    return videoRecognitionResult.json()

@app.route('/recognition/singe-image', methods=["POST"])
def recognitionImage():
    fileName = saveToUploadFolder(request)
        
    logging.info(f"processing uploaded image {fileName}")

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
