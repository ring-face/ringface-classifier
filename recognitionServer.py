from flask import Flask, flash, request, redirect, url_for, abort, jsonify
import glob
import os
from joblib import load
from PIL import Image
import logging

from werkzeug.utils import secure_filename

from ringFace.recogniser import singleImage, singleVideo
from ringFace.ringUtils import clfStorage

from ringFace.classifierRefit.encoder import processUnencoded
from ringFace.classifierRefit.fitter import fitEncodings
from ringFace.classifierRefit.qaTester import testClassifier


UPLOAD_FOLDER = '/tmp'
DATA_DIR = './data'
IMAGES_DIR      = DATA_DIR + "/images"
CLASSIFIER_DIR  = DATA_DIR + "/classifier"
RECOGNISER_DIR  = DATA_DIR + '/recogniser'
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
    return 'Pong'


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

@app.route('/recognition/singe-video', methods=["POST"])
def recognitionVideo():
    fileName = saveToUploadFolder(request)
        
    logging.info(f"processing uploaded video {fileName}")

    videoRecognitionResult = singleVideo.recognition(fileName, RECOGNISER_DIR, clf)

    return videoRecognitionResult.json()

@app.route('/recognition/singe-image', methods=["POST"])
def recognitionImage():
    fileName = saveToUploadFolder(request)
        
    logging.info(f"processing uploaded image {fileName}")

    fileRecognitionResult = singleImage.recognition(fileName, RECOGNISER_DIR, clf)

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


@app.route('/classifier/run')
def classifier():
        
    logging.info(f"Rerunning the classifier")

    processUnencoded(IMAGES_DIR)
    fitterData = fitEncodings(IMAGES_DIR, CLASSIFIER_DIR)
    testClassifier(fitterData.fittedClassifierFile, IMAGES_DIR)
    print(f"Result: {fitterData.json()}")

    clf = clfStorage.loadLatestClassifier()

    return fitterData.json()