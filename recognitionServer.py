import flask
import glob
import os
from joblib import load
from PIL import Image
import logging



app = flask.Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/recognition', methods=["POST"])
def recognition():

    clf = loadLatestClassifier()

    imageData = flask.request.get_data()

    

    image = Image.frombytes('RGB', (277,182), imageData, 'jpeg')




    return "{'person':'unknown'}"


def loadLatestClassifier():
    list_of_files = glob.glob('./data/classifier/*')
    latest_fitting = max(list_of_files, key=os.path.getctime)

    logging.debug(f"Loading the classifier from {latest_fitting}")
    clf = load(latest_fitting)

    return clf