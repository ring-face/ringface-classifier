# This is the repo for the classifier retrain service

The classifier server is the wrapper around the AI part of the solution. It exposes services 
* to fit a classifier to assing a name to a face encofing (SVC Classifier)
* to find face thumbnails in images (HOG algorithm)
* to find face thumbnails in frames of videos in parallel (HOG algorithm)
* to encode a face into an encoding (the actual AI work)
* and to apply the SVC classifier to the encoding.

# Glossary
A `face encoding` is a 200 dimensional array, derived from a face thumbnail, which is the output of the [dlib](http://dlib.net/)'s state-of-the-art face recognition. The training of the AI has been done externally, and its weights are loaded at server start. This is the actual AI in the solution. DLIB is used indirectly via the https://github.com/ageitgey/face_recognition module. 

The `SVC Classifier` is a linear classifier from [scikit](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

The `HOG` is the [Histogram of oriented gradients](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/) algorithm. It is used to find face thumbnails in a much larger image, or video frame. 

# How to run
Install and run the python virtual env
```bash
python3 -m venv venv
source venv/bin/activate
# your installation may fail due to missing cmake, which is needed to compile the https://github.com/davisking/dlib library
# install cmake before you run the next step (eg. brew install cmake)
pip3 install -r requirements.txt

```

## To start the server 
You will be interacting with the service over HTTP. 
```bash
./startServer.sh
```

## The fitting
Fitting is required to fit the `SVC Classifier` on your already known images. It has 3 phases under the hood, documented below. You can run it simply for the sample images of 2 presidents with

```bash
curl --location --request POST 'http://localhost:5001/classifier/fit' \
--header 'Content-Type: application/json' \
--data-raw '{
"persons":[
    {
        "personName": "Barack Obama",
        "imagePaths": [
            "./sample-data/images/barack/new-images/barack1.jpeg",
            "./sample-data/images/barack/new-images/barack2.jpeg"
        ]
    },
    {
        "personName": "Donald Trump",
        "imagePaths": [
            "sample-data/images/donald/new-images/donald1.jpeg",
            "sample-data/images/donald/new-images/donald2.jpeg"
        ]
    }
]
}'
```
At this point, your classifier is fitted, and can be tested on previously unseen faces or videos of these 2 gentlemen. 

## The recognition

To test the recognition, run 

```bash
curl --location --request POST 'http://localhost:5001/recognition/local-image' \
--header 'Content-Type: application/json' \
--data-raw '{
  "eventName": "20221003-200000",
  "imageFilePath": "./sample-data/images/barack/test-images/barack3.jpeg"
}'

### {"inputFile": "./sample-data/images/barack/test-images/barack3.jpeg", "recognisedPersons": ["Barack Obama"], "unknownPersons": []}
```


# How to use the trained classifier on a file without starting the server
```bash
python3 startRecogniserSingleImage.py ./sampe-data/images/barack/test-images/barack4.jpeg
# you can download a sample video here: https://www.youtube.com/watch?v=4P-4PlwTcoE
python3 startRecogniserSingleVideo.py ./sample-data/ring-sample-video.mp4
```

## The image recognition process does the following
* find all faces in the passed image
* print the name of the recognised faces
* create a thumbnail crop of the unrecognised face in a new file
* this thumbnail can be labelled
* and then the model can be retrained

## The video recognition process does the following
* find all faces in the each frame
* record the name of the recognised faces
* create a thumbnail crop of the unrecognised face in a new file
* compare this thubnail for similar faces, so the same person enters the same folder
* this thumbnail can be labelled
* and then the model can be retrained


