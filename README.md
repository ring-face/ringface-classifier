# This is the repo for the classifier and recogniser service

The classifier server is the wrapper around the AI part of the solution. It exposes apis 
* to fit a classifier to assing a name to a face encofing (SVC Classifier)
* to find face thumbnails in images (HOG algorithm)
* to find face thumbnails in frames of videos in parallel (HOG algorithm)
* to encode a face into an encoding (the actual AI work)
* and to apply the SVC classifier to the encoding.

The 3 main APIs are
* `POST /classifier/fit` which will process the known images, that will serve as a base for future recognition
* `POST /recognition/local-image` which will recognise the faces in the image
* `POST /recognition/local-video` which will roughly speaking process frames of the video as an image

See below, for details on how to call these APIs.

# Glossary
A `face encoding` is a 200 or so dimensional vector, derived from a face thumbnail, which is the output of the [dlib](http://dlib.net/)'s state-of-the-art face recognition. The training of the AI has been done externally, and its weights are loaded at server start. This is the actual AI in the solution. DLIB is used indirectly via the https://github.com/ageitgey/face_recognition module. 

The `SVC Classifier` is a linear classifier from [scikit](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html). It is used to group similar vectors near to each other, in order to predict the face in the image.

The `HOG` is the [Histogram of oriented gradients](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/) algorithm. It is used to find all face thumbnails in a much larger image, or video frame. 

# How to run

You will be interacting with the API over HTTP. Start the server via docker

```bash
docker-compose up # this may take couple of minutes 
```

## Alternatively, start from the command line

Install and run the python virtual env
```bash
python3 -m venv venv # will create your virtual env
source venv/bin/activate # will activate this virtual env
# the next step may fail due to missing cmake, which is required to compile the source distribution of the https://github.com/davisking/dlib library
# install cmake before you run the next step (eg. brew install cmake)
pip3 install -r requirements.txt
./startServer.sh
# Running on http://localhost:5001
```

## The fitting
Fitting is required teach the system your already known faces, eg to fit the `SVC Classifier` on your already known(tagged) images.  You can run it simply for the bundled sample images of 2 presidents (Obama, Trump) with the below command. Replace the `personName`, and the `imagePaths` and add more `persons` for your own database of faces.

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
At this point, your classifier is fitted, and can be tested on previously unseen faces of these 2 persons. In general the more faces you add, the better the recoginition is. However, it is sufficiently stable to start with just 1 image per person.  

## Image recognition

To test the recognition on an image, run the below, with `eventName` an arbitrary string, and the `imageFilePath` is the path relative to the `startServer.sh` script.

```bash
curl --location --request POST 'http://localhost:5001/recognition/local-image' \
--header 'Content-Type: application/json' \
--data-raw '{
  "eventName": "my-image-event",
  "imageFilePath": "./sample-data/images/barack/test-images/barack3.jpeg"
}'

### {"inputFile": "./sample-data/images/barack/test-images/barack3.jpeg", "recognisedPersons": ["Barack Obama"], "unknownPersons": []}
```

### The image recognition process does the following
* find all faces in the passed image
* print the name of the recognised faces
* create a thumbnail crop of the unrecognised face in a new file
* this thumbnail can be labelled
* and then the model can be retrained

## Video recognition

You can also run the recoginition on a video, in which case each frame will be tested for a face. Again, `eventName` an arbitrary string, and the `imageFilePath` is the path relative to the `startServer.sh` script.

```bash
curl --location --request POST 'http://localhost:5001/recognition/local-video' \
--header 'Content-Type: application/json' \
--data-raw '{
  "eventName": "my-video-event",
  "videoFileName": "./downloaeded-video-of-some-president.mp4"
}'

# {
#    "videoFile": "./downloaeded-video-of-some-president.mp4",
#    "eventName": "my-video-event",
#    "recognisedPersons": [Barack Obama],
#    "unknownPersons": []
#}
```

### The video recognition process does the following
* find all faces in the each frame
* record the name of the recognised faces
* create a thumbnail crop of the unrecognised face in a new file
* compare this thubnail for similar faces, so the same person enters the same folder
* this thumbnail can be labelled
* and then the model can be retrained

This structure can be relabelled, and fed back to the `fit` api.

# How to use the trained classifier on a file without starting the server
```bash
python3 startRecogniserSingleImage.py ./sample-data/images/barack/test-images/barack3.jpeg
# you can download a sample video here: https://www.youtubepi.com/watch?v=kVmG87FRsxY&list=PPSV 
python3 startRecogniserSingleVideo.py ./sample-data/downloaded-video.mp4
```


# Install on GCP cloudrun
https://cloud.google.com/run/docs/tutorials/network-filesystems-filestore

[Enable the apis](https://console.cloud.google.com/apis/enableflow?apiid=run.googleapis.com,file.googleapis.com,vpcaccess.googleapis.com,cloudbuild.googleapis.com,artifactregistry.googleapis.com)

Create the filestore to access
```bash

gcloud config set run/region europe-west3
gcloud config set filestore/zone europe-west3-a 

gcloud filestore instances create ringface \
    --tier=STANDARD \
    --file-share=name=vol1,capacity=1TiB \
    --network=name="default"
```

Set up a Serverless VPC Access connector
```bash
gcloud compute networks vpc-access connectors create firestore-from-cloudrun \
  --region europe-west3 \
  --range "10.8.0.0/28"
```

Get the IP of the filestore
```bash
export FILESTORE_IP_ADDRESS=$(gcloud filestore instances describe ringface --format "value(networks.ipAddresses[0])")
```

Create a service account to serve as the service identity. By default this has no privileges other than project membership.

```bash
gcloud iam service-accounts create fs-identity
```

Build and deploy
```bash
gcloud beta run deploy ringface-classifier --source . \
    --vpc-connector firestore-from-cloudrun \
    --execution-environment gen2 \
    --allow-unauthenticated \
    --service-account fs-identity \
    --update-env-vars FILESTORE_IP_ADDRESS=$FILESTORE_IP_ADDRESS,FILE_SHARE_NAME=vol1
```

