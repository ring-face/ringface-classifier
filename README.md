# This is the repo for the classifier retrain service

This service will use the directory structure that is sampled in the `sample-data` dir. It will refit the classifier, and persist the fitted classifier in a persistent file. 

# How to train the classifier
```bash
pip3 install -r requirements.txt
#rm -rf data
cp -r sample-data data
python3 startClassifier.py
```

# The training process will do the following steps.

Phase 1: Encode the unencoded images
* Encode images in the `new-images` dir and move the file to `encoded-images` dir
* The encoding information is saved as a json array to file in the `encodings` dir under the same file name as the image with `.json` extension
* encoding is using the face_recognition lib, which uses dlib as underlying
* the dlib pretrained model for encoding is loaded at startup from [face_recoginition_models](https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models) lib


Phase 2: Fit the [Support Vector Classifier](https://scikit-learn.org/stable/modules/svm.html#svm-classification) with the data from the `encodings` dir
* Load each file from the `encodings` folder into a numpy array
* create the same sized array with the name of the persons, where `encodings[i]` represents the encoded face of `persons[i]`
* run `newClassifier=SVC.fit(encodings, persons)`
* [persist](https://scikit-learn.org/stable/modules/model_persistence.html) the `newModel` to the `data/classifier` dir in as `fitting.timestamp` 

Phase 3: Test the new model
* loads the persisted classifier, and tests ist capabilities on the test set of the **/test-images folder
* in addition to the classification, it calculates an euclidian distance from the encodings of the found person
* only if the distance is below a treshold, is the classification valid
* this avoids recognising arbitrary face as known person


Once this is finished, the proces exits

# How to use the trained classifier on a file
```bash
python3 startRecogniserSingleImage.py ./data/images/barack/test-images/barack4.jpeg
# you can download a sample video here: https://www.youtube.com/watch?v=4P-4PlwTcoE
python3 startRecogniserSingleVideo.py ./sample-data/ring-sample-video.mp4
```

## The image recognition process does the following
* find all faces in the passed image
* print the name of the recognised faces
* create a thumbnail crop of the unrecognised face in a new file
** this thumbnail can be labelled
** and then the model can be retrained

## The video recognition process does the following
* find all faces in the each frame
* record the name of the recognised faces
* create a thumbnail crop of the unrecognised face in a new file
* compare this thubnail for similar faces, so the same person enters the same folder
** this thumbnail can be labelled
** and then the model can be retrained


# The recognition server
* loads the latest fitter setup at startup
* takes one file as input
* stores it in /tmp
* recognises the persons, and returns a HTTP 200 plus structured response similar to
```json
// image result
{"personImageFile": "/tmp/barack4.jpeg", "recognisedPersons": ["michele", "barack"], "unknownPersons": []}
// video result
{"videoFile": "sample-data/test.mp4", "recognisedPersons": [], "unknownPersons": [{"name": "unknown-1", "images": 2}, {"name": "unknown-2", "images": 2}, {"name": "unknown-3", "images": 1}]}
```

## To start the server 
```bash
./startServer.sh
# post image
curl -F "file=@./data/images/barack/test-images/barack3.jpeg" http://localhost:5000/recognition/singe-image
# post video
curl -F "file=@./sample-data/test.mp4" http://localhost:5000/recognition/singe-video

```
