import logging
import functions_framework
from flask import Response, jsonify


from ringFace.classifierRefit.fitter import fitClassifier
from ringFace.recogniser import singleVideo

# from ringFace.ringUtils.clfStorage import parseEncodingsAsNumpyArrays


def setup_logging():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)
    logging.getLogger('requests_oauthlib').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.info("Logging set up")

@functions_framework.http
def fit(request):
    
    try:
        fitClassifierRequest = request.json
        logging.debug(f"Running the fitting on request {fitClassifierRequest}")

        fitClassifierData, clf = fitClassifier(fitClassifierRequest)
        res = jsonify(fitClassifierData)

        return res
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {e}"}), 500   
     
@functions_framework.http
def recognition(request):
    try:
        event = request.json
        logging.info(f"will process event {event}")

        videoFilePath = event['videoFileName']
        
        videoRecognitionResult = singleVideo.recognition(videoFilePath, event)

        return Response(videoRecognitionResult, mimetype='application/json')
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {e}"}), 500   
    

setup_logging()
