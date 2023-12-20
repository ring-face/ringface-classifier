PROJECT=ringface-2
gcloud functions deploy recognition \
--gen2 \
--project $PROJECT \
--service-account classifier-sa@$PROJECT.iam.gserviceaccount.com \
--runtime python310 \
--source=. \
--entry-point=recognition \
--trigger-http \
--memory 4GiB \
--cpu=4 \
--region=europe-west3 \
--allow-unauthenticated \
--ingress-settings internal-and-gclb \
--env-vars-file .env.yaml

