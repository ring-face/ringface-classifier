PROJECT=ringface-2
gcloud functions deploy fit \
--gen2 \
--project $PROJECT \
--service-account classifier-sa@$PROJECT.iam.gserviceaccount.com \
--ingress-settings internal-and-gclb \
--runtime python310 \
--source=. \
--entry-point=fit \
--trigger-http \
--memory 1024MB \
--region=europe-west3 \
--allow-unauthenticated \
--env-vars-file .env.yaml
