#!/bin/bash

set -eo pipefail

# Create mount directory for service.
mkdir -p $MNT_DIR

echo "Mounting Cloud Filestore."
mount -o nolock $FILESTORE_IP_ADDRESS:/$FILE_SHARE_NAME $MNT_DIR
echo "Mounting completed."

export FLASK_APP=ringFace.recognitionServer
export FLASK_ENV=production
exec flask run --host=0.0.0.0 --no-reload --port $PORT

# Exit immediately when one of the background processes terminate.
wait -n