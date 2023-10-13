#!/bin/bash

cd "$(dirname "$0")"

docker build . --tag ringface/classifier-cr:latest

if [ -z ${1+x} ]
  then
  echo "semver is not defined"
  else
  echo "tagging to '$1'"
  docker tag ringface/classifier-cr:latest eu.gcr.io/ringface/classifier-cr:$1
  docker push eu.gcr.io/ringface/classifier-cr:$1

fi