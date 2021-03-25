#!/bin/bash

cd "$(dirname "$0")"

docker push ringface/classifier:latest

if [ -z ${1+x} ]
  then
  echo "semver is not defined"
  else
  echo "tagging to '$1'"
  docker push ringface/classifier:$1
fi