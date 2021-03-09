FROM python:3.8.8-slim-buster

WORKDIR /app

# modified from https://github.com/ageitgey/face_recognition/blob/master/Dockerfile
RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*


COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
# RUN pip3 install cmake
RUN pip3 install -r requirements.txt

COPY ringFace ringFace
COPY sample-data data


COPY startServerProd.sh .

CMD ./startServerProd.sh