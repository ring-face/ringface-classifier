# Installing a CUDA accelerated version on gcloud
https://towardsdatascience.com/installing-cuda-on-google-cloud-platform-in-10-minutes-9525d874c8c1
https://sparkle-mdm.medium.com/python-real-time-facial-recognition-identification-with-cuda-enabled-4819844ffc80

## Setup
* Ubuntu 20 LTS on 200 GB Boot Disk
* 4 vCPUs, 15 GB RAM
* NVidia Tesla T4
* approx 300 EUR Per Month
* Zone: us-central1-b

## Connect to the instance after provisionning
```bash
gcloud beta compute ssh --zone "us-central1-b" "ubuntu-gpu" --project "ringface-shared"
```

## Install Python3.8
* python 3.8 installed by default

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3-pip
sudo apt-get install python3-venv
```

## Install the requirements
```bash
sudo apt-get install -y --fix-missing \
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
    zip 

sudo apt-get install ffmpeg libsm6 libxext6  -y
```

## Run the project
```bash
git clone https://github.com/ring-face/ringface-classifier.git
cd ringface-classifier
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```
* this will take a while as it will compile dlib
* while waiting, copy some video file from the local machine
```bash
gcloud compute scp ../data/videos/20210319-123713.mp4 ubuntu-gpu:/home/csaba/python-test/repo/ringface-classifier/data/videos/
```

* at this point, the recoginition should be working WITHOUT the GPU acceleration, approx 17 sec
```bash
time ./startRecogniserSingleVideo.py data/videos/20210319-123713.mp4
```

# GPU Acceleration
* we will install cuda 11.2 and CuDNN 8.1
https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux

## Installing Cuda 
```bash
sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

sudo apt install cuda

echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
source ~/.bashrc

nvcc --version

# Test the cuda installation
nvcc -o hello hello.cu 
./hello 
```

## Installing CuDNN

* created nvidia account and donwloaded the v8 of CuDNN
```bash
# on local machine
gcloud compute scp ./cudnn-11.2-linux-x64-v8.1.1.33.tgz ubuntu-gpu:~/cuda/
# on the ubuntu-gpu
CUDNN_TAR_FILE="cudnn-11.2-linux-x64-v8.1.1.33.tgz"
sudo tar -xzvf ${CUDNN_TAR_FILE}
sudo cp cuda/include/cudnn* /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

## Compile dlib for GPU acceleration
https://sparkle-mdm.medium.com/python-real-time-facial-recognition-identification-with-cuda-enabled-4819844ffc80

```bash
sudo apt-get update
sudo apt-get install python3-dev
sudo apt-get install build-dep python3
sudo apt-get update
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev
sudo apt-get update
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev


cd ~/repo
git clone https://github.com/davisking/dlib.git
cd dlib

mkdir build && cd build

cmake .. \
-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/ -DDLIB_USE_CUDA=1 \
-DUSE_AVX_INSTRUCTIONS=1 -DUSE_F16C=1

cmake --build . --config Release

sudo ldconfig

cd ..
python3 setup.py install 

```

* at this point the dlib is compiled with cuda support
```python
import dlib
print(dlib.DLIB_USE_CUDA)
```

# Measurements

## T4 Tesla with Cuda enabled DLIB compilation
  - 4 parallelism: 17 sec
  - 8 parallelism: 17 sec
  
## c2 Standard with 8 CPUs and 32GB RAM, without graphic acceleration
  - 4 parallelism: 10 sec
  - 8 parallelism: 10 sec



