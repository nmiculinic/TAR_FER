FROM nvidia/cuda:8.0-cudnn5-devel

# Mostly copy/paste from tensorflow webpage
# Pick up some TF dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-setuptools \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-sklearn \
        rsync \
        software-properties-common \
        unzip \
        libhdf5-serial-dev \
        git \
        cmake \
        sshfs \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --build-arg tf=tensorflow for CPU only tensorflow
ARG tf=tensorflow-gpu
RUN pip3 --no-cache-dir install $tf git+https://github.com/tflearn/tflearn.git  python-dotenv slacker-log-handler

RUN mkdir /code
RUN mkdir /data
WORKDIR /code
