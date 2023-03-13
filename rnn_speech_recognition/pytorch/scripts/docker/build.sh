#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb

docker build . --rm -t mlperf/rnn_speech_recognition \
	--build-arg http_proxy=${http_proxy} \
	--build-arg https_proxy=${https_proxy}
