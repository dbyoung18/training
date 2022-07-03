#!/bin/bash

docker build . --rm -t mlperf/rnn_speech_recognition \
	--build-arg http_proxy="http://mlp-sdp-icx-1046.jf.intel.com:44333" \
	--build-arg https_proxy="http://mlp-sdp-icx-1046.jf.intel.com:44333"
