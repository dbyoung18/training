#!/bin/bash

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ ${DATASET_DIR} == '' ]]; then
  echo 'Warning: Please export DATASET_DIR=<path-to-dataset> first!'
  return 78
fi

if [[ ${MODEL_DIR} == '' ]]; then
  echo 'Warning: Please export MODEL_DIR=<path-to-backbone-model> first!'
  return 78
fi

set -e

# config hp and run vars
#source ./config_bs1024_1x8.sh  # 1 Tile / 1 GPU
source ./config_bs1024_2x4.sh  # 2 Tile / 2 GPU

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
NUMEPOCHS=${NUMEPOCHS:-70}

echo "running benchmark"

python -u bind_launch.py \
  --nsockets_per_node ${NUM_SOCKET_PER_NODE} \
  --ncores_per_socket ${NUM_CORE_PER_SOCKET} \
  --ngpu_per_node ${NUM_GPU_PER_NODE} \
  --nproc_per_node ${USE_GPU} \
  --ncore_per_proc ${USE_CORE_PER_PROC} \
 train.py \
  --epochs "${NUMEPOCHS}" \
  --threshold=0.23 \
  --data ${DATASET_DIR} \
  --device cuda \
  --precision fp16 \
  --pretrained-backbone ${MODEL_DIR} \
  --val-interval 5 \
  --val-epochs 46 47 48 49 \
  --log-interval 1 \
  --tb-epoch 1 \
  --tb-iter 100 \
  --workers 8 \
  ${EXTRA_PARAMS[@]} ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="OBJECT_DETECTION"

echo "RESULT,$result_name,,$result,intel,$start_fmt"
