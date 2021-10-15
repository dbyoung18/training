#!/bin/bash

set -x
## DL hyper params
EXTRA_PARAMS=(
               --batch-size        "512"
               --batch-splits      "4"
               --weight-decay      "0.00013"
               --lr                "0.003157"
               --warmup            "5"
               --warmup-factor     "0"
	       --lr-decay-schedule "44 55"
             )

## System run params
USE_NODE=1
USE_GPU=2
USE_CORE_PER_PROC=4

CONFIG=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=12:00:00

## System config params
NUM_NODE=1
NUM_GPU_PER_NODE=4
NUM_SOCKET_PER_NODE=1
NUM_CORE_PER_SOCKET=16
NUM_THREAD_PER_CORE=2 	# HT is on is 2, HT off is 1
set +x
