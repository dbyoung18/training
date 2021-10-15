#!/bin/bash

set -x
## DL params
EXTRA_PARAMS=(
               --batch-size      "1024"
               --batch-splits    "8"
               --weight-decay    "0.00013"
               --lr              "0.003157"
               --warmup          "5"
               --warmup-factor   "0"
	       --lr-decay-schedule "44 55"
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=12:00:00

## System config params
DGXNGPU=2
DGXSOCKETCORES=16
DGXNSOCKET=1
DGXHT=2 	# HT is on is 2, HT off is 1
set +x
