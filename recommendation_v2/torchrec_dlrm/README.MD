# TorchRec DLRM Example

`dlrm_main.py` trains, validates, and tests a [Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091) (DLRM) with TorchRec. The DLRM model contains both data parallel components (e.g. multi-layer perceptrons & interaction arch) and model parallel components (e.g. embedding tables). The DLRM model is pipelined so that dataloading, data-parallel to model-parallel comms, and forward/backward are overlapped. Can be run with either a random dataloader or [Criteo 1 TB click logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

It has been tested on the following cloud instance types:
| Cloud  | Instance Type       | GPUs             | vCPUs | Memory (GB) |
| ------ | ------------------- | ---------------- | ----- | ----------- |
| AWS    | p4d.24xlarge        | 8 x A100 (40GB)  | 96    | 1152        |
| Azure  | Standard_ND96asr_v4 | 8 x A100 (40GB)  | 96    | 900         |
| GCP    | a2-megagpu-16g      | 16 x A100 (40GB) | 96    | 1300        |

A basic understanding of [TorchRec](https://github.com/pytorch/torchrec) will help in understanding `dlrm_main.py`. See this [tutorial](https://pytorch.org/tutorials/intermediate/torchrec_tutorial.html).

# Running

## Install dependencies
`pip install tqdm torchmetrics`

## Torchx
We recommend using [torchx](https://pytorch.org/torchx/main/quickstart.html) to run. Here we use the [DDP builtin](https://pytorch.org/torchx/main/components/distributed.html)

1. pip install torchx
2. (optional) setup a slurm or kubernetes cluster
3.
    a. locally: `torchx run -s local_cwd dist.ddp -j 1x2 --script dlrm_main.py`
    b. remotely: `torchx run -s slurm dist.ddp -j 1x8 --script dlrm_main.py`

## TorchRun
You can also use [torchrun](https://pytorch.org/docs/stable/elastic/run.html).
* e.g. `torchrun --nnodes 1 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 54321 --role trainer dlrm_main.py`


## Preliminary Training Results

**Setup:**
* Dataset: Criteo 1TB Click Logs dataset
* CUDA 11.0, NCCL 2.10.3.
* AWS p4d24xlarge instances, each with 8 40GB NVIDIA A100s.

**Results**

Common settings across all runs:

```
--num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35" --embedding_dim 128 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --shuffle_batches
```

|Number of GPUs|Collective Size of Embedding Tables (GiB)|Local Batch Size|Global Batch Size|Learning Rate|AUROC over Val Set After 1 Epoch|AUROC Over Test Set After 1 Epoch|Train Records/Second|Time to Train 1 Epoch | Unique Flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|8|91.10|256|2048|1.0|0.8032480478286743|0.8032934069633484|~284,615 rec/s| 4h7m00s | `--batch_size 256 --learning_rate 1.0`|
|1|91.10|16384|16384|15.0|0.8025434017181396|0.8026024103164673|~740,065 rec/s| 1h35m29s | `--batch_size 16384 --learning_rate 15.0 --change_lr --lr_change_point 0.65 --lr_after_change_point 0.035` |
|4|91.10|4096|16384|15.0|0.8030692934989929|0.8030484914779663|~1,458,176 rec/s| 48m39s | `--batch_size 4096 --learning_rate 15.0 --change_lr --lr_change_point 0.80 --lr_after_change_point 0.20` |
|8|91.10|2048|16384|15.0|0.802501916885376|0.8025660514831543|~1,671,168 rec/s| 43m24s | `--batch_size 2048 --learning_rate 15.0 --change_lr --lr_change_point 0.80 --lr_after_change_point 0.20` |
|8|91.10|8192|65536|15.0|0.7996258735656738|0.7996508479118347|~5,373,952 rec/s| 13m40s | `--batch_size 8192 --learning_rate 15.0`|

QPS (train record/second) is calculated by using the following formula: `x it/s * local_batch_size * num_gpus`. The `it/s`
can be found within the logs of the training results.

The final row, using 8 GPUs with a batch size of 8192, was not tuned to hit the MLPerf benchmark but is shown to
highlight the QPS (train record/second) achievable with torchrec.

**Reproduce**

Run the following command to reproduce the results for a single node (8 GPUs) on AWS. This command makes use of the `aws_component.py` script.

Ensure to:
- set $PATH_TO_1TB_NUMPY_FILES to the path with the pre-processed .npy files of the Criteo 1TB dataset.
- set $TRAIN_QUEUE to the partition that handles training jobs

**NVTabular**
For an alternative way of preprocessing the dataset using NVTabular, which can decrease the time required from several days to just hours. See the run instructions [here] (https://github.com/pytorch/torchrec/tree/main/examples/nvt_dataloader).

Preprocessing command (numpy):

After downloading and uncompressing the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf), process the raw tsv files into the proper format for training by running `./scripts/process_Criteo_1TB_Click_Logs_dataset.sh` with necessary command line arguments.

Example usage:

```
bash ./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
./criteo_1tb/raw_input_dataset_dir \
./criteo_1tb/temp_intermediate_files_dir \
./criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir
```

The script requires 700GB of RAM and takes 1-2 days to run. We currently have features in development to reduce the preproccessing time and memory overhead.
MD5 checksums of the expected final preprocessed dataset files are in md5sums_preprocessed_criteo_click_logs_dataset.txt.

We are working on improving this experience, for updates about this see https://github.com/pytorch/torchrec/tree/main/examples/nvt_dataloader


Example command:
```
torchx run --scheduler slurm --scheduler_args partition=$TRAIN_QUEUE,time=5:00:00 aws_component.py:run_dlrm_main --num_trainers=8 -- --pin_memory --batch_size 2048 --epochs 1 --num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35" --embedding_dim 128 --dense_arch_layer_sizes "512,256,128" --over_arch_layer_sizes "1024,1024,512,256,1" --in_memory_binary_criteo_path $PATH_TO_1TB_NUMPY_FILES --learning_rate 15.0 --shuffle_batches --change_lr --lr_change_point 0.80 --lr_after_change_point 0.20
```
Upon scheduling the job, there should be an output that looks like this:

```
warnings.warn(
slurm://torchx/14731
torchx 2022-01-07 21:06:59 INFO     Launched app: slurm://torchx/14731
torchx 2022-01-07 21:06:59 INFO     AppStatus:
  msg: ''
  num_restarts: -1
  roles: []
  state: UNKNOWN (7)
  structured_error_msg: <NONE>
  ui_url: null

torchx 2022-01-07 21:06:59 INFO     Job URL: None
```

In this example, the job was launched to: `slurm://torchx/14731`.

Run the following commands to check the status of your job and read the logs:

```
# Status should be "RUNNING" if properly scheduled
torchx status slurm://torchx/14731

# Log file was automatically created in the directory where you launched the job from
cat slurm-14731.out

```

The results from the training can be found in the log file (e.g. `slurm-14731.out`).

**Debugging**

The `--validation_freq_within_epoch x` parameter can be used to print the AUROC every `x` iterations through an epoch.

The in-memory dataloader can take approximately 20-30 minutes to load the data into memory before training starts. The
`--mmap_mode` parameter can be used to load data from disk which reduces start-up time for training at the cost
of QPS.

**Inference**
A module which can be used for DLRM inference exists [here](https://github.com/pytorch/torchrec/blob/main/examples/inference/dlrm_predict.py#L49). Please see the [TorchRec inference examples](https://github.com/pytorch/torchrec/tree/main/examples/inference) for more information.

# Running the MLPerf DLRM v2 benchmark

## Create the synthetic multi-hot dataset
### Step 1: Download and uncompressing the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf)

### Step 2: Run the 1TB Criteo Preprocess script.
Example usage:

```
bash ./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
./criteo_1tb/raw_input_dataset_dir \
./criteo_1tb/temp_intermediate_files_dir \
./criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir
```

The script requires 700GB of RAM and takes 1-2 days to run. MD5 checksums for the output dataset files are in md5sums_preprocessed_criteo_click_logs_dataset.txt.

### Step 3: Run the `materialize_synthetic_multihot_dataset.py` script
```
python materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path $PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH \
    --output_path $MATERIALIZED_DATASET_PATH \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --multi_hot_distribution_type uniform
```

### Step 4: Run the MLPerf DLRM v2 benchmark, which uses the materialized multi-hot dataset
Example running 8 GPUs:
```
export TOTAL_TRAINING_SAMPLES=4195197692 ;
export GLOBAL_BATCH_SIZE=65536 ;
export WORLD_SIZE=8 ;
torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py -- \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --synthetic_multi_hot_criteo_path /home/ubuntu/mountpoint/multi_hot2 \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate 0.005
```
Note: The proposed target AUROC to reach within one epoch is 0.8030.

## (Alternative method that trains multi-hot data generated on-the-fly)

It is possible to use the 1-hot preprocessed dataset (the output of `./scripts/process_Criteo_1TB_Click_Logs_dataset.sh`) to create the synthetic multi-hot data on-the-fly during training. This is useful if your system does not have the space to store the 3.8 TB materialized multi-hot dataset. Example run command:

```
export TOTAL_TRAINING_SAMPLES=4195197692 ;
export BATCHSIZE=65536 ;
export WORLD_SIZE=8 ;
torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py -- \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --in_memory_binary_criteo_path $1tb_numpy_contiguous_shuffled_dataset \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (BATCHSIZE * 40))) \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate 0.005 \
    --multi_hot_distribution_type uniform \
    --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
