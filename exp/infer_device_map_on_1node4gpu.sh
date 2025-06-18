#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L node=1
#PJM -L elapse=1:00:00  # REWRITE ME!!!!!
#PJM -j
#PJM -S

module load gcc/11.3.0 cuda/12.4.1 openmpi_cuda/4.0.5 nccl/2.19.3

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate llm_on_supercomp
cd `dirname $0`

# 使用モデルの設定
export MODEL_NAME_OR_PATH="Qwen/Qwen3-32B"

# データパス
export INPUT_DATA_PATH="../data/metainfo_top30.jsonl"
export MODEL_OUTPUT_DIR="./output"

python infer_device_map_on_1node4gpu.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--input_data_path $INPUT_DATA_PATH \
--model_output_dir $MODEL_OUTPUT_DIR