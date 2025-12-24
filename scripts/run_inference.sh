#!/bin/bash

# Inference script
# 使用指定的 checkpoint 进行推理

# 忽略 TRANSFORMERS_CACHE 废弃警告
export PYTHONWARNINGS="ignore::FutureWarning:transformers.utils.hub"

accelerate launch --config_file accelerate_config.yaml inference.py \
    --config yaml/infer.yaml \
    --model_checkpoint checkpoints/ITFormer-0.5B \
    --output_dir inference_results \
    --batch_size 12 \
    --num_workers 4 \
    --max_new_tokens 128

