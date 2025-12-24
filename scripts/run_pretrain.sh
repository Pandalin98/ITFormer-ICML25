#!/bin/bash

# Pre-training script using accelerate
# 默认使用 accelerate 启动，配置文件为 accelerate_config.yaml

# 忽略 TRANSFORMERS_CACHE 废弃警告
export PYTHONWARNINGS="ignore::FutureWarning:transformers.utils.hub"

accelerate launch --config_file accelerate_config.yaml train_pretrain.py \
    --model TimeSeriesEncoder \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 4 \
    --patch_len 60 \
    --stride 60 \
    --input_len 600 \
    --output_dir save/pretrain_ts_small \
    --per_device_train_batch_size 1024 \
    --learning_rate 3e-4 \
    --num_train_epochs 10 \
    --dataloader_num_workers 8 \
    --report_to swanlab
