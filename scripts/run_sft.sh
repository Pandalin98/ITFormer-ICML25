#!/bin/bash

# SFT (Supervised Fine-Tuning) script using accelerate
# 默认加载预训练好的 ts_encoder 权重

# 忽略 TRANSFORMERS_CACHE 废弃警告
export PYTHONWARNINGS="ignore::FutureWarning:transformers.utils.hub"

accelerate launch --config_file accelerate_config.yaml train_sft.py \
    --model TimeSeriesEncoder \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 4 \
    --patch_len 60 \
    --stride 60 \
    --input_len 600 \
    --it_d_model 896 \
    --it_n_heads 16 \
    --it_layers 2 \
    --prefix_num 25 \
    --llm_model_path LLM/Qwen2.5-0.5B-Instruct \
    --load_ts_encoder save/pretrain/model.safetensors \
    --output_dir save/sft_qwen2.5_0.5B \
    --per_device_train_batch_size 24 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --dataloader_num_workers 4 \
    --report_to swanlab

