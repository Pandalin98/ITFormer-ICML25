#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${1:-save/sft_qwen2.5_0.5B/final}"
OUTPUT_DIR="${2:-runs/issue25_builtin_eval_2gpu_20260719}"
mkdir -p "${OUTPUT_DIR}"

accelerate launch --config_file accelerate_config_2gpu.yaml train_sft.py \
  --eval_only \
  --model_checkpoint "${CHECKPOINT}" \
  --eval_output_dir "${OUTPUT_DIR}" \
  --llm_model_path LLM/Qwen2.5-0.5B-Instruct \
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
  --per_device_eval_batch_size 12 \
  --bf16 \
  --dataloader_num_workers 4 \
  --report_to none \
  2>&1 | tee "${OUTPUT_DIR}/eval_builtin.log"
