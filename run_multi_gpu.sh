#!/bin/bash

# 在GPU 1-7上并行运行推理
# 每个GPU处理不同的数据分片

echo "🚀 Starting multi-GPU inference on GPUs 1-7..."

# 创建输出目录
mkdir -p inference_results

# 在后台启动7个进程，每个使用一个GPU
for gpu_id in {1..7}; do
    echo "Starting inference on GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python inference.py \
        --config yaml/infer.yaml \
        --model_checkpoint checkpoints/Qwen-0.5B \
        --output_dir inference_results/gpu_$gpu_id \
        --batch_size 4 \
        > logs/gpu_$gpu_id.log 2>&1 &
done

echo "✅ All inference processes started. Check logs/ directory for progress."
echo "Use 'ps aux | grep inference.py' to check running processes"
echo "Use 'tail -f logs/gpu_*.log' to monitor progress" 