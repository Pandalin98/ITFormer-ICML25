# ITFormer

This repository provides the official open-source implementation of ITFormer (Instruct Time Transformer), a novel framework for temporal-textual multimodal question answering (QA).

## Overview

ITFormer (Instruct Time Transformer) is a state-of-the-art model for temporal-textual multimodal question answering. This repository provides the official open-source implementation with inference and training scripts.

Our work introduces a large-scale multitask dataset (EngineMT-QA) and demonstrates ITFormer's superior performance in bridging time series data with natural language understanding. Remarkably, our 0.5B model is lightweight and efficient while achieving strong performance.

## Features

- 📊 Pre-trained Models: Ready-to-use ITFormer models (0.5B, 3B, 7B) available on Hugging Face
- 🚀 Lightweight & Efficient: The 0.5B model offers strong temporal QA capabilities and easy deployment
- 🎯 Easy Setup: Simple download-and-run process
- 📈 High Performance: State-of-the-art results on temporal-textual QA benchmarks
- 🌐 Open Source: Models and dataset freely available on Hugging Face
- 📚 Large-scale Dataset: EngineMT-QA dataset with 118K+ time series samples and QA pairs

## Quick Start

### 1. Download Models and Dataset

Models and Dataset are open-sourced on Hugging Face:

- Dataset: https://huggingface.co/datasets/pandalin98/EngineMT-QA  
- ITFormer-0.5B: https://huggingface.co/pandalin98/ITFormer-0.5B  
- ITFormer-3B: https://huggingface.co/pandalin98/ITFormer-3B  
- ITFormer-7B: https://huggingface.co/pandalin98/ITFormer-7B

Download via Git/HF Hub:

```bash
# Install Git LFS for large file downloads
git lfs install

# Download the EngineMT-QA dataset
git clone https://huggingface.co/datasets/pandalin98/EngineMT-QA

# Download ITFormer models (choose one)
git clone https://huggingface.co/pandalin98/ITFormer-0.5B
# OR
git clone https://huggingface.co/pandalin98/ITFormer-3B
# OR
git clone https://huggingface.co/pandalin98/ITFormer-7B

# Alternative: huggingface_hub
pip install -q huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/EngineMT-QA', repo_type='dataset')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/ITFormer-7B')"
```

### 2. Organize Directory Structure

After downloading, organize your files as follows:

<pre>
ITFormer-ICML25/
├── dataset/
│   └── datasets/                    # Place EngineMT-QA dataset files here
│       ├── time_series_data.h5      # Time series data
│       ├── train_qa.jsonl           # Training QA pairs
│       └── test_qa.jsonl            # Test QA pairs
├── LLM/                             # Base LLM models
│   ├── Qwen2.5-0.5B-Instruct/
│   ├── Qwen2.5-3B-Instruct/
│   └── Qwen2.5-7B-Instruct/
├── checkpoints/
│   ├── ITFormer-0.5B/               # ITFormer model checkpoints
│   ├── ITFormer-3B/
│   └── ITFormer-7B/
└── yaml/
    └── infer.yaml                   # Inference configuration
</pre>

### 3. Download Base LLM Models

```bash
mkdir -p LLM
cd LLM
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
# OR
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# OR
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

# Alternative via HF Hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct', local_dir='./Qwen2.5-7B-Instruct')"
```

### 4. Configure Paths (Inference)

The configuration file yaml/infer.yaml is set with defaults. Update if needed:

```yaml
# Time series data path
ts_path_test: dataset/datasets/time_series_data.h5

# QA data path
qa_path_test: dataset/datasets/test_qa.jsonl

# Model configuration (already set)
model: TimeSeriesEncoder
d_model: 512
n_heads: 8
e_layers: 4
patch_len: 60
stride: 60
input_len: 600
dropout: 0.1

tt_d_model: 896
tt_n_heads: 16
tt_layers: 2
tt_dropout: 0.1
prefix_num: 25
```

### 5. Run Inference

```bash
# ITFormer-0.5B (default)
python inference.py --config yaml/infer.yaml

# ITFormer-3B
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-3B

# ITFormer-7B
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-7B
```

The inference script will automatically:
- Load the ITFormer model from checkpoints/
- Load the corresponding Qwen2.5-Instruct from LLM/ (based on model size)
- Load time series data and QA pairs from dataset/datasets/
- Run inference on the test set

Note: The script detects model size from the checkpoint path and aligns the Qwen2.5-Instruct variant accordingly.

## Training

We provide two training stages: (A) pretraining the time-series encoder, and (B) supervised fine-tuning (SFT) of the full ITFormer.

Before you start:
- Make sure dataset/datasets/ contains time_series_data.h5, train_qa.jsonl, test_qa.jsonl.
- By default, logging is integrated with SwanLab (imported as wandb) and runs in offline mode unless you configure it otherwise.

### A. Pretraining Time-Series Encoder

Script: train_pretrain.py  
Objective: Masked modeling / pretraining of the time-series encoder.

Common arguments (subset):
- --model TimeSeriesEncoder
- --d_model 512 --n_heads 8 --e_layers 4 --patch_len 60 --stride 60 --input_len 600 --dropout 0.1
- --pretrain true --min_mask_ratio 0.7 --max_mask_ratio 0.8
- --per_device_train_batch_size 12 --per_device_eval_batch_size 12
- --learning_rate 1e-5 --num_train_epochs 10 --weight_decay 1e-5
- --output_dir save/pretrain_ts_small --save_steps 100 --logging_steps 10 --report_to swanlab

Example:

```bash
# Pretrain the time-series encoder
python train_pretrain.py \
  --model TimeSeriesEncoder \
  --d_model 512 --n_heads 8 --e_layers 4 \
  --patch_len 60 --stride 60 --input_len 600 --dropout 0.1 \
  --pretrain true --min_mask_ratio 0.7 --max_mask_ratio 0.8 \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
  --learning_rate 1e-5 --num_train_epochs 10 --weight_decay 1e-5 \
  --output_dir save/pretrain_ts_small --save_steps 100 --logging_steps 10 \
  --report_to swanlab
```

Outputs:
- Encoder checkpoints and safetensors in save/pretrain_ts_small (e.g., save/pretrain/model.safetensors)

### B. Supervised Fine-Tuning (SFT) of ITFormer

Script: train_sft.py  
Objective: End-to-end SFT with time series + language (Qwen2.5-Instruct).

Key arguments (subset):
- Time-series encoder:
  - --model TimeSeriesEncoder
  - --d_model 512 --n_heads 8 --e_layers 4 --patch_len 60 --stride 60 --input_len 600 --dropout 0.1
  - --load_ts_encoder save/pretrain/model.safetensors  # path from stage A
- Temporal-Text module:
  - --tt_d_model 896 --tt_n_heads 16 --tt_layers 2 --tt_dropout 0.1 --prefix_num 25
- LLM:
  - --llm_model_path LLM/Qwen2.5-0.5B-Instruct  # or 3B/7B variant
- Training:
  - --do_train true
  - --per_device_train_batch_size 12 --per_device_eval_batch_size 12

Examples:

```bash
# SFT with 0.5B LLM (lightweight)
python train_sft.py \
  --llm_model_path LLM/Qwen2.5-0.5B-Instruct \
  --load_ts_encoder save/pretrain/model.safetensors \
  --model TimeSeriesEncoder \
  --d_model 512 --n_heads 8 --e_layers 4 \
  --patch_len 60 --stride 60 --input_len 600 --dropout 0.1 \
  --tt_d_model 896 --tt_n_heads 16 --tt_layers 2 --tt_dropout 0.1 --prefix_num 25 \
  --do_train true \
  --per_device_train_batch_size 12 --per_device_eval_batch_size 12
```

```bash
# SFT with 7B LLM (best performance)
python train_sft.py \
  --llm_model_path LLM/Qwen2.5-7B-Instruct \
  --load_ts_encoder save/pretrain/model.safetensors \
  --model TimeSeriesEncoder \
  --d_model 512 --n_heads 8 --e_layers 4 \
  --patch_len 60 --stride 60 --input_len 600 --dropout 0.1 \
  --tt_d_model 896 --tt_n_heads 16 --tt_layers 2 --tt_dropout 0.1 --prefix_num 25 \
  --do_train true \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4
```

Notes:
- Choose --llm_model_path consistent with your intended model size (0.5B/3B/7B).
- Ensure the time-series encoder weights are correctly specified by --load_ts_encoder.
- By default, training uses single-node setup; adjust batch size for your GPU memory.
- For reproducibility, you can set --fix_seed to a fixed integer.

### Recommended Environment

- Python 3.10+
- PyTorch (CUDA)
- transformers, accelerate, datasets, swanlab (or wandb), numpy, h5py, tqdm, huggingface_hub

Example installation:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # choose your CUDA
pip install -U transformers accelerate datasets numpy h5py tqdm huggingface_hub swanlab
```

## Model Architecture

ITFormer leverages advanced temporal reasoning combined with multimodal language understanding to achieve superior performance on temporal-textual QA tasks.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{wang2025itformer,
  title={ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset},
  author={Yilin Wang and Peixuan Lei and Jie Song and Yuzhe Hao and Tao Chen and Yuxuan Zhang and Lei Jia and Yuanxiang Li and Zhongyu Wei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

Paper: https://arxiv.org/abs/2506.20093

## License

MIT License — see the LICENSE file for details.

## Contact

For questions and issues, please open a GitHub issue or contact the authors.