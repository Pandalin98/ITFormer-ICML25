# ITFormer

This repository provides the official open-source implementation of ITFormer (Instruct Time Transformer), a novel framework for temporal-textual multimodal question answering (QA).

## Overview

ITFormer (Instruct Time Transformer) is a state-of-the-art model for temporal-textual multimodal question answering. This repository provides the official open-source implementation with inference and training scripts.

Our work introduces a large-scale multitask dataset (EngineMT-QA) and demonstrates ITFormer's superior performance in bridging time series data with natural language understanding. Remarkably, our 0.5B model is lightweight and efficient while achieving strong performance.

## Features

- 📊 **Pre-trained Models**: Ready-to-use ITFormer models (0.5B, 3B, 7B) available on Hugging Face.
- 🚀 **Lightweight & Efficient**: The 0.5B model offers strong temporal QA capabilities and easy deployment.
- 🎯 **One-Click Scripts**: Automated scripts for pre-training, SFT, and parallel inference.
- 📈 **High Performance**: State-of-the-art results on temporal-textual QA benchmarks.
- 🌐 **Distributed Support**: Fully compatible with `accelerate` for multi-GPU training and inference.

## Quick Start

### 1. Organize Directory Structure

After downloading models and datasets, organize your files as follows:

<pre>
ITFormer-ICML25/
├── dataset/
│   └── datasets/                    # Place EngineMT-QA dataset files here
│       ├── time_series_data.h5
│       ├── train_qa.jsonl
│       └── test_qa.jsonl
├── LLM/                             # Base Qwen2.5-Instruct models
├── checkpoints/
│   └── ITFormer-0.5B/               # ITFormer model checkpoints
├── scripts/                         # One-click automation scripts
│   ├── run_pretrain.sh
│   ├── run_sft.sh
│   └── run_inference.sh
├── accelerate_config.yaml           # Configuration for distributed execution
└── yaml/
    └── infer.yaml                   # Inference configuration
</pre>

### 2. Run Inference

We now support **parallel inference** using `accelerate`. This automatically aggregates results from multiple GPUs.

```bash
# Using the automated script (Recommended)
bash scripts/run_inference.sh

# Or launch manually via accelerate
accelerate launch --config_file accelerate_config.yaml inference.py --config yaml/infer.yaml
```

The inference script will:
- Load ITFormer and the corresponding Qwen2.5-Instruct.
- Distribute data across all available GPUs.
- Aggregate and save results to `inference_results/` and `output_result_all.json`.

---

## Training

We provide a streamlined training pipeline using `accelerate`. Ensure your `accelerate_config.yaml` is properly configured for your hardware.

### A. Pre-training (Time-Series Encoder)

Stage A focuses on pre-training the `TimeSeriesEncoder` using masked modeling.

```bash
# One-click pre-training
bash scripts/run_pretrain.sh
```

### B. Supervised Fine-Tuning (SFT)

Stage B performs end-to-end SFT, bridging the time-series encoder with the LLM via ITFormer.

```bash
# One-click SFT (Requires pre-trained ts_encoder weights)
bash scripts/run_sft.sh
```

**Key Parameters in SFT:**
- `--it_d_model`, `--it_n_heads`, `--it_layers`: Configuration for the ITFormer module.
- `--load_ts_encoder`: Path to the weights generated in Stage A.
- `--llm_model_path`: Path to the base Qwen2.5-Instruct model.

---

## Model Architecture

ITFormer leverages an **Instruction-aware Time Series Transformer** to align temporal features with textual queries before feeding them into a Large Language Model. The framework is designed to be parameter-efficient, freezing the LLM and TS Encoder during SFT while training only the ITFormer and projection layers.

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

## License

MIT License — see the LICENSE file for details.
