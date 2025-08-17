# ITFormer

This repository provides the **official open-source implementation** of ITFormer (Instruct Time Transformer), a novel framework for temporal-textual multimodal question answering (QA), as presented in our paper "ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset".

## Overview

ITFormer (Instruct Time Transformer) is a state-of-the-art model for temporal-textual multimodal question answering. This repository provides the **official open-source implementation** with inference functionality, allowing users to run pre-trained ITFormer models on temporal-textual QA tasks.

Our work introduces a large-scale multitask dataset (EngineMT-QA) and demonstrates ITFormer's superior performance in bridging time series data with natural language understanding. **Remarkably, our lightweight 0.5B model achieves better temporal QA performance than ChatGPT-4o while being significantly easier to deploy and more resource-efficient.** All models and datasets are now freely available on Hugging Face for the research community.

## Features

- 📊 **Pre-trained Models**: Ready-to-use ITFormer models (0.5B, 3B, 7B) available on Hugging Face
- 🚀 **Lightweight & Efficient**: The 0.5B model offers superior temporal QA capabilities compared to ChatGPT-4o while being much easier to deploy
- 🎯 **Easy Setup**: Simple download and run process
- 📈 **High Performance**: State-of-the-art results on temporal-textual QA benchmarks
- 🌐 **Open Source**: Complete models and dataset freely available on Hugging Face
- 📚 **Large-scale Dataset**: EngineMT-QA dataset with 118K+ time series samples and QA pairs

## Quick Start

### 1. Download Models and Dataset

🎉 **Models and Dataset are now open-sourced on Hugging Face!**

- **Dataset**: [https://huggingface.co/datasets/pandalin98/EngineMT-QA](https://huggingface.co/datasets/pandalin98/EngineMT-QA)
- **ITFormer-0.5B**: [https://huggingface.co/pandalin98/ITFormer-0.5B](https://huggingface.co/pandalin98/ITFormer-0.5B)
- **ITFormer-3B**: [https://huggingface.co/pandalin98/ITFormer-3B](https://huggingface.co/pandalin98/ITFormer-3B)
- **ITFormer-7B**: [https://huggingface.co/pandalin98/ITFormer-7B](https://huggingface.co/pandalin98/ITFormer-7B)

#### Download from Hugging Face

```bash
# Install Git LFS for large file downloads
git lfs install

# Download the EngineMT-QA dataset
git clone https://huggingface.co/datasets/pandalin98/EngineMT-QA

# Download ITFormer models (choose one based on your needs)
git clone https://huggingface.co/pandalin98/ITFormer-0.5B  # Lightweight version (beats ChatGPT-4o, easy deployment)
# OR
git clone https://huggingface.co/pandalin98/ITFormer-3B    # Medium version  
# OR
git clone https://huggingface.co/pandalin98/ITFormer-7B    # Full version (best performance)

# Alternative: Use huggingface-hub Python library
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/EngineMT-QA', repo_type='dataset')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/ITFormer-7B')"
```
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
├── LLM/                             # Create this directory for base LLM models
│   ├── Qwen2.5-0.5B-Instruct/       # Download corresponding size
│   ├── Qwen2.5-3B-Instruct/         # Download corresponding size
│   └── Qwen2.5-7B-Instruct/         # Download corresponding size
├── checkpoints/
│   ├── ITFormer-0.5B/               # Downloaded ITFormer-0.5B model (lightweight, beats ChatGPT-4o)
│   ├── ITFormer-3B/                 # Downloaded ITFormer-3B model (medium)
│   └── ITFormer-7B/                 # Downloaded ITFormer-7B model (full, best performance)
└── yaml/
    └── infer.yaml                   # Configuration file
</pre>

#### Download Base LLM Models

Download the corresponding Qwen2.5-Instruct models to the `LLM/` directory:

```bash
# Create LLM directory
mkdir -p LLM

# Download Qwen2.5 models (choose one based on your ITFormer model size)
cd LLM
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct  # For ITFormer-0.5B
# OR
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct    # For ITFormer-3B
# OR
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct    # For ITFormer-7B (recommended)

# Alternative: Use huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct', local_dir='./LLM/Qwen2.5-7B-Instruct')"
```

### 3. Configure Paths

The configuration file `yaml/infer.yaml` is already set up with default paths. If you need to modify paths, update the following:

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

### 4. Run Inference

Execute the inference script:

```bash
# For ITFormer-0.5B (default) - Lightweight & efficient, beats ChatGPT-4o
python inference.py --config yaml/infer.yaml

# For ITFormer-3B - Medium performance
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-3B

# For ITFormer-7B - Best performance
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-7B
```

The inference script will automatically:
- Load the ITFormer model from the specified `checkpoints/` directory
- Load the corresponding Qwen2.5-Instruct model from `LLM/` directory based on model size
- Load time series data from `dataset/datasets/time_series_data.h5`
- Load QA pairs from `dataset/datasets/test_qa.jsonl`
- Perform inference on the test set

**Note**: The script automatically detects the model size from the checkpoint path and loads the corresponding Qwen2.5-Instruct model (0.5B, 3B, or 7B).

## Model Architecture

ITFormer leverages advanced temporal reasoning capabilities combined with multimodal understanding to achieve superior performance on temporal-textual question answering tasks. The model effectively bridges time series data with natural language processing, enabling comprehensive understanding of temporal patterns and textual context.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{wang2025itformer,
  title={ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset},
  author={Yilin Wang and Peixuan Lei and Jie Song and Yuzhe Hao and Tao Chen and Yuxuan Zhang and Lei Jia and Yuanxiang Li and Zhongyu Wei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

**Paper**: [ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset](https://arxiv.org/abs/2506.20093)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and issues, please open an issue on GitHub or contact the authors.
