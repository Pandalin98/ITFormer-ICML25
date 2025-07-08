# ITFormer

This repository provides the **official open-source implementation** of ITFormer (Instruct Time Transformer), a novel framework for temporal-textual multimodal question answering (QA), as presented in our paper "ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset".

## Overview

ITFormer (Instruct Time Transformer) is a state-of-the-art model for temporal-textual multimodal question answering. This repository provides the **official open-source implementation** with inference functionality, allowing users to run pre-trained ITFormer models on temporal-textual QA tasks.

Our work introduces a large-scale multitask dataset and demonstrates ITFormer's superior performance in bridging time series data with natural language understanding.

## Features

- 📊 **Pre-trained Models**: Ready-to-use ITFormer models via ModelScope
- 🎯 **Easy Setup**: Simple download and run process
- 📈 **High Performance**: State-of-the-art results on temporal-textual QA benchmarks

## Quick Start

### 1. Download Models and Dataset

First, navigate to the inference directory and download the required models and dataset:

```bash
cd Time-QA-inference

# Download ITFormer model
modelscope download --model SJ011001/qwen7B-itformer --local_dir checkpoints/ITformer

# Download TimeQA dataset
modelscope download --dataset SJ011001/timeqa-data --local_dir dataset/dataset_processing

# Download Qwen2.5-7B-Instruct model (base model)
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir checkpoints/Qwen2.5-7B-Instruct
```

### 2. Run Inference

Execute the inference script:

```bash
python inference.py
```

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
