# ITFormer - Temporal-Textual Multimodal Question Answering Framework

**ALWAYS follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.**

## Overview

ITFormer is a Python-based deep learning framework for temporal-textual multimodal question answering. It combines time series data with natural language processing using transformer architectures and large language models (Qwen2.5). The project requires GPU resources for optimal performance but can run on CPU for testing.

## Working Effectively

### Environment Setup and Dependencies

**NEVER CANCEL builds or installs - they may take 10+ minutes. Set timeout to 15+ minutes minimum.**

```bash
# Core Python dependencies - takes ~3-5 minutes. NEVER CANCEL.
pip install -r requirements.txt

# Additional required dependencies not in requirements.txt
pip install h5py nltk rouge-score scikit-learn
```

**Expected timing:**
- Main dependency installation: 3-5 minutes  
- Additional dependencies: 1-2 minutes
- **CRITICAL**: Set timeout to 15+ minutes for pip installs to avoid premature cancellation

### Project Structure Requirements

The project requires a specific directory structure to function correctly:

```
ITFormer-ICML25/
├── dataset/
│   └── datasets/                    # EngineMT-QA dataset files
│       ├── time_series_data.h5      # Time series data (required for inference)
│       ├── train_qa.jsonl           # Training QA pairs
│       └── test_qa.jsonl            # Test QA pairs (required for inference)
├── LLM/                             # Base LLM models directory
│   ├── Qwen2.5-0.5B-Instruct/       # For ITFormer-0.5B
│   ├── Qwen2.5-3B-Instruct/         # For ITFormer-3B  
│   └── Qwen2.5-7B-Instruct/         # For ITFormer-7B
├── checkpoints/                     # ITFormer model checkpoints
│   ├── ITFormer-0.5B/               # Lightweight model (beats ChatGPT-4o)
│   ├── ITFormer-3B/                 # Medium model
│   └── ITFormer-7B/                 # Best performance model
└── inference_results/               # Generated output directory
```

### Model and Dataset Downloads

**CRITICAL NETWORK REQUIREMENT**: Model downloads require internet access and may fail in restricted environments.

**Model download timing - NEVER CANCEL:**
- ITFormer-0.5B: ~15-30 minutes (smallest model)  
- ITFormer-7B: ~60-120 minutes (largest model)
- Qwen2.5 models: ~30-90 minutes each depending on size
- EngineMT-QA dataset: ~10-20 minutes

```bash
# Create required directories
mkdir -p LLM checkpoints dataset/datasets

# Install Git LFS for large model downloads  
git lfs install

# Download models (choose based on your needs - 0.5B recommended for testing)
# Set timeout to 120+ minutes for large model downloads
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/ITFormer-0.5B', local_dir='./checkpoints/ITFormer-0.5B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='./LLM/Qwen2.5-0.5B-Instruct')"

# Download dataset
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/EngineMT-QA', repo_type='dataset', local_dir='./dataset/datasets')"
```

### Running Inference

**Primary inference methods:**

```bash
# Method 1: Direct Python execution (single GPU/CPU)
python inference.py --config yaml/infer.yaml

# Method 2: Multi-GPU with Accelerate (recommended for production)
# Takes ~5-30 minutes depending on dataset size and model. NEVER CANCEL.
accelerate launch --config_file accelerate_config.yaml inference.py --config yaml/infer.yaml

# Method 3: Using provided script
bash infra.bash
```

**Model size options:**
```bash
# ITFormer-0.5B (default) - fastest, good performance
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-0.5B

# ITFormer-3B - medium performance  
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-3B

# ITFormer-7B - best performance, slowest
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-7B
```

**Expected inference timing - NEVER CANCEL:**
- Small dataset (100 samples): 2-5 minutes
- Medium dataset (1000 samples): 10-20 minutes  
- Full dataset (10k+ samples): 30-60 minutes
- **Set timeout to 90+ minutes for full inference runs**

### Validation and Testing

**Always run these validation steps after making changes:**

```bash
# 1. Test core imports (quick validation)
python -c "from models.TimeLanguageModel import TLM, TLMConfig; from dataset.dataset import TsQaDataset; from utils.metrics import open_question_metrics; print('✅ All modules import successfully')"

# 2. Test inference script help (validates argument parsing)
python inference.py --help

# 3. Syntax validation for all Python files
find . -name "*.py" -exec python -m py_compile {} \;

# 4. Test basic model initialization (requires models to be downloaded)
python -c "
import yaml
with open('yaml/infer.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Configuration file loads correctly')
"
```

## Configuration

**Main configuration file**: `yaml/infer.yaml`

Key parameters you might need to modify:
- `ts_path_test`: Path to time series data file
- `qa_path_test`: Path to QA pairs file  
- `batch_size`: Inference batch size (default: 12)
- GPU-related settings handled by `accelerate_config.yaml`

## Common Issues and Solutions

**Import errors**: Missing dependencies
```bash
# Error: ModuleNotFoundError: No module named 'h5py'
# Solution: Install additional dependencies
pip install h5py nltk rouge-score scikit-learn
```

**"Checkpoint path does not exist"**: Missing model files
```bash  
# Error: ValueError: Checkpoint path does not exist: checkpoints/ITFormer-0.5B
# Solution: Download required models or create directory structure
mkdir -p checkpoints/ITFormer-0.5B LLM/Qwen2.5-0.5B-Instruct dataset/datasets
```

**Network/download failures**: Restricted internet access
```bash
# Error: requests.exceptions.ConnectionError or LocalEntryNotFoundError
# Solution: Download models on a machine with internet access, then transfer
# Alternative: Use offline model loading if models are already available locally
```

**CUDA out of memory**: GPU memory insufficient
```bash
# Error: RuntimeError: CUDA out of memory
# Solution: Reduce batch size in yaml/infer.yaml
# Or run on CPU (slower but works): add --cpu flag to accelerate launch
```

**Accelerate multi-process errors**: Configuration mismatch
```bash
# Error: Process rank errors or device count mismatches
# Solution: Run with single process first, then debug multi-GPU setup
python inference.py --config yaml/infer.yaml  # Instead of accelerate launch
```

**Missing dataset files**: Data files not found
```bash
# Error: FileNotFoundError for time_series_data.h5 or test_qa.jsonl
# Solution: Ensure dataset is downloaded to correct location
ls -la dataset/datasets/  # Should show time_series_data.h5 and test_qa.jsonl
```

**Permission/filesystem errors**: Directory access issues
```bash
# Error: PermissionError or filesystem access denied
# Solution: Ensure you have write permissions to the working directory
chmod -R 755 . && mkdir -p checkpoints LLM dataset/datasets inference_results
```

## Key Components

**Core modules locations:**
- `inference.py`: Main inference script and entry point
- `models/TimeLanguageModel.py`: Core ITFormer model implementation
- `models/TimeSeriesEncoder.py`: Time series encoding components  
- `dataset/dataset.py`: Data loading and preprocessing
- `utils/metrics.py`: Evaluation metrics for QA performance
- `yaml/infer.yaml`: Main configuration file
- `accelerate_config.yaml`: Multi-GPU acceleration settings

**Model architecture**: ITFormer combines:
- Time series encoder (processes temporal data)
- Large language model (Qwen2.5 variants)
- Cross-modal attention mechanisms
- Question-answering head for generation

## Performance Expectations  

**ITFormer model performance:**
- 0.5B model: Beats ChatGPT-4o on temporal QA, fastest inference
- 3B model: Balanced performance and speed
- 7B model: Best accuracy, slowest inference

**Resource requirements:**
- Minimum RAM: 8GB (for 0.5B model)
- Recommended RAM: 16GB+ (for 3B+) 
- GPU memory: 4GB+ (0.5B), 12GB+ (7B)
- Storage: 50GB+ for all models and dataset

## Development Workflow

1. **Always validate imports first** - run the import validation command
2. **Test with smallest model (0.5B)** before trying larger models
3. **Use CPU mode for testing** when GPU resources are limited  
4. **Monitor memory usage** during inference runs
5. **Check output directory** `inference_results/` for generated results and metrics
6. **Verify configuration paths** in `yaml/infer.yaml` match your directory structure

## Troubleshooting Build/Runtime Issues

**For any "module not found" errors**: Re-run the dependency installation commands above
**For "file not found" errors**: Verify directory structure matches the required layout  
**For memory errors**: Reduce batch_size in configuration or use smaller model
**For slow performance**: Ensure GPU is available and properly configured with accelerate

## Complete Setup Workflow for New Clones

When working with a fresh clone, follow this exact sequence:

```bash
# 1. Install core dependencies (3-5 minutes, NEVER CANCEL)
pip install -r requirements.txt

# 2. Install additional dependencies (1-2 minutes) 
pip install h5py nltk rouge-score scikit-learn

# 3. Verify imports work
python -c "from models.TimeLanguageModel import TLM, TLMConfig; from dataset.dataset import TsQaDataset; from utils.metrics import open_question_metrics; print('✅ All modules import successfully')"

# 4. Create directory structure
mkdir -p LLM checkpoints dataset/datasets

# 5. Download models and dataset (30-120+ minutes total, NEVER CANCEL)
# Only if you have internet access - skip if in restricted environment
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/ITFormer-0.5B', local_dir='./checkpoints/ITFormer-0.5B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='./LLM/Qwen2.5-0.5B-Instruct')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/EngineMT-QA', repo_type='dataset', local_dir='./dataset/datasets')"

# 6. Test inference pipeline (only works if models/dataset are downloaded)
python inference.py --help  # Should show help without errors
```

## Repository Structure Reference

**Current repository contents (useful for navigation):**
```
ITFormer-ICML25/
├── .github/
│   └── copilot-instructions.md      # This file
├── models/                          # Core model implementations  
│   ├── TimeLanguageModel.py         # Main ITFormer model
│   ├── TimeSeriesEncoder.py         # Time series processing
│   ├── TT_Former.py                 # Transformer components
│   └── layers/                      # Neural network layers
├── dataset/                         # Data handling
│   └── dataset.py                   # Dataset loading logic
├── utils/                           # Utility functions
│   ├── metrics.py                   # Evaluation metrics
│   ├── position_coding.py           # Positional encodings
│   └── gradio_vlm.py               # UI components
├── yaml/
│   └── infer.yaml                   # Main configuration file
├── inference.py                     # Main entry point script
├── requirements.txt                 # Core dependencies
├── accelerate_config.yaml          # Multi-GPU settings
├── infra.bash                      # Shortcut script
└── README.md                       # Project documentation
```

**Key files you'll modify most often:**
- `inference.py`: Main inference logic and entry point
- `yaml/infer.yaml`: Configuration parameters  
- `models/TimeLanguageModel.py`: Core model architecture
- `dataset/dataset.py`: Data loading and preprocessing

## Common Development Tasks

**Testing changes without full model downloads:**
```bash
# Basic validation that doesn't require models
python -c "from models.TimeLanguageModel import TLM, TLMConfig; print('Core models loadable')"
python inference.py --help
find . -name "*.py" -exec python -m py_compile {} \;
```

**Monitoring inference progress:**
```bash
# Watch output directory for results
watch ls -la inference_results/

# Monitor GPU usage (if available)
nvidia-smi -l 5
```

**Quick syntax and import checks:**
```bash
# Check all Python files compile
find . -name "*.py" -exec python -m py_compile {} \; && echo "✅ All Python files compile successfully"

# Test core framework imports
python -c "
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
print('✅ Core ML libraries work')
"
```

Remember: This is a research framework optimized for temporal-textual QA tasks. Always start with the 0.5B model for testing and validation before moving to larger models.