# Troubleshooting Guide for ITFormer Inference Results

This guide helps diagnose and resolve discrepancies between your inference results and the reported paper results.

## Common Issues and Solutions

### 1. Low Performance on Stage 3 (Reasoning Task)

**Symptom**: Stage 3 metrics (precision, recall, F1, exact_match_accuracy) are significantly lower than expected.

**Possible causes**:
- **Model checkpoint mismatch**: Ensure you're using the correct checkpoint version
- **Base LLM version**: Verify the Qwen2.5-Instruct model version matches the checkpoint
- **Dataset version**: Confirm you have the latest EngineMT-QA dataset
- **Training state**: The model may not have completed full training
- **Environment differences**: Library versions, precision settings, or hardware differences

### 2. General Performance Below Paper Results

**Possible causes**:
- **Incomplete model weights**: Checkpoint may be corrupted or incomplete
- **Wrong configuration**: Check yaml/infer.yaml settings match the paper
- **Dataset preprocessing**: Ensure data preprocessing matches training
- **Tokenizer issues**: Special tokens may not be properly configured

## Diagnostic Information to Collect

When reporting inference result discrepancies, please provide the following information:

### A. Environment Information
```bash
python diagnostics.py --check-environment
```

This will collect:
- Python version
- PyTorch version and CUDA availability
- Transformers library version
- Other key dependencies (accelerate, datasets, h5py, etc.)
- GPU information (if available)

### B. Model Information
```bash
python diagnostics.py --check-model --model_checkpoint checkpoints/ITFormer-0.5B
```

This will verify:
- Model checkpoint integrity (file existence and sizes)
- Configuration files (config.json)
- Base LLM model path and version
- Parameter counts
- Model architecture configuration

### C. Dataset Information
```bash
python diagnostics.py --check-dataset
```

This will check:
- Dataset file existence and sizes
- Number of samples per stage
- Data format validation
- Sample data preview

### D. Inference Configuration
```bash
python diagnostics.py --check-config --config yaml/infer.yaml
```

This will review:
- All configuration parameters
- Batch size and inference settings
- Model hyperparameters (d_model, n_heads, etc.)

### E. Training Loss Information (if available)

If you trained the model yourself, please provide:
- Training logs showing loss curves
- Final training/validation loss values
- Number of training epochs completed
- Learning rate schedule used
- Any early stopping or convergence issues

## Step-by-Step Diagnostic Process

### Step 1: Run Full Diagnostics
```bash
python diagnostics.py --full --model_checkpoint checkpoints/ITFormer-0.5B
```

This generates a comprehensive diagnostic report saved as `diagnostic_report.json`.

### Step 2: Compare with Expected Results

Expected ranges for ITFormer models (from paper):

**ITFormer-0.5B** (approximate expected ranges):
- Stage 1 (Description): BLEU ~0.30-0.35, ROUGE-1 ~0.55-0.60
- Stage 2 (Classification): F1 ~0.65-0.70, Accuracy ~0.70-0.75
- Stage 3 (Reasoning): F1 ~0.55-0.65, Accuracy ~0.60-0.70
- Stage 4 (Summary): BLEU ~0.20-0.25, ROUGE-1 ~0.50-0.55

**ITFormer-3B** (approximate expected ranges):
- Stage 1: BLEU ~0.35-0.40, ROUGE-1 ~0.60-0.65
- Stage 2: F1 ~0.70-0.75, Accuracy ~0.75-0.80
- Stage 3: F1 ~0.60-0.70, Accuracy ~0.65-0.75
- Stage 4: BLEU ~0.25-0.30, ROUGE-1 ~0.55-0.60

**ITFormer-7B** (best performance):
- Stage 1: BLEU ~0.40-0.45, ROUGE-1 ~0.65-0.70
- Stage 2: F1 ~0.75-0.80, Accuracy ~0.80-0.85
- Stage 3: F1 ~0.65-0.75, Accuracy ~0.70-0.80
- Stage 4: BLEU ~0.30-0.35, ROUGE-1 ~0.60-0.65

### Step 3: Verify Checkpoint Integrity

```bash
# Check file sizes match expected
python diagnostics.py --verify-checkpoint --model_checkpoint checkpoints/ITFormer-0.5B
```

Expected checkpoint sizes:
- ITFormer-0.5B: ~1.5-2.5 GB total
- ITFormer-3B: ~6-8 GB total
- ITFormer-7B: ~14-16 GB total

### Step 4: Test with Sample Data

```bash
# Run inference on a small sample to verify basic functionality
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-0.5B --max_samples 10
```

## Reporting an Issue

When opening a GitHub issue about inference results, please include:

1. **Your Results**: Complete JSON output from inference
2. **Diagnostic Report**: Output from `python diagnostics.py --full`
3. **Environment Details**: 
   - OS (Linux/Windows/Mac)
   - Python version
   - GPU type and CUDA version (if applicable)
   - Installation method (pip, conda, etc.)
4. **Steps to Reproduce**:
   - Exact commands used
   - Model checkpoint used
   - Dataset version
5. **Training Information** (if you trained the model):
   - Training logs
   - Final loss values
   - Number of epochs
   - Any modifications to training scripts

## Known Issues and Workarounds

### Issue 1: Stage 3 Performance Drop
- **Cause**: Model may need additional training on reasoning tasks
- **Workaround**: Try ITFormer-7B model which has better reasoning capabilities

### Issue 2: Special Token Mismatch
- **Cause**: Tokenizer special tokens not properly configured
- **Workaround**: Ensure `<|image_pad|>` token is added to tokenizer

### Issue 3: Precision Differences
- **Cause**: Different hardware or FP16 vs FP32
- **Workaround**: Try running with FP32 for more consistent results

## Getting Help

If issues persist after following this guide:

1. Run the full diagnostic: `python diagnostics.py --full --model_checkpoint <your_checkpoint>`
2. Open a GitHub issue with:
   - Title: "Inference Results Discrepancy: [Brief Description]"
   - Body: Include all information from "Reporting an Issue" section above
   - Attach: diagnostic_report.json

## Additional Resources

- Paper: https://arxiv.org/abs/2506.20093
- Dataset: https://huggingface.co/datasets/pandalin98/EngineMT-QA
- Models: https://huggingface.co/pandalin98/
- Issues: https://github.com/Pandalin98/ITFormer-ICML25/issues
