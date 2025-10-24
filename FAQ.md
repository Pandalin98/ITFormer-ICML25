# Frequently Asked Questions - Inference Results

## General Questions

### Q1: Why are my inference results different from the paper?

Several factors can cause discrepancies:

1. **Model Version**: Ensure you're using the correct checkpoint version from HuggingFace
2. **Base LLM**: The Qwen2.5-Instruct model must match the size of your ITFormer checkpoint
3. **Dataset Version**: Use the latest EngineMT-QA dataset
4. **Environment Differences**: Different PyTorch/CUDA versions may produce slightly different results
5. **Precision**: FP16 vs FP32 can cause minor variations

**Action**: Run `python diagnostics.py --full --model_checkpoint <path>` to identify issues.

### Q2: Stage 3 (Reasoning) performance is significantly lower than expected. Why?

Stage 3 tasks require complex reasoning capabilities. Common causes:

1. **Model Size**: Smaller models (0.5B) naturally have lower reasoning performance than larger ones (7B)
2. **Incomplete Training**: If you trained the model yourself, it may need more epochs
3. **Data Preprocessing**: Ensure the test data matches the training data format
4. **Special Token Issues**: The `<|image_pad|>` token must be properly configured

**Expected Stage 3 F1 scores**:
- ITFormer-0.5B: ~0.55-0.65
- ITFormer-3B: ~0.60-0.70
- ITFormer-7B: ~0.65-0.75

**Action**: Try using a larger model (ITFormer-7B) for better reasoning performance.

### Q3: What are the expected performance ranges for each model?

Based on the paper and expected results:

**ITFormer-0.5B** (Lightweight, efficient):
- Stage 1 BLEU: 0.30-0.35, ROUGE-1: 0.55-0.60
- Stage 2 F1: 0.65-0.70, Accuracy: 0.70-0.75
- Stage 3 F1: 0.55-0.65, Accuracy: 0.60-0.70
- Stage 4 BLEU: 0.20-0.25, ROUGE-1: 0.50-0.55

**ITFormer-3B** (Balanced):
- Stage 1 BLEU: 0.35-0.40, ROUGE-1: 0.60-0.65
- Stage 2 F1: 0.70-0.75, Accuracy: 0.75-0.80
- Stage 3 F1: 0.60-0.70, Accuracy: 0.65-0.75
- Stage 4 BLEU: 0.25-0.30, ROUGE-1: 0.55-0.60

**ITFormer-7B** (Best performance):
- Stage 1 BLEU: 0.40-0.45, ROUGE-1: 0.65-0.70
- Stage 2 F1: 0.75-0.80, Accuracy: 0.80-0.85
- Stage 3 F1: 0.65-0.75, Accuracy: 0.70-0.80
- Stage 4 BLEU: 0.30-0.35, ROUGE-1: 0.60-0.65

Note: These are approximate ranges. Small variations (±0.02-0.05) are normal due to environmental differences.

## Model and Checkpoint Questions

### Q4: How do I verify my checkpoint is correct?

Run the verification tool:

```bash
python diagnostics.py --verify-checkpoint --model_checkpoint checkpoints/ITFormer-0.5B
```

Check:
1. Total checkpoint size matches expected (0.5B: ~1.5-2.5GB, 3B: ~6-8GB, 7B: ~14-16GB)
2. All required files exist (config.json, model.safetensors or pytorch_model.bin)
3. Model loads without errors
4. Parameter counts are reasonable

### Q5: Can I use a different base LLM (not Qwen2.5)?

No, ITFormer is specifically trained with Qwen2.5-Instruct models. Using a different base LLM will likely result in poor performance or errors. The model checkpoint includes adapters and weights specifically tuned for Qwen2.5.

### Q6: What if I trained the model myself and results are poor?

Check training logs for:
1. **Loss convergence**: Training loss should decrease steadily
2. **Final loss values**: Compare with expected ranges
3. **Number of epochs**: Ensure you completed the recommended number of epochs
4. **Learning rate**: Verify learning rate was appropriate (typically 1e-5 for SFT)
5. **Data quality**: Ensure training data preprocessing matches the official setup

**Action**: If training loss is high or didn't converge, consider:
- Training for more epochs
- Adjusting learning rate
- Checking data preprocessing pipeline
- Verifying pretrained time series encoder was loaded correctly

## Dataset Questions

### Q7: Does the dataset version matter?

Yes! Always use the latest version from HuggingFace:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pandalin98/EngineMT-QA', repo_type='dataset', local_dir='./dataset/datasets')"
```

Check dataset integrity:
```bash
python diagnostics.py --check-dataset
```

Expected:
- time_series_data.h5: ~1-2 GB
- test_qa.jsonl: Contains samples for all 4 stages

### Q8: Can I use my own dataset?

Yes, but you'll need to:
1. Follow the exact format of EngineMT-QA
2. Preprocess data the same way
3. Retrain or fine-tune the model
4. Results may differ significantly from paper

For inference with pretrained models, use the official dataset.

## Configuration Questions

### Q9: Should I modify yaml/infer.yaml?

The default configuration should work out-of-the-box. Only modify if:
- Your dataset files are in different locations (update `ts_path_test`, `qa_path_test`)
- You have memory constraints (reduce `batch_size`)
- You're using custom model architecture parameters (not recommended)

Most users should NOT need to modify other parameters like `d_model`, `n_heads`, etc.

### Q10: What batch size should I use?

Default batch size is 12, which works for most GPUs with 16GB+ memory.

Adjust based on your GPU:
- 8GB GPU: batch_size=4-6
- 16GB GPU: batch_size=12-16
- 24GB+ GPU: batch_size=16-24

Batch size doesn't significantly affect final metrics, only inference speed.

## Technical Questions

### Q11: Should I use FP16 or FP32?

- **FP16** (default): Faster inference, lower memory, may have tiny precision differences
- **FP32**: Slower, more memory, slightly more precise

For most use cases, FP16 is recommended. If you suspect precision issues, try FP32:

```yaml
# In yaml/infer.yaml
fp16: false
```

### Q12: CPU vs GPU inference - does it matter?

GPU is strongly recommended for reasonable inference speed. CPU will work but be much slower.

Results should be identical between CPU and GPU (with same precision settings).

### Q13: I get CUDA out of memory errors. What should I do?

1. Reduce batch size in yaml/infer.yaml
2. Use a smaller model (0.5B instead of 7B)
3. Enable FP16 if not already enabled
4. Close other GPU-consuming processes
5. Use gradient checkpointing (requires code modification)

### Q14: Special token warnings appear. Is this normal?

Some warnings about special tokens are expected, such as:
```
Adding <|image_pad|> token to tokenizer
```

This is normal and handled automatically by the code.

### Q15: Can I run inference with multiple GPUs?

Yes, use accelerate:

```bash
accelerate launch --config_file accelerate_config.yaml inference.py --config yaml/infer.yaml
```

Configure `accelerate_config.yaml` for your multi-GPU setup.

## Troubleshooting

### Q16: Where can I get more detailed diagnostics?

Run the comprehensive diagnostic tool:

```bash
python diagnostics.py --full --model_checkpoint checkpoints/ITFormer-0.5B
```

This generates `diagnostic_report.json` with complete environment, model, and dataset information.

### Q17: How do I report an issue effectively?

1. Run diagnostics: `python diagnostics.py --full`
2. Use the issue template: [Inference Results Discrepancy](.github/ISSUE_TEMPLATE/inference_results_discrepancy.md)
3. Attach `diagnostic_report.json`
4. Include complete inference results
5. Specify exact commands used

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed guidance.

### Q18: Are there known issues with specific environments?

Known issues:
- **Windows**: May require different path separators
- **M1/M2 Mac**: CUDA unavailable, use CPU or MPS backend
- **Old PyTorch (<2.0)**: May have compatibility issues
- **Transformers <4.30**: Special token handling may differ

Recommended environment:
- Linux (Ubuntu 20.04+)
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+

## Getting Help

### Q19: I've tried everything. Where can I get help?

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Run diagnostics and review the report
3. Search existing GitHub issues
4. Open a new issue with:
   - Diagnostic report
   - Complete results
   - Steps to reproduce
   - Environment details

### Q20: Can I contact the authors directly?

For general questions, please use GitHub issues. For collaboration or research inquiries, see contact information in the paper or repository README.

---

**Still have questions?** 

Open an issue: https://github.com/Pandalin98/ITFormer-ICML25/issues

Check the paper: https://arxiv.org/abs/2506.20093
