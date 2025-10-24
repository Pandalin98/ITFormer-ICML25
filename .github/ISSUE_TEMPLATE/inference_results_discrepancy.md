---
name: Inference Results Discrepancy
about: Report inference results that differ from expected/paper results
title: '[Inference] '
labels: 'inference, help wanted'
assignees: ''
---

## Issue Description

Thank you for using ITFormer! To help us diagnose inference result discrepancies, please provide the information below.

### Your Inference Results

Please paste your complete inference results (JSON format):

```json
{
  "stage1_BLEU": 0.xxx,
  "stage1_rouge1": 0.xxx,
  ...
}
```

### Expected vs Actual

Which stage(s) show the largest discrepancy?
- [ ] Stage 1 (Description)
- [ ] Stage 2 (Classification)
- [ ] Stage 3 (Reasoning)
- [ ] Stage 4 (Summary)

### Environment Information

**IMPORTANT**: Please run the diagnostic tool and attach the report:

```bash
python diagnostics.py --full --model_checkpoint <your_checkpoint_path>
```

Then attach the generated `diagnostic_report.json` file to this issue.

**Manual Information** (if diagnostic tool cannot run):

- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- Python version: [e.g., 3.10.12]
- PyTorch version: [e.g., 2.1.0]
- Transformers version: [e.g., 4.36.0]
- CUDA version (if using GPU): [e.g., 12.1]
- GPU model (if applicable): [e.g., NVIDIA RTX 3090]

### Model Information

- Model used: [e.g., ITFormer-0.5B, ITFormer-3B, ITFormer-7B]
- Checkpoint source: [e.g., downloaded from HuggingFace, self-trained]
- Base LLM version: [e.g., Qwen2.5-0.5B-Instruct]

**If self-trained**, please provide:
- Training logs (loss curves)
- Final training/validation loss
- Number of epochs completed
- Any early stopping or convergence issues

### Dataset Information

- Dataset source: [e.g., downloaded from HuggingFace]
- Dataset version/date: [e.g., downloaded on 2024-XX-XX]
- Number of test samples: [can be found in diagnostic report]

### Inference Configuration

Please provide your inference configuration (yaml/infer.yaml):

```yaml
# Paste your configuration here
```

Or attach the file.

### Steps to Reproduce

Please provide the exact commands you used:

```bash
# Example:
python inference.py --config yaml/infer.yaml --model_checkpoint checkpoints/ITFormer-0.5B
```

### Additional Context

Any other information that might be helpful:
- Are you using CPU or GPU?
- Did you modify any code?
- Are there any error messages or warnings?
- Have you tried with different model sizes?

### Checklist

Before submitting, please ensure you have:
- [ ] Run the diagnostic tool: `python diagnostics.py --full`
- [ ] Attached the `diagnostic_report.json` file
- [ ] Provided complete inference results (all stages)
- [ ] Specified which model checkpoint you're using
- [ ] Checked the [Troubleshooting Guide](../TROUBLESHOOTING.md)

---

**For Maintainers**: Please review the diagnostic report and check:
- [ ] Environment matches expected setup
- [ ] Model checkpoint integrity verified
- [ ] Dataset version is correct
- [ ] Configuration parameters are standard
- [ ] Compare with known baseline results
