# Maintainer Guide: Handling Inference Result Issues

This guide helps maintainers respond to users reporting inference result discrepancies.

## Quick Response Workflow

When a user reports inference results that differ from the paper:

### Step 1: Initial Response (Within 24 hours)

Use the template from `.github/SAMPLE_ISSUE_RESPONSE.md` to:
1. Thank the user
2. Request diagnostic information
3. Point to troubleshooting resources

**Key request**: Ask user to run diagnostic tool and provide:
```bash
python diagnostics.py --full --model_checkpoint <their_checkpoint>
```

### Step 2: Analysis (After receiving diagnostic report)

Review the `diagnostic_report.json` file the user provides. Check for:

#### Common Issues to Look For:

1. **Checkpoint Issues**
   - File size doesn't match expected (0.5B: ~1.5-2.5GB, 3B: ~6-8GB, 7B: ~14-16GB)
   - Missing required files (config.json, model.safetensors)
   - Verification failed when loading model

2. **Environment Issues**
   - PyTorch version < 2.0
   - Transformers version < 4.30
   - Missing required libraries (h5py, nltk, rouge_score, sklearn)
   - CUDA version mismatch

3. **Dataset Issues**
   - Dataset files missing or corrupted
   - Wrong number of samples per stage
   - Old dataset version

4. **Configuration Issues**
   - Wrong model parameters in yaml/infer.yaml
   - Batch size too large causing memory issues
   - Wrong paths to dataset files

5. **Training Issues (if self-trained)**
   - Training loss didn't converge
   - Insufficient training epochs
   - Pretrained encoder not loaded
   - Wrong learning rate or optimizer settings

### Step 3: Provide Specific Guidance

Based on the diagnostic report, provide targeted solutions:

#### If checkpoint integrity issue:
```markdown
Based on your diagnostic report, the checkpoint appears to be [incomplete/corrupted]. 
Please try:
1. Re-downloading from HuggingFace: [link]
2. Verifying file checksums
3. Ensuring sufficient disk space during download
```

#### If environment issue:
```markdown
Your environment shows [specific issue]. Please:
1. Update PyTorch to 2.0+: `pip install -U torch`
2. Update Transformers to 4.30+: `pip install -U transformers`
3. Install missing dependencies: `pip install h5py nltk rouge-score scikit-learn`
```

#### If dataset issue:
```markdown
The dataset appears to have [specific issue]. Please:
1. Re-download the latest EngineMT-QA dataset
2. Verify the file sizes match expected values
3. Check the number of samples per stage
```

#### If self-training issue:
```markdown
Looking at your training information, I see [specific issue]. This suggests:
1. The model may need more training epochs
2. Verify the pretrained time series encoder was loaded correctly
3. Check if the training loss converged properly
4. Consider adjusting the learning rate
```

### Step 4: Performance Expectations

Compare user's results against expected ranges:

**Stage 3 Performance** (most commonly problematic):
- ITFormer-0.5B: F1 ~0.55-0.65, Accuracy ~0.60-0.70
- ITFormer-3B: F1 ~0.60-0.70, Accuracy ~0.65-0.75
- ITFormer-7B: F1 ~0.65-0.75, Accuracy ~0.70-0.80

If user's Stage 3 results are:
- **Within ±0.05 of expected**: Normal variation, likely due to environment differences
- **0.10-0.20 below expected**: Checkpoint or training issue
- **>0.20 below expected**: Critical issue (wrong checkpoint, corrupted data, or incomplete training)

### Step 5: Follow-up

If initial solutions don't work:
1. Request more detailed logs
2. Ask for specific training checkpoints at different epochs
3. Consider scheduling a call for complex issues
4. Check if this is a systematic issue affecting multiple users

## Common User Scenarios

### Scenario 1: Downloaded Checkpoint, Poor Stage 3 Performance

**Likely causes**:
- Corrupted download
- Wrong base LLM version
- Old checkpoint version

**Solution**:
1. Re-download both ITFormer checkpoint and matching Qwen2.5-Instruct
2. Verify file integrity
3. Clear cache and retry

### Scenario 2: Self-Trained Model, All Stages Underperform

**Likely causes**:
- Training not converged
- Pretrained encoder not loaded
- Wrong training configuration

**Solution**:
1. Request training logs and loss curves
2. Verify pretrained encoder was loaded with `--load_ts_encoder`
3. Check training ran for sufficient epochs
4. Verify learning rate and optimizer settings

### Scenario 3: Good Results for Stages 1,2,4 but Poor Stage 3

**Likely causes**:
- Model size limitation (0.5B has lower reasoning capability)
- Training data imbalance
- Evaluation metric issue

**Solution**:
1. Suggest trying ITFormer-7B if using smaller model
2. Verify Stage 3 samples in dataset are correct
3. Check if special token handling is correct

### Scenario 4: All Stages Slightly Lower (~0.03-0.05)

**Likely causes**:
- Environmental differences (PyTorch version, hardware)
- FP16 vs FP32 precision
- Random seed effects

**Solution**:
This is normal variation. Suggest:
1. Verify this is consistent across multiple runs
2. Try FP32 for more precision (slower)
3. Acceptable if within expected range

## Resources for Users

Always point users to:
1. **TROUBLESHOOTING.md**: Detailed diagnostic guide
2. **FAQ.md**: Common questions and expected performance
3. **diagnostics.py**: Automated diagnostic tool
4. **Issue template**: Structured way to report problems

## Escalation

If the issue cannot be resolved after 2-3 back-and-forth exchanges:
1. Label as `needs-investigation`
2. Consider if it's a bug in the checkpoint or code
3. Check if multiple users report similar issues
4. May need to re-release checkpoint or update code

## Templates

Use these template files:
- `.github/RESPONSE_TEMPLATE.md`: Detailed response templates
- `.github/SAMPLE_ISSUE_RESPONSE.md`: Sample initial response
- `.github/ISSUE_TEMPLATE/inference_results_discrepancy.md`: User-facing template

## Metrics to Track

Keep track of:
- Most common causes of discrepancies
- Which stages are most problematic
- Success rate of diagnostic tool in identifying issues
- Average time to resolution

## Continuous Improvement

Based on recurring issues:
1. Update FAQ with new common questions
2. Enhance diagnostic tool to catch more issues
3. Improve checkpoint distribution process
4. Add more validation checks in inference code
5. Consider adding automated checkpoint integrity verification

---

Remember: Most inference discrepancies are due to:
1. Checkpoint download/integrity issues (40%)
2. Environment/dependency issues (30%)
3. Self-training problems (20%)
4. Dataset issues (10%)

The diagnostic tool should catch 80%+ of these issues automatically.
