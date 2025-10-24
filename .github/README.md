# .github Documentation Overview

This directory contains maintainer documentation and templates for handling user issues, particularly inference result discrepancies.

## Quick Navigation

### For Maintainers Responding to Issues

**Start here:** 
- 📋 **[QUICK_RESPONSE_GUIDE.md](QUICK_RESPONSE_GUIDE.md)** - Immediate action guide for current issue
- 📘 **[MAINTAINER_GUIDE.md](MAINTAINER_GUIDE.md)** - Complete workflow for handling inference issues

**Templates:**
- 💬 **[SAMPLE_ISSUE_RESPONSE.md](SAMPLE_ISSUE_RESPONSE.md)** - Ready-to-use bilingual response
- 📝 **[RESPONSE_TEMPLATE.md](RESPONSE_TEMPLATE.md)** - Templates for various scenarios

**Reference:**
- 📊 **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What was built and why

### For Users Reporting Issues

**Issue Template:**
- 🐛 **[ISSUE_TEMPLATE/inference_results_discrepancy.md](ISSUE_TEMPLATE/inference_results_discrepancy.md)** - Template for reporting inference issues

**User-facing documentation** (in repository root):
- [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md) - Troubleshooting guide
- [../FAQ.md](../FAQ.md) - Frequently asked questions
- [../diagnostics.py](../diagnostics.py) - Diagnostic tool

## File Structure

```
.github/
├── README.md                              # This file
├── QUICK_RESPONSE_GUIDE.md                # Quick start for responding to current issue
├── MAINTAINER_GUIDE.md                    # Complete maintainer workflow
├── SAMPLE_ISSUE_RESPONSE.md               # Sample response (English + Chinese)
├── RESPONSE_TEMPLATE.md                   # Response templates for common scenarios
├── IMPLEMENTATION_SUMMARY.md              # What was implemented and why
├── ISSUE_TEMPLATE/
│   └── inference_results_discrepancy.md   # User-facing issue template
└── copilot-instructions.md                # Copilot instructions (do not modify)
```

## Workflow for Handling Inference Issues

### Step 1: User Reports Issue
User opens issue using template or reports results in free form.

### Step 2: Initial Response (You)
1. Read **[QUICK_RESPONSE_GUIDE.md](QUICK_RESPONSE_GUIDE.md)**
2. Copy template response (bilingual)
3. Request diagnostic information
4. Point to resources (FAQ, Troubleshooting)

### Step 3: User Provides Information
User runs `diagnostics.py` and provides:
- `diagnostic_report.json`
- Training logs (if self-trained)
- Environment details

### Step 4: Analysis (You)
1. Review diagnostic report
2. Identify root cause
3. Use **[MAINTAINER_GUIDE.md](MAINTAINER_GUIDE.md)** for common scenarios

### Step 5: Provide Solution
Use templates from **[RESPONSE_TEMPLATE.md](RESPONSE_TEMPLATE.md)** for:
- Checkpoint issues
- Environment issues  
- Training issues
- Dataset issues

### Step 6: Follow Up
- Verify solution worked
- Update FAQ if new pattern found
- Close issue when resolved

## Common Scenarios

### Scenario 1: Downloaded Model, Poor Results
➡️ See MAINTAINER_GUIDE.md "Scenario 1"
- Likely: Corrupted download or wrong base LLM
- Action: Re-download and verify integrity

### Scenario 2: Self-Trained Model
➡️ See MAINTAINER_GUIDE.md "Scenario 2"
- Likely: Training convergence issues
- Action: Review training logs and loss values

### Scenario 3: Good Performance Except Stage 3
➡️ See MAINTAINER_GUIDE.md "Scenario 3"
- Likely: Model size limitation or reasoning capability
- Action: Suggest larger model (7B)

### Scenario 4: All Stages Slightly Low
➡️ See MAINTAINER_GUIDE.md "Scenario 4"
- Likely: Normal environmental variation
- Action: Confirm within acceptable range

## Key Tools

### For Users:
```bash
# Full diagnostic
python diagnostics.py --full --model_checkpoint checkpoints/ITFormer-0.5B

# Individual checks
python diagnostics.py --check-environment
python diagnostics.py --check-model --model_checkpoint <path>
python diagnostics.py --check-dataset
python diagnostics.py --verify-checkpoint --model_checkpoint <path>
```

### For Maintainers:
- Review `diagnostic_report.json` from users
- Use response templates
- Track common issues
- Update FAQ as needed

## Expected Performance Ranges

Quick reference for all stages:

### ITFormer-0.5B
- Stage 1: BLEU 0.30-0.35, ROUGE-1 0.55-0.60
- Stage 2: F1 0.65-0.70, Accuracy 0.70-0.75
- **Stage 3: F1 0.55-0.65, Accuracy 0.60-0.70** ⭐
- Stage 4: BLEU 0.20-0.25, ROUGE-1 0.50-0.55

### ITFormer-3B
- Stage 1: BLEU 0.35-0.40, ROUGE-1 0.60-0.65
- Stage 2: F1 0.70-0.75, Accuracy 0.75-0.80
- **Stage 3: F1 0.60-0.70, Accuracy 0.65-0.75** ⭐
- Stage 4: BLEU 0.25-0.30, ROUGE-1 0.55-0.60

### ITFormer-7B
- Stage 1: BLEU 0.40-0.45, ROUGE-1 0.65-0.70
- Stage 2: F1 0.75-0.80, Accuracy 0.80-0.85
- **Stage 3: F1 0.65-0.75, Accuracy 0.70-0.80** ⭐
- Stage 4: BLEU 0.30-0.35, ROUGE-1 0.60-0.65

⭐ Stage 3 is the most commonly reported problematic stage

## Variations Considered Normal

- ±0.02-0.05 difference: Environmental (PyTorch version, hardware)
- ±0.05-0.10 difference: May indicate minor issues (FP16/FP32, random seed)
- >0.10 difference: Significant issue requiring investigation
- >0.20 difference: Critical issue (wrong model, corrupted data, failed training)

## Response Time Guidelines

- **Initial response**: Within 24 hours
- **After diagnostic report**: Within 24 hours of receipt
- **Follow-up**: 1-2 days for complex issues
- **Expected resolution**: 1-7 days depending on complexity

## Escalation

If issue persists after 2-3 exchanges:
1. Label as `needs-investigation`
2. Check if multiple users report similar issue
3. Consider if checkpoint or code bug
4. May need to re-release checkpoint or update code

## Language Support

All maintainer documentation includes:
- **English** - Primary documentation
- **Chinese (中文)** - Key templates and responses
- Users can report issues in either language

## Updates and Maintenance

When updating this documentation:
1. Keep all templates synchronized
2. Update FAQ with new common issues
3. Enhance diagnostic tool as needed
4. Track metrics on issue resolution
5. Update expected performance ranges if models updated

## Questions?

For questions about this documentation:
- Review the full MAINTAINER_GUIDE.md
- Check IMPLEMENTATION_SUMMARY.md for context
- Refer to copilot-instructions.md for project overview

---

**Quick Start**: Read QUICK_RESPONSE_GUIDE.md and respond to the user's issue!
