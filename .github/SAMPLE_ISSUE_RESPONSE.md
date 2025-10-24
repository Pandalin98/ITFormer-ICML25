# Sample GitHub Issue Response

This is a sample response to the user who reported inference result discrepancies. Copy and paste this into the GitHub issue comment.

---

## English Response

Thank you for using ITFormer and reporting this issue! 

I notice that your results, particularly for Stage 3 (reasoning task), are below the expected ranges. To help diagnose this discrepancy, I need some additional information about your environment and setup.

### Please provide the following information:

#### 1. Run Diagnostic Tool

We've created a diagnostic tool to help identify issues. Please run:

```bash
python diagnostics.py --full --model_checkpoint checkpoints/ITFormer-0.5B
```

This will generate a `diagnostic_report.json` file. Please attach this file to your response.

#### 2. Model Information

- **Which model are you using?** ITFormer-0.5B / ITFormer-3B / ITFormer-7B
- **Checkpoint source**: Did you download from HuggingFace or train it yourself?
- **Base LLM version**: Which Qwen2.5-Instruct are you using?

#### 3. Training Information (if applicable)

If you trained the model yourself, please provide:
- **Training logs**: Loss curves and convergence behavior
- **Final loss values**: What were your final training/validation loss values?
- **Number of epochs**: How many epochs did you complete?
- **Pretrained encoder**: Did you load the pretrained time series encoder correctly?

#### 4. Dataset Information

- Where did you download the EngineMT-QA dataset from?
- When did you download it?
- Can you verify the number of samples per stage?

#### 5. Environment Details

- Operating System (Linux/Windows/Mac)
- Python version
- PyTorch version
- Transformers version
- GPU type (if applicable)

### Reference Resources

While waiting, you may find these helpful:
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Detailed diagnostic steps
- **[FAQ](FAQ.md)**: Common questions and expected performance ranges

### Expected Performance Ranges

For comparison, here are the expected ranges for each model:

**ITFormer-0.5B**:
- Stage 3 F1: ~0.55-0.65, Accuracy: ~0.60-0.70

**ITFormer-3B**:
- Stage 3 F1: ~0.60-0.70, Accuracy: ~0.65-0.75

Your reported Stage 3 results (F1: 0.345 for 0.5B, 0.237 for 3B) are significantly below these ranges, which suggests there may be an issue with the checkpoint, dataset, or training process.

Looking forward to your diagnostic information so we can help resolve this issue!

---

## Chinese Response (中文回复)

感谢您使用ITFormer并报告这个问题！

我注意到您的结果，特别是Stage 3（推理任务）的表现低于预期范围。为了帮助诊断这个差异，我需要一些关于您的环境和设置的额外信息。

### 请提供以下信息：

#### 1. 运行诊断工具

我们创建了一个诊断工具来帮助识别问题。请运行：

```bash
python diagnostics.py --full --model_checkpoint checkpoints/ITFormer-0.5B
```

这将生成一个 `diagnostic_report.json` 文件。请将此文件附在您的回复中。

#### 2. 模型信息

- **您使用的是哪个模型？** ITFormer-0.5B / ITFormer-3B / ITFormer-7B
- **Checkpoint来源**：是从HuggingFace下载的还是自己训练的？
- **Base LLM版本**：您使用的是哪个版本的Qwen2.5-Instruct？

#### 3. 训练信息（如果适用）

如果您自己训练了模型，请提供：
- **训练日志**：损失曲线和收敛行为
- **最终损失值**：您的最终训练/验证损失值是多少？
- **训练轮数**：您完成了多少个epoch？
- **预训练编码器**：是否正确加载了预训练的时序编码器？

#### 4. 数据集信息

- 您从哪里下载的EngineMT-QA数据集？
- 下载时间？
- 能否确认每个stage的样本数量？

#### 5. 环境详情

- 操作系统（Linux/Windows/Mac）
- Python版本
- PyTorch版本
- Transformers版本
- GPU型号（如果适用）

### 参考资源

在等待的同时，您可能会发现这些资源很有帮助：
- **[故障排除指南](TROUBLESHOOTING.md)**：详细的诊断步骤
- **[常见问题解答](FAQ.md)**：常见问题和预期性能范围

### 预期性能范围

供参考，以下是每个模型的预期范围：

**ITFormer-0.5B**：
- Stage 3 F1: ~0.55-0.65，准确率: ~0.60-0.70

**ITFormer-3B**：
- Stage 3 F1: ~0.60-0.70，准确率: ~0.65-0.75

您报告的Stage 3结果（F1: 0.5B为0.345，3B为0.237）明显低于这些范围，这表明checkpoint、数据集或训练过程可能存在问题。

期待您的诊断信息，以便我们能帮助解决这个问题！

---

## Follow-up Based on Diagnostic Report

After receiving the diagnostic report, analyze it and provide specific guidance:

### If checkpoint is incomplete or corrupted:
```
Based on your diagnostic report, I can see that [specific issue]. I recommend:
1. Re-downloading the checkpoint from HuggingFace
2. Verifying the file integrity after download
3. Ensuring you have enough disk space
```

### If model was self-trained with poor convergence:
```
Looking at your training logs, I notice [specific issue]. This suggests:
1. The training may not have converged properly
2. Consider training for more epochs
3. Verify the pretrained time series encoder was loaded correctly
4. Check if the learning rate was appropriate
```

### If environment issues detected:
```
Your diagnostic report shows [specific environment issue]. Please:
1. Update [specific library] to version X.X or higher
2. Ensure CUDA version is compatible with PyTorch
3. Try running inference with FP32 instead of FP16
```

### If dataset issues detected:
```
The diagnostic report indicates [dataset issue]. Please:
1. Re-download the EngineMT-QA dataset from HuggingFace
2. Verify the dataset has the correct number of samples per stage
3. Check that the data files are not corrupted
```
