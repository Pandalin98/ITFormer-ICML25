# Response Template for Inference Result Issues

Use this template when responding to users reporting inference result discrepancies.

## Initial Response Template (Chinese)

感谢您使用ITFormer并报告这个问题！为了更好地帮助您诊断推理结果的差异，请提供以下信息：

### 1. 环境信息 (Environment Information)

请运行诊断工具来收集您的环境信息：

```bash
python diagnostics.py --full --model_checkpoint <您使用的checkpoint路径>
```

这将生成一个 `diagnostic_report.json` 文件。请将此文件附在回复中。

### 2. 模型信息 (Model Information)

- **使用的模型**：ITFormer-0.5B / ITFormer-3B / ITFormer-7B
- **checkpoint来源**：从HuggingFace下载 / 自己训练
- **Base LLM版本**：Qwen2.5-0.5B-Instruct / Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct

如果是**自己训练的模型**，请额外提供：
- 训练日志（training logs）
- 最终的训练/验证损失值（final training/validation loss）
- 完成的训练轮数（number of epochs completed）
- 是否有早停或收敛问题（any early stopping or convergence issues）

### 3. 数据集信息 (Dataset Information)

- 数据集来源：从HuggingFace下载 / 其他
- 下载日期：
- 测试集样本数量：

### 4. 推理配置 (Inference Configuration)

请提供您的配置文件内容（`yaml/infer.yaml`）或确认使用的是默认配置。

### 5. 具体问题 (Specific Issues)

哪个阶段的表现与预期差异最大？
- Stage 1 (描述任务 / Description)
- Stage 2 (分类任务 / Classification)  
- Stage 3 (推理任务 / Reasoning)
- Stage 4 (摘要任务 / Summary)

### 参考资源 (Reference Resources)

在等待回复的同时，您可以参考：
- [故障排除指南 (Troubleshooting Guide)](../TROUBLESHOOTING.md)
- [常见问题解答 (FAQ)](../FAQ.md)

---

## Initial Response Template (English)

Thank you for using ITFormer and reporting this issue! To help diagnose the discrepancy in your inference results, please provide the following information:

### 1. Environment Information

Please run the diagnostic tool to collect your environment information:

```bash
python diagnostics.py --full --model_checkpoint <your_checkpoint_path>
```

This will generate a `diagnostic_report.json` file. Please attach this file in your response.

### 2. Model Information

- **Model used**: ITFormer-0.5B / ITFormer-3B / ITFormer-7B
- **Checkpoint source**: Downloaded from HuggingFace / Self-trained
- **Base LLM version**: Qwen2.5-0.5B-Instruct / Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct

If **self-trained**, please also provide:
- Training logs
- Final training/validation loss values
- Number of epochs completed
- Any early stopping or convergence issues

### 3. Dataset Information

- Dataset source: Downloaded from HuggingFace / Other
- Download date:
- Number of test samples:

### 4. Inference Configuration

Please provide your configuration file contents (`yaml/infer.yaml`) or confirm you're using the default configuration.

### 5. Specific Issues

Which stage(s) show the largest discrepancy from expectations?
- Stage 1 (Description)
- Stage 2 (Classification)
- Stage 3 (Reasoning)
- Stage 4 (Summary)

### Reference Resources

While waiting for a response, you may find these resources helpful:
- [Troubleshooting Guide](../TROUBLESHOOTING.md)
- [FAQ](../FAQ.md)

---

## Follow-up Questions Based on Common Issues

### If Stage 3 performance is low:

Stage 3 (推理任务) 对模型的推理能力要求较高。请确认：

1. 是否使用了正确的模型checkpoint？较小的模型（0.5B）在Stage 3上的表现会比大模型（7B）差一些
2. 如果是自训练模型，训练loss是否收敛？
3. base LLM (Qwen2.5-Instruct) 的版本是否与checkpoint匹配？

Stage 3 (Reasoning task) requires stronger reasoning capabilities. Please confirm:

1. Are you using the correct model checkpoint? Smaller models (0.5B) naturally have lower Stage 3 performance than larger models (7B)
2. If self-trained, did the training loss converge properly?
3. Does your base LLM (Qwen2.5-Instruct) version match the checkpoint?

### If all stages are lower than expected:

整体性能低于预期可能是由于：

1. **Checkpoint问题**：文件可能损坏或不完整，请重新下载
2. **数据集版本**：确保使用最新的EngineMT-QA数据集
3. **环境问题**：库版本不兼容，请查看diagnostic report

Overall lower performance may be due to:

1. **Checkpoint issues**: Files may be corrupted or incomplete, try re-downloading
2. **Dataset version**: Ensure you're using the latest EngineMT-QA dataset
3. **Environment issues**: Incompatible library versions, check the diagnostic report

### If checkpoint is self-trained:

请提供训练相关信息：

1. 预训练的时序编码器是否正确加载？（`--load_ts_encoder` 参数）
2. 训练loss曲线如何？最终loss值是多少？
3. 训练了多少个epoch？是否完成了推荐的训练轮数？
4. 使用的学习率是多少？

Please provide training information:

1. Was the pretrained time series encoder loaded correctly? (`--load_ts_encoder` parameter)
2. What does the training loss curve look like? What was the final loss value?
3. How many epochs did you train? Did you complete the recommended number of epochs?
4. What learning rate did you use?

---

## Resolution Templates

### After receiving diagnostic report:

根据您提供的diagnostic report，我看到：

[具体问题分析，例如：]
- 模型checkpoint文件完整性正常
- 环境配置符合要求
- 但是注意到数据集的Stage 3样本数量偏少

建议：
[具体建议]

Based on your diagnostic report, I can see:

[Specific issue analysis, e.g.:]
- Model checkpoint files are intact
- Environment configuration meets requirements  
- However, I notice the number of Stage 3 samples in the dataset is low

Recommendation:
[Specific recommendations]

### If issue is resolved:

很高兴问题已经解决！如果您愿意，可以分享一下最终的解决方案，这可能对其他用户有帮助。

Glad the issue is resolved! If you'd like, please share the final solution - it may help other users.

### If more investigation needed:

我们需要进一步调查这个问题。能否提供：
[额外需要的信息]

同时，您可以尝试：
[临时解决方案或workaround]

We need to investigate this further. Could you provide:
[Additional information needed]

Meanwhile, you could try:
[Temporary solutions or workarounds]
