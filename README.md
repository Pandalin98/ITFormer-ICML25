# ITFormer-ICML25-

 This repository provides the official implementation of ITFormer, a novel framework for temporal-textual multimodal question answering (QA), as presented in our ICML 2025 paper.


## Quick_start

Download model and dataset

```
cd Time-QA-inference
modelscope download --model SJ011001/qwen7B-itformer --local_dir checkpoints/ITformer
modelscope download --dataset SJ011001/timeqa-data  --local_dir dataset/dataset_processing
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir checkpoints/Qwen2.5-7B-Instruct
```

Inference

```
python /dataYYF/dataWX/SJ/aipt/Time-QA-inference/inference.py
```
