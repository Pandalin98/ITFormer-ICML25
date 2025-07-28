#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import json
import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer,AutoProcessor
from dataset.dataset import TsQaDataset, DataCollator
from models.TimeLanguageModel import TLM, TLMConfig
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import open_question_metrics,closed_question_metrics,compute_rul
from typing import List, Dict, Any
from accelerate import Accelerator  

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_results(results, output_dir, config_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inference_results_{config_name}_{timestamp}.json"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Inference results saved to: {filepath}")


def main_inference(args):

    set_seed(args.seed)
    print("🚀 Starting inference process...")
    print(f"🔧 Using config file: {args.config}")
    print(f"💾 Results will be saved to: {args.output_dir}")
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)

    print("\n🔨 Building model configuration...")
    tlmconfig = TLMConfig(
        ts_pad_num=args.prefix_num
    )
    
    print("⚙️ Initializing model...")
    
    # Simple model loading like original code
    model = TLM.from_pretrained(args.model_checkpoint, config=tlmconfig, ts_config=args)
    
    # Initialize LLM components first
    model._initialize_llm_components(args.model_checkpoint)
    
    # 打印model的总参数量
    # 分别记录llm和itformer和ts_encoder的参数，用M做单位
    def count_params(module):
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters())

    llm_params = count_params(model.llm_model) / 1e6
    itformer_params = count_params(model.itformer) / 1e6
    ts_encoder_params = count_params(model.ts_encoder) / 1e6
    total_params = count_params(model) / 1e6

    print(f"\n🔢 Model parameter counts (in M):")
    print(f"   LLM:         {llm_params:.2f}M")
    print(f"   ITFormer:    {itformer_params:.2f}M")
    print(f"   TS_Encoder:  {ts_encoder_params:.2f}M")
    print(f"   Total:       {total_params:.2f}M")
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    model.eval()
    print(f"✅ Model loaded successfully! Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    print("\n📊 Preparing test dataset...")
    # Load tokenizer directly from LLM/Qwen2.5-0.5B-Instruct
    llm_path = "LLM/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(llm_path, trust_remote_code=True)
    
    tokenizer.padding_side = 'left'
    
    test_dataset = TsQaDataset(
        args.ts_path_test,
        args.qa_path_test,
        tokenizer,
        processor,
        tlmconfig
    )
    data_collator = DataCollator(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers
    )
    
    # Prepare dataloader with accelerator
    test_loader = accelerator.prepare(test_loader)
    
    print(f"📁 Test set size: {len(test_dataset)} samples")
    print(f"🔢 Batch size: {args.batch_size}, Total batches: {len(test_loader)}")
    print("\n🔍 Starting test set inference...")
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference progress")):
            # if batch_idx==20:
            #     break
            # Data is already on the correct device thanks to accelerator
            
            generated_ids = model.generate(
                input_ids=batch['input_ids'],
                query_ids=batch['query_ids'],
                ts_values=batch['ts_values'],
                stage=batch['stage'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=args.do_sample,
                temperature=args.temperature,
                num_beams=args.num_beams
            )
            
            batch_predictions = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            batch_labels = tokenizer.batch_decode(
                batch['labels'], 
                skip_special_tokens=True
            )
            
            for i in range(len(batch_predictions)):
                prediction = batch_predictions[i].split('assistant\n')[-1]
                results.append({
                    "index": batch['index'][i].item(),
                    "stage": batch['stage'][i].item(),
                    "input": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                    "prediction": prediction,
                    "label": batch_labels[i],
                    "is_correct": prediction.strip() == batch_labels[i].strip()
                })
            
    print("\n📊 Calculating evaluation metrics...")
    metrics = compute_metrics_from_results(results)
    print_metrics(metrics)

    config_base = os.path.basename(args.config).split('.yaml')[0]
    save_results(results, args.output_dir, config_base)
    save_metrics(metrics, args.output_dir, config_base)

def compute_metrics_from_results(results: List[Dict]) -> Dict[str, Any]:

    stage1_data = [r for r in results if r['stage'] == 1]
    stage2_data = [r for r in results if r['stage'] == 2]
    stage3_data = [r for r in results if r['stage'] == 3]
    stage4_data = [r for r in results if r['stage'] == 4]
    
    metrics = {}
    
    if stage1_data:
        stage1_metrics = open_question_metrics(
            [r['prediction'] for r in stage1_data],
            [r['label'] for r in stage1_data]
        )
        metrics.update({f"stage1_{k}": v for k, v in stage1_metrics.items()})
    
    if stage2_data:
        stage2_metrics = closed_question_metrics(
            [r['prediction'] for r in stage2_data],
            [r['label'] for r in stage2_data]
        )
        metrics.update({f"stage2_{k}": v for k, v in stage2_metrics.items()})
    
    if stage3_data:
        stage3_metrics = closed_question_metrics(
            [r['prediction'] for r in stage3_data],
            [r['label'] for r in stage3_data]
        )
        metrics.update({f"stage3_{k}": v for k, v in stage3_metrics.items()})
    
    if stage4_data:
        stage4_metrics = open_question_metrics(
            [r['prediction'] for r in stage4_data],
            [r['label'] for r in stage4_data]
        )
        metrics.update({f"stage4_{k}": v for k, v in stage4_metrics.items()})
    
    # Calculate global accuracy
    correct_counts = sum(r['is_correct'] for r in results)
    total_counts = len(results)
    metrics['overall_accuracy'] = correct_counts / total_counts if total_counts > 0 else 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    print("\n📈 Evaluation results:")
    for stage in [1, 2, 3, 4]:
        stage_metrics = {k.replace(f"stage{stage}_", ""): v 
                        for k, v in metrics.items() if k.startswith(f"stage{stage}_")}
        if stage_metrics:
            print(f"\n🔹 Stage {stage} metrics:")
            for metric, value in stage_metrics.items():
                print(f"   {metric}: {value:.4f}")
    
    print(f"\n🔹 Overall accuracy: {metrics.get('overall_accuracy', 0.0):.4f}")


def save_metrics(metrics: Dict[str, Any], output_dir: str, config_base: str):
    metrics_file = os.path.join(output_dir, f"{config_base}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Evaluation metrics saved to: {metrics_file}") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yaml/infer.yaml', help='YAML config')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='output_dir')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/Qwen-0.5B', help='checkpoint path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--do_sample', type=bool,default=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_beams', type=int, default=1)
    
    args = parser.parse_args()
    main_inference(args)