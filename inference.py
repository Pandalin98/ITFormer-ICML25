#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for Time Language Model (TLM).
Performs inference on the test set and saves results and evaluation metrics.
"""
import os
import yaml
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TRANSFORMERS_CACHE.*")
import argparse
import random
import logging
from transformers.utils import logging as transformers_logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
from dataset.dataset import TsQaDataset, DataCollator
from models.TimeLanguageModel import TLM
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import open_question_metrics, closed_question_metrics
from typing import List, Dict, Any
from accelerate import Accelerator  
from utils.result_utils import (
    dataset_sample_indices,
    merge_unique_results,
    rank_strided_positions,
)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_results(results, output_dir, config_name):
    """Save inference results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inference_results_{config_name}_{timestamp}.json"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Inference results saved to: {filepath}")

def count_model_parameters(model):
    """Count the number of parameters in each model component (in millions)."""
    def count_params(module):
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters())
    llm_params = count_params(model.llm_model) / 1e6
    itformer_params = count_params(model.itformer) / 1e6
    ts_encoder_params = count_params(model.ts_encoder) / 1e6
    total_params = count_params(model) / 1e6
    return {
        'llm': llm_params,
        'itformer': itformer_params,
        'ts_encoder': ts_encoder_params,
        'total': total_params
    }

def main_inference(args):
    """Main inference pipeline."""
    # Initialize accelerator
    accelerator = Accelerator()
    
    # 设置日志级别，主进程显示进度，从进程静默
    if accelerator.is_local_main_process:
        transformers_logging.set_verbosity_info()
    else:
        transformers_logging.set_verbosity_error()
        logging.disable(logging.CRITICAL)

    set_seed(args.seed)
    # Print only in the main process
    if accelerator.is_main_process:
        print("🚀 Starting inference process...")
        print(f"🔧 Using config file: {args.config}")
        print(f"💾 Results will be saved to: {args.output_dir}")
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)
    if accelerator.is_main_process:
        print("\n⚙️ Loading model from checkpoint...")
    # Import TLMConfig
    from models.TimeLanguageModel import TLMConfig
    
    # Determine LLM model path based on ITFormer model size
    if '0.5B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct'
    elif '3B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-3B-Instruct'
    elif '7B' in args.model_checkpoint:
        llm_model_path = 'LLM/Qwen2.5-7B-Instruct'
    else:
        # Default to 7B if size cannot be determined
        llm_model_path = 'LLM/Qwen2.5-7B-Instruct'
    
    # 将推断出的 llm_model_path 存入 args，防止后续 compute_metrics_from_results 报错
    args.llm_model_path = llm_model_path
    
    if accelerator.is_main_process:
        print(f"🔗 Using LLM model: {llm_model_path}")
    
    # Create TLMConfig
    tlm_config = TLMConfig(
        llm_model_path=llm_model_path,
        freeze_ts_model=True,
        ts_pad_num=args.prefix_num,
        # 显式补全 ITFormer 和 TS Encoder 需要的字段名
        prefix_num=args.prefix_num,
        it_d_model=config.get('it_d_model', 896),
        it_n_heads=config.get('it_n_heads', 16),
        it_layers=config.get('it_layers', 2),
        it_dropout=config.get('it_dropout', 0.1),
        dropout=config.get('dropout', 0.1),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        e_layers=config.get('e_layers', 4),
        patch_len=config.get('patch_len', 60),
        stride=config.get('stride', 60),
        input_len=config.get('input_len', 600),
    )
    
    # Load model - let from_pretrained handle configuration loading
    from models.TimeLanguageModel import TLM
    model = TLM.from_pretrained(args.model_checkpoint, config=tlm_config, ts_config=tlm_config)
    # Print model parameter statistics only in the main process
    if accelerator.is_main_process:
        param_counts = count_model_parameters(model)
        print(f"\n🔢 Model parameter counts (in M):")
        print(f"   LLM:         {param_counts['llm']:.2f}M")
        print(f"   ITFormer:    {param_counts['itformer']:.2f}M")
        print(f"   TS_Encoder:  {param_counts['ts_encoder']:.2f}M")
        print(f"   Total:       {param_counts['total']:.2f}M")
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    model.eval()
    if accelerator.is_main_process:
        print(f"✅ Model loaded successfully!")
        print("\n📊 Preparing test dataset...")
    # The TLM implementation reconstructs its LLM from ``llm_model_path`` and
    # intentionally ignores checkpoint LLM weights.  Use that model's tokenizer
    # as well so standalone inference obeys the same tokenization contract as
    # SFT evaluation.  Loading a separately serialized checkpoint tokenizer can
    # otherwise introduce version-dependent differences.
    tokenizer = accelerator.unwrap_model(model).tokenizer
    if accelerator.is_main_process:
        print("Loading tokenizer from the reconstructed LLM model")
    # Add special token if not present
    if '<|image_pad|>' not in tokenizer.get_vocab():
        if accelerator.is_main_process:
            print("Adding <|image_pad|> token to tokenizer")
        tokenizer.add_tokens(['<|image_pad|>'])
        model.llm_model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = 'left'
    # Simple config for dataset
    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    tlmconfig = SimpleConfig(ts_pad_num=args.prefix_num)
    test_dataset = TsQaDataset(
        args.ts_path_test,
        args.qa_path_test,
        tokenizer,
        tokenizer,  # Use tokenizer as processor
        tlmconfig,
        strict_preprocessing=True,
    )
    if args.max_samples is not None:
        from torch.utils.data import Subset
        test_dataset = Subset(
            test_dataset,
            range(min(args.max_samples, len(test_dataset))),
        )
    expected_indices = sorted(dataset_sample_indices(test_dataset))
    from torch.utils.data import Subset

    local_positions = rank_strided_positions(
        len(test_dataset),
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
    )
    local_test_dataset = Subset(test_dataset, local_positions)
    data_collator = DataCollator(tokenizer=tokenizer)
    test_loader = DataLoader(
        local_test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers
    )
    if accelerator.is_main_process:
        print(f"📁 Test set size: {len(test_dataset)} samples")
        print(
            f"🔢 Batch size: {args.batch_size}, "
            f"rank-0 batches: {len(test_loader)}"
        )
        print(
            "🧩 Distributed inference uses an explicit padding-free strided "
            "dataset partition."
        )
        print("\n🔍 Starting test set inference...")
    results = []
    with torch.no_grad():
        # Show progress bar only in the main process
        if accelerator.is_main_process:
            batch_iterator = tqdm(test_loader, desc="Inference progress")
        else:
            batch_iterator = test_loader
        for batch_idx, batch in enumerate(batch_iterator):
            batch = {
                key: value.to(accelerator.device)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            
            # 使用 unwrap_model 来调用原始模型的 generate 方法
            unwrapped_model = accelerator.unwrap_model(model)
            generated_ids = unwrapped_model.generate(
                input_ids=batch['input_ids'],
                query_ids=batch['query_ids'],
                ts_values=batch['ts_values'],
                stage=batch['stage'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                # Optimization parameters, consistent with Time-QA
                num_beams=1,                    # Greedy search, fastest
                temperature=1.0,                # Avoid extra computation
                top_p=None,                     # Disable nucleus sampling
                top_k=None,                     # Disable top-k sampling
                repetition_penalty=1.0,         # Disable repetition penalty
                length_penalty=1.0,             # Disable length penalty
                no_repeat_ngram_size=0,         # Disable n-gram repetition check
                output_scores=False,            # Do not output scores
                output_attentions=False,        # Do not output attention
                output_hidden_states=False,     # Do not output hidden states
                return_dict_in_generate=False,  # Simplify return format
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
                    "source_line": batch['source_line'][i].item(),
                    "qa_offset": batch['qa_offset'][i].item(),
                    "stage": batch['stage'][i].item(),
                    "input": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                    "prediction": prediction,
                    # Preserve the source JSON reference verbatim. Tokenizer
                    # round-trips are useful diagnostics but are not the
                    # benchmark ground truth contract.
                    "label": batch['reference'][i],
                    "decoded_label": batch_labels[i],
                    "is_correct": (
                        prediction.strip() == batch['reference'][i].strip()
                    )
                })
    if accelerator.is_main_process:
        print("\n📊 Calculating evaluation metrics...")

    # Persist every rank before the collective merge. If a later metric or
    # validation step fails, completed generation remains recoverable.
    rank_shard_dir = os.path.join(args.output_dir, "rank_shards")
    os.makedirs(rank_shard_dir, exist_ok=True)
    rank_shard_path = os.path.join(
        rank_shard_dir,
        f"rank_{accelerator.process_index:05d}.json",
    )
    with open(rank_shard_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False)
    accelerator.wait_for_everyone()
    
    # 汇总所有进程的结果
    result_groups = [results]
    # 使用 gather_object 汇总每个 rank 的 list
    import torch.distributed as dist
    if dist.is_initialized():
        result_groups = [None] * accelerator.num_processes
        dist.all_gather_object(result_groups, results)
    
    if accelerator.is_main_process:
        # ``index`` is the flattened QA index.  This removes only distributed
        # sampler padding and never collapses QA pairs from the same JSONL line.
        results = merge_unique_results(result_groups)
        actual_indices = [item["index"] for item in results]
        if actual_indices != expected_indices:
            missing = sorted(set(expected_indices) - set(actual_indices))
            unexpected = sorted(set(actual_indices) - set(expected_indices))
            raise RuntimeError(
                "Distributed inference returned the wrong QA index set. "
                f"Missing indices: {missing[:10]}; unexpected indices: "
                f"{unexpected[:10]}."
            )
        if len(results) != len(test_dataset):
            raise RuntimeError(
                "Distributed inference produced "
                f"{len(results)} unique QA results, expected {len(test_dataset)}. "
                "Check rank gathering, sampler padding, and flattened QA indices."
            )

        metrics = compute_metrics_from_results(results, args)
        print_metrics(metrics)
        config_base = os.path.basename(args.config).split('.yaml')[0]
        save_results(results, args.output_dir, config_base)
        save_metrics(metrics, args.output_dir, config_base)

def compute_metrics_from_results(results: List[Dict],args) -> Dict[str, Any]:
    """Compute evaluation metrics for each stage from inference results."""
    stage1_data = [r for r in results if r['stage'] == 1]
    stage2_data = [r for r in results if r['stage'] == 2]
    stage3_data = [r for r in results if r['stage'] == 3]
    stage4_data = [r for r in results if r['stage'] == 4]
    # Create special_id, consistent with Time-QA
    from transformers import AutoTokenizer
    # 为了和设置的模型对齐，tokenizer应与主模型保持一致
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    special_id = tokenizer.all_special_ids  
    common_punctuations = [".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", "\"", "'"]
    punctuation_ids = tokenizer.convert_tokens_to_ids(common_punctuations)
    special_id.extend(punctuation_ids)
    metrics = {}
    if stage1_data:
        stage1_metrics = open_question_metrics(
            [r['prediction'] for r in stage1_data],
            [r['label'] for r in stage1_data],
            special_id
        )
        metrics.update({f"stage1_{k}": v for k, v in stage1_metrics.items()})
    if stage2_data:
        stage2_metrics = closed_question_metrics(
            [r['prediction'] for r in stage2_data],
            [r['label'] for r in stage2_data],
            special_id
        )
        metrics.update({f"stage2_{k}": v for k, v in stage2_metrics.items()})
    if stage3_data:
        stage3_metrics = closed_question_metrics(
            [r['prediction'] for r in stage3_data],
            [r['label'] for r in stage3_data],
            special_id
        )
        metrics.update({f"stage3_{k}": v for k, v in stage3_metrics.items()})
    if stage4_data:
        stage4_metrics = open_question_metrics(
            [r['prediction'] for r in stage4_data],
            [r['label'] for r in stage4_data],
            special_id
        )
        metrics.update({f"stage4_{k}": v for k, v in stage4_metrics.items()})
    return metrics

def print_metrics(metrics: Dict[str, Any]):
    """Print evaluation metrics for each stage."""
    print("\n📈 Evaluation results:")
    for stage in [1, 2, 3, 4]:
        stage_metrics = {k.replace(f"stage{stage}_", ""): v 
                        for k, v in metrics.items() if k.startswith(f"stage{stage}_")}
        if stage_metrics:
            print(f"\n🔹 Stage {stage} metrics:")
            for metric, value in stage_metrics.items():
                print(f"   {metric}: {value:.4f}")

def save_metrics(metrics: Dict[str, Any], output_dir: str, config_base: str):
    """Save evaluation metrics to a JSON file."""
    metrics_file = os.path.join(output_dir, f"{config_base}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Evaluation metrics saved to: {metrics_file}") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TLM Inference')
    parser.add_argument('--config', type=str, default='yaml/infer.yaml', help='YAML config file')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/ITFormer-0.5B', help='Model checkpoint path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum new tokens to generate')
    parser.add_argument('--max_samples', type=int, default=None, help='Optional leading subset size for smoke evaluation')
    args = parser.parse_args()
    main_inference(args)
