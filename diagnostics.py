#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script for ITFormer inference issues.
Collects environment, model, and dataset information for troubleshooting.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

def check_environment():
    """Check and report Python environment information."""
    print("=" * 70)
    print("ENVIRONMENT INFORMATION")
    print("=" * 70)
    
    info = {}
    
    # Python version
    info['python_version'] = sys.version
    print(f"\n📌 Python Version: {sys.version}")
    
    # PyTorch
    try:
        import torch
        info['pytorch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
        info['cudnn_version'] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        info['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        print(f"\n📌 PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {info['cuda_available']}")
        if info['cuda_available']:
            print(f"   CUDA Version: {info['cuda_version']}")
            print(f"   cuDNN Version: {info['cudnn_version']}")
            print(f"   Number of GPUs: {info['num_gpus']}")
            for i in range(info['num_gpus']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
                info[f'gpu_{i}_name'] = gpu_name
                info[f'gpu_{i}_memory_gb'] = gpu_memory
    except ImportError as e:
        print(f"\n⚠️  PyTorch not found: {e}")
        info['pytorch_error'] = str(e)
    
    # Transformers
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
        print(f"\n📌 Transformers Version: {transformers.__version__}")
    except ImportError as e:
        print(f"\n⚠️  Transformers not found: {e}")
        info['transformers_error'] = str(e)
    
    # Accelerate
    try:
        import accelerate
        info['accelerate_version'] = accelerate.__version__
        print(f"\n📌 Accelerate Version: {accelerate.__version__}")
    except ImportError as e:
        print(f"\n⚠️  Accelerate not found: {e}")
        info['accelerate_error'] = str(e)
    
    # Other key libraries
    libraries = ['numpy', 'h5py', 'nltk', 'rouge_score', 'sklearn', 'yaml', 'tqdm']
    print("\n📌 Other Libraries:")
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'unknown')
            info[f'{lib}_version'] = version
            print(f"   {lib}: {version}")
        except ImportError:
            print(f"   {lib}: ⚠️  NOT INSTALLED")
            info[f'{lib}_installed'] = False
    
    return info

def check_model_checkpoint(checkpoint_path):
    """Check model checkpoint integrity and configuration."""
    print("\n" + "=" * 70)
    print("MODEL CHECKPOINT INFORMATION")
    print("=" * 70)
    
    info = {}
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint path does not exist: {checkpoint_path}")
        info['exists'] = False
        return info
    
    info['exists'] = True
    info['path'] = checkpoint_path
    print(f"\n📁 Checkpoint Path: {checkpoint_path}")
    
    # List all files
    files = []
    for root, dirs, filenames in os.walk(checkpoint_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, checkpoint_path)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            files.append({'path': rel_path, 'size_mb': size_mb})
    
    info['files'] = files
    total_size = sum(f['size_mb'] for f in files)
    info['total_size_mb'] = total_size
    
    print(f"\n📊 Total Size: {total_size:.2f} MB")
    print(f"📊 Number of Files: {len(files)}")
    
    # Check for key files
    key_files = [
        'config.json',
        'model.safetensors',
        'pytorch_model.bin',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    print("\n📋 Key Files:")
    for key_file in key_files:
        exists = any(f['path'] == key_file for f in files)
        if exists:
            file_info = next(f for f in files if f['path'] == key_file)
            print(f"   ✅ {key_file} ({file_info['size_mb']:.2f} MB)")
            info[f'has_{key_file.replace(".", "_")}'] = True
        else:
            print(f"   ❌ {key_file} (missing)")
            info[f'has_{key_file.replace(".", "_")}'] = False
    
    # Check for split safetensors
    safetensors_files = [f for f in files if 'model-' in f['path'] and f['path'].endswith('.safetensors')]
    if safetensors_files:
        print(f"\n📦 Found {len(safetensors_files)} split safetensors files:")
        for f in safetensors_files[:5]:  # Show first 5
            print(f"   {f['path']} ({f['size_mb']:.2f} MB)")
        if len(safetensors_files) > 5:
            print(f"   ... and {len(safetensors_files) - 5} more")
        info['split_safetensors_count'] = len(safetensors_files)
    
    # Load and display config.json
    config_path = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            info['config'] = config
            print(f"\n⚙️  Model Configuration:")
            for key, value in config.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"\n⚠️  Error loading config.json: {e}")
            info['config_error'] = str(e)
    
    return info

def check_dataset(ts_path='dataset/datasets/time_series_data.h5', 
                  qa_path='dataset/datasets/test_qa.jsonl'):
    """Check dataset files and basic statistics."""
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    
    info = {}
    
    # Check time series data
    print(f"\n📊 Time Series Data: {ts_path}")
    if os.path.exists(ts_path):
        size_mb = os.path.getsize(ts_path) / (1024 * 1024)
        print(f"   ✅ File exists ({size_mb:.2f} MB)")
        info['ts_data_exists'] = True
        info['ts_data_size_mb'] = size_mb
        
        try:
            import h5py
            with h5py.File(ts_path, 'r') as f:
                keys = list(f.keys())
                print(f"   📋 HDF5 Keys: {keys[:10]}")  # Show first 10 keys
                print(f"   📊 Total number of time series: {len(keys)}")
                info['ts_data_keys_count'] = len(keys)
                info['ts_data_sample_keys'] = keys[:10]
        except Exception as e:
            print(f"   ⚠️  Error reading HDF5 file: {e}")
            info['ts_data_error'] = str(e)
    else:
        print(f"   ❌ File not found")
        info['ts_data_exists'] = False
    
    # Check QA data
    print(f"\n📝 QA Data: {qa_path}")
    if os.path.exists(qa_path):
        size_mb = os.path.getsize(qa_path) / (1024 * 1024)
        print(f"   ✅ File exists ({size_mb:.2f} MB)")
        info['qa_data_exists'] = True
        info['qa_data_size_mb'] = size_mb
        
        try:
            import json
            samples = []
            stage_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            
            with open(qa_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                    stage = sample.get('stage', 0)
                    if stage in stage_counts:
                        stage_counts[stage] += 1
            
            info['qa_total_samples'] = len(samples)
            info['qa_stage_counts'] = stage_counts
            
            print(f"   📊 Total samples: {len(samples)}")
            print(f"   📊 Samples by stage:")
            for stage, count in sorted(stage_counts.items()):
                print(f"      Stage {stage}: {count} samples")
            
            # Show a sample
            if samples:
                print(f"\n   📄 Sample QA pair (Stage {samples[0].get('stage', 'unknown')}):")
                print(f"      Question: {samples[0].get('question', 'N/A')[:100]}...")
                print(f"      Answer: {samples[0].get('answer', 'N/A')[:100]}...")
                info['qa_sample'] = {
                    'stage': samples[0].get('stage'),
                    'question': samples[0].get('question', '')[:200],
                    'answer': samples[0].get('answer', '')[:200]
                }
        except Exception as e:
            print(f"   ⚠️  Error reading QA file: {e}")
            info['qa_data_error'] = str(e)
    else:
        print(f"   ❌ File not found")
        info['qa_data_exists'] = False
    
    return info

def check_config(config_path='yaml/infer.yaml'):
    """Check inference configuration."""
    print("\n" + "=" * 70)
    print("INFERENCE CONFIGURATION")
    print("=" * 70)
    
    info = {}
    
    print(f"\n⚙️  Config File: {config_path}")
    if os.path.exists(config_path):
        print(f"   ✅ File exists")
        info['exists'] = True
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            info['config'] = config
            print(f"\n   Configuration Parameters:")
            for key, value in config.items():
                print(f"      {key}: {value}")
        except Exception as e:
            print(f"   ⚠️  Error loading config: {e}")
            info['error'] = str(e)
    else:
        print(f"   ❌ File not found")
        info['exists'] = False
    
    return info

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint can be loaded."""
    print("\n" + "=" * 70)
    print("CHECKPOINT VERIFICATION")
    print("=" * 70)
    
    info = {'verification_passed': False}
    
    try:
        print("\n🔍 Attempting to load checkpoint...")
        
        # Try importing the model
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.TimeLanguageModel import TLM, TLMConfig
        
        # Determine LLM path
        if '0.5B' in checkpoint_path:
            llm_path = 'LLM/Qwen2.5-0.5B-Instruct'
        elif '3B' in checkpoint_path:
            llm_path = 'LLM/Qwen2.5-3B-Instruct'
        elif '7B' in checkpoint_path:
            llm_path = 'LLM/Qwen2.5-7B-Instruct'
        else:
            llm_path = 'LLM/Qwen2.5-0.5B-Instruct'
        
        print(f"   Using LLM: {llm_path}")
        
        if not os.path.exists(llm_path):
            print(f"   ⚠️  Warning: LLM path does not exist: {llm_path}")
            info['llm_exists'] = False
            return info
        
        info['llm_exists'] = True
        info['llm_path'] = llm_path
        
        config = TLMConfig(llm_model_path=llm_path, freeze_ts_model=True, ts_pad_num=25)
        
        print(f"   Loading model (this may take a moment)...")
        model = TLM.from_pretrained(checkpoint_path, config=config)
        
        # Count parameters
        def count_params(module):
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters())
        
        llm_params = count_params(model.llm_model) / 1e6
        itformer_params = count_params(model.itformer) / 1e6
        ts_encoder_params = count_params(model.ts_encoder) / 1e6
        total_params = count_params(model) / 1e6
        
        info['llm_params_M'] = llm_params
        info['itformer_params_M'] = itformer_params
        info['ts_encoder_params_M'] = ts_encoder_params
        info['total_params_M'] = total_params
        
        print(f"\n   ✅ Model loaded successfully!")
        print(f"\n   📊 Parameter Counts:")
        print(f"      LLM:        {llm_params:.2f}M")
        print(f"      ITFormer:   {itformer_params:.2f}M")
        print(f"      TS Encoder: {ts_encoder_params:.2f}M")
        print(f"      Total:      {total_params:.2f}M")
        
        info['verification_passed'] = True
        
    except Exception as e:
        print(f"\n   ❌ Error loading model: {e}")
        import traceback
        print(f"\n   Traceback:")
        print(traceback.format_exc())
        info['error'] = str(e)
        info['traceback'] = traceback.format_exc()
    
    return info

def save_report(report, output_path='diagnostic_report.json'):
    """Save diagnostic report to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full diagnostic report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='ITFormer Diagnostics Tool')
    parser.add_argument('--check-environment', action='store_true', 
                        help='Check Python environment and libraries')
    parser.add_argument('--check-model', action='store_true',
                        help='Check model checkpoint')
    parser.add_argument('--check-dataset', action='store_true',
                        help='Check dataset files')
    parser.add_argument('--check-config', action='store_true',
                        help='Check inference configuration')
    parser.add_argument('--verify-checkpoint', action='store_true',
                        help='Verify checkpoint can be loaded')
    parser.add_argument('--full', action='store_true',
                        help='Run all diagnostics')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/ITFormer-0.5B',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='yaml/infer.yaml',
                        help='Path to inference config')
    parser.add_argument('--ts_path', type=str, default='dataset/datasets/time_series_data.h5',
                        help='Path to time series data')
    parser.add_argument('--qa_path', type=str, default='dataset/datasets/test_qa.jsonl',
                        help='Path to QA data')
    parser.add_argument('--output', type=str, default='diagnostic_report.json',
                        help='Output path for diagnostic report')
    
    args = parser.parse_args()
    
    # If no specific check is requested, show help
    if not any([args.check_environment, args.check_model, args.check_dataset, 
                args.check_config, args.verify_checkpoint, args.full]):
        parser.print_help()
        return
    
    print("\n" + "=" * 70)
    print("ITFormer Diagnostic Tool")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'diagnostics': {}
    }
    
    if args.full or args.check_environment:
        report['diagnostics']['environment'] = check_environment()
    
    if args.full or args.check_model:
        report['diagnostics']['model'] = check_model_checkpoint(args.model_checkpoint)
    
    if args.full or args.check_dataset:
        report['diagnostics']['dataset'] = check_dataset(args.ts_path, args.qa_path)
    
    if args.full or args.check_config:
        report['diagnostics']['config'] = check_config(args.config)
    
    if args.full or args.verify_checkpoint:
        report['diagnostics']['verification'] = verify_checkpoint(args.model_checkpoint)
    
    # Save report
    save_report(report, args.output)
    
    print("\n" + "=" * 70)
    print("Diagnostic Summary")
    print("=" * 70)
    
    # Print summary
    env_ok = report['diagnostics'].get('environment', {}).get('pytorch_version') is not None
    model_ok = report['diagnostics'].get('model', {}).get('exists', False)
    dataset_ok = (report['diagnostics'].get('dataset', {}).get('ts_data_exists', False) and 
                  report['diagnostics'].get('dataset', {}).get('qa_data_exists', False))
    config_ok = report['diagnostics'].get('config', {}).get('exists', False)
    verify_ok = report['diagnostics'].get('verification', {}).get('verification_passed', False)
    
    print(f"\n{'✅' if env_ok else '❌'} Environment: {'OK' if env_ok else 'Issues detected'}")
    if 'model' in report['diagnostics']:
        print(f"{'✅' if model_ok else '❌'} Model Checkpoint: {'OK' if model_ok else 'Issues detected'}")
    if 'dataset' in report['diagnostics']:
        print(f"{'✅' if dataset_ok else '❌'} Dataset: {'OK' if dataset_ok else 'Issues detected'}")
    if 'config' in report['diagnostics']:
        print(f"{'✅' if config_ok else '❌'} Configuration: {'OK' if config_ok else 'Issues detected'}")
    if 'verification' in report['diagnostics']:
        print(f"{'✅' if verify_ok else '❌'} Checkpoint Verification: {'Passed' if verify_ok else 'Failed'}")
    
    print(f"\n📋 For detailed information, see: {args.output}")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
