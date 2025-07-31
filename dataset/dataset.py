#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Question Answering Dataset.
Handles loading and preprocessing of time series data and question-answer pairs.
"""
import sys
from transformers import PretrainedConfig, AutoTokenizer
from transformers import AutoProcessor
import torch
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import h5py
import re
from models.TimeLanguageModel import TLMConfig
from accelerate import Accelerator

# Get accelerator instance for main process checks
accelerator = Accelerator()


def find_assistant_tokens(tokenizer, target):
    """Find assistant token positions in the target sequence.
    
    Args:
        tokenizer: Tokenizer instance
        target: Target token sequence
        
    Returns:
        List of tuples containing start and end positions of assistant tokens
    """
    result = []
    start_index = 0
    end_index = 0
    while start_index <= len(target) - 1:
        if target[start_index] != tokenizer('assistant')['input_ids'][0]:
            start_index += 1
            end_index += 1
        else:
            end_index += 1
            if target[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index + 1, end_index + 1))
                start_index = end_index + 1
    return result


class TsQaDataset(Dataset):
    """Time Series Question Answering Dataset with token ID range validation."""
    
    def __init__(self, ts_path, data_path, tokenizer, processor, config, pretrain=False, sft=False):
        """Initialize the dataset.
        
        Args:
            ts_path: Path to time series data file
            data_path: Path to question-answer data file
            tokenizer: Tokenizer instance
            processor: Processor instance
            config: Configuration object
            pretrain: Whether in pretraining mode
            sft: Whether in supervised fine-tuning mode
        """
        super().__init__()
        self.ts_path = ts_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.pretrain = pretrain
        self.sft = sft
        self.h5_file = None
        
        # Key fix: Ensure vocab_size is correct
        self.vocab_size = len(self.tokenizer)
        if accelerator.is_main_process:
            accelerator.print(f"📊 Vocab size: {self.vocab_size}")
        
        # Ensure tokenizer settings are consistent
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Validate special tokens
        self._validate_special_tokens()
        self._build_index()

    def _validate_special_tokens(self):
        """Validate that all special token IDs are within valid range."""
        special_tokens = {
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None),
        }
        
        if accelerator.is_main_process:
            accelerator.print("🔍 Validating special tokens:")
        for name, token_id in special_tokens.items():
            if token_id is not None:
                if token_id >= self.vocab_size or token_id < 0:
                    if accelerator.is_main_process:
                        accelerator.print(f"❌ {name} = {token_id} out of range [0, {self.vocab_size})")
                    # Fix invalid special tokens
                    if name == 'pad_token_id':
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        if accelerator.is_main_process:
                            accelerator.print(f"🔧 Fixed: pad_token_id -> {self.tokenizer.pad_token_id}")
                else:
                    if accelerator.is_main_process:
                        accelerator.print(f"✅ {name} = {token_id}")

    def _validate_token_ids(self, token_ids, context=""):
        """Validate token IDs for validity.
        
        Args:
            token_ids: List of token IDs to validate
            context: Context string for error messages
            
        Returns:
            List of validated token IDs
        """
        if not isinstance(token_ids, list):
            return token_ids
            
        valid_ids = []
        for i, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= self.vocab_size:
                if accelerator.is_main_process:
                    accelerator.print(f"⚠️ {context} position {i}: invalid token_id {token_id}, replacing with unk_token")
                # Replace with unk_token, if not available use eos_token
                replacement = getattr(self.tokenizer, 'unk_token_id', self.tokenizer.eos_token_id)
                valid_ids.append(replacement)
            else:
                valid_ids.append(token_id)
        return valid_ids

    def _build_index(self):
        """Build dataset index by loading and processing data files."""
        self.datas = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                item = json.loads(line)
                for i in range(0, len(item['conversations']), 2):
                    if item['conversations'][i]['stage'] in ['1', '2', '3', '4']:
                        self.datas.append({
                            'id': item['id'],
                            'stage': int(item['conversations'][i]['stage']),
                            'form': item['conversations'][i]['attribute'],
                            'question': item['conversations'][i]['value'],
                            'answer': item['conversations'][i + 1]['value'],
                            'line_num': line_num
                        })

    def _get_h5_file(self):
        """Get HDF5 file handle for time series data."""
        if self.h5_file is None and os.path.exists(self.ts_path):
            self.h5_file = h5py.File(self.ts_path, 'r')
        return self.h5_file

    def __len__(self):
        """Return dataset length."""
        return len(self.datas)

    def add_adaptive_prompt(self, sample):
        sample = sample.copy()
        
        if sample['stage'] == 1:
            sample['question'] += " Please analyze the change in this signal and explain its physical implication, such as component load, airflow, or temperature stability."
        elif sample['stage'] == 2:
            sample['question'] += " Carefully analyze the signal pattern (e.g., stability, oscillation, drops) to determine the correct fault status or root cause. Select the most likely option based on observed signal behavior."
        elif sample['stage'] == 3:
            sample['question'] += " Review the trends across 10 cycles and evaluate the degradation pattern. Select the option that best reflects the long-term health status or risk level indicated by the signal."
        elif sample['stage'] == 4:
            sample['question'] += " Based on the 10-cycle degradation pattern, propose concrete maintenance actions (e.g., replace, inspect) to ensure safe and efficient operation."
        return sample

    def _create_chat_input(self, question):
        """Unified chat input creation method."""
        messages = [
            {"role": "system", "content": 'You are a helpful assistant.'},
            {"role": "user", "content": question}
        ]
        
        try:
            # Use a safer tokenization method
            chat_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # Replace time series placeholder
            chat_text = chat_text.replace('<ts>', '<|image_pad|>' * self.config.ts_pad_num)
            return chat_text
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"❌ Chat template error: {e}")
            # Fallback to a simple format
            return f"You are a helpful assistant.\nuser\n{question}\nassistant\n"

    def _safe_tokenize(self, text, add_special_tokens=True):
        """Safe tokenization, ensure results are within valid range."""
        try:
            # Add more tokenization parameters
            result = self.tokenizer(
                text, 
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_tensors=None
            )
            token_ids = result['input_ids']
            
            # Validate token_ids
            token_ids = self._validate_token_ids(token_ids, f"tokenize: {text[:50]}...")
            return token_ids
            
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"❌ Tokenization error for text: {text[:100]}...")
                accelerator.print(f"Error: {e}")
            # Return a safe default value
            return [self.tokenizer.eos_token_id]

    def __getitem__(self, idx):
        try:
            sample = self.datas[idx]
            # sample = self.add_adaptive_prompt(sample)

            # Load time series data
            h5f = self._get_h5_file()
            if isinstance(sample['id'], str):
                ts = h5f['seq_data'][int(sample['id']) - 1]
            elif isinstance(sample['id'], list):
                ts_list = [h5f['seq_data'][int(i) - 1][:len(h5f['seq_data'][int(i) - 1]) // 10] for i in sample['id']]
                ts = np.concatenate(ts_list, axis=0)

            # =========================== Mode 1: Pretraining ===========================
            if getattr(self, 'pretrain', False):
                return {
                    'ts_values': torch.tensor(ts, dtype=torch.float)
                }

            # =========================== Mode 2: SFT Training ===========================
            elif getattr(self, 'sft', False):
                # Create query_ids: only the original question text, no other information
                original_question = sample['question']
                query_ids = self._safe_tokenize(original_question, add_special_tokens=False)
                
                # Create input_ids: full input including time series placeholder
                q_text = self._create_chat_input(sample['question'])  # This includes <|image_pad|>
                q_input_ids = self._safe_tokenize(q_text, add_special_tokens=False)
                
                # Ensure answer format is consistent and safe
                a_text = sample['answer']
                if not a_text.endswith(self.tokenizer.eos_token):
                    a_text += self.tokenizer.eos_token
                a_input_ids = self._safe_tokenize(a_text, add_special_tokens=False)

                # Construct training data
                input_ids = q_input_ids + a_input_ids
                labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids

                # Final validation
                query_ids = self._validate_token_ids(query_ids, f"query_ids_sample_{idx}")
                input_ids = self._validate_token_ids(input_ids, f"final_input_sample_{idx}")
                labels = self._validate_token_ids(labels, f"final_labels_sample_{idx}")

                # Ensure length matches
                final_input_ids = input_ids[:-1] if len(input_ids) > 1 else input_ids
                final_labels = labels[1:] if len(labels) > 1 else labels

                return {
                    'form': sample['form'],
                    'stage': sample['stage'],
                    'query_ids': query_ids,  # Only contains the original question text
                    'input_ids': final_input_ids,
                    'labels': final_labels,
                    'ts_values': torch.tensor(ts, dtype=torch.float),
                    'index': sample['line_num']
                }

            # =========================== Mode 3: Inference/Evaluation ===========================
            else:
                # Create query_ids: only the original question text, no other information
                original_question = sample['question']
                query_ids = self._safe_tokenize(original_question, add_special_tokens=False)
                
                # Create input_ids: includes time series placeholder
                q_text = self._create_chat_input(sample['question'])
                q_input_ids = self._safe_tokenize(q_text, add_special_tokens=False)
                
                a_text = sample['answer']
                if not a_text.endswith(self.tokenizer.eos_token):
                    a_text += self.tokenizer.eos_token
                a_input_ids = self._safe_tokenize(a_text, add_special_tokens=False)

                # Validate results
                query_ids = self._validate_token_ids(query_ids, f"infer_query_sample_{idx}")
                q_input_ids = self._validate_token_ids(q_input_ids, f"infer_q_sample_{idx}")
                a_input_ids = self._validate_token_ids(a_input_ids, f"infer_a_sample_{idx}")

                return {
                    'form': sample['form'],
                    'stage': sample['stage'],
                    'query_ids': query_ids,  # Only contains the original question text
                    'input_ids': q_input_ids,
                    'labels': a_input_ids,
                    'ts_values': torch.tensor(ts, dtype=torch.float),
                    'index': sample['line_num']
                }
                
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"❌ Error processing sample {idx}: {e}")
            # Return a safe default sample
            return self._get_safe_default_sample()

    def _get_safe_default_sample(self):
        """Return a safe default sample."""
        return {
            'form': 'default',
            'stage': 1,
            'query_ids': [self.tokenizer.eos_token_id],  # Simple default query
            'input_ids': [self.tokenizer.eos_token_id],
            'labels': [self.tokenizer.eos_token_id],
            'ts_values': torch.zeros(100, dtype=torch.float),
            'index': 0
        }

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Ensure tokenizer settings are correct
        if self.tokenizer.padding_side != 'left':
            if accelerator.is_main_process:
                accelerator.print("⚠️  Warning: Setting tokenizer.padding_side to 'left' for decoder-only model")
            self.tokenizer.padding_side = 'left'
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len_inputs = max(len(feature['input_ids']) for feature in features)
        max_len_labels = max(len(feature['labels']) for feature in features)
        max_len_querys = max(len(feature['query_ids']) for feature in features)    
        input_ids = []
        attention_mask = []
        labels = []
        ts_values = []
        stages = []
        index = []
        query_ids = []
        for feature in features:
            input_len = len(feature['input_ids'])
            label_len = len(feature['labels'])
            query_ids_len = len(feature['query_ids'])
            # Left padding is correct (keep original logic)
            padded_input = [self.tokenizer.pad_token_id] * (max_len_inputs - input_len) + feature['input_ids']
            input_ids.append(padded_input)
            
            # Corresponding attention mask
            attention_mask.append([0] * (max_len_inputs - input_len) + [1] * input_len)
            
            # Labels also left-padded
            padded_labels = [self.tokenizer.pad_token_id] * (max_len_labels - label_len) + feature['labels']  # Use -100 to ignore pad positions in loss
            labels.append(padded_labels)
            
            # query_ids also left-padded
            padded_query_ids = [self.tokenizer.pad_token_id] * (max_len_querys - query_ids_len) + feature['query_ids']
            query_ids.append(padded_query_ids)

            ts_values.append(feature['ts_values'])
            stages.append(feature['stage'])
            index.append(feature['index'])


        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'ts_values': torch.stack(ts_values, dim=0),
            'stage': torch.tensor(stages, dtype=torch.int8),
            'index': torch.tensor(index, dtype=torch.int32),
            'query_ids': torch.tensor(query_ids, dtype=torch.long)
        }


