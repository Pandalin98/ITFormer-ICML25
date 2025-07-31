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
        print(f"📊 Vocab size: {self.vocab_size}")
        
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
        
        print("🔍 Validating special tokens:")
        for name, token_id in special_tokens.items():
            if token_id is not None:
                if token_id >= self.vocab_size or token_id < 0:
                    print(f"❌ {name} = {token_id} out of range [0, {self.vocab_size})")
                    # Fix invalid special tokens
                    if name == 'pad_token_id':
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        print(f"🔧 Fixed: pad_token_id -> {self.tokenizer.pad_token_id}")
                else:
                    print(f"✅ {name} = {token_id}")

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
                print(f"⚠️ {context} position {i}: invalid token_id {token_id}, replacing with unk_token")
                # Replace with unk_token, if not available use eos_token
                replacement = getattr(self.tokenizer, 'unk_token_id', self.tokenizer.eos_token_id)
                valid_ids.append(replacement)
            else:
                valid_ids.append(token_id)
        return valid_ids

    def _build_index(self):
        """Build dataset index by loading and processing data files."""
        self.datas = []
        
        # Load time series data
        if os.path.exists(self.ts_path):
            print(f"Loading time series data from: {self.ts_path}")
            self.h5_file = h5py.File(self.ts_path, 'r')
            ts_keys = list(self.h5_file.keys())
            print(f"Found {len(ts_keys)} time series samples")
        else:
            print(f"Warning: Time series file not found: {self.ts_path}")
            self.h5_file = None
        
        # Load question-answer data
        if os.path.exists(self.data_path):
            print(f"Loading QA data from: {self.data_path}")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            self.datas.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON line: {e}")
                            continue
            print(f"Loaded {len(self.datas)} QA samples")
        else:
            print(f"Warning: QA file not found: {self.data_path}")
        
        print(f"Total dataset size: {len(self.datas)}")

    def _get_h5_file(self):
        """Get HDF5 file handle for time series data."""
        if self.h5_file is None and os.path.exists(self.ts_path):
            self.h5_file = h5py.File(self.ts_path, 'r')
        return self.h5_file

    def __len__(self):
        """Return dataset length."""
        return len(self.datas)

    def add_adaptive_prompt(self, sample):
        """Add adaptive prompt to the sample.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            str: Question with adaptive prompt
        """
        question = sample.get('question', '')
        stage = sample.get('stage', 1)
        
        # Add stage-specific prompts
        if stage == 1:
            return f"Based on the time series data, please analyze and answer: {question}"
        elif stage == 2:
            return f"Given the time series information, answer this multiple choice question: {question}"
        elif stage == 3:
            return f"Using the time series data, select the correct answer: {question}"
        elif stage == 4:
            return f"Based on the time series analysis, provide a detailed answer: {question}"
        else:
            return question

    def _create_chat_input(self, question):
        """Create chat format input.
        
        Args:
            question: Question text
            
        Returns:
            str: Formatted chat input
        """
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    def _safe_tokenize(self, text, add_special_tokens=True):
        """Safely tokenize text with error handling.
        
        Args:
            text: Text to tokenize
            add_special_tokens: Whether to add special tokens
            
        Returns:
            dict: Tokenization result
        """
        try:
            return self.tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")
        except Exception as e:
            print(f"Tokenization error: {e}")
            # Return safe default
            return self.tokenizer("", add_special_tokens=add_special_tokens, return_tensors="pt")

    def __getitem__(self, idx):
        """Get dataset item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Processed sample
        """
        try:
            sample = self.datas[idx]
            
            # Extract basic information
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            stage = sample.get('stage', 1)
            index = sample.get('index', idx)
            
            # Add adaptive prompt
            question = self.add_adaptive_prompt(sample)
            
            # Create chat input
            chat_input = self._create_chat_input(question)
            
            # Tokenize input and target
            input_tokens = self._safe_tokenize(chat_input, add_special_tokens=False)
            target_tokens = self._safe_tokenize(answer, add_special_tokens=False)
            
            # Validate token IDs
            input_ids = self._validate_token_ids(input_tokens['input_ids'][0].tolist(), "input")
            target_ids = self._validate_token_ids(target_tokens['input_ids'][0].tolist(), "target")
            
            # Load time series data
            ts_values = None
            if self.h5_file is not None and 'ts_key' in sample:
                ts_key = sample['ts_key']
                if ts_key in self.h5_file:
                    ts_data = self.h5_file[ts_key][:]
                    ts_values = torch.tensor(ts_data, dtype=torch.float32)
                else:
                    print(f"Warning: TS key {ts_key} not found in HDF5 file")
                    ts_values = torch.zeros(600, 33, dtype=torch.float32)  # Default shape
            else:
                # Create dummy time series data if not available
                ts_values = torch.zeros(600, 33, dtype=torch.float32)
            
            # Create input_ids with time series padding tokens
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            
            # Add time series padding tokens
            ts_pad_token_id = self.tokenizer('<|image_pad|>')['input_ids'][0]
            ts_pad_tokens = [ts_pad_token_id] * self.config.ts_pad_num
            ts_pad_tensor = torch.tensor(ts_pad_tokens, dtype=torch.long)
            
            # Combine input_ids with time series padding
            final_input_ids = torch.cat([input_ids_tensor, ts_pad_tensor], dim=0)
            
            # Create attention mask
            attention_mask = torch.ones_like(final_input_ids)
            
            # Create labels (same as input_ids for causal LM)
            labels = final_input_ids.clone()
            
            return {
                'input_ids': final_input_ids,
                'query_ids': input_ids_tensor,
                'ts_values': ts_values,
                'stage': torch.tensor(stage, dtype=torch.long),
                'index': torch.tensor(index, dtype=torch.long),
                'attention_mask': attention_mask,
                'labels': labels,
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._get_safe_default_sample()

    def _get_safe_default_sample(self):
        """Get a safe default sample in case of errors.
        
        Returns:
            dict: Default sample
        """
        return {
            'input_ids': torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long),
            'query_ids': torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long),
            'ts_values': torch.zeros(600, 33, dtype=torch.float32),
            'stage': torch.tensor(1, dtype=torch.long),
            'index': torch.tensor(0, dtype=torch.long),
            'attention_mask': torch.tensor([1], dtype=torch.long),
            'labels': torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long),
        }

    def __del__(self):
        """Cleanup method to close HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()


class DataCollator:
    """Data collator for batching samples."""
    
    def __init__(self, tokenizer):
        """Initialize data collator.
        
        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate features into a batch.
        
        Args:
            features: List of sample dictionaries
            
        Returns:
            dict: Batched data
        """
        # Extract all fields
        input_ids = [f['input_ids'] for f in features]
        query_ids = [f['query_ids'] for f in features]
        ts_values = [f['ts_values'] for f in features]
        stages = [f['stage'] for f in features]
        indices = [f['index'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        query_ids = torch.nn.utils.rnn.pad_sequence(query_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # Stack other tensors
        ts_values = torch.stack(ts_values)
        stages = torch.stack(stages)
        indices = torch.stack(indices)
        
        return {
            'input_ids': input_ids,
            'query_ids': query_ids,
            'ts_values': ts_values,
            'stage': stages,
            'index': indices,
            'attention_mask': attention_masks,
            'labels': labels,
        }


def debug_tokenizer_and_vocab(tokenizer):
    """Debug function to print tokenizer and vocabulary information.
    
    Args:
        tokenizer: Tokenizer instance
    """
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"BOS token: {tokenizer.bos_token}")
    print(f"UNK token: {tokenizer.unk_token}")