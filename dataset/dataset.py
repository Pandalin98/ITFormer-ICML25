import sys
sys.path.append('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/weizhongyu-24036/time_series/Time-QA-new/')
sys.path.append('/dataYYF/dataWX/SJ/aipt/Time-QA/')

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
    """修复token ID范围问题的数据集"""
    
    def __init__(self, ts_path, data_path, tokenizer, processor, config, pretrain=False, sft=False):
        super().__init__()
        self.ts_path = ts_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.pretrain = pretrain
        self.sft = sft
        self.h5_file = None
        
        # 🔧 关键修复：确保vocab_size正确
        self.vocab_size = len(self.tokenizer)
        print(f"📊 Vocab size: {self.vocab_size}")
        
        # 确保tokenizer设置一致
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 🔧 验证特殊token的有效性
        self._validate_special_tokens()
        self._build_index()

    def _validate_special_tokens(self):
        """验证所有特殊token的ID都在有效范围内"""
        special_tokens = {
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None),
        }
        
        print("🔍 验证特殊token:")
        for name, token_id in special_tokens.items():
            if token_id is not None:
                if token_id >= self.vocab_size or token_id < 0:
                    print(f"❌ {name} = {token_id} 超出范围 [0, {self.vocab_size})")
                    # 修复无效的特殊token
                    if name == 'pad_token_id':
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        print(f"🔧 修复: pad_token_id -> {self.tokenizer.pad_token_id}")
                else:
                    print(f"✅ {name} = {token_id}")

    def _validate_token_ids(self, token_ids, context=""):
        """验证token IDs的有效性"""
        if not isinstance(token_ids, list):
            return token_ids
            
        valid_ids = []
        for i, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= self.vocab_size:
                print(f"⚠️ {context} 位置 {i}: 无效token_id {token_id}, 替换为unk_token")
                # 替换为unk_token，如果没有则用eos_token
                replacement = getattr(self.tokenizer, 'unk_token_id', self.tokenizer.eos_token_id)
                valid_ids.append(replacement)
            else:
                valid_ids.append(token_id)
        return valid_ids

    def _build_index(self):
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
        if self.h5_file is None:
            self.h5_file = h5py.File(self.ts_path, 'r', swmr=True)
        return self.h5_file

    def __len__(self):
        return len(self.datas)
    
    def add_adaptive_prompt(self, sample):
        """确保prompt添加的一致性"""
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
        """统一的聊天输入创建方法"""
        messages = [
            {"role": "system", "content": 'You are a helpful assistant.'},
            {"role": "user", "content": question}
        ]
        
        try:
            # 🔧 使用更安全的tokenization方法
            chat_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # 替换时间序列占位符
            chat_text = chat_text.replace('<ts>', '<|image_pad|>' * self.config.ts_pad_num)
            return chat_text
        except Exception as e:
            print(f"❌ Chat template error: {e}")
            # 降级到简单格式
            return f"You are a helpful assistant.\nuser\n{question}\nassistant\n"

    def _safe_tokenize(self, text, add_special_tokens=True):
        """安全的tokenization，确保结果在有效范围内"""
        try:
            # 🔧 添加更多的tokenization参数
            result = self.tokenizer(
                text, 
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_tensors=None
            )
            token_ids = result['input_ids']
            
            # 验证token_ids
            token_ids = self._validate_token_ids(token_ids, f"tokenize: {text[:50]}...")
            return token_ids
            
        except Exception as e:
            print(f"❌ Tokenization error for text: {text[:100]}...")
            print(f"Error: {e}")
            # 返回一个安全的默认值
            return [self.tokenizer.eos_token_id]

    def __getitem__(self, idx):
        try:
            sample = self.datas[idx]
            # sample = self.add_adaptive_prompt(sample)

            # 加载时间序列数据
            h5f = self._get_h5_file()
            if isinstance(sample['id'], str):
                ts = h5f['seq_data'][int(sample['id']) - 1]
            elif isinstance(sample['id'], list):
                ts_list = [h5f['seq_data'][int(i) - 1][:len(h5f['seq_data'][int(i) - 1]) // 10] for i in sample['id']]
                ts = np.concatenate(ts_list, axis=0)

            # =========================== 模式 1：预训练 ===========================
            if getattr(self, 'pretrain', False):
                return {
                    'ts_values': torch.tensor(ts, dtype=torch.float)
                }

            # =========================== 模式 2：SFT 训练 ===========================
            elif getattr(self, 'sft', False):
                # 🔧 创建query_ids：仅原始问题文本，不包含任何其他信息
                original_question = sample['question']
                query_ids = self._safe_tokenize(original_question, add_special_tokens=False)
                
                # 🔧 创建input_ids：包含时间序列占位符的完整输入
                q_text = self._create_chat_input(sample['question'])  # 这个包含<|image_pad|>
                q_input_ids = self._safe_tokenize(q_text, add_special_tokens=False)
                
                # 🔧 确保答案格式一致和安全
                a_text = sample['answer']
                if not a_text.endswith(self.tokenizer.eos_token):
                    a_text += self.tokenizer.eos_token
                a_input_ids = self._safe_tokenize(a_text, add_special_tokens=False)

                # 🔧 构造训练数据
                input_ids = q_input_ids + a_input_ids
                labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids

                # 🔧 最终验证
                query_ids = self._validate_token_ids(query_ids, f"query_ids_sample_{idx}")
                input_ids = self._validate_token_ids(input_ids, f"final_input_sample_{idx}")
                labels = self._validate_token_ids(labels, f"final_labels_sample_{idx}")

                # 🔧 确保长度匹配
                final_input_ids = input_ids[:-1] if len(input_ids) > 1 else input_ids
                final_labels = labels[1:] if len(labels) > 1 else labels

                return {
                    'form': sample['form'],
                    'stage': sample['stage'],
                    'query_ids': query_ids,  # 🔧 只包含原始问题文本
                    'input_ids': final_input_ids,
                    'labels': final_labels,
                    'ts_values': torch.tensor(ts, dtype=torch.float),
                    'index': sample['line_num']
                }

            # =========================== 模式 3：推理评估 ===========================
            else:
                # 🔧 创建query_ids：仅原始问题文本，不包含任何其他信息
                original_question = sample['question']
                query_ids = self._safe_tokenize(original_question, add_special_tokens=False)
                
                # 🔧 创建input_ids：包含时间序列占位符
                q_text = self._create_chat_input(sample['question'])
                q_input_ids = self._safe_tokenize(q_text, add_special_tokens=False)
                
                a_text = sample['answer']
                if not a_text.endswith(self.tokenizer.eos_token):
                    a_text += self.tokenizer.eos_token
                a_input_ids = self._safe_tokenize(a_text, add_special_tokens=False)

                # 验证结果
                query_ids = self._validate_token_ids(query_ids, f"infer_query_sample_{idx}")
                q_input_ids = self._validate_token_ids(q_input_ids, f"infer_q_sample_{idx}")
                a_input_ids = self._validate_token_ids(a_input_ids, f"infer_a_sample_{idx}")

                return {
                    'form': sample['form'],
                    'stage': sample['stage'],
                    'query_ids': query_ids,  # 🔧 只包含原始问题文本
                    'input_ids': q_input_ids,
                    'labels': a_input_ids,
                    'ts_values': torch.tensor(ts, dtype=torch.float),
                    'index': sample['line_num']
                }
                
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            # 返回一个安全的默认样本
            return self._get_safe_default_sample()

    def _get_safe_default_sample(self):
        """返回一个安全的默认样本"""
        return {
            'form': 'default',
            'stage': 1,
            'query_ids': [self.tokenizer.eos_token_id],  # 🔧 简单的默认query
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
        # 确保tokenizer设置正确
        if self.tokenizer.padding_side != 'left':
            print("⚠️  Warning: Setting tokenizer.padding_side to 'left' for decoder-only model")
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
            # ✅ 左padding是正确的（保持原有逻辑）
            padded_input = [self.tokenizer.pad_token_id] * (max_len_inputs - input_len) + feature['input_ids']
            input_ids.append(padded_input)
            
            # ✅ 对应的attention mask
            attention_mask.append([0] * (max_len_inputs - input_len) + [1] * input_len)
            
            # ✅ labels也左padding
            padded_labels = [self.tokenizer.pad_token_id] * (max_len_labels - label_len) + feature['labels']  # 用-100忽略pad位置的loss
            labels.append(padded_labels)
            
            # ✅ query_ids也左padding
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


def debug_tokenizer_and_vocab(tokenizer):
    """调试tokenizer和词汇表"""
    print("=== Tokenizer调试信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"BOS token: '{getattr(tokenizer, 'bos_token', 'None')}' (ID: {getattr(tokenizer, 'bos_token_id', 'None')})")
    print(f"UNK token: '{getattr(tokenizer, 'unk_token', 'None')}' (ID: {getattr(tokenizer, 'unk_token_id', 'None')})")
    
    # 测试一些特殊token
    test_tokens = ['<|image_pad|>', 'assistant', '<|im_end|>']
    for token in test_tokens:
        try:
            token_ids = tokenizer(token, add_special_tokens=False)['input_ids']
            print(f"Token '{token}': {token_ids}")
            for tid in token_ids:
                if tid >= len(tokenizer):
                    print(f"  ❌ Token ID {tid} >= vocab_size {len(tokenizer)}")
        except Exception as e:
            print(f"  ❌ Error tokenizing '{token}': {e}")


if __name__ == "__main__":
    # 配置
    tlmconfig = TLMConfig()
    ts_path = 'dataset/dataset_processing/data_merged_new.h5'
    qa_path = 'dataset/dataset_processing/test_sw3000.jsonl'
    
    # 初始化tokenizer
    model_path = 'checkpoints/Qwen-0.5B'  # 使用默认模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 调试tokenizer
    debug_tokenizer_and_vocab(tokenizer)
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 创建训练数据集进行测试
    print("\n=== 创建训练数据集 ===")
    train_dataset = TsQaDataset(ts_path, qa_path, tokenizer, processor, tlmconfig, sft=True)
    data_collator = DataCollator(tokenizer)
    
    print(f"Dataset length: {len(train_dataset)}")
    
    # 测试单个样本
    print("\n=== 测试单个样本 ===")
    try:
        sample = train_dataset[0]
        print("Sample keys:", sample.keys())
        print("Input_ids length:", len(sample['input_ids']))
        print("Labels length:", len(sample['labels']))
        print("Query_ids length:", len(sample['query_ids']))
        print("Input_ids range:", f"[{min(sample['input_ids'])}, {max(sample['input_ids'])}]")
        print("Query_ids range:", f"[{min(sample['query_ids'])}, {max(sample['query_ids'])}]")
        
        labels_without_ignore = [x for x in sample['labels'] if x != -100]
        if labels_without_ignore:
            print("Labels range (without -100):", f"[{min(labels_without_ignore)}, {max(labels_without_ignore)}]")
        
        # 验证query_ids是否只包含原始问题
        query_text = tokenizer.decode(sample['query_ids'], skip_special_tokens=True)
        print("Query text preview:", query_text[:100] + "..." if len(query_text) > 100 else query_text)
        
    except Exception as e:
        print(f"❌ Error testing single sample: {e}")
    
    # 测试批处理
    print("\n=== 测试批处理 ===")
    try:
        batch_size = 2
        features = [train_dataset[i] for i in range(batch_size)]
        
        # 检查每个feature的query_ids
        for i, feature in enumerate(features):
            print(f"Feature {i} query_ids length: {len(feature['query_ids'])}")
            print(f"Feature {i} query_ids range: [{min(feature['query_ids'])}, {max(feature['query_ids'])}]")
            # 解码查看内容
            query_text = tokenizer.decode(feature['query_ids'], skip_special_tokens=True)
            print(f"Feature {i} query text: {query_text[:50]}...")
        
        batch = data_collator(features)
        
        if batch:
            print("✅ Batch creation successful!")
            print("Batch keys:", batch.keys())
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}, dtype: {value.dtype}")
                    if key in ['query_ids', 'input_ids', 'labels']:
                        print(f"    Range: [{value.min().item()}, {value.max().item()}]")
            
            # 🔧 特别验证query_ids
            if 'query_ids' in batch:
                print(f"\n🔍 Query_ids 详细信息:")
                print(f"  Shape: {batch['query_ids'].shape}")
                # print(f"  Query attention mask shape: {batch['query_attention_mask'].shape}")
                # 解码第一个样本的query_ids
                first_query = batch['query_ids'][0]
                # 移除padding tokens
                valid_tokens = first_query[first_query != tokenizer.pad_token_id]
                decoded_query = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                print(f"  First sample decoded: {decoded_query}")
        else:
            print("❌ Batch creation failed!")
            
    except Exception as e:
        print(f"❌ Error testing batch: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 脚本完成 ===")