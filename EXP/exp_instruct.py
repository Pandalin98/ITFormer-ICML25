#!/usr/bin/env python
# -*- coding:utf-8 _*-
import importlib
import json
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
import os
import torch
from models.TimeLanguageModel import TLM, TLMConfig
from dataset.dataset import DataCollator
from typing import Dict, List, Any, NamedTuple, Optional, Tuple, Union
from datasets import load_metric
import numpy as np
from utils.metrics import open_question_metrics,closed_question_metrics,compute_rul
import warnings
from tqdm import tqdm
import pickle
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from accelerate import Accelerator
accelerator = Accelerator()
from models.TimeLanguageModel import GradientAndActivationMonitor 
import torch.distributed as dist    
from datetime import datetime

def distributed_tqdm(iterable, desc=None):
    if not dist.is_initialized() or dist.get_rank() == 0:
        return tqdm(iterable, desc=desc)
    else:
        return iterable

class OutputWrapper:
    def __init__(self, original_output):
        self.original_output = original_output

    def __getattr__(self, name):
        # 如果属性不存在于自身，则尝试从原始对象中获取
        return getattr(self.original_output, name)
    
class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    pred_extra: Optional[Dict[str, Any]] = None



class Exp_Instruct(Trainer):
    def __init__(self, args, train_dataset, tlm_config=None, eval_dataset=None):
        # Build the model
        self.tlmconfig = tlm_config
        model = self._build_model(args)
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            dataloader_num_workers = args.dataloader_num_workers,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_strategy='no',
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            ddp_find_unused_parameters=False,  
            fp16=args.fp16,  
            num_train_epochs=args.num_train_epochs,
            report_to=args.report_to,  # Example: Integrate TensorBoard
            prediction_loss_only=False,
            max_grad_norm=0.1,
            dataloader_drop_last=True)

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollator(tokenizer=train_dataset.tokenizer),
            eval_dataset=eval_dataset,
            # compute_metrics=self._compute_metrics if eval_dataset else None,
        )
        self.compute_metrics  = self.custom_compute_metrics if eval_dataset else None
        self.special_id = train_dataset.processor.all_special_ids  
        self.processor = train_dataset.processor
        self.padding_idx = self.processor.pad_token_id
        # 常用标点符号列表
        common_punctuations = [".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", "\"", "'"]
        punctuation_ids = self.processor.convert_tokens_to_ids(common_punctuations)
        # 将标点符号 ID 合并到特殊标记 ID 列表中
        self.special_id.extend(punctuation_ids)
        self.tlmargs = args
        
        # 定义stage权重
        self.stage_weights = {
            1: 1.0,   # 开放式问题 - 基础权重
            2: 1.5,   # 封闭式问题 - 稍高权重
            3: 1.5,   # 封闭式问题 - 中等权重
            4: 1.1    # 开放式问题 - 稍低权重
        }
        # 初始化损失函数，不使用ignore_index（我们将手动处理）
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=self.padding_idx)
        # self.args.remove_unused_columns = True  # 添加这一行
    def load_model(self, checkpoint_path):
        self.model = TLM.from_pretrained(checkpoint_path, config=self.tlmconfig, ts_config=self.tlmargs).cuda()

    def _build_model(self, args):
        """Load the model dynamically based on the configuration."""
        # self.tlmconfig = TLMConfig(llm_model_path = args.llm_model_path)
        model = TLM(self.tlmconfig,args).cuda()
        if accelerator.is_main_process:
            # 输出模型结构
            print("模型结构：")
            print(model)

            # 计算并输出可训练参数量（以百万为单位）
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_params_m = trainable_params / 1e6  # 转换为百万（M）

            print(f"模型训练参数量为：{trainable_params_m:.2f}M")
        # monitor = GradientAndActivationMonitor(model,track_outputs=False,verbose=True)
        return model
    
    def concat_np_array(self, array_list,num_samples):
        """
        对传入的列表进行 Concat 操作。
        
        Args:
            array_list (List[List[int]]): 每个子列表为需要 Padding 的序列。
            num_samples (int): 样本数量。
        Returns:
            np.ndarray: Padding 后的二维数组。
        """
        # 获取最大长度
        max_length = max(arr.shape[-1] for arr in array_list)
        
        # 初始化 Padding 后的数组，填充为 padding_idx
        padded_array = np.full((num_samples, max_length), self.padding_idx, dtype=np.int32)
        
        # 填充每个序列
        for i, arr in enumerate(array_list):
            padded_array[:arr.shape[0], :arr.shape[1]] = arr
        concat_array = np.stack(padded_array, axis=0)
        return concat_array    

    def debug_generate(self, input_ids, query_ids,ts_values, stage, attention_mask):
        # 生成阶段
        import time
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                query_ids=query_ids,
                ts_values=ts_values,
                stage=stage,
                past_key_values=None,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=self.processor.eos_token_id,
                pad_token_id=self.processor.pad_token_id,
                attention_mask=attention_mask,
                use_cache=True,
                # 新增优化参数
                num_beams=1,                    # 贪婪搜索，最快
                temperature=1.0,                # 避免额外计算
                top_p=None,                     # 关闭nucleus sampling
                top_k=None,                     # 关闭top-k sampling
                repetition_penalty=1.0,         # 关闭重复惩罚
                length_penalty=1.0,             # 关闭长度惩罚
                no_repeat_ngram_size=0,         # 关闭n-gram重复检查
                output_scores=False,            # 不输出分数
                output_attentions=False,        # 不输出attention
                output_hidden_states=False,     # 不输出隐藏状态
                return_dict_in_generate=False,  # 简化返回格式
            )
        end_time = time.time()
        if accelerator.is_main_process:
            # generated_ids = output[:, input_ids.shape[1]:]
            print(f"[生成耗时] {end_time - start_time:.2f} 秒")
            # 解码输出为文字
            try:
                generated_text = self.processor.decode(output[0], skip_special_tokens=True)
                print(f"[生成结果] {generated_text}")
            except Exception as e:
                print(f"[错误] 无法解码 generated_ids: {e}")
        return output
    def generate(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        all_predictions = []
        all_labels = []
        all_losses = []
        all_index = []

        model = self._wrap_model(self.model, training=False)
        model.eval()
        sample_num = len(dataloader.dataset)
        # forms = []
        stages = []
        with torch.no_grad():
            for step, inputs in enumerate(distributed_tqdm(dataloader, desc=description)):
                # if step==50:
                #     break

                input_ids = inputs['input_ids']
                ts_values = inputs['ts_values']
                stage = inputs['stage']
                index = inputs['index']
                query_ids = inputs['query_ids']
                attention_mask =inputs['attention_mask']
                generated_ids = self.debug_generate(input_ids, 
                                                    query_ids,ts_values, stage, attention_mask)

                prediction = generated_ids.cpu().numpy()
                all_predictions.extend(prediction)
                all_labels.extend(inputs["labels"].cpu().numpy())

                # forms.extend(inputs['form'])
                stages.extend(inputs['stage'].tolist())

                all_index.extend(inputs['index'].tolist())

        filtered_preds, filtered_labels = [], []
        str_predictions = self.processor.batch_decode(all_predictions,skip_special_tokens=True)
        str_labels = self.processor.batch_decode(all_labels,skip_special_tokens=True)
        #取出assistant\n后的内容
        str_predictions = [pred.split('assistant\n')[-1] for pred in str_predictions]
        output_data = {
            "predictions": str_predictions,
            "labels": str_labels,
            "stages": stages,
            "index": all_index,
            "num_samples": sample_num
        }

        with open('output_result_all.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        pred_extra = {'stages': stages}
        avg_loss = np.mean(all_losses) if all_losses else None
        return EvalLoopOutput(predictions=str_predictions, label_ids=str_labels,
                               metrics=avg_loss, num_samples=sample_num,pred_extra=pred_extra)


    #写一个过滤str_predictions和str_labels的函数

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        eval_dataset = eval_dataset or self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.generate(
            eval_dataloader, 'eval'
        )

        metrics = self.custom_compute_metrics(output)

        if accelerator.is_main_process:
            # 打印到控制台
            print(metrics)
                # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_eval_{timestamp}.txt'
            # 同时写入文件

            with open(filename, 'w', encoding='utf-8') as f:
                print(metrics, file=f)


    def custom_compute_metrics(self,eval_pred: EvalLoopOutput) -> Dict[str, Any]:
        """
        针对 stages 为 1 或 2 的样本，计算 BLEU 和 ROUGE 指标。
        Args:
            eval_pred (EvalPrediction): 包含 predictions 和 labels，以及附加信息 pred_extra。
        
        Returns:
            Dict[str, Any]: BLEU 和 ROUGE 指标结果字典。
        """
        # 解析预测和标签
        labels =  eval_pred.label_ids
        stages = eval_pred.pred_extra['stages']        
        # 解析附加信息
        # 筛选 stages 为 1 
        stage1_indices = [i for i, stage in enumerate(stages) if stage in [1]]
        if  len(stage1_indices) >=1:
            # 提取对应的预测和标签
            stage1_labels = [labels[i] for i in stage1_indices]
            stage1_metrics = open_question_metrics([eval_pred.predictions[i] for i in stage1_indices], 
                                                   stage1_labels,self.special_id)

        #筛选出stage为2的样本
        stage2_indices = [i for i, stage in enumerate(stages) if stage in [2]]
        if len(stage2_indices) >=1:
            # 提取对应的预测和标签
            stage2_labels = [labels[i] for i in stage2_indices]
            stage2_predictions = [eval_pred.predictions[i] for i in stage2_indices] 
            stage2_metrics = closed_question_metrics( stage2_predictions,
                                                     stage2_labels,self.special_id)

        #筛选出stage为3的样本
        stage3_indices = [i for i, stage in enumerate(stages) if stage in [3]]
        if  len(stage3_indices)>=1 :
            # 提取对应的预测和标签
            stage3_labels = [labels[i] for i in stage3_indices]
            stage3_predictions = [eval_pred.predictions[i] for i in stage3_indices]
            stage3_metrics = closed_question_metrics( stage3_predictions, 
                                                     stage3_labels,self.special_id)

        #筛选出stage为4的样本
        stage4_indices = [i for i, stage in enumerate(stages) if stage in [4]]
        if len(stage4_indices) >=1:
            # 提取对应的预测和标签
            stage4_labels = [labels[i] for i in stage4_indices]
            stage4_metrics = open_question_metrics([eval_pred.predictions[i] for i in stage4_indices],
                                                    stage4_labels,self.special_id)
        
        #合并存在的指标
        metrics = {}
        if stage1_indices:
            metrics.update({f"stage1_{k}": v for k, v in stage1_metrics.items()})
        if stage2_indices:
            metrics.update({f"stage2_{k}": v for k, v in stage2_metrics.items()})
        if stage3_indices:
            metrics.update({f"stage3_{k}": v for k, v in stage3_metrics.items()})
        if stage4_indices:
            metrics.update({f"stage4_{k}": v for k, v in stage4_metrics.items()})


        return metrics
    
    def compute_stage_weighted_loss(self, logits, labels, stages, attention_mask=None):
        """
        修正版本 - 不需要shift，因为Dataset已经处理了
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 🔧 不需要shift，直接使用
        flat_logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        flat_labels = labels.view(-1)              # [batch_size * seq_len]
        
        # 计算基础损失（padding会被自动ignore）
        token_losses = self.base_loss_fn(flat_logits, flat_labels)  # [batch_size * seq_len]
        token_losses = token_losses.view(batch_size, seq_len)       # [batch_size, seq_len]
        
        # 创建有效token掩码
        valid_mask = (labels != self.padding_idx).float()  # [batch_size, seq_len]
        
        # 应用stage权重
        stage_weights = torch.tensor([self.stage_weights.get(stage.item(), 1.0) 
                                    for stage in stages], 
                                    device=logits.device, dtype=torch.float32)
        
        # 计算每个样本的加权损失
        sample_losses = []
        for i in range(batch_size):
            valid_tokens = valid_mask[i].sum()  # 有效token数量
            if valid_tokens > 0:
                # 🔧 只对有效token计算平均损失
                sample_loss = (token_losses[i] * valid_mask[i]).sum() / valid_tokens * stage_weights[i]
            else:
                sample_loss = torch.tensor(0.0, device=logits.device)
            sample_losses.append(sample_loss)
        
        return torch.stack(sample_losses).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        内存优化版本的损失计算
        """
        # 使用torch.no_grad()减少不必要的梯度计算
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            # 前向传播
            outputs = model(
                input_ids=inputs.get('input_ids'),
                query_ids=inputs.get('query_ids'),
                ts_values=inputs.get('ts_values'),
                stage=inputs.get('stage'),
                attention_mask=inputs.get('attention_mask'),
                labels=inputs.get('labels')
            )
        
        # 获取logits
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # 计算损失
        loss = self.compute_stage_weighted_loss(
            logits=logits,
            labels=inputs.get('labels'),
            stages=inputs.get('stage'),
            attention_mask=inputs.get('attention_mask')
        )
        
        if return_outputs:
            # 清理不必要的输出以节省内存
            if hasattr(outputs, 'past_key_values'):
                outputs.past_key_values = None
            if hasattr(outputs, 'hidden_states'):
                outputs.hidden_states = None
            if hasattr(outputs, 'attentions'):
                outputs.attentions = None
            
            wrapped_outputs = OutputWrapper(outputs)
            wrapped_outputs.loss = loss
            return loss, wrapped_outputs
        
        return loss
    
    def get_stage_loss_statistics(self, dataloader, num_samples=100):
        """
        分析不同stage的损失分布，用于调整权重
        
        Args:
            dataloader: 数据加载器
            num_samples: 分析的样本数量
        
        Returns:
            Dict: 包含各stage损失统计信息的字典
        """
        self.model.eval()
        stage_losses = {1: [], 2: [], 3: [], 4: []}
        
        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                # 移动到正确的设备
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.model.device)
                
                # 前向传播
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # 计算每个样本的损失
                labels = inputs['labels']
                stages = inputs['stage']
                attention_mask = inputs.get('attention_mask')
                
                batch_size, seq_len, vocab_size = logits.shape
                flat_logits = logits.view(-1, vocab_size)
                flat_labels = labels.view(-1)
                
                token_losses = self.base_loss_fn(flat_logits, flat_labels)
                token_losses = token_losses.view(batch_size, seq_len)
                
                if attention_mask is not None:
                    valid_mask = attention_mask.bool()
                else:
                    valid_mask = (labels != self.padding_idx)
                
                masked_losses = token_losses * valid_mask.float()
                valid_token_counts = valid_mask.sum(dim=1).float()
                valid_token_counts = torch.clamp(valid_token_counts, min=1.0)
                sample_losses = masked_losses.sum(dim=1) / valid_token_counts
                
                # 按stage收集损失
                for j, stage in enumerate(stages):
                    stage_val = stage.item()
                    if stage_val in stage_losses:
                        stage_losses[stage_val].append(sample_losses[j].item())
        
        # 计算统计信息
        statistics = {}
        for stage, losses in stage_losses.items():
            if losses:
                statistics[f'stage_{stage}'] = {
                    'mean': np.mean(losses),
                    'std': np.std(losses),
                    'count': len(losses),
                    'min': np.min(losses),
                    'max': np.max(losses)
                }
        
        return statistics

