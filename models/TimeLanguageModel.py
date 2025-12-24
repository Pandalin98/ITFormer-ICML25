#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Language Model (TLM) for inference.
A multimodal model that combines time series data with language model for time series question answering.
"""
import os
import json
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from safetensors.torch import load_file
from models.TimeSeriesEncoder import Model
from models.ITFormer import ITFormer
from accelerate import Accelerator

accelerator = Accelerator()


class TLMConfig(PretrainedConfig):
    """Configuration class for Time Language Model."""
    model_type = "vlm_model"
    
    def __init__(self, llm_model_path='LLM/Qwen2.5-0.5B-Instruct',
                 freeze_ts_model=True,
                 ts_pad_num=25,
                 **kwargs):
        """Initialize TLM configuration.
        
        Args:
            llm_model_path: Path to the language model
            freeze_ts_model: Whether to freeze time series model parameters
            ts_pad_num: Number of time series padding tokens
            **kwargs: Additional configuration parameters
        """
        self.llm_model_path = llm_model_path
        self.freeze_ts_model = freeze_ts_model
        self.ts_pad_num = ts_pad_num
        super().__init__(**kwargs)


class TLM(PreTrainedModel, GenerationMixin):
    """Time Language Model for inference."""
    config_class = TLMConfig
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        """Load model from pretrained checkpoint.
        
        Args:
            pretrained_model_name_or_path: Path to the checkpoint
            config: Model configuration
            **kwargs: Additional arguments, including ts_config
            
        Returns:
            TLM: Loaded model instance
        """
        if not os.path.exists(pretrained_model_name_or_path):
            raise ValueError(f"Checkpoint path does not exist: {pretrained_model_name_or_path}")

        # Load config.json
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            if config is None:
                config = TLMConfig(**config_dict)
        else:
            if config is None:
                config = TLMConfig()

        # Create model instance with potential ts_config from kwargs
        model = cls(config, **kwargs)

        # Load model weights
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")

        state_dict = None
        # 1. Try normal files
        if os.path.exists(model_path):
            if accelerator.is_main_process:
                print(f"Loading model weights from: {model_path}")
            if model_path.endswith('.safetensors'):
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
        else:
            # 2. Try split safetensors in the same directory
            all_files = os.listdir(pretrained_model_name_or_path)
            safetensors_files = [f for f in all_files if f.startswith('model-') and f.endswith('.safetensors')]
            safetensors_files.sort()  # Ensure order
            if safetensors_files:
                if accelerator.is_main_process:
                    print(f"Loading split safetensors from: {pretrained_model_name_or_path}")
                state_dict = {}
                for fname in safetensors_files:
                    fpath = os.path.join(pretrained_model_name_or_path, fname)
                    part = load_file(fpath)
                    state_dict.update(part)
                if accelerator.is_main_process:
                    print(f"Successfully loaded {len(safetensors_files)} split safetensors files.")
        if state_dict is not None:
            # Separate LLM weights from other weights
            llm_weights = {}
            other_weights = {}
            for k, v in state_dict.items():
                if k.startswith('llm_model.'):
                    llm_weights[k] = v
                else:
                    other_weights[k] = v
            if accelerator.is_main_process:
                print(f"Found {len(llm_weights)} LLM weights (will be ignored)")
                print(f"Found {len(other_weights)} non-LLM weights (will be loaded)")
            # Load only non-LLM weights
            missing_keys, unexpected_keys = model.load_state_dict(other_weights, strict=False)
            # Filter out LLM-related missing keys since we're not loading LLM weights
            llm_missing_keys = [k for k in missing_keys if k.startswith('llm_model.')]
            non_llm_missing_keys = [k for k in missing_keys if not k.startswith('llm_model.')]
            if llm_missing_keys and accelerator.is_main_process:
                print(f"LLM missing keys (ignored): {len(llm_missing_keys)} keys")
            if non_llm_missing_keys and accelerator.is_main_process:
                print(f"Non-LLM missing keys: {non_llm_missing_keys}")
            if unexpected_keys and accelerator.is_main_process:
                print(f"Unexpected keys: {unexpected_keys}")
        else:
            if accelerator.is_main_process:
                print(f"Warning: No model weights found at {model_path} or in split safetensors.")

        return model

    def __init__(self, config, ts_config=None):
        """Initialize TLM model.
        
        Args:
            config: TLM configuration
            ts_config: Optional time series configuration (args)
        """
        super().__init__(config)
        self.config = config
        
        if ts_config is None:
            # Create default ts_config if not provided
            class DefaultTSConfig:
                def __init__(self):
                    self.model = 'TimeSeriesEncoder'
                    self.d_model = 512
                    self.n_heads = 8
                    self.e_layers = 4
                    self.patch_len = 60
                    self.stride = 60
                    self.input_len = 600
                    self.dropout = 0.1
                    self.it_d_model = 896
                    self.it_n_heads = 16
                    self.it_layers = 2
                    self.it_dropout = 0.1
                    self.prefix_num = 25
            ts_config = DefaultTSConfig()
        
        self.ts_config = ts_config
        
        # 统一属性名对齐逻辑：确保 ts_pad_num 和 prefix_num 存在且一致
        if hasattr(self.ts_config, 'ts_pad_num') and not hasattr(self.ts_config, 'prefix_num'):
            setattr(self.ts_config, 'prefix_num', self.ts_config.ts_pad_num)
        elif hasattr(self.ts_config, 'prefix_num') and not hasattr(self.ts_config, 'ts_pad_num'):
            setattr(self.ts_config, 'ts_pad_num', self.ts_config.prefix_num)
        
        # Initialize LLM model from external path
        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
            if accelerator.is_main_process:
                print(f"✅ Loaded LLM model from: {self.config.llm_model_path}")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"❌ Failed to load LLM model from {self.config.llm_model_path}: {e}")
            raise e
        
        if self.llm_model is not None:
            self.llm_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Set LLM hidden layer dimension
        ts_config.llm_d_model = self.llm_model.config.hidden_size
        
        # Initialize components
        self.ts_encoder = Model(ts_config)
        
        # 加载预训练的 TS Encoder 权重
        load_path = getattr(ts_config, 'load_ts_encoder', None)
        if load_path and os.path.exists(load_path):
            if accelerator.is_main_process:
                from utils.log_util import adaptive_print
                adaptive_print(f"📥 Loading pre-trained TimeSeries Encoder from: {load_path}")
            
            try:
                if load_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    ts_state_dict = load_file(load_path)
                else:
                    ts_state_dict = torch.load(load_path, map_location='cpu')
                
                # 兼容性处理：如果权重包含前缀，进行移除
                new_state_dict = {}
                for k, v in ts_state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                
                msg = self.ts_encoder.load_state_dict(new_state_dict, strict=False)
                if accelerator.is_main_process:
                    adaptive_print(f"✅ TS Encoder weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            except Exception as e:
                if accelerator.is_main_process:
                    adaptive_print(f"❌ Failed to load TS Encoder weights: {e}")
        elif load_path:
            if accelerator.is_main_process:
                from utils.log_util import adaptive_print
                adaptive_print(f"⚠️ Warning: TS Encoder load path '{load_path}' does not exist. Using random initialization.")

        self.itformer = ITFormer(ts_config)
        
        # Projection layers
        self.ts_project = nn.Linear(ts_config.d_model, ts_config.it_d_model)
        self.query_project = nn.Linear(ts_config.llm_d_model, ts_config.it_d_model)
        self.fusion_project = nn.Linear(ts_config.it_d_model, ts_config.llm_d_model)
        
        # 根据配置冻结参数
        self._freeze_layers()

    def _freeze_layers(self):
        """根据配置冻结特定层，保留中间件的可训练性。"""
        # 1. 强制冻结 LLM (通常 SFT 只训练适配器)
        if self.llm_model is not None:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        # 2. 根据配置冻结 TS Encoder
        if self.config.freeze_ts_model:
            for param in self.ts_encoder.parameters():
                param.requires_grad = False
        else:
            pass

        # 3. 确保中间件是可训练的 (ITFormer 和 Projections)
        # 这些层默认 requires_grad=True，所以不需要额外操作，
        # 除非之前调用了 _setup_inference_mode()

    def _setup_inference_mode(self):
        """Set inference mode, freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        if accelerator.is_main_process:
            print('🧊 Model set to inference mode - all parameters frozen')

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        if self.llm_model is not None:
            self.llm_model.eval()
        if self.ts_encoder is not None:
            self.ts_encoder.eval()
        if self.itformer is not None:
            self.itformer.eval()
        if self.ts_project is not None:
            self.ts_project.eval()
        if self.query_project is not None:
            self.query_project.eval()
        if self.fusion_project is not None:
            self.fusion_project.eval()

    def prepare_inputs_for_generation(self, input_ids, query_ids, past_key_values=None, attention_mask=None, **kwargs):
        """Prepare inputs for text generation.
        
        Args:
            input_ids: Input token IDs
            query_ids: Query token IDs
            past_key_values: Past key values for caching
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            dict: Prepared inputs for generation
        """
        ts_values = kwargs.get("ts_values", None)
        stage = kwargs.get("stage", None)
        
        if input_ids is None or input_ids.numel() == 0 or ts_values is None or ts_values.numel() == 0:
            return {
                "inputs_embeds": torch.empty(0, self.llm_model.config.hidden_size, device=input_ids.device),
                "attention_mask": attention_mask,
            }
        
        device = next(self.llm_model.parameters()).device
        input_ids = input_ids.to(device)
        ts_values = ts_values.to(device)             
        attention_mask = attention_mask.to(device) 
        
        if ts_values is None:
            raise ValueError("`ts_values` must be provided for generation.")
        
        # Process time series and query
        query_embeds = self.llm_model.get_input_embeddings()(query_ids)
        ts_embeds = self.ts_encoder(ts_values).logits
        ts_embeds = self.ts_project(ts_embeds)
        query_embeds_f = self.query_project(query_embeds)
        it_embeds = self.itformer(query_embeds_f, ts_embeds, stage)
        it_embeds = self.fusion_project(it_embeds)
        
        # Generate inputs_embeds
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        inputs_embeds = self.merge_input_ids_with_ts_features(it_embeds, inputs_embeds, input_ids)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids=None, query_ids=None, 
                ts_values=None, inputs_embeds=None, stage=None, index=None,
                attention_mask=None, past_key_values=None, labels=None, **kwargs):
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            query_ids: Query token IDs
            ts_values: Time series values
            inputs_embeds: Pre-computed input embeddings
            stage: Processing stage
            index: Sample index
            attention_mask: Attention mask
            past_key_values: Past key values for caching
            labels: Ground truth labels for loss calculation
            **kwargs: Additional arguments
            
        Returns:
            CausalLMOutputWithPast: Model output
        """
        if inputs_embeds is None:
            # Get query embedding
            query_embeds = self.llm_model.get_input_embeddings()(query_ids)
            # Time series encoding
            ts_embeds = self.ts_encoder(ts_values).logits
            ts_embeds = self.ts_project(ts_embeds)
            query_embeds_f = self.query_project(query_embeds)
            it_embeds = self.itformer(query_embeds_f, ts_embeds, stage)
            it_embeds = self.fusion_project(it_embeds)
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
            inputs_embeds = self.merge_input_ids_with_ts_features(it_embeds, inputs_embeds, input_ids)

        # Forward through LLM
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, 
                                 use_cache=True)
        
        logits = outputs.logits
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)

    def merge_input_ids_with_ts_features(self, ts_features, inputs_embeds, input_ids):
        batch_size, seq_len, embed_dim = inputs_embeds.shape
        num_tss, num_ts_patches, embed_dim_ = ts_features.shape
        assert embed_dim == embed_dim_, "Embedding dimensions must match."

        pad_token_id = self.tokenizer('<|image_pad|>')['input_ids'][0]
        batch_indices, seq_indices = torch.where(input_ids == pad_token_id)

        if len(batch_indices) != num_tss * num_ts_patches:
            raise ValueError(f"Mismatch: found {len(batch_indices)} pad positions but got {num_tss * num_ts_patches} ts_features.")
        ts_features_flat = ts_features.view(-1, embed_dim).to(
            dtype=inputs_embeds.dtype, 
            device=inputs_embeds.device
        )
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_indices, seq_indices] = ts_features_flat

        return inputs_embeds
