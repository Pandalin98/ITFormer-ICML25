#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Language Model (TLM) implementation.
A multimodal model that combines time series data with language model for time series question answering.
"""
import sys
import re
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, GenerationMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import argparse
from models.TimeSeriesEncoder import Model
from safetensors.torch import load_file
from models.TT_Former import ITformer
import accelerate
accelerator = accelerate.Accelerator()  
from peft import LoraConfig, get_peft_model, TaskType
from safetensors import safe_open
import traceback
import os
import json

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
    """Time Language Model that combines time series data with language model."""
    config_class = TLMConfig
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, ts_config=None, **kwargs):
        """Load model from pretrained checkpoint.
        
        Args:
            pretrained_model_name_or_path: Path to the checkpoint
            config: Model configuration
            ts_config: Time series encoder configuration
            **kwargs: Additional arguments
            
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
                # Check LLM model path existence and use default path
                if 'llm_model_path' in config_dict:
                    llm_path = config_dict['llm_model_path']
                    if not os.path.exists(llm_path):
                        print(f"Warning: LLM model path '{llm_path}' does not exist, using default path")
                        config_dict['llm_model_path'] = 'LLM/Qwen2.5-0.5B-Instruct'
                config = TLMConfig(**config_dict)
        else:
            if config is None:
                config = TLMConfig()

        # Ensure LLM model path exists (redundant check for robustness)
        if not os.path.exists(config.llm_model_path):
            print(f"Warning: LLM model path '{config.llm_model_path}' does not exist, using default path")
            config.llm_model_path = 'LLM/Qwen2.5-0.5B-Instruct'

        # Load ts_config.json or use default values
        ts_config_path = os.path.join(pretrained_model_name_or_path, "ts_config.json")
        if os.path.exists(ts_config_path):
            with open(ts_config_path, 'r') as f:
                ts_config_dict = json.load(f)
            if ts_config is None:
                class TSConfig:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                ts_config = TSConfig(**ts_config_dict)
        if ts_config is None:  # If ts_config.json not found, use default configuration
            print("No ts_config.json found, using default configuration based on checkpoint analysis")
            class TSConfig:
                def __init__(self):
                    # Configuration based on checkpoint analysis
                    self.d_model = 512  # ts_encoder parameters
                    self.n_heads = 8
                    self.e_layers = 4   # ts_encoder has 4 layers
                    self.patch_len = 60
                    self.stride = 60
                    self.input_len = 600
                    self.dropout = 0.1
                    self.tt_d_model = 896  # itformer parameters
                    self.tt_n_heads = 8
                    self.tt_layers = 2     # itformer has only 2 layers
                    self.tt_dropout = 0.1
                    self.prefix_num = 25   # From config.json
            ts_config = TSConfig()

        model = cls(config, ts_config)

        # Load model weights
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")

        if os.path.exists(model_path):
            print(f"Loading model weights from: {model_path}")
            if model_path.endswith('.safetensors'):
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            
            # Load weights
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Handle missing lm_head.weight - fix based on actual checkpoint content
            if 'llm_model.lm_head.weight' in missing_keys:
                print("🔧 Fixing missing lm_head.weight by tying with embed_tokens.weight")
                print(f"🔍 Debug: llm_model type: {type(model.llm_model)}")
                
                # According to checkpoint analysis, embed_tokens is in model.model
                if hasattr(model.llm_model, 'model') and hasattr(model.llm_model.model, 'embed_tokens'):
                    embed_layer = model.llm_model.model.embed_tokens
                    print(f"✅ Found embedding layer: model.model.embed_tokens")
                    
                    if hasattr(model.llm_model, 'lm_head'):
                        lm_head_layer = model.llm_model.lm_head
                        print(f"✅ Found lm_head layer: lm_head")
                        
                        # Tie weights
                        if hasattr(embed_layer, 'weight') and hasattr(lm_head_layer, 'weight'):
                            lm_head_layer.weight = embed_layer.weight
                            missing_keys.remove('llm_model.lm_head.weight')
                            print("✅ Successfully tied lm_head.weight with embed_tokens.weight")
                        else:
                            print("⚠️ Warning: embed_tokens or lm_head missing weight attribute")
                    else:
                        print("⚠️ Warning: lm_head not found in model")
                else:
                    print("⚠️ Warning: embed_tokens not found in model.model")
                    print("💡 Note: This missing key is often expected and does not affect inference")
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(f"Warning: No model weights found at {model_path}")

        # If LLM model is not initialized, load from checkpoint
        if model.llm_model is None:
            print("Loading LLM model from checkpoint...")
            try:
                model.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
                model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
                model.llm_model.config.pad_token_id = model.tokenizer.pad_token_id
            except Exception as e:
                print(f"Failed to load LLM model from checkpoint: {e}")
                raise e

        # Set tokenizer based on checkpoint content
        if model.tokenizer is None:
            print("Loading tokenizer from checkpoint...")
            try:
                model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
                # Set special tokens based on added_tokens.json
                added_tokens_path = os.path.join(pretrained_model_name_or_path, "added_tokens.json")
                if os.path.exists(added_tokens_path):
                    with open(added_tokens_path, 'r') as f:
                        added_tokens = json.load(f)
                    print(f"Found {len(added_tokens)} added tokens in checkpoint")
                    # Ensure <|image_pad|> token exists
                    if '<|image_pad|>' in added_tokens:
                        print(f"✅ <|image_pad|> token found with ID: {added_tokens['<|image_pad|>']}")
            except Exception as e:
                print(f"Failed to load tokenizer from checkpoint: {e}")
                raise e

        # Set inference mode
        model._setup_inference_mode()
        
        # Save checkpoint path for later use
        model.checkpoint_path = pretrained_model_name_or_path
        
        return model
    
    def save_pretrained(self, save_directory):
        """Save model to specified directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save time series encoder configuration
        ts_config_dict = {}
        for attr in dir(self.ts_encoder.config):
            if not attr.startswith('_'):
                value = getattr(self.ts_encoder.config, attr)
                if isinstance(value, (int, float, str, bool, list, dict)):
                    ts_config_dict[attr] = value
        
        ts_config_path = os.path.join(save_directory, "ts_config.json")
        with open(ts_config_path, 'w') as f:
            json.dump(ts_config_dict, f, indent=2)
        
        # Save model weights
        model_path = os.path.join(save_directory, "model.safetensors")
        from safetensors.torch import save_file
        save_file(self.state_dict(), model_path)
        
        print(f"Model saved to: {save_directory}")
    
    def __init__(self, config, ts_config):
        """Initialize TLM model.
        
        Args:
            config: TLM configuration
            ts_config: Time series encoder configuration
        """
        super().__init__(config)
        self.config = config
        
        # Initialize LLM model
        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        except Exception as e:
            print(f"Warning: Failed to load LLM model from {self.config.llm_model_path}: {e}")
            print("This is expected if the model was saved with a different transformers version.")
            print("The LLM model will be loaded from the checkpoint weights.")
            # Create an empty LLM model, weights will be loaded later
            self.llm_model = None
            self.tokenizer = None
        
        if self.llm_model is not None:
            self.llm_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Set LLM hidden layer dimension (if LLM model is loaded)
        if self.llm_model is not None:
            ts_config.llm_d_model = self.llm_model.config.hidden_size
        else:
            # Use default value, will be inferred from weights later
            ts_config.llm_d_model = 896
        
        # Initialize time series encoder
        self.ts_encoder = Model(ts_config)
        
        # Initialize other components
        self.itformer = ITformer(ts_config)
        
        # Projection layers (if LLM model is loaded)
        self.ts_project = nn.Linear(ts_config.d_model, ts_config.tt_d_model)
        if self.llm_model is not None:
            self.query_project = nn.Linear(self.llm_model.config.hidden_size, ts_config.tt_d_model)
            self.fusion_project = nn.Linear(ts_config.tt_d_model, self.llm_model.config.hidden_size)
        else:
            # Use default dimensions, will be inferred from weights later
            self.query_project = nn.Linear(896, ts_config.tt_d_model)
            self.fusion_project = nn.Linear(ts_config.tt_d_model, 896)
        
        # Set to inference mode, freeze all parameters
        self._setup_inference_mode()

    def _setup_inference_mode(self):
        """Set inference mode, freeze all parameters."""
        # Freeze all parameters for inference
        for param in self.parameters():
            param.requires_grad = False
        
        # Set to evaluation mode
        self.eval()
        
        if accelerator.is_main_process:
            print('🧊 Model set to inference mode - all parameters frozen')

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
        
        # Use query_ids to get query_embeds
        query_embeds = self.llm_model.get_input_embeddings()(query_ids)
        ts_embeds = self.ts_encoder(ts_values).logits
        ts_embeds = self.ts_project(ts_embeds)
        query_embeds_f = self.query_project(query_embeds)
        tt_embeds = self.itformer(query_embeds_f, ts_embeds, stage)
        tt_embeds = self.fusion_project(tt_embeds)
        
        # Generate inputs_embeds
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        inputs_embeds = self.merge_input_ids_with_ts_features(tt_embeds, inputs_embeds, input_ids)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids=None, query_ids=None, 
                ts_values=None, inputs_embeds=None, stage=None, index=None,
                attention_mask=None, past_key_values=None, **kwargs):
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
            tt_embeds = self.itformer(query_embeds_f, ts_embeds, stage)
            tt_embeds = self.fusion_project(tt_embeds)
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
            inputs_embeds = self.merge_input_ids_with_ts_features(tt_embeds, inputs_embeds, input_ids)

        # Key: Let gradients flow normally, don't use torch.no_grad()
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, 
                                 use_cache=True)
        
        logits = outputs.logits

        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)

    def merge_input_ids_with_ts_features(self, ts_features, inputs_embeds, input_ids):
        """Merge time series features with input embeddings.
        
        Args:
            ts_features: Time series features
            inputs_embeds: Input embeddings
            input_ids: Input token IDs
            
        Returns:
            torch.Tensor: Merged embeddings
        """
        batch_size, seq_len, embed_dim = inputs_embeds.shape
        num_tss, num_ts_patches, embed_dim_ = ts_features.shape
        assert embed_dim == embed_dim_, "Embedding dimensions must match."

        pad_token_id = self.tokenizer('<|image_pad|>')['input_ids'][0]
        batch_indices, seq_indices = torch.where(input_ids == pad_token_id)

        if len(batch_indices) != num_tss * num_ts_patches:
            raise ValueError(f"Mismatch: found {len(batch_indices)} pad positions but got {num_tss * num_ts_patches} ts_features.")

        # Ensure dtype and device consistency, maintain gradients
        ts_features_flat = ts_features.view(-1, embed_dim).to(
            dtype=inputs_embeds.dtype, 
            device=inputs_embeds.device
        )

        # Use clone() to ensure correct gradient propagation
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_indices, seq_indices] = ts_features_flat

        return inputs_embeds

    def eval(self):
        """Set to evaluation mode."""
        super().eval()
        if self.llm_model is not None:
            self.llm_model.eval()
        self.ts_encoder.eval()
        self.itformer.eval()
        self.ts_project.eval()
        self.query_project.eval()
        self.fusion_project.eval()
        return self

if __name__ == '__main__':
    # Inference parameter settings
    parser = argparse.ArgumentParser(description='TLM Inference')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/ITformer-0.5B', 
                        help='Path to saved model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum generation length')
    parser.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy decoding')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p for sampling')
    
    args = parser.parse_args()

    # Model loading
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = TLM.from_pretrained(args.checkpoint_path)
    model.to(args.device)
    model.eval()

    # Load tokenizer
    if os.path.exists(os.path.join(args.checkpoint_path, "tokenizer.json")):
        print("Loading tokenizer from checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    else:
        print("Loading tokenizer from original model")
        tokenizer = model.tokenizer
    
    # Check and add special token
    if '<|image_pad|>' not in tokenizer.get_vocab():
        print("Adding <|image_pad|> token to tokenizer")
        tokenizer.add_tokens(['<|image_pad|>'])
        model.llm_model.resize_token_embeddings(len(tokenizer))
    
    ts_pad_token_id = tokenizer('<|image_pad|>')['input_ids'][0]
    print(f"Time series pad token ID: {ts_pad_token_id}, vocab_size: {len(tokenizer)}")
    
    # Inference test
    print("\n=== Inference Test ===")
    
    # Prepare test data
    query_text = "Based on the engine signal data, what should we do?"
    query_ids = tokenizer(query_text, return_tensors="pt").input_ids.to(args.device)
    
    # Create input with time series placeholders
    input_ids = torch.tensor([[tokenizer.pad_token_id] + [ts_pad_token_id] * model.config.ts_pad_num + [tokenizer.pad_token_id, tokenizer.pad_token_id]]).to(args.device)
    input_ids = torch.cat([query_ids, input_ids], dim=1)
    
    # Time series data (batch_size=1, sequence_length=600, features=33)
    ts_values = torch.randn(1, 600, 33).to(args.device)
    stage = [torch.tensor(0).to(args.device)]

    print(f"Query text: {query_text}")
    print(f"Input shape: {input_ids.shape}")
    print(f"TS values shape: {ts_values.shape}")

    # Forward inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            query_ids=query_ids,
            ts_values=ts_values, 
            stage=stage
        )
    print(f"Logits shape: {outputs.logits.shape}")

    # Text generation test
    print("\n=== Text Generation Test ===")
    
    # Prepare generation input
    prompt = "Based on the engine signal data over 10 cycles, what should we do?"
    gen_query_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    input_ids_gen = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    
    # Insert time series placeholders
    pad_tokens = torch.tensor([[ts_pad_token_id] * model.config.ts_pad_num]).to(args.device)
    input_ids_gen = torch.cat([input_ids_gen[:, :-1], pad_tokens, input_ids_gen[:, -1:]], dim=1)
    attention_mask = (input_ids_gen != tokenizer.pad_token_id).long()

    print(f"Generation prompt: {prompt}")
    print(f"Input shape: {input_ids_gen.shape}")

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids_gen,
            query_ids=gen_query_ids,
            ts_values=ts_values,
            stage=stage,
            max_length=args.max_length,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

    print("\n=== Inference Complete ===")
