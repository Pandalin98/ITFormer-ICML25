import sys,re
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
class TLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, ts_pad_num = 25, **kwargs):
        self.ts_pad_num = ts_pad_num
        super().__init__(**kwargs)

class GradientAndActivationMonitor:
    def __init__(self, model, track_outputs=True, whitelist_patterns=None, verbose=False):
        self.model = model
        self.track_outputs = track_outputs
        self.verbose = verbose
        self.forward_hooks = []
        self.backward_hooks = []

        self.whitelist_patterns = whitelist_patterns or [
            r"^ts_project", r"^fusion_project", r"^itformer"
        ]

        self._register_hooks()

    def _is_whitelisted(self, name):
        return any(re.match(pattern, name) for pattern in self.whitelist_patterns)

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if not name or not self._is_whitelisted(name):
                continue

            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention, nn.Conv1d)):
                if self.verbose:
                    print(f"✅ Registering hooks on: {name}")
                if self.track_outputs:
                    self.forward_hooks.append(
                        module.register_forward_hook(self._get_forward_hook(name))
                    )
                self.backward_hooks.append(
                    module.register_full_backward_hook(self._get_backward_hook(name))
                )

    def _get_forward_hook(self, name):
        def hook(module, input, output):
            tensors = output if isinstance(output, (tuple, list)) else [output]
            for out in tensors:
                if isinstance(out, torch.Tensor):
                    with torch.no_grad():
                        if torch.isnan(out).any() or torch.isinf(out).any():
                            print(f"🚨 NaN or Inf in forward output of {name}")
                            print(f"[Forward:{name}] mean={out.mean():.4f}, std={out.std():.4f}, max={out.max():.2f}")
        return hook

    def _get_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            tensors = grad_output if isinstance(grad_output, (tuple, list)) else [grad_output]
            for grad in tensors:
                if isinstance(grad, torch.Tensor):
                    with torch.no_grad():
                        if torch.isnan(grad).any():
                            print(f"🔥 NaN in gradient of {name}")
                            print(f"[Backward:{name}] grad norm: {grad.norm():.4f}")
                        elif torch.isinf(grad).any():
                            print(f"🔥 Inf in gradient of {name}")
                            print(f"[Backward:{name}] grad norm: {grad.norm():.4f}")
        return hook

    def remove_hooks(self):
        for h in self.forward_hooks + self.backward_hooks:
            h.remove()
        self.forward_hooks.clear()
        self.backward_hooks.clear()


class TLM(PreTrainedModel, GenerationMixin):
    config_class = TLMConfig
    
    def __init__(self, config, ts_config):
        super().__init__(config)
        self.config = config
        self.ts_config = ts_config
        
        # Initialize tsencoder first
        self.ts_encoder = Model(ts_config)
        # load ts checkpoints
        if hasattr(ts_config, 'load_ts_encoder') and ts_config.load_ts_encoder:
            self._load_ts_encoder_weights(ts_config.load_ts_encoder)
            
        self.itformer = ITformer(ts_config)
        
        # Initialize LLM model and other components
        self.llm_model = None
        self.tokenizer = None
        self.ts_project = None
        self.query_project = None
        self.fusion_project = None
        self.loss_fct = None
        
        # These will be initialized after LLM model is loaded
        self._initialized = False

    def _initialize_llm_components(self, model_path):
        """Initialize LLM model and related components"""
        if self._initialized:
            return
            
        # Load LLM directly from LLM/Qwen2.5-0.5B-Instruct
        llm_path = "LLM/Qwen2.5-0.5B-Instruct"
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.llm_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Update ts_config with LLM model info
        self.ts_config.llm_d_model = self.llm_model.config.hidden_size
        
        # Initialize projection layers
        self.ts_project = nn.Linear(self.ts_config.d_model, self.ts_config.tt_d_model)
        self.query_project = nn.Linear(self.llm_model.config.hidden_size, self.ts_config.tt_d_model)
        self.fusion_project = nn.Linear(self.ts_config.tt_d_model, self.llm_model.config.hidden_size)
        
        # Set all parameters to eval mode for inference
        self.llm_model.eval()
        self.ts_encoder.eval()
        self.itformer.eval()
        self.ts_project.eval()
        self.query_project.eval()
        self.fusion_project.eval()
        
        # Freeze all parameters for inference
        for param in self.parameters():
            param.requires_grad = False
            
        if accelerator.is_main_process:
            print("✅ Model initialized for inference - all parameters frozen")
            
        self._initialized = True

    def _load_ts_encoder_weights(self, checkpoint_path):
        try:
            if accelerator.is_main_process:
                print(f"Loading TS encoder from: {checkpoint_path}")
            
            state_dict = load_file(checkpoint_path)
            model_state_dict = self.ts_encoder.state_dict()
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    if accelerator.is_main_process:
                        print(f"⚠️ Skipping {k}: shape mismatch or not found")
            
            self.ts_encoder.load_state_dict(filtered_state_dict, strict=False)
            
            if accelerator.is_main_process:
                print(f"✅ Loaded {len(filtered_state_dict)}/{len(model_state_dict)} parameters")
                
        except Exception as e:
            if accelerator.is_main_process:
                print(f"❌ Failed to load TS encoder weights: {e}")

    def prepare_inputs_for_generation(self, input_ids, query_ids, past_key_values=None, attention_mask=None, **kwargs):
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
        query_embeds = self.llm_model.get_input_embeddings()(query_ids)
        ts_embeds = self.ts_encoder(ts_values).logits
        ts_embeds = self.ts_project(ts_embeds)
        query_embeds_f = self.query_project(query_embeds)
        tt_embeds = self.itformer(query_embeds_f, ts_embeds, stage)
        tt_embeds = self.fusion_project(tt_embeds)
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        inputs_embeds = self.merge_input_ids_with_ts_features(tt_embeds, inputs_embeds, input_ids)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids=None, labels=None, query_ids=None, 
                ts_values=None, inputs_embeds=None, stage=None, index=None,
                attention_mask=None, past_key_values=None, **kwargs):
        
        if inputs_embeds is None:
            query_embeds = self.llm_model.get_input_embeddings()(query_ids)
            ts_embeds = self.ts_encoder(ts_values).logits
            ts_embeds = self.ts_project(ts_embeds)
            query_embeds_f = self.query_project(query_embeds)
            tt_embeds = self.itformer(query_embeds_f, ts_embeds, stage)
            tt_embeds = self.fusion_project(tt_embeds)
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
            inputs_embeds = self.merge_input_ids_with_ts_features(tt_embeds, inputs_embeds, input_ids)

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



