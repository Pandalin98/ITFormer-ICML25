#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Encoder implementation.
Provides various encoding mechanisms for time series data processing.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import Mlp, DropPath
from timm.layers.helpers import to_2tuple
from transformers.modeling_outputs import CausalLMOutputWithPast


def calculate_unfold_output_length(input_length, size, step):
    """Calculate output length after unfolding operation.
    
    Args:
        input_length: Length of input sequence
        size: Size of each window
        step: Step size for sliding window
        
    Returns:
        int: Number of output windows
    """
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
    """Cross-attention mechanism for time series processing."""
    
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        """Initialize cross-attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            qk_norm: Whether to normalize Q and K
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            norm_layer: Normalization layer type
            var_num: Number of variables
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        """Forward pass for cross-attention.
        
        Args:
            x: Input tensor of shape (B, N, C)
            query: Optional query tensor
            
        Returns:
            torch.Tensor: Attention output
        """
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """Feed-forward network for time series processing."""
    
    def __init__(
            self,
            dim,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            prefix_token_length=None,
            group=1,
    ):
        """Initialize feed-forward network.
        
        Args:
            dim: Input dimension
            hidden_features: Hidden layer dimension
            out_features: Output dimension
            act_layer: Activation layer type
            norm_layer: Normalization layer type
            bias: Whether to use bias
            drop: Dropout rate
            prefix_token_length: Length of prefix tokens
            group: Group size for grouped convolution
        """
        super().__init__()
        dim = dim
        hidden_features = hidden_features or 4*dim
        out_features = out_features or dim
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(dim, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
 
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def forward(self, x):
        """Forward pass for feed-forward network.
        
        Args:
            x: Input tensor of shape (n, var, l, d)
            
        Returns:
            torch.Tensor: Output tensor
        """
        n, var, l, d = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.drop2(x) + x
        return x


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embedding for time series."""
    
    def __init__(self, d_model, max_len=5000):
        """Initialize learnable positional embedding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        """Forward pass for learnable positional embedding.
        
        Args:
            x: Input tensor
            offset: Position offset
            
        Returns:
            torch.Tensor: Positional embeddings
        """
        return self.pe[:, :, offset:offset+x.size(2)]


class SeqAttention(nn.Module):
    """Sequential attention mechanism."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        """Initialize sequential attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            qk_norm: Whether to normalize Q and K
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            norm_layer: Normalization layer type
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        """Forward pass for sequential attention.
        
        Args:
            x: Input tensor of shape (B, N, C)
            attn_mask: Optional attention mask
            
        Returns:
            torch.Tensor: Attention output
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,  # attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):
    """Variable attention mechanism."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        """Initialize variable attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            qk_norm: Whether to normalize Q and K
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            norm_layer: Normalization layer type
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward pass for variable attention.
        
        Args:
            x: Input tensor of shape (B, N, P, C)
            
        Returns:
            torch.Tensor: Attention output
        """
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False)
        k = k.mean(dim=1, keepdim=False)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    """Gate layer for controlling information flow."""
    
    def __init__(self, dim, init_values=1e-5, inplace=False):
        """Initialize gate layer.
        
        Args:
            dim: Input dimension
            init_values: Initial values for gate
            inplace: Whether to perform operation in-place
        """
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        """Forward pass for gate layer.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Gated output
        """
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class SeqAttBlock(nn.Module):
    """Sequential attention block."""

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        """Initialize sequential attention block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            qk_norm: Whether to normalize Q and K
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate
            init_values: Initial values for scaling
            drop_path: Drop path rate
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask):
        """Forward pass for sequential attention block.
        
        Args:
            x: Input tensor
            attn_mask: Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        x_input = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask)
        x = x_input + self.drop_path(x)
        return x


class VarAttBlock(nn.Module):
    """Variable attention block."""

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        """Initialize variable attention block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            qk_norm: Whether to normalize Q and K
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate
            init_values: Initial values for scaling
            drop_path: Drop path rate
            norm_layer: Normalization layer type
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass for variable attention block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x_input = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x_input + self.drop_path(x)
        return x


class MLPBlock(nn.Module):
    """MLP block for time series processing."""

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,
            prefix_token_length=0,
    ):
        """Initialize MLP block.
        
        Args:
            dim: Input dimension
            mlp_ratio: MLP expansion ratio
            proj_drop: Projection dropout rate
            init_values: Initial values for scaling
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
            mlp_layer: MLP layer type
            prefix_token_length: Length of prefix tokens
        """
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is FeedForward:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        """Forward pass for MLP block.
        
        Args:
            x: Input tensor
            prefix_seq_len: Length of prefix sequence
            
        Returns:
            torch.Tensor: Output tensor
        """
        x_input = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_input + self.drop_path(x)
        return x


class BasicBlock(nn.Module):
    """Basic transformer block combining attention and MLP."""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            prefix_token_length=0,
    ):
        """Initialize basic transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projections
            qk_norm: Whether to normalize Q and K
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate
            init_values: Initial values for scaling
            drop_path: Drop path rate
            act_layer: Activation layer type
            norm_layer: Normalization layer type
            prefix_token_length: Length of prefix tokens
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_features=dim*4, act_layer=act_layer, drop=proj_drop)

    def forward(self, x, prefix_seq_len, attn_mask):
        """Forward pass for basic transformer block.
        
        Args:
            x: Input tensor
            prefix_seq_len: Length of prefix sequence
            attn_mask: Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        x_input = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask)
        x = x_input + x
        x_input = x
        x = self.feed_forward(x)
        x = x_input + x
        return x


class Patchfy(nn.Module):
    """Patchify layer for time series data."""
    
    def __init__(self, patch_len, stride):
        """Initialize patchify layer.
        
        Args:
            patch_len: Length of each patch
            stride: Stride for sliding window
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """Forward pass for patchify layer.
        
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            torch.Tensor: Patched tensor of shape (B, N, P, D)
        """
        B, L, D = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)
        return patches.permute(0, 1, 3, 2)


class Model(nn.Module):
    """Main time series encoder model."""

    def __init__(self, args):
        """Initialize time series encoder model.
        
        Args:
            args: Configuration arguments
        """
        super().__init__()
        self.config = args
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.input_len = args.input_len
        self.dropout = args.dropout
        self.model = args.model

        # Patchify layer
        self.patchify = Patchfy(self.patch_len, self.stride)
        
        # Positional encoding
        self.pos_embed = LearnablePositionalEmbedding(self.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BasicBlock(
                dim=self.d_model,
                num_heads=self.n_heads,
                mlp_ratio=8.,
                qkv_bias=False,
                qk_norm=False,
                proj_drop=self.dropout,
                attn_drop=self.dropout,
                drop_path=0.0,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                prefix_token_length=0,
            )
            for _ in range(self.e_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(self.d_model)

    def choose_masking(self, x, min_mask_ratio, max_mask_ratio):
        """Choose masking strategy for training.
        
        Args:
            x: Input tensor
            min_mask_ratio: Minimum masking ratio
            max_mask_ratio: Maximum masking ratio
            
        Returns:
            torch.Tensor: Masked tensor
        """
        # Generate a random number to decide which masking function to use
        mask_type = torch.randint(0, 2, (1,)).item()
        if mask_type == 0:
            return self.random_masking(x, min_mask_ratio, max_mask_ratio)
        else:
            return self.random_masking(x, min_mask_ratio, max_mask_ratio)

    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """Apply random masking to input tensor.
        
        Args:
            x: Input tensor
            min_mask_ratio: Minimum masking ratio
            max_mask_ratio: Maximum masking ratio
            
        Returns:
            torch.Tensor: Masked tensor
        """
        B, N, P, D = x.shape
        mask_ratio = torch.rand(1).item() * (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        mask_len = int(P * mask_ratio)
        
        # Create random mask
        mask = torch.ones(B, N, P, D, device=x.device)
        for b in range(B):
            for n in range(N):
                mask_indices = torch.randperm(P)[:mask_len]
                mask[b, n, mask_indices, :] = 0
        
        return x * mask

    def encode(self, x):
        """Encode time series data.
        
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            torch.Tensor: Encoded tensor
        """
        # Apply patches
        x = self.patchify(x)  # (B, N, P, D)
        
        # Add positional encoding
        x = x + self.pos_embed(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, prefix_seq_len=0, attn_mask=None)
        
        # Final normalization
        x = self.norm(x)
        
        return x

    def informer_encode(self, x):
        """Informer-style encoding.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Encoded tensor
        """
        return self.encode(x)

    def crossformer_encode(self, x):
        """Crossformer-style encoding.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Encoded tensor
        """
        return self.encode(x)

    def patchtst_encode(self, x):
        """PatchTST-style encoding.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Encoded tensor
        """
        return self.encode(x)

    def forward(self, ts_values):
        """Forward pass for time series encoder.
        
        Args:
            ts_values: Time series values of shape (B, L, D)
            
        Returns:
            torch.Tensor: Encoded time series features
        """
        # X_enc [B,L,D]
        if self.model == 'TimeSeriesEncoder':
            return self.encode(ts_values)
        elif self.model == 'Informer':
            return self.informer_encode(ts_values)
        elif self.model == 'Crossformer':
            return self.crossformer_encode(ts_values)
        elif self.model == 'PatchTST':
            return self.patchtst_encode(ts_values)
        else:
            raise ValueError(f"Unknown model type: {self.model}")
