#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Positional encoding implementations for transformer models.
Includes rotary, sinusoidal, and learnable positional encodings.
"""
import torch
import math
from torch import nn


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        """Initialize rotary positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(RotaryPositionalEncoding, self).__init__()
        
        # Ensure d_model is even
        assert d_model % 2 == 0

        # Create position indices
        position = torch.arange(0, max_len).float().unsqueeze(1) 
        dim = torch.arange(0, d_model // 2).float() 
        div_term = torch.exp(dim * -(math.log(10000.0) / (d_model // 2)))  

        # Calculate angles
        angle = position * div_term 
        sin_part = torch.sin(angle)  
        cos_part = torch.cos(angle)  

        # Combine sin and cos parts
        pe = torch.cat([sin_part, cos_part], dim=-1)  
        pe = pe.unsqueeze(0).unsqueeze(0) 
        self.register_buffer('pe', pe)  

    def forward(self, x, offset=0):
        """Apply rotary positional encoding.
        
        Args:
            x: Input tensor
            offset: Position offset
            
        Returns:
            torch.Tensor: Encoded tensor
        """
        seq_len = x.size(1)
        pe = self.pe[0, :, offset:offset + seq_len, :] 

        # Split input into two halves
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:] 

        # Apply rotary transformation
        x_rotated = torch.cat([
            x1 * pe[..., :x.size(-1)//2] - x2 * pe[..., x.size(-1)//2:],
            x1 * pe[..., x.size(-1)//2:] + x2 * pe[..., :x.size(-1)//2]
        ], dim=-1)  

        return x_rotated


class ReRoPE:
    """Relative Rotary Positional Encoding."""
    
    def __init__(self, dim: int):
        """Initialize ReRoPE.
        
        Args:
            dim: Model dimension
        """
        assert dim % 2 == 0
        self.dim = dim
        self.theta = self._compute_base_theta(dim)

    @staticmethod
    def _compute_base_theta(dim: int):
        """Compute base theta values.
        
        Args:
            dim: Model dimension
            
        Returns:
            torch.Tensor: Base theta values
        """
        theta = torch.tensor([10000 ** (-2 * (i // 2) / dim) for i in range(dim)])
        return theta

    def forward(self, pos: torch.Tensor):
        """Forward pass for ReRoPE.
        
        Args:
            pos: Position tensor
            
        Returns:
            torch.Tensor: Sinusoidal embeddings
        """
        seq_len = pos.size(-1)

        angles = pos.unsqueeze(-1) * self.theta
        sinusoidal_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return sinusoidal_embedding

    @staticmethod
    def apply_rotary_embedding(query, key, sincos):
        """Apply rotary embedding to query and key.
        
        Args:
            query: Query tensor
            key: Key tensor
            sincos: Sinusoidal embeddings
            
        Returns:
            tuple: Rotated query and key tensors
        """
        sin, cos = sincos[..., :query.size(-1)], sincos[..., query.size(-1):]
        query_rotated = query * cos + torch.roll(query, shifts=1, dims=-1) * sin
        key_rotated = key * cos + torch.roll(key, shifts=1, dims=-1) * sin
        return query_rotated, key_rotated
    
    
class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embedding."""
    
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
        return self.pe[0, :, offset:offset+x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        """Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # [d_model//2]

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).contiguous()
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        """Forward pass for sinusoidal positional encoding.
        
        Args:
            x: Input tensor
            offset: Position offset
            
        Returns:
            torch.Tensor: Positional embeddings
        """
        return self.pe[offset:offset + x.size(0), :]

