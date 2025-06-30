import torch
import math
from torch import nn
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):

        super(RotaryPositionalEncoding, self).__init__()
        
 
        assert d_model % 2 == 0

 
        position = torch.arange(0, max_len).float().unsqueeze(1) 
        dim = torch.arange(0, d_model // 2).float() 
        div_term = torch.exp(dim * -(math.log(10000.0) / (d_model // 2)))  

  
        angle = position * div_term 
        sin_part = torch.sin(angle)  
        cos_part = torch.cos(angle)  

        pe = torch.cat([sin_part, cos_part], dim=-1)  
        pe = pe.unsqueeze(0).unsqueeze(0) 
        self.register_buffer('pe', pe)  

    def forward(self, x, offset=0):

        seq_len = x.size(1)
        pe = self.pe[0, :, offset:offset + seq_len, :] 

        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:] 

        x_rotated = torch.cat([
            x1 * pe[..., :x.size(-1)//2] - x2 * pe[..., x.size(-1)//2:],
            x1 * pe[..., x.size(-1)//2:] + x2 * pe[..., :x.size(-1)//2]
        ], dim=-1)  

        return x_rotated

class ReRoPE:
    def __init__(self, dim: int):

        assert dim % 2 == 0
        self.dim = dim
        self.theta = self._compute_base_theta(dim)

    @staticmethod
    def _compute_base_theta(dim: int):

        theta = torch.tensor([10000 ** (-2 * (i // 2) / dim) for i in range(dim)])
        return theta

    def forward(self, pos: torch.Tensor):

        seq_len = pos.size(-1)

        angles = pos.unsqueeze(-1) * self.theta
        sinusoidal_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return sinusoidal_embedding

    @staticmethod
    def apply_rotary_embedding(query, key, sincos):

        sin, cos = sincos[..., :query.size(-1)], sincos[..., query.size(-1):]
        query_rotated = query * cos + torch.roll(query, shifts=1, dims=-1) * sin
        key_rotated = key * cos + torch.roll(key, shifts=1, dims=-1) * sin
        return query_rotated, key_rotated
    
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
        return self.pe[0, :, offset:offset+x.size(1), :]

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):

        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # [d_model//2]

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)  

        pe = pe.unsqueeze(0).unsqueeze(0)  
        self.register_buffer('pe', pe) 

    def forward(self, x, offset=0):
        return self.pe[0, :, offset:offset + x.size(1), :]

