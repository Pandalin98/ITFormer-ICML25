import torch
import math
from torch import nn
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化 RotaryPositionalEncoding。

        Args:
            d_model (int): 特征维度。
            max_len (int): 支持的最大序列长度。
        """
        super(RotaryPositionalEncoding, self).__init__()
        
        # 确保特征维度为偶数
        assert d_model % 2 == 0, "d_model must be even for RotaryPositionalEncoding."

        # 创建旋转位置编码矩阵
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        dim = torch.arange(0, d_model // 2).float()  # [d_model // 2]
        div_term = torch.exp(dim * -(math.log(10000.0) / (d_model // 2)))  # [d_model // 2]

        # 计算正弦和余弦部分
        angle = position * div_term  # [max_len, d_model // 2]
        sin_part = torch.sin(angle)  # 正弦部分
        cos_part = torch.cos(angle)  # 余弦部分

        # 将 sin 和 cos 部分堆叠
        pe = torch.cat([sin_part, cos_part], dim=-1)  # [max_len, d_model]
        pe = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为非参数张量

    def forward(self, x, offset=0):
        """
        前向传播。

        Args:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]。
            offset (int): 位置偏移量，默认为 0。

        Returns:
            Tensor: 应用旋转位置编码的张量。
        """
        # 获取位置编码
        seq_len = x.size(1)
        pe = self.pe[0, :, offset:offset + seq_len, :]  # [1, seq_len, d_model]

        # 将输入张量拆分为两部分：前半部分与后半部分
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]  # [batch_size, seq_len, d_model//2]

        # 应用旋转操作
        x_rotated = torch.cat([
            x1 * pe[..., :x.size(-1)//2] - x2 * pe[..., x.size(-1)//2:],
            x1 * pe[..., x.size(-1)//2:] + x2 * pe[..., :x.size(-1)//2]
        ], dim=-1)  # [batch_size, seq_len, d_model]

        return x_rotated

class ReRoPE:
    def __init__(self, dim: int):
        """
        初始化 ReRoPE 编码器。

        Args:
            dim (int): 特征向量的维度（必须为偶数）。
        """
        assert dim % 2 == 0, "Dimension must be even for ReRoPE."
        self.dim = dim
        self.theta = self._compute_base_theta(dim)

    @staticmethod
    def _compute_base_theta(dim: int):
        """
        计算基本的 θ 值，用于旋转位置编码。

        Args:
            dim (int): 特征向量的维度。
        
        Returns:
            torch.Tensor: θ 值的张量。
        """
        theta = torch.tensor([10000 ** (-2 * (i // 2) / dim) for i in range(dim)])
        return theta

    def forward(self, pos: torch.Tensor):
        """
        计算给定位置的 ReRoPE 编码。

        Args:
            pos (torch.Tensor): 位置索引的张量，形状为 [seq_len] 或 [batch_size, seq_len]。

        Returns:
            torch.Tensor: ReRoPE 编码，形状为 [seq_len, dim] 或 [batch_size, seq_len, dim]。
        """
        seq_len = pos.size(-1)
        # 获取正弦和余弦部分
        angles = pos.unsqueeze(-1) * self.theta
        sinusoidal_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return sinusoidal_embedding

    @staticmethod
    def apply_rotary_embedding(query, key, sincos):
        """
        应用旋转位置编码到查询和键。

        Args:
            query (torch.Tensor): 查询向量，形状为 [batch_size, seq_len, dim].
            key (torch.Tensor): 键向量，形状为 [batch_size, seq_len, dim].
            sincos (torch.Tensor): ReRoPE 编码，形状为 [seq_len, dim] 或 [batch_size, seq_len, dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 编码后的查询和键。
        """
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
        """
        初始化 SinusoidalPositionalEncoding。

        Args:
            d_model (int): 特征维度。
            max_len (int): 支持的最大序列长度。
        """
        super(SinusoidalPositionalEncoding, self).__init__()

        # 创建一个固定的位置编码矩阵
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # [d_model//2]

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 将矩阵增加批量维度，方便在 forward 中直接使用
        pe = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为非参数张量

    def forward(self, x, offset=0):
        """
        前向传播。

        Args:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
            offset (int): 位置偏移量，默认为 0。

        Returns:
            Tensor: 添加了位置编码的张量。
        """
        return self.pe[0, :, offset:offset + x.size(1), :]

if __name__ == "__main__":
    d_model = 64
    seq_len = 128
    batch_size = 32

    pos_encoder = LearnablePositionalEmbedding(d_model=d_model, max_len=5000)
    x = torch.randn(batch_size, seq_len, d_model)  # 随机输入张量
    pos_encoded_x = pos_encoder(x)
    print("Shape of Positional Encoded Output:", pos_encoded_x.shape)