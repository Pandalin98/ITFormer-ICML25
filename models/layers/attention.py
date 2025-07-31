import torch
import torch.nn as nn
import torch.nn.functional as F

class InstructTimeAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scale factor for dot-product attention
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # Query, Key, Value projections
        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.channel_query_proj = nn.Linear(dim, dim, bias=qkv_bias)  # Projection for channel attention query
        self.channel_key_proj = nn.Linear(dim, dim, bias=qkv_bias)    # Projection for channel attention key
        self.channel_value_proj = nn.Linear(dim, dim, bias=qkv_bias)  # Projection for channel attention value

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, memory, mask=None):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, L_q, d).
            memory (torch.Tensor): Memory tensor of shape (batch_size, L_p, V, d).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, L_q, d).
        """
        batch_size, L_q, d = query.size()
        _, L_p, V, _ = memory.size()

        # Step 1: Compute attention over channels (V) guided by query
        query_channel = self.channel_query_proj(query)  # Shape: (batch_size, L_q, d)
        key_channel = self.channel_key_proj(memory).mean(dim=1)     # Shape: (batch_size, V, d)
        value_channel = self.channel_value_proj(memory) # Shape: (batch_size, L_p, V, d)

        # Reshape for multi-head attention
        query_channel = self.q_norm(query_channel.view(batch_size, self.num_heads, L_q, self.head_dim))  # (batch_size, num_heads, L_q, head_dim)
        key_channel = self.k_norm(key_channel.view(batch_size, self.num_heads, V, self.head_dim)) # (batch_size, num_heads, V, head_dim)
        value_channel = value_channel.view(batch_size, self.num_heads, L_p, V, self.head_dim)  # (batch_size, num_heads, L_p, V, head_dim)
        

        # Compute scaled dot-product attention for channels
        channel_scores = torch.matmul(query_channel, key_channel.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, L_q, V)
        channel_weights = F.softmax(channel_scores, dim=-1)  # Normalize across channels
        channel_output = torch.einsum('bnqv,bnlvd->bnld', channel_weights, value_channel)  # (batch_size, num_heads, L_p, head_dim)

        # Reshape channel output back to memory dimension
        channel_output = channel_output.transpose(1, 2).contiguous()  # (batch_size, L_p, num_heads, head_dim)
        memory_aggregated = channel_output.view(batch_size, L_p, d)  # (batch_size, L_p, d)

        # Step 2: Compute Query, Key, and Value projections for time-step attention
        Q = self.query_proj(query)  # Shape: (batch_size, L_q, d)
        K = self.key_proj(memory_aggregated)  # Shape: (batch_size, L_p, d)
        V = self.value_proj(memory_aggregated)  # Shape: (batch_size, L_p, d)

        # Step 3: Reshape for multi-head attention
        Q = self.q_norm(Q.view(batch_size, L_q, self.num_heads, self.head_dim).transpose(1, 2))  # (batch_size, num_heads, L_q, head_dim)
        K = self.k_norm(K.view(batch_size, L_p, self.num_heads, self.head_dim).transpose(1, 2))  # (batch_size, num_heads, L_p, head_dim)
        V = V.view(batch_size, L_p, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, L_p, head_dim)

        # Step 4: Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, L_q, L_p)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, L_q, L_p)
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, L_q, head_dim)

        # Step 5: Combine multi-head outputs
        attention_output = attention_output.transpose(1, 2).contiguous()  # (batch_size, L_q, num_heads, head_dim)
        output = attention_output.view(batch_size, L_q, d)  # (batch_size, L_q, d)

        # Final linear projection
        output = self.proj(output)

        return output

if __name__ == '__main__':
    # Example usage
    batch_size = 8
    seq_len = 10  # Length of the query sequence
    mem_len = 20  # Length of the memory sequence
    mem_channels = 5  # Number of memory channels
    dim = 64  # Feature dimension
    num_heads = 4  # Number of attention heads

    # Query and Memory tensors
    query = torch.rand(batch_size, seq_len, dim)  # Shape: (batch_size, L_q, d)
    memory = torch.rand(batch_size, mem_len, mem_channels, dim)  # Shape: (batch_size, L_p, V, d)

    # Initialize the InstructTimeAttention module
    model = InstructTimeAttention(dim=dim, num_heads=num_heads, qkv_bias=True)

    # Compute the output
    output = model(query, memory)  # Shape: (batch_size, L_q, d)

    # Print output shape
    print("Output shape:", output.shape)
