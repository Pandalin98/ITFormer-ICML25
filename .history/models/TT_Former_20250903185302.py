"""
Time Text Former
"""
import sys
sys.path.append('/dataYYF/dataWX/SJ/Time-QA/')
import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import Mlp, DropPath
from timm.layers.helpers import to_2tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils.position_coding import LearnablePositionalEmbedding, SinusoidalPositionalEncoding,RotaryPositionalEncoding
from models.layers.attention import InstructTimeAttention
class SeqCrossAttention(nn.Module):
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
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value, attn_mask=None):
        B, N, C = query.shape
        _, V, L, _ = key_value.shape

        # Reshape Key and Value to focus only on L (time) dimension
        key_value = key_value.view(B * V, L, C)

        # Compute Query, Key, and Value projections
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(key_value).reshape(B * V, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Adjust batch size for Key and Value
        k = k.view(B, V, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, V * L, self.head_dim)
        v = v.view(B, V, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, V * L, self.head_dim)

        # Apply normalization (if any)
        q = self.q_norm(q).reshape(B * self.num_heads, N, self.head_dim)
        k = self.k_norm(k)

        # Scaled Dot-Product Attention over L dimension
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.
        )

        # Reshape and project output
        x = x.view(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SeqAttBlock(nn.Module):

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
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_value, attn_mask):
        x_input = x
        x = self.norm1(x)
        key_value = self.norm1(key_value)

        # key_value = torch.reshape(
        #     key_value, (-1, key_value.shape[-2], key_value.shape[-1]))
        x = self.attn_seq(x, key_value, attn_mask)
        x = x_input + self.drop_path1(x)
        return x

class VarCrossAttention(nn.Module):

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
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value, attn_mask=None):
        B, N, C = query.shape
        _, V, L, _ = key_value.shape

        # Reshape Key and Value to focus only on V (variable) dimension
        key_value = key_value.view(B * L, V, C)

        # Compute Query, Key, and Value projections
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(key_value).reshape(B * L, V, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Adjust batch size for Key and Value
        k = k.view(B, L, self.num_heads, V, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, L * V, self.head_dim)
        v = v.view(B, L, self.num_heads, V, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, L * V, self.head_dim)

        # Apply normalization (if any)
        q = self.q_norm(q).reshape(B * self.num_heads, N, self.head_dim)
        k = self.k_norm(k)

        # Scaled Dot-Product Attention over V dimension
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.
        )

        # Reshape and project output
        x = x.view(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class VarAttBlock(nn.Module):

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
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_value, attn_mask):
        x_input = x
        x = self.norm1(x)
        key_value = self.norm1(key_value)

        x = self.attn_var(x, key_value, attn_mask)
        x = x_input + self.drop_path1(x)
        return x

class SeqAttention(nn.Module):

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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,  attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttBlock(nn.Module):

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
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask=None):
        x_input = x
        x = self.norm1(x)
        x = self.attn_seq(x, attn_mask)
        x = x_input + self.drop_path1(x)
        return x


class ITAttBlock(nn.Module):

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
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_it = InstructTimeAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,

        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x,memory, attn_mask=None):
        x_input = x
        x = self.attn_it(x, memory,attn_mask)
        x = x_input + self.norm1(self.drop_path1(x))
        return x

class DecoderBasicBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            prefix_num=10,
    ):
        super().__init__()

        self.prefix_num = prefix_num

        self.self_attn = SelfAttBlock(
              dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, norm_layer=norm_layer
        ) 

        self.it_attn = ITAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, norm_layer=norm_layer
        )

        
        self.feed_forward_prefix = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.feed_forward_instruct = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
        )


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, memory, attn_mask=None):

        # Self-attention block
        x = x + self.self_attn(x, attn_mask)
        prefix  =x[:, :self.prefix_num, :]
        x = self.feed_forward_instruct(x)+x
        # Cross-attention block 10token vs b,n,c,d中的n
        prefix = prefix + self.it_attn(prefix, memory, attn_mask)
        # 10token vs b,n,c,d中的c
        # Feed forward block
        prefix = prefix + self.feed_forward_prefix(prefix)
        # Concatenate prefix and x
        x = torch.cat([prefix, x[:, self.prefix_num:, :]], dim=1)
        return x


class ITformer(nn.Module):
    def __init__(self, args):
        super(ITformer, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBasicBlock(
                dim=args.tt_d_model,
                num_heads=args.tt_n_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=args.tt_dropout,
                attn_drop=args.tt_dropout,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                prefix_num=args.prefix_num
            ) for _ in range(args.tt_layers)
        ])
        self.norm = nn.LayerNorm(args.tt_d_model)


        #time posi
        self.time_pos = SinusoidalPositionalEncoding(args.tt_d_model)
        #variable posi
        self.var_pos = LearnablePositionalEmbedding(args.tt_d_model)
        #instruction posi
        self.instruc_pos = SinusoidalPositionalEncoding(args.tt_d_model)
        # cycle posi
        self.cycle_pos = RotaryPositionalEncoding(args.tt_d_model)

        #prefix num
        self.prefix_num = args.prefix_num
        self.prefix_token = nn.Parameter(torch.randn(1, args.prefix_num, args.tt_d_model))
    def forward(self, x, memory, stage=None,attn_mask=None):

        # Add prefix token to x 
        x = torch.cat([self.prefix_token.repeat(x.shape[0], 1, 1), x], dim=1)
        # Positional encoding
        # Apply positional encoding to x
        x = x + self.instruc_pos(x)

        #Stage是list,找出stage中等于3,4的位置
        cycle_index = [i for i in stage if i != 3 and i != 4]
        cross_cycle_index = [i for i in stage if i == 3 or i == 4]

        cycle_memory = memory[cycle_index, :, :, :]
        cross_cycle_memory = memory[cross_cycle_index, :, :, :]

        # Reshape and apply positional encoding to memory at time dimension
        b, l, v, d = cycle_memory.shape
        cycle_memory = cycle_memory.view(b * l, v, d)
        cycle_memory = cycle_memory + self.time_pos(cycle_memory)
        cycle_memory = cycle_memory.view(b, l, v, d)


        # Reshape and apply positional encoding to memory at cycle dimension
        b, l, v, d = cross_cycle_memory.shape
        cross_cycle_memory = cross_cycle_memory.view(b * v, l, d)
        cross_cycle_memory = cross_cycle_memory + self.cycle_pos(cross_cycle_memory)
        cross_cycle_memory = cross_cycle_memory.view(b, l, v, d)

        memory = torch.cat([cycle_memory, cross_cycle_memory], dim=0)

        # Reshape and apply positional encoding to memory at var dimension
        b, v, l, d = memory.shape
        memory = memory.view(b * l, v, d)
        memory = memory + self.var_pos(memory)
        memory = memory.view(b, l, v, d)


        for layer in self.layers:
            x = layer(x, memory, attn_mask)
        x = self.norm(x)


        return x[:, :self.prefix_num, :]

def count_parameters(model):
    """统计模型中可训练参数的总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":

    # dim = 64
    # num_heads = 8
    # seq_len = 20
    # var_num = 5
    # memory_len = 30
    # batch_size = 2

    # x = torch.randn(batch_size, seq_len, dim)
    # memory = torch.randn(batch_size, var_num,memory_len, dim)
    # attn_mask = None

    # decoder_block = DecoderBasicBlock(
    #     dim=dim, num_heads=num_heads, qkv_bias=True, proj_drop=0.1, attn_drop=0.1
    # )
    # output = decoder_block(x, memory, attn_mask)
    # print("DecoderBasicBlock Output Shape:", output.shape)
    # class Args:
    #     def __init__(self):
    #         self.tt_d_model = 64
    #         self.tt_n_heads = 8
    #         self.tt_layers = 6
    #         self.tt_dropout = 0.1
    #         self.prefix_num = 10

    # args = Args()
    # model = TTformer(args)

    # x = torch.randn(batch_size, seq_len, dim)
    # memory = torch.randn(batch_size, var_num, memory_len, dim)
    # attn_mask = None
    # stage = [1,2]
    # output = model(x, memory,stage, attn_mask)
    # print("Model Output Shape:", output.shape)
    class Args:
        def __init__(self):
            self.tt_d_model = 512
            self.tt_n_heads = 8
            self.tt_layers = 4
            self.tt_dropout = 0.1
            self.prefix_num = 10

    args = Args()
    model = ITformer(args)

    # 打印可训练参数量
    total_trainable_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_trainable_params:,}")

    # # 可选：打印每一层的参数量
    # print("\nLayer-wise Parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.numel():,}")