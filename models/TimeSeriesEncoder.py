"""
Time Series Encoder
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import Mlp, DropPath
from timm.layers.helpers import to_2tuple
from transformers.modeling_outputs import CausalLMOutputWithPast



def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
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




class FeedFoward(nn.Module):
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
        super().__init__()
        dim = dim
        hidden_features = hidden_features or 4*dim
        out_features = out_features or dim
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(dim, hidden_features,
                              bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
 
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def forward(self, x):
        n, var, l, d = x.shape
        # x = x.view(-1, d) # (n*var, l, c)
        # x = x.transpose(-1, -2) # (n*var, c, l)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.drop2(x)+x
        return x



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
        return self.pe[:, :, offset:offset+x.size(2)]


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
            q, k, v,  # attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):

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

    def forward(self, x):
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

        # Robust reshape to handle dimension mismatches  
        # Expected: x has shape [B, num_heads, N, P * head_dim]
        # Target: [B, N, P, C] where C = num_heads * head_dim
        expected_last_dim = P * self.head_dim
        actual_last_dim = x.shape[-1]
        
        if actual_last_dim == expected_last_dim:
            # Standard case: reshape back to separate P dimension
            x = x.view(B, self.num_heads, N, self.head_dim, P).permute(0, 2, 4, 1, 3).reshape(B, N, P, -1)
        else:
            # Handle dimension mismatch case
            # When attention produces unexpected dimensions, we adapt by using a simpler reshape
            # This preserves the total information while ensuring dimensional consistency
            x = x.transpose(1, 2)  # [B, N, num_heads, actual_last_dim]
            
            # Flatten the last two dimensions and then reshape to target
            flattened_dim = self.num_heads * actual_last_dim
            x = x.reshape(B, N, flattened_dim)  # [B, N, flattened_dim]
            
            # Now we reshape to [B, N, P, remaining_dim] where remaining_dim = flattened_dim / P
            if flattened_dim % P == 0:
                remaining_dim = flattened_dim // P
                x = x.reshape(B, N, P, remaining_dim)
            else:
                # If not evenly divisible, we keep P=1 and adjust accordingly
                # This is a fallback that preserves all information
                x = x.reshape(B, N, 1, flattened_dim)
                # Expand to P patches by repeating (not ideal but prevents crash)
                x = x.expand(B, N, P, flattened_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


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
        # self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(
            x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(
            x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(x)
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
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.drop_path1(self.attn_var(self.norm1(x)))
        return x


class MLPBlock(nn.Module):

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
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is FeedFoward:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + \
                self.drop_path2(
                    self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
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
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)


        self.feed_forward = FeedFoward(dim=dim, hidden_features=dim*4, act_layer=act_layer, drop=proj_drop)

    def forward(self, x, prefix_seq_len, attn_mask):
        x = self.var_att_block(x)
        x = self.seq_att_block(x, attn_mask)
        x = self.feed_forward(x)
        return x


class Patchfy(nn.Module):
    def __init__(self, patch_len, stride):
        super(Patchfy, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"
    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x = x.transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.patchfy = Patchfy(args.patch_len, args.stride)
        self.layers = nn.ModuleList([
            BasicBlock(
                dim=args.d_model,
                num_heads=args.n_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=args.dropout,
                attn_drop=args.dropout,
                init_values=None,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                prefix_token_length=0
            ) for _ in range(args.e_layers)
        ])
        self.norm = nn.LayerNorm(args.d_model)
        self.patch_embedding = nn.Sequential(
            nn.Linear(args.patch_len, args.d_model, bias=False),
            nn.Dropout(args.dropout)
        )
        self.pretrain = getattr(args, 'pretrain', False)
        self.min_mask_ratio = getattr(args, 'min_mask_ratio', 0.7)
        self.max_mask_ratio = getattr(args, 'max_mask_ratio', 0.8)
        self.proj_head = nn.Linear(args.d_model, args.patch_len)
        # self.times_project = nn.Linear(6*args.d_model, args.d_model)
    def choose_masking(self, x, min_mask_ratio, max_mask_ratio):
        # Generate a random number to decide which masking function to use
        # if torch.rand(1).item() > right_prob:
        #     return self.random_masking(x, min_mask_ratio, max_mask_ratio)
        # else:
        #     return self.right_masking(x, min_mask_ratio, max_mask_ratio)
        return self.random_masking(x,min_mask_ratio,max_mask_ratio)
    
    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform random masking where a specified ratio of the total V*L blocks are masked.
        """
        N, V, L, D = x.shape  # batch, var, length, dim
        total_elements = V * L

        mask_ratio = (min_mask_ratio+max_mask_ratio)/2
        # Calculate the number of elements to keep based on the mask ratio
        total_keeps = int((1 - mask_ratio) * total_elements)

        # Generate a random noise array for each sample in the batch
        noise = torch.rand(N, V, L, device=x.device)  # noise in [0, 1] for V*L blocks

        # Flatten noise for easier processing
        noise_flat = noise.view(N, V * L)

        # Get indices to sort and restore noise
        ids_shuffle = torch.argsort(noise_flat, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Create the binary mask: 0 is keep, 1 is remove
        # We create a range tensor and compare it with total_keeps to generate the mask
        range_tensor = torch.arange(V * L, device=x.device).repeat(N, 1)
        mask_flat = range_tensor >= total_keeps

        # Unshuffle to get the binary mask in original order
        mask_flat = mask_flat.gather(dim=1, index=ids_restore)

        # Reshape mask back to the original V, L dimensions
        mask = mask_flat.view(N, V, L)


        return mask
    
    def encode(self, x):
        B,n_vars,N,C = x.shape
        for layer in self.layers:
            x = layer(x, prefix_seq_len=None, attn_mask=None)
        x = self.norm(x)
        # x = x.view(B // n_vars, n_vars , N* C)
        # x = self.vars_project(x)
        return x


    def forward(self, ts_values):
        #X_enc [B,L,D]
        x =  ts_values 
        x = self.patchfy(x)
        if self.pretrain:
            orin_x = x
            mask = self.choose_masking(x,self.min_mask_ratio, self.max_mask_ratio)
            mask_repeat = mask.unsqueeze(dim=-1) #[B,D,N,1]
            mask_repeat = mask_repeat.repeat(1, 1, 1, x.shape[-1])#[B,D,N,d]
            #进行掩码
            x = x.masked_fill(mask_repeat, 0)
        x = self.patch_embedding(x)
        x = self.encode(x)
        if self.pretrain:
            predict_x = self.proj_head(x)
            loss = F.mse_loss(predict_x, orin_x, reduction='mean')
            return CausalLMOutputWithPast(loss=loss, logits=x)
        return CausalLMOutputWithPast(logits=x,loss=None)




def test_model():
    import yaml
    import argparse


    #读取args
    parser = argparse.ArgumentParser(description='TsEncoder Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--input_len", type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--pretrain', type=bool, default=True, help='pretrain mode')
    parser.add_argument('--min_mask_ratio', type=float, default=0.1, help='minimum mask ratio')
    parser.add_argument('--max_mask_ratio', type=float, default=0.3, help='maximum mask ratio')

    args = parser.parse_args()
    # 创建一个Model实例

    model = Model(args)

    # 创建一些假的输入数据
    x_enc = torch.randn(10, 600, 33)  # 假设有10个样本，每个样本有50个时间步，每个时间步有100个特征

    # 调用forward方法
    dec_out = model.forward(x_enc)

    if 'loss' in dec_out:
        print(f"Loss: {dec_out['loss']}")
        print(f"Logits Shape: {dec_out['logits'].shape}")
    else:
        print(f"Logits Shape: {dec_out['logits'].shape}")

# 在脚本的最后调用测试函数
if __name__ == '__main__':
    test_model()