import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):

    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        scale_factor = 1 / math.sqrt(C)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        v = self.value(x)
        out = attn_weights @ v
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return x
    
if __name__=="__main__":
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_length = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    multihead_attn = MultiHeadAttention(n_head=num_heads, n_embed=embedding_dim, context_length=context_length)
    output_tensor = multihead_attn(input_tensor)

    print("MultiHeadAttention Input Shape:", input_tensor.shape)
    print("MultiHeadAttention Output Shape:", output_tensor.shape)