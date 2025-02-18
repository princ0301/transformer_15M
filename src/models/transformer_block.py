import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.mlp import MLP
from typing import Tuple

class Block(nn.Module):

    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        res = x + self.attn(self.ln1(x))
        x = self.mlp.forward_embedding(self.ln2(res))
        return x, res

if __name__ == '__main__': 
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    transformer_block = Block(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)
    output_tensor = transformer_block(input_tensor)

    print("Transformer Block Input Shape:", input_tensor.shape)
    print("Transformer Block Output Shape:", output_tensor.shape)