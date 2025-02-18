import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):

    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_embed, 4*n_embed)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4*n_embed, n_embed)

    def forward_embedding(self, x: Tensor) -> Tensor:
        x = self.relu(self.hidden(x))
        return x
    
    def project_embedding(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_embedding(x)
        x = self.project_embedding(x)
        return x

if __name__=="__main__":
    batch_size = 2
    sequence_length = 3
    embedding_dim = 16
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    mlp_module = MLP(n_embed=embedding_dim)
    output_tensor = mlp_module(input_tensor)

    print("MLP Input Shape:", input_tensor.shape)
    print("MLP Output Shape:", output_tensor.shape)

     