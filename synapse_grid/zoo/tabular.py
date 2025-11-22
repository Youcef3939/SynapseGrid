import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_tabular_model(name: str, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
    if name == "mlp":
        return MLP(input_dim, output_dim, hidden_dims)
    else:
        raise ValueError(f"Unknown tabular model: {name}")