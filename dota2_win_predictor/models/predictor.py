import torch
import torch.nn as nn
import torch.nn.functional as F

class Dota2MLP(nn.Module):
    def __init__(self, hero_count: int = 126, hidden_dims: list[int] = [256, 128, 64]):
        super(Dota2MLP, self).__init__()

        input_dim = hero_count * 2  # 126 героев Radiant + 126 героев Dire

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], 1)  # бинарная классификация

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.output(x)
        return x  # используется с BCEWithLogitsLoss
