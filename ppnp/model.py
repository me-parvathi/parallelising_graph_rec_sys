import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.utils import dropout_adj

class Net(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = dropout

        # Glorot / Xavier initialization
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x, edge_index):
        # feature + adjacency dropout exactly as in notebook
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.training:
            edge_index, _ = dropout_adj(edge_index, p=self.dropout)

        return self.prop(x, edge_index)
