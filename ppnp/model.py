import torch, torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.utils import dropout_adj

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden)
        self.lin2 = torch.nn.Linear(hidden,  out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x, edge_index, training=True):
        x = F.dropout(x, p=0.5, training=training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=training)
        x = self.lin2(x)
        if training:
            edge_index, _ = dropout_adj(edge_index, p=0.5)
        return self.prop(x, edge_index)
