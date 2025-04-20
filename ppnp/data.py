import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def get_cora(root, device):
    ds = Planetoid(root=root, name='Cora', transform=NormalizeFeatures())
    data = ds[0].to(device)
    return data, ds.num_features, ds.num_classes
