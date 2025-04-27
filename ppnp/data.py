import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_data(name: str, root: str = "data/Planetoid"):
    """
    Loads a Planetoid dataset (Cora/Citeseer/PubMed) and returns
    (dataset, data_obj) where data_obj is already on the CPU.
    """
    name = name.capitalize()
    transform = NormalizeFeatures()
    dataset = Planetoid(root=root, name=name, transform=transform)
    data = dataset[0]
    return dataset, data
