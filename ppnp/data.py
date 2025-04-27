import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

class NormalizeFeaturesL1:
    """Alternative L1‐normalization transform (not used by default)."""
    # NOT USED: This class is kept for reference but not used in the codebase
    def __call__(self, data):
        data.x = data.x / data.x.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return data

def load_data(name: str, root: str = "data/Planetoid"):
    """
    Loads a Planetoid dataset (Cora/Citeseer/PubMed) and returns
    (dataset, data_obj) where data_obj is already on the CPU.
    """
    name = name.capitalize()
    transform = NormalizeFeatures()  # same as in your notebook
    dataset = Planetoid(root=root, name=name, transform=transform)
    data = dataset[0]
    return dataset, data
