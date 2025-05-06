import torch
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import NeighborLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import numpy as np
import os
import gc

class NormalizeFeaturesL1:
    """Alternative L1‐normalization transform (not used by default)."""
    # NOT USED: This class is kept for reference but not used in the codebase
    def __call__(self, data):
        data.x = data.x / data.x.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return data

class RedditPreProcess:
    """Transform for pre-processing Reddit features more effectively."""
    def __call__(self, data):
        # Normalize features with L2 normalization (often better for Reddit)
        data.x = F.normalize(data.x, p=2, dim=1)
        
        # Store original feature size for reference
        data.original_x_size = data.x.size()
        
        # Memory-efficient PCA dimensionality reduction (helps with Reddit's 602 features)
        if data.x.size(1) > 200:  # If features are high-dimensional
            print("Applying dimensionality reduction to Reddit features...")
            # Use batched SVD for better memory efficiency
            batch_size = min(10000, data.x.size(0))
            num_batches = (data.x.size(0) + batch_size - 1) // batch_size
            
            # Calculate covariance matrix in chunks
            cov = torch.zeros((data.x.size(1), data.x.size(1)), dtype=torch.float32)
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, data.x.size(0))
                batch = data.x[start_idx:end_idx]
                # Update covariance matrix with this batch contribution
                cov += torch.mm(batch.t(), batch)
                del batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            # Get eigenvalues and eigenvectors
            print("Computing SVD for feature reduction...")
            e, v = torch.linalg.eigh(cov)
            # Sort eigenvalues in descending order
            e, indices = torch.sort(e, descending=True)
            v = v[:, indices]
            
            # Keep top 128 components (reduced from 200 for memory efficiency)
            reduced_dim = min(128, data.x.size(1))
            projection = v[:, :reduced_dim]
            
            # Project features in batches to reduce memory usage
            x_reduced = torch.zeros((data.x.size(0), reduced_dim), dtype=torch.float32)
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, data.x.size(0))
                x_reduced[start_idx:end_idx] = torch.mm(data.x[start_idx:end_idx], projection)
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            # Replace original features with reduced ones
            del data.x
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            data.x = x_reduced
            
            # Renormalize after projection
            data.x = F.normalize(data.x, p=2, dim=1)
            print(f"Reduced feature dimension from {data.original_x_size[1]} to {data.x.size(1)}")
        
        return data

def load_data(name: str, root: str = "data", world_size=1, rank=0, distributed=False):
    """
    Loads a dataset and returns (dataset, data_obj) where data_obj is already on the CPU.
    Supports Planetoid datasets (Cora/Citeseer/PubMed) and Reddit.
    
    Args:
        name (str): Dataset name ('Cora', 'Citeseer', 'PubMed', or 'Reddit')
        root (str): Root directory where datasets should be saved
        world_size (int, optional): Number of processes for distributed training
        rank (int, optional): Process rank for distributed training
        distributed (bool, optional): Whether to use distributed samplers
    """
    # name = name.capitalize() # Removed this line
    
    if name in ['Cora', 'Citeseer', 'PubMed']:
        transform = NormalizeFeatures()
        dataset = Planetoid(root=f"{root}/Planetoid", name=name, transform=transform)
    elif name == 'Reddit':
        # Use both normalization and Reddit-specific pre-processing
        transform = RedditPreProcess()
        dataset = Reddit(root=f"{root}/Reddit", transform=transform)
        data = dataset[0]
        train_loader = get_reddit_loader(data, distributed=distributed, rank=rank, world_size=world_size)
        val_loader = get_reddit_val_loader(data, distributed=distributed, rank=rank, world_size=world_size)
        test_loader = get_reddit_test_loader(data, distributed=distributed, rank=rank, world_size=world_size)
        return dataset, data, train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Dataset {name} not supported. Choose from: Cora, Citeseer, PubMed, Reddit")
    
    data = dataset[0]
    return dataset, data

def get_reddit_config():
    """Get Reddit-specific configuration from environment variables."""
    batch_size = int(os.environ.get('REDDIT_BATCH_SIZE', '512'))
    num_neighbors_str = os.environ.get('REDDIT_NUM_NEIGHBORS', '10,10')
    num_neighbors = [int(x) for x in num_neighbors_str.split(',')]
    return batch_size, num_neighbors

def get_reddit_loader(data, batch_size=None, num_neighbors=None, num_workers=0, distributed=False, rank=0, world_size=1):
    """Get a loader for the training set."""
    # Use environment variables if available
    if batch_size is None or num_neighbors is None:
        env_batch_size, env_num_neighbors = get_reddit_config()
        batch_size = batch_size or env_batch_size
        num_neighbors = num_neighbors or env_num_neighbors
    
    print(f"Creating Reddit train loader (batch_size={batch_size}, num_neighbors={num_neighbors})")
    
    if distributed:
        # For distributed training, split the training data
        # We'll use a DistributedSampler over the training mask nodes
        input_nodes = data.train_mask.nonzero().squeeze()
        sampler = DistributedSampler(
            input_nodes, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.train_mask,
            shuffle=False,  # Sampler handles shuffling
            num_workers=num_workers,
            sampler=sampler,
            persistent_workers=num_workers > 0,
            drop_last=False
        )
    else:
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False
        )

def get_reddit_val_loader(data, batch_size=None, num_neighbors=None, num_workers=0, distributed=False, rank=0, world_size=1):
    """Get a loader for validation set evaluation."""
    # Use environment variables if available
    if batch_size is None or num_neighbors is None:
        env_batch_size, env_num_neighbors = get_reddit_config()
        batch_size = batch_size or env_batch_size * 2  # Double batch size for validation
        num_neighbors = num_neighbors or env_num_neighbors
    
    print(f"Creating Reddit validation loader (batch_size={batch_size}, num_neighbors={num_neighbors})")
    
    if distributed:
        # For distributed evaluation, split the validation data
        input_nodes = data.val_mask.nonzero().squeeze()
        sampler = DistributedSampler(
            input_nodes, 
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.val_mask,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            persistent_workers=num_workers > 0,
            drop_last=False
        )
    else:
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.val_mask,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False
        )

def get_reddit_test_loader(data, batch_size=None, num_neighbors=None, num_workers=0, distributed=False, rank=0, world_size=1):
    """Get a loader for test set evaluation."""
    # Use environment variables if available
    if batch_size is None or num_neighbors is None:
        env_batch_size, env_num_neighbors = get_reddit_config()
        batch_size = batch_size or env_batch_size * 2  # Double batch size for testing
        num_neighbors = num_neighbors or env_num_neighbors
    
    print(f"Creating Reddit test loader (batch_size={batch_size}, num_neighbors={num_neighbors})")
    
    if distributed:
        # For distributed evaluation, split the test data
        input_nodes = data.test_mask.nonzero().squeeze()
        sampler = DistributedSampler(
            input_nodes, 
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.test_mask,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            persistent_workers=num_workers > 0,
            drop_last=False
        )
    else:
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.test_mask,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False
        )
