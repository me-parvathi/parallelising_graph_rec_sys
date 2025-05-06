import torch
import numpy as np
import pymetis
from typing import List, Tuple

def create_adjacency_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Convert edge_index to adjacency list format required by METIS."""
    # Ensure edge_index is on CPU
    edge_index = edge_index.cpu()
    row, col = edge_index.numpy()
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in zip(row, col):
        adj_list[i].append(j)
    return adj_list

def partition_graph(edge_index: torch.Tensor, num_nodes: int, num_parts: int) -> Tuple[List[int], List[int]]:
    """
    Partition the graph using METIS to minimize edge cuts.
    
    Args:
        edge_index: [2, E] tensor containing edge indices
        num_nodes: Number of nodes in the graph
        num_parts: Number of partitions to create
        
    Returns:
        Tuple of (partition assignments, edge cuts)
    """
    # Convert to adjacency list format
    adj_list = create_adjacency_list(edge_index, num_nodes)
    
    # Run METIS partitioning
    _, partition = pymetis.part_graph(num_parts, adjacency=adj_list)
    
    # Convert partition assignments to tensor on CPU
    partition_tensor = torch.tensor(partition, dtype=torch.long, device='cpu')
    
    # Calculate edge cuts (ensure edge_index is on CPU)
    edge_index = edge_index.cpu()
    row, col = edge_index
    edge_cuts = (partition_tensor[row] != partition_tensor[col]).sum().item()
    
    return partition, edge_cuts

def get_partition_indices(partition: List[int], num_parts: int, device: torch.device = None) -> List[torch.Tensor]:
    """
    Convert partition assignments to lists of node indices for each partition.
    
    Args:
        partition: List of partition assignments for each node
        num_parts: Number of partitions
        device: Device to place the tensors on (optional)
        
    Returns:
        List of tensors containing node indices for each partition
    """
    partition_tensor = torch.tensor(partition, dtype=torch.long, device='cpu')
    indices = [torch.where(partition_tensor == i)[0] for i in range(num_parts)]
    
    # Move indices to specified device if provided
    if device is not None:
        indices = [idx.to(device) for idx in indices]
    
    return indices

def calculate_partition_metrics(edge_index: torch.Tensor, partition: List[int], num_parts: int) -> dict:
    """
    Calculate various metrics about the partitioning quality.
    
    Args:
        edge_index: [2, E] tensor containing edge indices
        partition: List of partition assignments for each node
        num_parts: Number of partitions
        
    Returns:
        Dictionary containing partitioning metrics
    """
    # Ensure tensors are on CPU for calculations
    edge_index = edge_index.cpu()
    partition_tensor = torch.tensor(partition, dtype=torch.long, device='cpu')
    row, col = edge_index
    
    # Calculate edge cuts
    edge_cuts = (partition_tensor[row] != partition_tensor[col]).sum().item()
    
    # Calculate partition sizes
    partition_sizes = torch.bincount(partition_tensor, minlength=num_parts)
    
    # Calculate communication volume
    comm_volume = torch.zeros(num_parts, num_parts, dtype=torch.long, device='cpu')
    for i, j in zip(row, col):
        p1, p2 = partition_tensor[i], partition_tensor[j]
        if p1 != p2:
            comm_volume[p1, p2] += 1
            comm_volume[p2, p1] += 1
    
    return {
        'edge_cuts': edge_cuts,
        'partition_sizes': partition_sizes.tolist(),
        'communication_volume': comm_volume.tolist(),
        'max_partition_size': partition_sizes.max().item(),
        'min_partition_size': partition_sizes.min().item(),
        'partition_imbalance': (partition_sizes.max() - partition_sizes.min()).item()
    } 