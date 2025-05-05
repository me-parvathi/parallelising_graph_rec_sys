import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.utils import dropout_adj
from torch.utils.checkpoint import checkpoint, _StopRecomputationError
import gc

class Net(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        use_parallel: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = dropout
        self.use_parallel = use_parallel
        self.world_size = world_size
        self.rank = rank
        self.K = K
        self.alpha = alpha
        
        # For large datasets like Reddit, use memory-efficient settings
        self.is_large_dataset = in_channels > 100 or out_channels > 10
        self.use_checkpointing = self.is_large_dataset
        
        # Special cases for very large datasets like Reddit
        self.is_reddit = in_channels > 500
        
        # Mix-precision settings
        self.use_amp = True
        
        # Memory optimization for large datasets
        if self.is_large_dataset and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Glorot / Xavier initialization
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)

    def _forward_mlp(self, x):
        # Ensure x is on the same device as the model parameters
        device = self.lin1.weight.device
        x = x.to(device)
        
        # Apply dropout with additional safety for numerical stability
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # First linear layer with activation
        x = F.relu(self.lin1(x))
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Apply dropout again
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Second linear layer
        x = self.lin2(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return x

    def _forward_prop(self, x, edge_index):
        """Propagation step with safety checks for large graphs."""
        # Use sparse operations for propagation
        if self.use_parallel:
            from parallel_bsp import bsp_appnp_propagation
            # For Reddit dataset, reduce K if needed
            k_steps = min(5, self.K) if self.is_reddit else self.K
            return bsp_appnp_propagation(x, edge_index, self.alpha, k_steps, self.world_size, self.rank)
        else:
            # For Reddit dataset, use memory-efficient settings
            if self.is_reddit:
                # If it's a very large graph like Reddit, add safety measures
                try:
                    # Keep normalization and safety in APPNP for Reddit
                    x = F.normalize(x, p=2, dim=1)
                    result = self.prop(x, edge_index)
                    return torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
                except Exception as e:
                    # Check if the exception is the one used by checkpointing
                    if isinstance(e, _StopRecomputationError):
                        raise # Re-raise if it is, so checkpointing handles it
                    # Handle other exceptions as before
                    print(f"Error in APPNP propagation: {repr(e)}. Using safer fallback.")
                    # Fallback to a simpler propagation for very large graphs
                    return x  # In case of error, return input features
            else:
                # Normal behavior for smaller datasets
                return self.prop(x, edge_index)

    def forward(self, x, edge_index):
        """Forward pass with memory optimization for large graphs."""
        # Ensure inputs are on the same device as the model parameters
        device = self.lin1.weight.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        
        # Make sure x requires gradient
        if self.training and not x.requires_grad:
            x.requires_grad_(True)
        
        # Memory efficient checkpointing
        if self.training and self.use_checkpointing:
            # For Reddit, add custom memory cleanup between operations
            if self.is_reddit and torch.cuda.is_available():
                # More aggressive memory management for Reddit
                torch.cuda.empty_cache()
            
            x = checkpoint(self._forward_mlp, x, use_reentrant=False)
        else:
            x = self._forward_mlp(x)

        if self.training:
            try:
                # Add safety handling for edge dropout
                if edge_index.shape[1] > 1000000:  # For very large edge indexes
                    # Random uniform sampling for large graphs instead of dropout_adj
                    edge_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.dropout
                    edge_index_sampled = edge_index[:, edge_mask]
                    if edge_index_sampled.shape[1] == 0:  # Safety check
                        # If all edges were dropped (unlikely), keep at least 10%
                        keep_edges = max(int(0.1 * edge_index.shape[1]), 1)
                        indices = torch.randperm(edge_index.shape[1], device=edge_index.device)[:keep_edges]
                        edge_index_sampled = edge_index[:, indices]
                    edge_index = edge_index_sampled
                else:
                    # Use standard dropout_adj for smaller graphs
                    edge_index, _ = dropout_adj(edge_index, p=self.dropout)
            except Exception as e:
                print(f"Warning: Error in edge_index processing: {e}")
                # If dropout fails, use original edge_index
        
        # Second checkpoint for propagation step
        if self.training and self.is_large_dataset and not self.use_parallel:
            out = checkpoint(self._forward_prop, x, edge_index, use_reentrant=False)
        else:
            out = self._forward_prop(x, edge_index)
            
        # Final safety check
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # For large datasets, free memory explicitly
        if self.is_large_dataset and torch.cuda.is_available():
            del x
            torch.cuda.empty_cache()
            
        return out
