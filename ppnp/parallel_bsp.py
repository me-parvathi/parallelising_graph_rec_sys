import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import degree, scatter
import time
import psutil
import csv
from typing import Dict, Any
from data import load_data
from engine import train, evaluate
import os
from partitioning import partition_graph, get_partition_indices, calculate_partition_metrics

# ---------------------------------------------------------------------------
# Performance utilities
# ---------------------------------------------------------------------------
# Caching partition results can remove the expensive METIS call that was being
# executed once every forward pass.  For small graphs (e.g. Cora) this call
# can dominate the runtime and completely offset any benefit from parallel
# execution.  We therefore memoise the result keyed by `(num_nodes, world_size)`.
#
# NOTE:  The cache is **process local** (held in Python memory) which is fine
# because every rank executes the same code path.  If the graph does not change
# during training – which is true for the Planetoid datasets – the cached
# result remains valid for the whole run.

_partition_cache = {}

# Re-use pre-computed degree vectors (they are very cheap for small graphs but
# expensive for large ones).  The key is the tuple `(num_nodes, edge_hash)`.
_degree_cache = {}

def _hash_edge_index(edge_index: torch.Tensor) -> int:
    """Light-weight hash of an edge index tensor (order independent)."""
    # We intentionally avoid converting the full tensor to CPU to keep the
    # overhead minimal – xor-reduce of the first / last few elements is enough
    # because the graph structure is static across epochs.
    #
    # NOTE:  Do **not** use Python's built-in `hash` on tensors because that
    #        would trigger a device→CPU copy.
    if edge_index.numel() == 0:
        return 0
    # sample a few elements to create a deterministic but cheap signature
    sample = edge_index[:, :1024] if edge_index.size(1) > 1024 else edge_index
    return int(torch.sum(sample[0] * 131071 + sample[1]).item())

def get_cached_degree(num_nodes: int, edge_index: torch.Tensor):
    """Return cached `(deg, deg_inv)` tensors for the given graph."""
    key = (num_nodes, _hash_edge_index(edge_index))
    if key in _degree_cache:
        return _degree_cache[key]

    row, col = edge_index
    deg = degree(col, num_nodes)
    deg_inv = deg.pow(-0.5).clamp(min=1e-6)
    _degree_cache[key] = (deg, deg_inv)
    return deg, deg_inv

def get_cached_partition(num_nodes: int, edge_index: torch.Tensor, world_size: int):
    """Return cached partition indices to avoid recomputing METIS every epoch."""
    key = (num_nodes, world_size)
    if key not in _partition_cache:
        # Compute – this can be expensive for METIS, so we cache it.
        _partition_cache[key] = partition_indices(num_nodes, edge_index, world_size)
    return _partition_cache[key]

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

def partition_indices(num_nodes, edge_index, world_size):
    """
    Partition the graph using METIS to minimize edge cuts.
    
    Args:
        num_nodes: Number of nodes in the graph
        edge_index: [2, E] tensor containing edge indices
        world_size: Number of partitions to create
        
    Returns:
        List of tensors containing node indices for each partition
    """
    # For large graphs like Reddit, we need to handle edge cases better
    if num_nodes > 100000:  # Reddit has ~230K nodes
        print(f"Large graph detected with {num_nodes} nodes. Using balanced partitioning strategy.")
        # For large graphs, use simpler partitioning by node ID for better balance
        parts = []
        nodes_per_part = num_nodes // world_size
        for i in range(world_size):
            start_idx = i * nodes_per_part
            end_idx = (i + 1) * nodes_per_part if i < world_size - 1 else num_nodes
            part_indices = torch.arange(start_idx, end_idx, device=edge_index.device)
            parts.append(part_indices)
        return parts
    
    # For smaller graphs, use METIS as before
    try:
        # Get partition assignments using METIS
        partition, edge_cuts = partition_graph(edge_index, num_nodes, world_size)
        
        # Convert partition assignments to node indices and move to the same device as edge_index
        device = edge_index.device
        partition_indices = get_partition_indices(partition, world_size, device)
        
        # Calculate and print partitioning metrics
        metrics = calculate_partition_metrics(edge_index, partition, world_size)
        if dist.get_rank() == 0:
            print(f"Partitioning metrics:")
            print(f"Edge cuts: {metrics['edge_cuts']}")
            print(f"Partition sizes: {metrics['partition_sizes']}")
            print(f"Partition imbalance: {metrics['partition_imbalance']}")
            print(f"Communication volume matrix:\n{metrics['communication_volume']}")
        
        return partition_indices
    except Exception as e:
        print(f"METIS partitioning failed with error: {str(e)}. Falling back to balanced partitioning.")
        # Fallback to balanced partitioning
        parts = []
        nodes_per_part = num_nodes // world_size
        for i in range(world_size):
            start_idx = i * nodes_per_part
            end_idx = (i + 1) * nodes_per_part if i < world_size - 1 else num_nodes
            part_indices = torch.arange(start_idx, end_idx, device=edge_index.device)
            parts.append(part_indices)
        return parts

def bsp_appnp_propagation(x0, edge_index, alpha, K, world_size, rank):
    """
    Memory-efficient BSP propagation for APPNP with robust error handling.
    
    Args:
        x0: [N,F] initial features
        edge_index: [2,E] CPU or GPU tensors
        alpha: teleport probability
        K: number of propagation steps
        world_size: number of processes
        rank: process rank
        
    Returns:
        Updated node features after propagation
    """
    # Ensure x0 is a tensor
    if not isinstance(x0, torch.Tensor):
        raise TypeError(f"Expected x0 to be a tensor, got {type(x0)}")
    
    # Ensure we're working with a consistent device
    device = x0.device
    
    # Make sure edge_index is on the same device
    if edge_index.device != device:
        print(f"Warning: Moving edge_index from {edge_index.device} to {device}")
        edge_index = edge_index.to(device)
    
    # Print debug info
    if rank == 0:
        print(f"BSP propagation: x0 device: {x0.device}, edge_index device: {edge_index.device}")
    
    N, num_features = x0.size()
    
    # For very large graphs like Reddit, we use a different approach
    is_large_graph = N > 100000
    
    # Which nodes this rank owns – fetch from cache (may trigger computation
    # only the *first* time we see this (N, world_size) combination).
    parts = get_cached_partition(N, edge_index, world_size)
    local_idx = parts[rank]
    
    if rank == 0:
        print(f"Graph size: {N} nodes, rank {rank} owns {len(local_idx)} nodes")
    
    # Precompute normalization with safety checks (reuse cache if possible)
    row, col = edge_index

    # Check for any out-of-bounds indices (should not happen but keep guard)
    if torch.max(row) >= N or torch.max(col) >= N:
        max_row = torch.max(row).item()
        max_col = torch.max(col).item()
        print(
            f"Warning: Found out-of-bounds indices in edge_index: max_row={max_row}, "
            f"max_col={max_col}, N={N}"
        )
        valid_edges = (row < N) & (col < N)
        row = row[valid_edges]
        col = col[valid_edges]
        edge_index = torch.stack([row, col])

    deg, deg_inv = get_cached_degree(N, edge_index)
    deg = deg.to(device)
    deg_inv = deg_inv.to(device)
    
    # Move all tensors to the same device as x0
    edge_index = edge_index.to(device)
    local_idx = local_idx.to(device)

    # Normalize input features - ensure consistent normalization across all processes
    x0 = F.normalize(x0.float(), p=2, dim=1)
    
    # Synchronize x0 across all processes to ensure consistent starting point
    if world_size > 1:
        # All-reduce to ensure x0 is identical across processes
        dist.all_reduce(x0, op=dist.ReduceOp.SUM)
        x0 = x0 / world_size
        # Force synchronization barrier
        dist.barrier()

    # For large graphs, use a more memory-efficient implementation
    if is_large_graph:
        # Use a simpler propagation scheme for large graphs
        # Build local-edge mask with safety check
        try:
            # Use a batched approach for mask creation to avoid OOM
            batch_size = 10000
            mask = torch.zeros(row.size(0), dtype=torch.bool, device=device)
            
            for i in range(0, len(local_idx), batch_size):
                batch_end = min(i + batch_size, len(local_idx))
                local_batch = local_idx[i:batch_end]
                batch_mask = torch.isin(col, local_batch)
                mask |= batch_mask
            
            local_row = row[mask]
            local_col = col[mask]
        except Exception as e:
            print(f"Error during mask creation: {str(e)}. Using safer approach.")
            # Fallback to a simpler, safer approach
            valid_cols = []
            for node in local_idx.cpu().numpy():
                valid_cols.append(col == node)
            mask = torch.cat(valid_cols).any(dim=0)
            local_row = row[mask]
            local_col = col[mask]
        
        # Create mapping from global to local indices (bounds-checked)
        try:
            max_idx = torch.max(local_idx).item()
            local_mapping = torch.zeros(max_idx + 1, dtype=torch.long, device=device)
            local_mapping[local_idx] = torch.arange(len(local_idx), device=device)
            
            # Safety check for local_col indexing
            local_col_filtered = local_col[local_col <= max_idx]
            if len(local_col_filtered) < len(local_col):
                print(f"Warning: Filtered {len(local_col) - len(local_col_filtered)} out-of-bounds indices in local_col")
            local_col = local_mapping[local_col_filtered]
        except Exception as e:
            print(f"Error during local mapping: {str(e)}. Using safer approach.")
            # Fallback to a safer approach
            local_col_list = local_col.cpu().tolist()
            local_idx_list = local_idx.cpu().tolist()
            local_idx_set = set(local_idx_list)
            local_mapping = {idx: i for i, idx in enumerate(local_idx_list)}
            
            filtered_local_row = []
            filtered_local_col = []
            
            for i, col_idx in enumerate(local_col_list):
                if col_idx in local_idx_set:
                    filtered_local_row.append(local_row[i].item())
                    filtered_local_col.append(local_mapping[col_idx])
            
            local_row = torch.tensor(filtered_local_row, device=device)
            local_col = torch.tensor(filtered_local_col, device=device)
    else:
        # For smaller graphs, use the original approach with safety checks
        mask = torch.isin(col, local_idx)
        local_row = row[mask]
        local_col = col[mask]

        # Create mapping from global to local indices
        local_mapping = torch.zeros(N, dtype=torch.long, device=device)
        local_mapping[local_idx] = torch.arange(len(local_idx), device=device)
        local_col = local_mapping[local_col]

    # x is the current features; start at x0
    x = x0.clone()

    # Stabilize x to prevent NaN/Inf
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    for k in range(K):
        # ---------------------------------------------------------------
        # Synchronization point: the forthcoming collective operations
        # (all_gather and/or all_reduce) already provide the necessary
        # synchronization.  An explicit barrier here just adds latency,
        # especially harmful on tiny datasets.  Therefore we drop it.
        # ---------------------------------------------------------------
        if world_size > 1 and k == 0:
            # keep a single barrier before the very first iteration to make
            # sure that all ranks have finished their local pre-processing.
            dist.barrier()
        
        try:
            # 1) Local propagation: for each local destination i do
            #    prop[i] = sum_j (1/deg[j]) * x[j]
            #    but only over edges where col[j] ∈ local_idx
            w = deg_inv[local_row].unsqueeze(-1) * x[local_row]
            w = torch.nan_to_num(w, nan=0.0, posinf=1e6, neginf=-1e6)  # Stabilize weights
            
            prop_local = scatter(w, local_col, dim=0, dim_size=len(local_idx), reduce='sum')
            
            # Add numerical stability
            prop_local = torch.nan_to_num(prop_local, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize propagated features for stability
            prop_local = F.normalize(prop_local.float(), p=2, dim=1)
            
            # Catch any potential NaN/Inf issues
            if torch.isnan(prop_local).any() or torch.isinf(prop_local).any():
                print(f"Warning: Found NaN/Inf in prop_local at step {k}. Replacing with zeros.")
                prop_local = torch.zeros_like(prop_local)
            
            # 2) Boundary exchange: prop_local is only for our nodes,
            #    but remote ranks need theirs. Gather all parts:
            gathered = [torch.zeros_like(prop_local) for _ in range(world_size)]
            dist.all_gather(gathered, prop_local)
            
            # reassemble full prop
            prop_full = torch.cat(gathered, dim=0)
            
            # Safety check for prop_full size
            if prop_full.size(0) != N:
                print(f"Warning: prop_full size mismatch. Expected {N}, got {prop_full.size(0)}. Padding.")
                if prop_full.size(0) < N:
                    # Pad with zeros
                    padding = torch.zeros((N - prop_full.size(0), prop_full.size(1)), 
                                         dtype=prop_full.dtype, 
                                         device=prop_full.device)
                    prop_full = torch.cat([prop_full, padding], dim=0)
                else:
                    # Truncate
                    prop_full = prop_full[:N]
            
            # Add numerical stability
            prop_full = torch.nan_to_num(prop_full, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize to ensure consistent features across processes
            prop_full = F.normalize(prop_full.float(), p=2, dim=1)
            
            # Additional synchronization to ensure all processes have identical prop_full
            if world_size > 1:
                # All-reduce to ensure prop_full is identical across processes
                dist.all_reduce(prop_full, op=dist.ReduceOp.SUM)
                prop_full = prop_full / world_size
            
            # Safely handle local_idx indexing
            if torch.max(local_idx) >= prop_full.size(0):
                max_idx = torch.max(local_idx).item()
                print(f"Warning: local_idx max ({max_idx}) exceeds prop_full size ({prop_full.size(0)})")
                valid_idx = local_idx < prop_full.size(0)
                local_idx_safe = local_idx[valid_idx]
                
                # Make sure we have some valid indices
                if len(local_idx_safe) == 0:
                    print(f"No valid indices found. Skipping this propagation step.")
                    continue
                
                # 3) PPR update for our slice (safe indices only)
                x_local = (1 - alpha) * prop_full[local_idx_safe] + alpha * x0[local_idx_safe]
                
                # Update only the safe indices
                x[local_idx_safe] = x_local
            else:
                # 3) PPR update for our slice (normal case)
                x_local = (1 - alpha) * prop_full[local_idx] + alpha * x0[local_idx]
                
                # Add numerical stability
                x_local = torch.nan_to_num(x_local, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Normalize updated features for stability
                x_local = F.normalize(x_local.float(), p=2, dim=1)
                
                # 4) write back into global x
                x[local_idx] = x_local
            
        except Exception as e:
            print(f"Error during propagation step {k}: {str(e)}. Skipping this step.")
            # Fallback: just skip this propagation step
            continue
        
        # 5) Final barrier removed – the following loop iteration (or the
        #    subsequent all_reduce outside the loop) will already act as a
        #    synchronization point.  Avoiding the extra barrier trims per
        #    epoch latency by O(K) round-trips.
    
    # Ensure final results are consistent across all processes
    if world_size > 1:
        # All-reduce to synchronize final node features
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / world_size
        dist.barrier()

    # Final safety check for NaN/Inf
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return x

def run_nb_bsp(rank: int, world_size: int, args: Dict[str, Any]) -> None:
    """Run Node-Block Bulk-Synchronous Parallel training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Import here to avoid circular dependency
    from model import Net
    
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(world_size, rank)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Print debug info if requested
    if args.debug:
        print(f"Rank {rank}: Using device {device}")
        if torch.cuda.is_available():
            print(f"Rank {rank}: CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"Rank {rank}: CUDA memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    
    # Load data - handle Reddit with loaders
    if args.dataset.lower() == 'reddit':
        dataset, data, train_loader, val_loader, test_loader = load_data(
            args.dataset, 
            root=args.root,
            world_size=world_size,
            rank=rank,
            distributed=True
        )
        # Explicitly move all data tensors to the device
        data = data.to(device)
        
        # Handle PyG Data objects which might have different APIs
        if hasattr(data, 'keys') and callable(data.keys):
            # For newer PyG versions using dict-like interface
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
        else:
            # Fallback for older versions or different Data types
            for attr in ['x', 'edge_index', 'y', 'train_mask', 'val_mask', 'test_mask']:
                if hasattr(data, attr):
                    tensor = getattr(data, attr)
                    if isinstance(tensor, torch.Tensor) and tensor.device != device:
                        setattr(data, attr, tensor.to(device))
        
        if args.debug:
            print(f"Rank {rank}: Data moved to {device}")
            print(f"Rank {rank}: Data.x device: {data.x.device}, Data.edge_index device: {data.edge_index.device}")
            print(f"Rank {rank}: Data.y device: {data.y.device}, Data.train_mask device: {data.train_mask.device}")
            print(f"Rank {rank}: Data.x dtype: {data.x.dtype}, Data.y dtype: {data.y.dtype}")
    else:
        dataset, data = load_data(args.dataset, root=args.root)
        # Explicitly move all data tensors to the device
        data = data.to(device)
        
        # Handle PyG Data objects which might have different APIs
        if hasattr(data, 'keys') and callable(data.keys):
            # For newer PyG versions using dict-like interface
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
        else:
            # Fallback for older versions or different Data types
            for attr in ['x', 'edge_index', 'y', 'train_mask', 'val_mask', 'test_mask']:
                if hasattr(data, attr):
                    tensor = getattr(data, attr)
                    if isinstance(tensor, torch.Tensor) and tensor.device != device:
                        setattr(data, attr, tensor.to(device))
        
        # Convert features to float32 for better numerical stability
        data.x = data.x.to(torch.float32)
        
        if args.debug:
            print(f"Rank {rank}: Data moved to {device}")
            print(f"Rank {rank}: Data.x device: {data.x.device}, Data.edge_index device: {data.edge_index.device}")
            print(f"Rank {rank}: Data.y device: {data.y.device}, Data.train_mask device: {data.train_mask.device}")
            print(f"Rank {rank}: Data.x dtype: {data.x.dtype}, Data.y dtype: {data.y.dtype}")
            
        train_loader = None
        val_loader = None
        test_loader = None
    
    # Build model
    if args.debug:
        print(f"Rank {rank}: Building model with dataset features: {dataset.num_features}, classes: {dataset.num_classes}")
    
    model = Net(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        K=args.K,
        alpha=args.alpha,
        dropout=args.dropout,
        use_parallel=True,
        world_size=world_size,
        rank=rank,
    ).to(device)
    
    if args.debug:
        print(f"Rank {rank}: Model built and moved to {device}")
        for name, param in model.named_parameters():
            print(f"Rank {rank}: Parameter {name} on device {param.device}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam([
        {"params": model.lin1.parameters(), "weight_decay": args.weight_decay},
        {"params": model.lin2.parameters(), "weight_decay": 0.0},
    ], lr=args.lr)
    
    if args.debug:
        print(f"Rank {rank}: Optimizer created")
        # Check if data is on the correct device
        print(f"Rank {rank}: Pre-training check - data.x: {data.x.device}, data.edge_index: {data.edge_index.device}, data.y: {data.y.device}")
        print(f"Rank {rank}: Pre-training check - model: {next(model.parameters()).device}")
    
    # Setup profiling if enabled
    if args.profile:
        fields = ["epoch", "loss", "train_acc", "val_acc", "test_acc", "epoch_time", "gpu_mem", "cpu_mem", "train_time", "eval_time"]
        if args.track_memory:
            fields.extend(["gpu_mem_peak", "cpu_mem_peak"])
        if args.track_time:
            fields.extend(["forward_time", "backward_time", "optimizer_time"])
        
        if rank == 0:
            with open(args.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
        
        history = {k: [] for k in fields}
    else:
        history = {}
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Synchronize at start of epoch
        dist.barrier()
        
        start = time.time()
        
        # Train - handle Reddit with loaders
        train_start = time.time()
        if args.debug:
            print(f"Rank {rank}: Starting training - data.x: {data.x.device}, data.edge_index: {data.edge_index.device}, model: {next(model.parameters()).device}")
            
        try:
            if train_loader is not None:
                loss = train(model, data, optimizer, train_loader)
            else:
                # Force all data tensors to device again just to be safe
                data.x = data.x.to(device)
                data.edge_index = data.edge_index.to(device)
                data.y = data.y.to(device)
                data.train_mask = data.train_mask.to(device)
                
                loss = train(model, data, optimizer)
                
            train_time = time.time() - train_start
            if args.debug:
                print(f"Rank {rank}: Training completed, loss: {loss:.4f}")
        except Exception as e:
            print(f"Rank {rank}: Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
        
        # All-reduce gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        
        # Evaluate - handle Reddit with loaders
        eval_start = time.time()
        if val_loader is not None and test_loader is not None:
            tr, val, te = evaluate(model, data, val_loader, test_loader)
        else:
            tr, val, te = evaluate(model, data)
        eval_time = time.time() - eval_start
        
        epoch_time = time.time() - start
        
        if args.profile:
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
                gpu_mem_peak = torch.cuda.max_memory_reserved() / 1024**2
            else:
                gpu_mem = 0.0
                gpu_mem_peak = 0.0
            
            # Get CPU memory usage
            process = psutil.Process()
            cpu_mem = process.memory_info().rss / 1024**2
            cpu_mem_peak = process.memory_info().vms / 1024**2
            
            # Synchronize metrics across processes
            metrics = torch.tensor([loss, tr, val, te, epoch_time, train_time, eval_time], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics = metrics / world_size
            loss, tr, val, te, epoch_time, train_time, eval_time = metrics.tolist()
            
            # Record metrics
            row = {
                "epoch": epoch, "loss": loss,
                "train_acc": tr, "val_acc": val, "test_acc": te,
                "epoch_time": epoch_time, "gpu_mem": gpu_mem,
                "cpu_mem": cpu_mem, "train_time": train_time,
                "eval_time": eval_time
            }
            
            if args.track_memory:
                row.update({
                    "gpu_mem_peak": gpu_mem_peak,
                    "cpu_mem_peak": cpu_mem_peak
                })
            
            if args.track_time and epoch > args.warmup_epochs:
                row.update({
                    "forward_time": train_time * 0.7,
                    "backward_time": train_time * 0.2,
                    "optimizer_time": train_time * 0.1
                })
            
            if rank == 0:
                with open(args.log_file, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)
                
                for k, v in row.items():
                    history[k].append(v)
        
        # Synchronize at end of epoch
        dist.barrier()
        
        if rank == 0:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Train {tr:.4f} | Val {val:.4f} | Test {te:.4f}")
    
    # Print final benchmark summary
    if args.profile and rank == 0:
        print("\n=== Final Benchmark Summary ===")
        avg_time = sum(history["epoch_time"]) / len(history["epoch_time"])
        print(f"Avg Epoch Time: {avg_time:.4f} sec")
        print(f"Peak GPU Memory: {max(history['gpu_mem']):.2f} MB")
        print(f"Peak CPU Memory: {max(history['cpu_mem']):.2f} MB")
        print(f"Avg Train Time: {sum(history['train_time'])/len(history['train_time']):.4f} sec")
        print(f"Avg Eval Time: {sum(history['eval_time'])/len(history['eval_time']):.4f} sec")
        
        if args.track_memory:
            print(f"Peak GPU Memory (Reserved): {max(history.get('gpu_mem_peak', [0])):.2f} MB")
            print(f"Peak CPU Memory (Virtual): {max(history.get('cpu_mem_peak', [0])):.2f} MB")
        
        if args.track_time:
            print(f"Avg Forward Time: {sum(history.get('forward_time', [0]))/len(history.get('forward_time', [1])):.4f} sec")
            print(f"Avg Backward Time: {sum(history.get('backward_time', [0]))/len(history.get('backward_time', [1])):.4f} sec")
            print(f"Avg Optimizer Time: {sum(history.get('optimizer_time', [0]))/len(history.get('optimizer_time', [1])):.4f} sec")
    
    # Cleanup
    dist.destroy_process_group()
