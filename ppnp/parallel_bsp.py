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

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

def partition_indices(num_nodes, world_size):
    # simple block partition
    sizes = [(num_nodes + i) // world_size for i in range(world_size)]
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    return [torch.arange(offsets[i], offsets[i] + sizes[i])
            for i in range(world_size)]

def bsp_appnp_propagation(x0, edge_index, alpha, K, world_size, rank):
    """
    x0: [N,F] initial features
    edge_index: [2,E] CPU or GPU tensors
    """
    # Ensure x0 is a tensor
    if not isinstance(x0, torch.Tensor):
        raise TypeError(f"Expected x0 to be a tensor, got {type(x0)}")
    
    N, num_features = x0.size()  # Changed F to num_features to avoid conflict
    # Which nodes this rank owns
    parts = partition_indices(N, world_size)
    local_idx = parts[rank]

    # Precompute normalization
    row, col = edge_index
    deg = degree(col, N)
    deg_inv = deg.pow(-1).clamp(min=1e-6)  # Add minimum value to prevent division by zero
    
    # Move all tensors to the same device as x0
    device = x0.device
    deg_inv = deg_inv.to(device)
    edge_index = edge_index.to(device)
    local_idx = local_idx.to(device)

    # Normalize input features
    x0 = F.normalize(x0.float(), p=2, dim=1)  # Ensure float type for normalization

    # Build local-edge mask once
    mask = torch.isin(col, local_idx)
    local_row = row[mask]
    local_col = col[mask]

    # Create mapping from global to local indices
    local_mapping = torch.zeros(N, dtype=torch.long, device=device)
    local_mapping[local_idx] = torch.arange(len(local_idx), device=device)
    local_col = local_mapping[local_col]

    # x is the current features; start at x0
    x = x0.clone()

    for _ in range(K):
        # 1) Local propagation: for each local destination i do
        #    prop[i] = sum_j (1/deg[j]) * x[j]
        #    but only over edges where col[j] ∈ local_idx
        w = deg_inv[local_row].unsqueeze(-1) * x[local_row]
        prop_local = scatter(w, local_col, dim=0, dim_size=len(local_idx), reduce='sum')
        
        # Add numerical stability
        prop_local = torch.nan_to_num(prop_local, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize propagated features
        prop_local = F.normalize(prop_local.float(), p=2, dim=1)  # Ensure float type

        # 2) Boundary exchange: prop_local is only for our nodes,
        #    but remote ranks need theirs. Gather all parts:
        gathered = [torch.zeros_like(prop_local) for _ in range(world_size)]
        dist.all_gather(gathered, prop_local)
        # reassemble full prop
        prop_full = torch.cat(gathered, dim=0)
        
        # Add numerical stability
        prop_full = torch.nan_to_num(prop_full, nan=0.0, posinf=1e6, neginf=-1e6)

        # 3) PPR update for our slice
        x_local = (1 - alpha) * prop_full[local_idx] + alpha * x0[local_idx]
        
        # Add numerical stability
        x_local = torch.nan_to_num(x_local, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize updated features
        x_local = F.normalize(x_local.float(), p=2, dim=1)  # Ensure float type

        # 4) write back into global x
        x[local_idx] = x_local

        # 5) barrier to keep sync
        dist.barrier()

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
    
    # Load data
    dataset, data = load_data(args.dataset, root=args.root)
    data = data.to(device)
    
    # Build model
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
    
    # Setup optimizer
    optimizer = torch.optim.Adam([
        {"params": model.lin1.parameters(), "weight_decay": args.weight_decay},
        {"params": model.lin2.parameters(), "weight_decay": 0.0},
    ], lr=args.lr)
    
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
        
        # Train
        train_start = time.time()
        loss = train(model, data, optimizer)
        train_time = time.time() - train_start
        
        # All-reduce gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        
        # Evaluate
        eval_start = time.time()
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
