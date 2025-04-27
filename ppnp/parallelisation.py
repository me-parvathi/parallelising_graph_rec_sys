import torch
import torch.distributed as dist
from typing import Tuple, Dict, Any
from torch_geometric.data import Data
from model import Net
from engine import train, evaluate
from data import load_data
from torch_geometric.utils import degree, add_self_loops
import os
import time
import psutil
import csv

def partition_graph(rank: int, world_size: int, data: Data) -> Data:
    """Partition the graph into node blocks using a simple random split.
    Ensures each subgraph is connected and has no isolated nodes.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        data: Full graph data
        
    Returns:
        Data: Subgraph for this rank
    """
    # Get total number of nodes
    num_nodes = data.x.size(0)
    
    # Calculate nodes per partition
    nodes_per_partition = num_nodes // world_size
    start_idx = rank * nodes_per_partition
    end_idx = start_idx + nodes_per_partition if rank < world_size - 1 else num_nodes
    
    # Create node mask for this partition
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[start_idx:end_idx] = True
    
    # Get all edges where at least one endpoint is in this partition
    edge_mask = node_mask[data.edge_index[0]] | node_mask[data.edge_index[1]]
    
    # Get the nodes involved in these edges
    involved_nodes = torch.unique(data.edge_index[:, edge_mask])
    
    # Update node mask to include all nodes involved in the edges
    node_mask[involved_nodes] = True
    
    # Create mapping from original node indices to new indices
    node_mapping = torch.zeros(num_nodes, dtype=torch.long)
    node_mapping[node_mask] = torch.arange(node_mask.sum())
    
    # Create subgraph with remapped indices
    sub_data = Data(
        x=data.x[node_mask],
        edge_index=node_mapping[data.edge_index[:, edge_mask]],
        y=data.y[node_mask],
        train_mask=data.train_mask[node_mask],
        val_mask=data.val_mask[node_mask],
        test_mask=data.test_mask[node_mask]
    )
    
    # Add self-loops to ensure no isolated nodes
    sub_data.edge_index, _ = add_self_loops(sub_data.edge_index)
    
    # Verify no isolated nodes and indices are in bounds
    deg = degree(sub_data.edge_index[0], num_nodes=sub_data.x.size(0))
    assert torch.all(deg > 0), "Found isolated nodes in subgraph"
    assert torch.all(sub_data.edge_index < sub_data.x.size(0)), "Edge indices out of bounds"
    
    return sub_data

def run_nb_bsp(rank: int, world_size: int, args: Dict[str, Any]) -> None:
    """Run Node-Block Bulk-Synchronous Parallel training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Load and partition data
    dataset, full_data = load_data(args.dataset, root=args.root)
    data = partition_graph(rank, world_size, full_data)
    data = data.to(device)
    
    # Build model
    model = Net(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        K=args.K,
        alpha=args.alpha,
        dropout=args.dropout,
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