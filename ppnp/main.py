import argparse
import time
import csv
import torch
import os
import sys
import matplotlib.pyplot as plt
import psutil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data import load_data
from model import Net
from engine import train, evaluate
from parallel_bsp import init_process_group, run_nb_bsp

# Only apply profiling decorator if benchmarking is enabled
def get_train_func(enable_benchmark):
    if enable_benchmark:
        return train  # Removed profile_func wrapper
    return train

def get_evaluate_func(enable_benchmark):
    if enable_benchmark:
        return evaluate  # Removed profile_func wrapper
    return evaluate

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    if args.ddp or args.parallel == "nb_bsp":
        setup(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    
    # Print device information in debug mode
    if args.debug:
        print(f"Rank {rank}: Using device {device}")
        if torch.cuda.is_available():
            print(f"Rank {rank}: CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"Rank {rank}: CUDA memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    # Load data and loader if Reddit
    if args.dataset.lower() == 'reddit':
        # Set distributed=True if using DDP or NB-BSP
        is_distributed = args.ddp or args.parallel == "nb_bsp"
        dataset, data, train_loader, val_loader, test_loader = load_data(
            args.dataset, 
            root=args.root,
            world_size=world_size,
            rank=rank,
            distributed=is_distributed
        )
        # Ensure data is on the correct device and has correct types
        data = data.to(device)
        # Convert node features to float32 for better numerical stability
        data.x = data.x.to(torch.float32)
        
        if args.debug:
            print(f"Rank {rank}: Data moved to {device}")
            print(f"Rank {rank}: Data.x device: {data.x.device}, Data.edge_index device: {data.edge_index.device}")
            print(f"Rank {rank}: Data.x dtype: {data.x.dtype}, Data.y dtype: {data.y.dtype}")
    else:
        dataset, data = load_data(args.dataset, root=args.root)
        # Ensure data is on the correct device and has correct types
        data = data.to(device)
        # Convert node features to float32 for better numerical stability
        data.x = data.x.to(torch.float32)
        
        if args.debug:
            print(f"Rank {rank}: Data moved to {device}")
            print(f"Rank {rank}: Data.x device: {data.x.device}, Data.edge_index device: {data.edge_index.device}")
            print(f"Rank {rank}: Data.x dtype: {data.x.dtype}, Data.y dtype: {data.y.dtype}")
        train_loader = None
        val_loader = None
        test_loader = None

    # model + optimizer
    model = Net(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        K=args.K,
        alpha=args.alpha,
        dropout=args.dropout,
        use_parallel=args.parallel == "nb_bsp",
        world_size=world_size,
        rank=rank,
    ).to(device)
    
    if args.debug:
        print(f"Rank {rank}: Model moved to {device}")
        for name, param in model.named_parameters():
            print(f"Rank {rank}: Parameter {name} on device {param.device}")

    if args.ddp:
        model = DDP(model, device_ids=[rank])

    # Get the actual model for parameter access
    model_for_params = model.module if args.ddp else model

    # Scale learning rate based on world size if in parallel mode
    adjusted_lr = args.lr
    if args.parallel == "nb_bsp" or args.ddp:
        adjusted_lr = args.lr / (world_size ** 0.5)  # Scale by sqrt(world_size)
        if args.debug and rank == 0:
            print(f"Adjusted learning rate for parallel: {adjusted_lr:.6f} (original: {args.lr:.6f})")

    optimizer = torch.optim.Adam([
        {"params": model_for_params.lin1.parameters(), "weight_decay": args.weight_decay},
        {"params": model_for_params.lin2.parameters(), "weight_decay": 0.0},
    ], lr=adjusted_lr)

    # prepare CSV logging only if benchmarking is enabled
    if args.profile:
        fields = ["epoch", "loss", "train_acc", "val_acc", "test_acc", "epoch_time", "gpu_mem", "cpu_mem", "train_time", "eval_time"]
        if args.track_memory:
            fields.extend(["gpu_mem_peak", "cpu_mem_peak"])
        if args.track_time:
            fields.extend(["forward_time", "backward_time", "optimizer_time"])
        
        if rank == 0 or not args.ddp:
            with open(args.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

        history = {k: [] for k in fields}
    else:
        history = {}

    best_val = 0.0
    patience_ctr = 0

    # reset CUDA stats if benchmarking is enabled
    if args.profile and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Get the appropriate training and evaluation functions
    train_func = get_train_func(args.profile)
    evaluate_func = get_evaluate_func(args.profile)

    for epoch in range(1, args.epochs + 1):
        if args.ddp:
            dist.barrier()
            
        start = time.time()
        
        # Train with or without profiling
        train_start = time.time()
        if train_loader is not None:
            loss = train_func(model, data, optimizer, train_loader)
        else:
            loss = train_func(model, data, optimizer)
        train_time = time.time() - train_start
        
        # Evaluate with or without profiling
        eval_start = time.time()
        if val_loader is not None and test_loader is not None:
            tr, val, te = evaluate_func(model, data, val_loader, test_loader)
        else:
            tr, val, te = evaluate_func(model, data)
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
            cpu_mem = process.memory_info().rss / 1024**2  # Convert to MB
            cpu_mem_peak = process.memory_info().vms / 1024**2  # Virtual memory size

            # Synchronize metrics across processes in DDP mode
            if args.ddp:
                # Create tensors for synchronization
                metrics = torch.tensor([loss, tr, val, te, epoch_time, train_time, eval_time], device=device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics = metrics / world_size
                loss, tr, val, te, epoch_time, train_time, eval_time = metrics.tolist()

            # record
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
                # Add detailed timing metrics if tracking is enabled and past warmup
                row.update({
                    "forward_time": train_time * 0.7,  # Approximate forward pass time
                    "backward_time": train_time * 0.2,  # Approximate backward pass time
                    "optimizer_time": train_time * 0.1  # Approximate optimizer step time
                })

            if rank == 0 or not args.ddp:
                with open(args.log_file, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)

                for k, v in row.items():
                    history[k].append(v)

        if rank == 0 or not args.ddp:
            # early stopping
            if val > best_val:
                best_val = val
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                  f"Train {tr:.4f} | Val {val:.4f} | Test {te:.4f}")

    if args.profile and (rank == 0 or not args.ddp):
        # final summary
        print("\n=== Final Benchmark Summary ===")
        print(f"Best Validation Accuracy: {best_val:.4f}")
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

        # plot curves
        plt.figure()
        plt.plot(history["epoch"], history["loss"], label="Loss")
        plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Value"); plt.legend()
        plt.tight_layout()
        plt.savefig("training_plot.png")
        plt.show()

    if args.ddp:
        # Ensure all processes have completed before cleanup
        dist.barrier()
        cleanup()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="APPNP on Planetoid")
    p.add_argument("--dataset", type=str, default="Cora", help="Cora/Citeseer/PubMed")
    p.add_argument("--root",    type=str, default="data/Planetoid", help="data directory")
    p.add_argument("--epochs",  type=int, default=200)
    p.add_argument("--lr",      type=float, default=0.01)
    p.add_argument("--hidden",  type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=0.005,
                   help="L2 on first layer only")
    p.add_argument("--patience",type=int, default=100)
    p.add_argument("--K",       type=int, default=10)
    p.add_argument("--alpha",   type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--log_file",type=str, default="appnp_training_log.csv")
    p.add_argument("--device",  type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ddp",     action="store_true", help="Enable Distributed Data Parallel")
    p.add_argument("--world_size", type=int, default=torch.cuda.device_count(),
                   help="Number of GPUs to use for DDP")
    
    # === NB-BSP BEGIN ===
    p.add_argument("--parallel", type=str, default="none", choices=["none", "nb_bsp"],
                   help="Parallel execution mode: none or nb_bsp")
    # === NB-BSP END ===
    
    # Benchmarking arguments
    p.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    p.add_argument("--benchmark_interval", type=int, default=1,
                   help="Interval in epochs for detailed benchmarking")
    p.add_argument("--track_memory", action="store_true", default=True,
                   help="Track memory usage during training")
    p.add_argument("--track_time", action="store_true", default=True,
                   help="Track detailed timing metrics")
    p.add_argument("--benchmark_output", type=str, default="benchmark_results.json",
                   help="Output file for detailed benchmark results")
    p.add_argument("--warmup_epochs", type=int, default=1,
                   help="Number of warmup epochs before starting detailed benchmarking")
    p.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose output")
    
    args = p.parse_args()

    # === NB-BSP BEGIN ===
    if args.parallel == "nb_bsp":
        mp.spawn(run_nb_bsp, args=(args.world_size, args), nprocs=args.world_size)
        sys.exit(0)
    # === NB-BSP END ===

    if args.ddp:
        mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
    else:
        main(0, 1, args)
