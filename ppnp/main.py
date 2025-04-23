import argparse
import time
import csv
import torch
import os
import matplotlib.pyplot as plt

from data import load_data
from model import Net
from engine import train, evaluate
from engine import run

from profiling import profile_func
from metrics    import Benchmark

def main():
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
    args = p.parse_args()

    device = torch.device(args.device)
    dataset, data = load_data(args.dataset, root=args.root)
    data = data.to(device)

    # model + optimizer
    model = Net(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        K=args.K,
        alpha=args.alpha,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam([
        {"params": model.lin1.parameters(), "weight_decay": args.weight_decay},
        {"params": model.lin2.parameters(), "weight_decay": 0.0},
    ], lr=args.lr)

    # prepare CSV logging
    fields = ["epoch", "loss", "train_acc", "val_acc", "test_acc", "epoch_time", "gpu_mem"]
    with open(args.log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    history = {k: [] for k in fields}
    best_val = 0.0
    patience_ctr = 0

    # reset CUDA stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(model, data, optimizer)
        tr, val, te = evaluate(model, data)
        epoch_time = time.time() - start

        if device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
        else:
            gpu_mem = 0.0

        # record
        row = {
            "epoch": epoch, "loss": loss,
            "train_acc": tr, "val_acc": val, "test_acc": te,
            "epoch_time": epoch_time, "gpu_mem": gpu_mem
        }
        with open(args.log_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        for k, v in row.items():
            history[k].append(v)

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

    # final summary
    print("\n=== Final Benchmark Summary ===")
    print(f"Best Validation Accuracy: {best_val:.4f}")
    avg_time = sum(history["epoch_time"]) / len(history["epoch_time"])
    print(f"Avg Epoch Time: {avg_time:.4f} sec")
    print(f"Peak GPU Memory: {max(history['gpu_mem']):.2f} MB")

    # plot curves
    plt.figure()
    plt.plot(history["epoch"], history["loss"], label="Loss")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
