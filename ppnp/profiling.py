# profiling.py

import cProfile
import pstats
import io
from memory_profiler import memory_usage
from functools import wraps
import time
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import re

# NOT USED: This functionality is now covered by the main profiling system in main.py
"""
def profile_func(func):
    '''
    Decorator to:
      - run cProfile and print top 10 cumulative calls
      - measure simple CPU memory delta via memory_profiler
      - measure wall-clock time
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        # start profiling
        pr = cProfile.Profile()
        pr.enable()

        # capture starting memory (list of floats, MiB)
        mem_start = memory_usage(-1, interval=0.1, timeout=1)
        t0 = time.time()

        result = func(*args, **kwargs)

        # stop profiling
        pr.disable()
        elapsed = time.time() - t0
        mem_end = memory_usage(-1, interval=0.1, timeout=1)

        # dump top‐10 by cumulative time
        s = io.StringIO()
        stats = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        stats.print_stats(10)
        print("\n=== cProfile top 10 cumulative ===\n", s.getvalue())

        # simple memory delta
        delta_mem = max(mem_end) - min(mem_start)
        print(f"[PROFILE] {func.__name__} took {elapsed:.3f}s, "
              f"≈{delta_mem:.2f} MiB RAM increase\n")

        return result

    return wrapper
"""

def compare_performance(
    parallel_log_file: str = "appnp_training_log.csv",
    sequential_log_file: str = "appnp_training_log_not_parallel.csv",
    output_file: str = "speedup_analysis.json",
    plot_file: str = "speedup_plot.png"
) -> Dict:
    """
    Compare performance between parallel and sequential runs.
    
    Args:
        parallel_log_file: Path to CSV file with parallel run metrics
        sequential_log_file: Path to CSV file with sequential run metrics
        output_file: Path to save the analysis results as JSON
        plot_file: Path to save the comparison plot
        
    Returns:
        Dictionary with speedup metrics
    """
    # Load data from CSV files
    parallel_data = load_csv_data(parallel_log_file)
    sequential_data = load_csv_data(sequential_log_file)
    
    if not parallel_data or not sequential_data:
        print(f"Error: Could not load data from log files")
        return {}
    
    # Calculate average metrics for stable comparison (exclude first few epochs as warmup)
    warmup = 5
    p_metrics = calculate_avg_metrics(parallel_data[warmup:])
    s_metrics = calculate_avg_metrics(sequential_data[warmup:])
    
    # Print detailed metric comparison for debugging
    print("\n=== Detailed Metric Comparison ===")
    print(f"{'Metric':<15} {'Parallel':<15} {'Sequential':<15} {'Ratio (Seq/Par)':<15}")
    print("-" * 60)
    
    for metric in ["epoch_time", "train_time", "eval_time"]:
        if metric in p_metrics and metric in s_metrics:
            ratio = s_metrics[metric] / p_metrics[metric] if p_metrics[metric] > 0 else 0
            print(f"{metric:<15} {p_metrics[metric]:<15.6f} {s_metrics[metric]:<15.6f} {ratio:<15.6f}")
    
    # Calculate speedups for each metric
    speedups = {
        "epoch_time": s_metrics["epoch_time"] / p_metrics["epoch_time"] if p_metrics["epoch_time"] > 0 else 0,
        "train_time": s_metrics["train_time"] / p_metrics["train_time"] if p_metrics["train_time"] > 0 else 0,
        "eval_time": s_metrics["eval_time"] / p_metrics["eval_time"] if p_metrics["eval_time"] > 0 else 0,
        "forward_time": s_metrics.get("forward_time", 0) / max(p_metrics.get("forward_time", 1), 1e-6),
        "backward_time": s_metrics.get("backward_time", 0) / max(p_metrics.get("backward_time", 1), 1e-6),
        "optimizer_time": s_metrics.get("optimizer_time", 0) / max(p_metrics.get("optimizer_time", 1), 1e-6)
    }
    
    # Check if any speedups are less than 1.0 (parallel slower than sequential)
    if any(s < 1.0 for s in [speedups["epoch_time"], speedups["train_time"], speedups["eval_time"]]):
        print("\n⚠️ WARNING: Some speedups are less than 1.0, meaning the parallel version is SLOWER than sequential!")
        print("This could be due to:")
        print("  - Communication overhead exceeding computation benefits")
        print("  - Inefficient data partitioning")
        print("  - Synchronization bottlenecks")
        print("  - Small dataset where parallelism overhead dominates")
        print("Consider profiling the communication patterns or trying with a larger dataset.")
    
    # Add accuracy comparison
    accuracy_diff = {
        "final_train_acc": parallel_data[-1]["train_acc"] - sequential_data[-1]["train_acc"],
        "final_val_acc": parallel_data[-1]["val_acc"] - sequential_data[-1]["val_acc"],
        "final_test_acc": parallel_data[-1]["test_acc"] - sequential_data[-1]["test_acc"]
    }
    
    # Print accuracy differences
    print("\n=== Accuracy Comparison ===")
    for metric, diff in accuracy_diff.items():
        p_acc = parallel_data[-1][metric.replace("final_", "")]
        s_acc = sequential_data[-1][metric.replace("final_", "")]
        print(f"{metric:<15}: Parallel: {p_acc:.4f}, Sequential: {s_acc:.4f}, Difference: {diff:.4f}")
    
    if any(abs(diff) > 0.01 for diff in accuracy_diff.values()):
        print("\n⚠️ WARNING: There are significant accuracy differences between parallel and sequential runs.")
        print("This could indicate numerical instability or synchronization issues in the parallel implementation.")
    
    # Add memory usage comparison
    memory_ratio = {
        "gpu_mem": p_metrics["gpu_mem"] / s_metrics["gpu_mem"] if s_metrics["gpu_mem"] > 0 else 1.0,
        "cpu_mem": p_metrics["cpu_mem"] / s_metrics["cpu_mem"] if s_metrics["cpu_mem"] > 0 else 1.0
    }
    
    # Add epochs to convergence if available
    best_val_parallel = find_best_val_epoch(parallel_data)
    best_val_sequential = find_best_val_epoch(sequential_data)
    
    convergence = {
        "epochs_to_best_parallel": best_val_parallel["epoch"],
        "epochs_to_best_sequential": best_val_sequential["epoch"],
        "convergence_speedup": best_val_sequential["epoch"] / max(best_val_parallel["epoch"], 1)
    }
    
    # Collect all metrics
    results = {
        "speedups": speedups,
        "accuracy_diff": accuracy_diff,
        "memory_ratio": memory_ratio,
        "convergence": convergence,
        "parallel_avg_metrics": p_metrics,
        "sequential_avg_metrics": s_metrics,
        "parallel_world_size": find_world_size(parallel_log_file)
    }
    
    # Save results to JSON
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create visualization
    create_speedup_plot(results, plot_file)
    
    # Generate a proper file path for the time comparison plot
    time_plot_dir = os.path.dirname(plot_file)
    time_plot_filename = f"epoch_time_comparison_{os.path.basename(plot_file)}"
    time_plot_path = os.path.join(time_plot_dir, time_plot_filename) if time_plot_dir else time_plot_filename
    
    create_time_comparison_plot(parallel_data, sequential_data, time_plot_path)
    
    print("\n=== Performance Comparison Summary ===")
    print(f"Overall Speedup: {speedups['epoch_time']:.2f}x")
    print(f"Training Speedup: {speedups['train_time']:.2f}x")
    print(f"Evaluation Speedup: {speedups['eval_time']:.2f}x")
    if "forward_time" in p_metrics and "forward_time" in s_metrics:
        print(f"Forward Pass Speedup: {speedups['forward_time']:.2f}x")
    print(f"Converged in {convergence['epochs_to_best_parallel']} epochs (parallel) vs "
          f"{convergence['epochs_to_best_sequential']} epochs (sequential)")
    print(f"Test Accuracy Difference: {accuracy_diff['final_test_acc']:.4f}")
    
    return results

def load_csv_data(file_path: str) -> List[Dict]:
    """Load data from a CSV file into a list of dictionaries."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to floats
            numeric_row = {}
            for k, v in row.items():
                try:
                    numeric_row[k] = float(v)
                except (ValueError, TypeError):
                    numeric_row[k] = v
            data.append(numeric_row)
    return data

def calculate_avg_metrics(data: List[Dict]) -> Dict[str, float]:
    """Calculate average values for each metric in the data."""
    if not data:
        return {}
    
    # Initialize with keys from first data point
    sums = {k: 0.0 for k in data[0].keys()}
    
    # Sum all values
    for entry in data:
        for k, v in entry.items():
            try:
                sums[k] += float(v)
            except (ValueError, TypeError):
                pass  # Skip non-numeric fields
    
    # Calculate averages
    avgs = {}
    for k, total in sums.items():
        try:
            avgs[k] = total / len(data)
        except (ZeroDivisionError, TypeError):
            avgs[k] = 0
    
    return avgs

def find_best_val_epoch(data: List[Dict]) -> Dict:
    """Find the epoch with the best validation accuracy."""
    best_val = 0
    best_epoch = None
    
    for entry in data:
        try:
            val_acc = float(entry.get("val_acc", 0))
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = entry
        except (ValueError, TypeError):
            continue
    
    return best_epoch or {}

def find_world_size(parallel_log_file: str) -> int:
    """Extract world size from the log file name."""
    try:
        filename = os.path.basename(parallel_log_file)
        # Look for pattern like "world2" in the filename
        match = re.search(r'world(\d+)', filename)
        if match:
            return int(match.group(1))
    except:
        pass
    
    # Default world size if we can't determine
    return 2  # Assumption based on typical setup

def create_speedup_plot(results: Dict, output_file: str) -> None:
    """Create a bar chart showing speedups for different metrics."""
    speedups = results["speedups"]
    
    # Select metrics to display
    metrics = ["epoch_time", "train_time", "eval_time"]
    if "forward_time" in speedups and speedups["forward_time"] > 0:
        metrics.extend(["forward_time", "backward_time", "optimizer_time"])
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    # Define colors based on speedup (green for speedup, red for slowdown)
    colors = ['g' if speedups[m] >= 1.0 else 'r' for m in metrics]
    
    bars = plt.bar(metrics, [speedups[m] for m in metrics], color=colors)
    
    # Add world size info
    world_size = results.get("parallel_world_size", "unknown")
    plt.title(f"Performance Comparison with Parallel Implementation (World Size: {world_size})")
    plt.ylabel("Speedup Factor (Sequential Time / Parallel Time)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add speedup values on top of bars
    for bar in bars:
        height = bar.get_height()
        # Position the text based on whether it's a speedup or slowdown
        if height >= 1.0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}x', ha='center', va='bottom')
        else:
            # For slowdowns, add a "slower" label
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}x\n({1/height:.1f}x slower)', ha='center', va='bottom', color='r')
    
    # Add horizontal line at y=1 (no speedup/slowdown)
    plt.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Sequential baseline')
    
    # Add a note if overall performance is worse
    if speedups["epoch_time"] < 1.0:
        plt.figtext(0.5, 0.01, 
                   "Note: Values below 1.0 indicate the parallel version is slower than sequential",
                   ha='center', bbox={"facecolor":"yellow", "alpha":0.2, "pad":5})
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file)

def create_time_comparison_plot(
    parallel_data: List[Dict],
    sequential_data: List[Dict],
    output_file: str
) -> None:
    """Create a line plot comparing epoch times between parallel and sequential runs."""
    # Extract epoch times
    p_epochs = [d.get("epoch", i) for i, d in enumerate(parallel_data)]
    p_times = [d.get("epoch_time", 0) for d in parallel_data]
    
    s_epochs = [d.get("epoch", i) for i, d in enumerate(sequential_data)]
    s_times = [d.get("epoch_time", 0) for d in sequential_data]
    
    # Create line plot
    plt.figure(figsize=(12, 6))
    plt.plot(p_epochs, p_times, 'b-', label='Parallel')
    plt.plot(s_epochs, s_times, 'r-', label='Sequential')
    
    plt.title("Epoch Time Comparison: Parallel vs Sequential")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add average time annotations
    avg_p_time = np.mean(p_times)
    avg_s_time = np.mean(s_times)
    speedup = avg_s_time / avg_p_time if avg_p_time > 0 else 0
    
    plt.annotate(f"Avg Parallel: {avg_p_time:.4f}s\nAvg Sequential: {avg_s_time:.4f}s\nSpeedup: {speedup:.2f}x",
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_file)

def strong_scaling_analysis(
    log_files: Dict[int, str],
    output_file: str = "scaling_analysis.json",
    plot_file: str = "scaling_plot.png"
) -> Dict:
    """
    Analyze strong scaling efficiency across different world sizes.
    
    Args:
        log_files: Dictionary mapping world_size to log file path
        output_file: Path to save analysis results
        plot_file: Path to save scaling plot
        
    Returns:
        Dictionary with scaling metrics
    """
    # Load data for each world size
    data_by_size = {}
    for world_size, log_file in log_files.items():
        data = load_csv_data(log_file)
        if data:
            data_by_size[world_size] = calculate_avg_metrics(data[5:])  # Skip warmup epochs
    
    if len(data_by_size) < 2:
        print("Need at least two different world sizes for scaling analysis")
        return {}
    
    # Use world_size=1 as baseline
    baseline = data_by_size.get(1)
    if not baseline and 1 not in log_files:
        # Use the smallest world_size as baseline
        min_size = min(data_by_size.keys())
        baseline = data_by_size[min_size]
        baseline_size = min_size
    else:
        baseline_size = 1
    
    # Calculate speedup and efficiency for each world size
    results = {"baseline_world_size": baseline_size, "scaling": {}}
    
    for world_size, metrics in data_by_size.items():
        if world_size == baseline_size:
            continue
            
        speedup = baseline["epoch_time"] / metrics["epoch_time"]
        efficiency = speedup / (world_size / baseline_size)
        
        results["scaling"][world_size] = {
            "speedup": speedup,
            "efficiency": efficiency,
            "epoch_time": metrics["epoch_time"],
            "train_time": metrics["train_time"],
            "eval_time": metrics["eval_time"]
        }
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create scaling plots
    create_scaling_plot(results, baseline_size, baseline["epoch_time"], plot_file)
    
    print("\n=== Strong Scaling Analysis ===")
    print(f"Baseline: World Size {baseline_size}")
    for world_size, data in sorted(results["scaling"].items()):
        print(f"World Size {world_size}: Speedup {data['speedup']:.2f}x, Efficiency {data['efficiency']:.2f}")
    
    return results

def create_scaling_plot(
    results: Dict,
    baseline_size: int,
    baseline_time: float,
    output_file: str
) -> None:
    """Create plots showing speedup and efficiency vs world size."""
    world_sizes = sorted([int(ws) for ws in results["scaling"].keys()])
    speedups = [results["scaling"][ws]["speedup"] for ws in world_sizes]
    efficiencies = [results["scaling"][ws]["efficiency"] for ws in world_sizes]
    
    # Include baseline point
    all_sizes = [baseline_size] + world_sizes
    all_speedups = [1.0] + speedups
    all_efficiencies = [1.0] + efficiencies
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Speedup (left y-axis)
    ax1.set_xlabel('World Size (Number of Processes)')
    ax1.set_ylabel('Speedup', color='blue')
    line1 = ax1.plot(all_sizes, all_speedups, 'o-', color='blue', label='Speedup')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add ideal speedup line
    ideal_speedups = [size/baseline_size for size in all_sizes]
    ax1.plot(all_sizes, ideal_speedups, '--', color='lightblue', label='Ideal Speedup')
    
    # Efficiency (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Parallel Efficiency', color='red')
    line2 = ax2.plot(all_sizes, all_efficiencies, 'o-', color='red', label='Efficiency')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 1.1])  # Efficiency between 0 and 1
    
    # Add a horizontal line at efficiency=1
    ax2.axhline(y=1, color='lightcoral', linestyle='--', alpha=0.5)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    plt.title(f'Strong Scaling Analysis (Baseline: {baseline_size} process)')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    
    # Create additional plots
    amdahl_plot_path = os.path.join(os.path.dirname(output_file), f"amdahls_fit_{os.path.basename(output_file)}")
    create_amdahls_law_fit(results, baseline_size, baseline_time, amdahl_plot_path)

def create_amdahls_law_fit(
    results: Dict,
    baseline_size: int,
    baseline_time: float,
    output_file: str
) -> None:
    """Create a plot showing measured speedups vs. Amdahl's Law predictions."""
    # Extract world sizes and speedups
    world_sizes = sorted([int(ws) for ws in results["scaling"].keys()])
    all_sizes = [baseline_size] + world_sizes
    all_speedups = [1.0] + [results["scaling"][ws]["speedup"] for ws in world_sizes]
    
    # Fit Amdahl's Law: S(p) = 1 / (s + (1-s)/p) where s is serial fraction
    # Use a simple approach to estimate the serial fraction
    try:
        from scipy.optimize import curve_fit
        
        def amdahls_law(p, s):
            return 1 / (s + (1-s)/p)
        
        # Fit the model
        popt, _ = curve_fit(amdahls_law, all_sizes, all_speedups, bounds=(0, 1))
        serial_fraction = popt[0]
        
        # Generate predictions
        p_range = np.linspace(baseline_size, max(all_sizes)*1.5, 100)
        predicted_speedups = amdahls_law(p_range, serial_fraction)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(all_sizes, all_speedups, 'o', color='blue', label='Measured Speedup')
        plt.plot(p_range, predicted_speedups, '-', color='red', 
                 label=f"Amdahl's Law (s={serial_fraction:.4f})")
        
        # Add ideal speedup
        ideal_speedups = [p/baseline_size for p in p_range]
        plt.plot(p_range, ideal_speedups, '--', color='lightblue', label='Ideal Linear Speedup')
        
        plt.xlabel('World Size (Number of Processes)')
        plt.ylabel('Speedup')
        plt.title("Measured Speedup vs. Amdahl's Law Prediction")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        
        # Print the findings
        print(f"\nAmdahl's Law Analysis:")
        print(f"Estimated Serial Fraction: {serial_fraction:.4f}")
        print(f"Maximum Theoretical Speedup: {1/serial_fraction:.2f}x")
        
    except ImportError:
        print("SciPy not available, skipping Amdahl's Law fit")
    except Exception as e:
        print(f"Error fitting Amdahl's Law: {e}")
